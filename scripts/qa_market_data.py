# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     notebook_metadata_filter: jupytext,text_representation,kernelspec
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
# ---

# %%
import argparse, json, math
from pathlib import Path
import pandas as pd, numpy as np

# %%
def pct(n, d): return float(n)/d*100 if d else 0.0

# %%
def load_parquet(path, ts_col=None):
    df = pd.read_parquet(path)
    if ts_col and ts_col in df.columns:
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
        df = df.set_index(ts_col).sort_index()
    elif isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        df = df.sort_index()
    return df

# %%
def ohlcv_checks(df):
    probs = []
    cols = set(df.columns)
    needed = {"open","high","low","close","volume"}
    missing = sorted(list(needed - cols))
    if missing: probs.append(f"Missing OHLCV columns: {missing}")

    if {"open","high","low","close"} <= cols:
        bad_high = (df["high"] < df[["open","close","low"]].max(axis=1)).sum()
        bad_low  = (df["low"]  > df[["open","close","high"]].min(axis=1)).sum()
        if bad_high: probs.append(f"{bad_high} bars have high < max(open,close,low)")
        if bad_low:  probs.append(f"{bad_low} bars have low  > min(open,close,high)")
        neg_px = (df[["open","high","low","close"]] <= 0).sum().sum()
        if neg_px: probs.append(f"{neg_px} non-positive price fields")

    if "volume" in cols:
        neg_v = (df["volume"] < 0).sum()
        if neg_v: probs.append(f"{neg_v} negative volume rows")
    return probs

# %%
def gap_dup_checks(idx, expected_freq=None):
    out = {}
    out["tz_aware"] = idx.tz is not None
    out["start"] = str(idx.min()) if len(idx) else None
    out["end"] = str(idx.max()) if len(idx) else None
    out["rows"] = int(len(idx))
    out["dupes"] = int(idx.duplicated().sum())
    if expected_freq:
        full = pd.date_range(idx.min(), idx.max(), freq=expected_freq, tz="UTC") if len(idx) else pd.DatetimeIndex([], tz="UTC")
        out["expected_bars"] = int(len(full))
        out["missing_bars"] = int(len(full.difference(idx)))
        out["missing_pct"] = pct(out["missing_bars"], out["expected_bars"]) if out["expected_bars"] else 0.0
    return out

# %%
def jump_checks(close, label):
    if close.isnull().all():
        return { "label": label, "note": "close all NA" }
    ret = np.log(close).diff()
    q = ret.quantile([0.01,0.05,0.5,0.95,0.99]).to_dict()
    big = ret.abs() > 0.15  # >15% log-move in one bar
    return {
        "label": label,
        "ret_count": int(ret.notna().sum()),
        "ret_abs_gt15pct": int(big.sum()),
        "quantiles": {str(k): float(v) for k,v in q.items()}
    }

# %%
def basis_checks(spot_c, perp_c):
    df = pd.concat({"spot": spot_c, "perp": perp_c}, axis=1).dropna()
    if df.empty: return {"rows": 0}
    basis = (df["perp"]/df["spot"] - 1.0)
    out = {
        "rows": int(len(df)),
        "basis_mean_bp": float(basis.mean()*1e4),
        "basis_p5_bp": float(basis.quantile(0.05)*1e4),
        "basis_p95_bp": float(basis.quantile(0.95)*1e4),
        "basis_abs_gt_200bp": int((basis.abs()>0.02).sum())
    }
    return out

# %%
def funding_checks(funding_df):
    # Expect Binance perp funding every 8h; check spacing and range
    out = {"rows": int(len(funding_df))}
    if "funding_rate" in funding_df.columns:
        fr = funding_df["funding_rate"].dropna()
        out["rate_min_bp"] = float(fr.min()*1e4) if len(fr) else None
        out["rate_max_bp"] = float(fr.max()*1e4) if len(fr) else None
        # spacing
        if isinstance(funding_df.index, pd.DatetimeIndex) and len(funding_df)>1:
            dt = funding_df.index.to_series().diff().dropna().dt.total_seconds()/3600
            out["spacing_hours_top3"] = list(map(float, dt.value_counts().head(3).index))
            out["pct_spacing_8h"] = float((dt.round()==8).mean()*100)
    return out

# %%
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spot",   required=True, help="spot parquet path (OHLCV)")
    ap.add_argument("--perp",   required=True, help="perp parquet path (OHLCV)")
    ap.add_argument("--funding",required=True, help="Binance perp funding parquet")
    ap.add_argument("--timeframe", default="5min", help="expected bar freq for spot/perp (e.g., 5min, 1h)")
    ap.add_argument("--outdir", default="runs/_qa", help="output directory")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    spot = load_parquet(args.spot)
    perp = load_parquet(args.perp)
    fund = load_parquet(args.funding, ts_col="timestamp" if "timestamp" in pd.read_parquet(args.funding).columns else None)

    # Basic shape & gaps
    spot_gap = gap_dup_checks(spot.index, expected_freq=args.timeframe)
    perp_gap = gap_dup_checks(perp.index, expected_freq=args.timeframe)
    fund_gap = gap_dup_checks(fund.index)

    # OHLCV sanity
    spot_prob = ohlcv_checks(spot)
    perp_prob = ohlcv_checks(perp)

    # Jump anomalies
    spot_jump = jump_checks(spot.get("close"), "spot")
    perp_jump = jump_checks(perp.get("close"), "perp")

    # Spot vs Perp basis
    basis = basis_checks(spot.get("close"), perp.get("close"))

    # Funding
    funding = funding_checks(fund)

    # Compose report
    summary = {
        "spot_path": args.spot,
        "perp_path": args.perp,
        "funding_path": args.funding,
        "timeframe_expectation": args.timeframe,
        "spot_gaps": spot_gap,
        "perp_gaps": perp_gap,
        "funding_gaps": fund_gap,
        "spot_problems": spot_prob,
        "perp_problems": perp_prob,
        "spot_jumps": spot_jump,
        "perp_jumps": perp_jump,
        "basis": basis,
        "funding": funding,
    }

    # Write JSON + Markdown
    (outdir / "dq_report.json").write_text(json.dumps(summary, indent=2))

    md = []
    md.append("# Data Quality Report\n")
    md.append(f"**Spot:** `{args.spot}`  \n**Perp:** `{args.perp}`  \n**Funding:** `{args.funding}`  \n")
    md.append("\n## 1) Time index & gaps\n")
    for name, g in [("Spot", spot_gap), ("Perp", perp_gap), ("Funding", fund_gap)]:
        line = f"- **{name}**: rows={g.get('rows')} tz_aware={g.get('tz_aware')} start={g.get('start')} end={g.get('end')}"
        if "expected_bars" in g:
            line += f" expected={g['expected_bars']} missing={g['missing_bars']} ({g.get('missing_pct',0):.2f}%)"
        line += f" dupes={g.get('dupes',0)}"
        md.append(line)
    md.append("\n## 2) OHLCV sanity\n")
    md.append(f"- Spot: {'; '.join(spot_prob) if spot_prob else 'OK'}")
    md.append(f"- Perp: {'; '.join(perp_prob) if perp_prob else 'OK'}")
    md.append("\n## 3) Return jumps (log)\n")
    md.append(f"- Spot: {spot_jump}")
    md.append(f"- Perp: {perp_jump}")
    md.append("\n## 4) Spot vs Perp basis\n")
    md.append(f"- {basis}")
    md.append("\n## 5) Funding cadence & range\n")
    md.append(f"- {funding}")
    (outdir / "dq_report.md").write_text("\n".join(md))
    print(f"Wrote: {outdir/'dq_report.md'} and {outdir/'dq_report.json'}")

# %%
if __name__ == "__main__":
    main()
