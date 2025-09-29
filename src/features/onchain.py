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
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# src/features/onchain.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence, Dict
import json
import pandas as pd
import numpy as np

HOUR = "1H"

def _read_parquet(path: Optional[Path], cols: Optional[Sequence[str]] = None) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    df = pd.read_parquet(path)
    if cols:
        missing = set(cols) - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")
        df = df[list(cols)]
    return df

def _ensure_ts_index(df: pd.DataFrame, ts_col: str, freq: str = HOUR) -> pd.DataFrame:
    out = df.copy()
    out[ts_col] = pd.to_datetime(out[ts_col], utc=True)
    out = out.set_index(ts_col).sort_index()
    return out

def _resample_sum(df: pd.DataFrame, freq: str = HOUR) -> pd.DataFrame:
    return df.resample(freq).sum(min_count=1)

def _resample_mean(df: pd.DataFrame, freq: str = HOUR) -> pd.DataFrame:
    return df.resample(freq).mean()

def _resample_median(df: pd.DataFrame, freq: str = HOUR) -> pd.DataFrame:
    return df.resample(freq).median()

def _resample_quantile(df: pd.DataFrame, q: float, freq: str = HOUR) -> pd.DataFrame:
    return df.resample(freq).quantile(q)

def _zscore_rolling(x: pd.Series, window: str = "168H", min_periods: int = 24) -> pd.Series:
    # 7d = 168 hours
    mean = x.rolling(window, min_periods=min_periods).mean()
    std = x.rolling(window, min_periods=min_periods).std(ddof=0)
    return (x - mean) / std

def _load_labels(path: Optional[Path]) -> Dict[str, Dict]:
    if path is None:
        return {}
    with open(path, "r") as f:
        return json.load(f)

def _is_cex(addr: str, labels: Dict[str, Dict]) -> bool:
    meta = labels.get(addr.lower())
    return bool(meta and meta.get("is_exchange", False))

def _align_index(df: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
    return df.reindex(idx)

def build_onchain_mv_feature_set(
    ts_start: str | pd.Timestamp,
    ts_end: str | pd.Timestamp,
    freq: str = HOUR,
    # inputs
    transfers_path: Optional[Path] = None,
    labels_exchanges_path: Optional[Path] = None,
    uniswap_swaps_path: Optional[Path] = None,
    eth_blocks_path: Optional[Path] = None,
    address_first_seen_path: Optional[Path] = None,
    stable_tokens: Sequence[str] = ("USDT", "USDC", "DAI"),
) -> pd.DataFrame:
    """
    Returns an hourly DataFrame with the 6 minimum-viable on-chain features (plus a few helpful extras).
    Missing inputs are skipped gracefully (columns filled with NaN, index still returned).
    """
    ts_start = pd.to_datetime(ts_start, utc=True)
    ts_end = pd.to_datetime(ts_end, utc=True)
    idx = pd.date_range(ts_start.floor(freq), ts_end.ceil(freq), freq=freq, tz="UTC", inclusive="left")
    out = pd.DataFrame(index=idx)

    # ==== Labels (exchanges) ====
    labels = _load_labels(labels_exchanges_path)

    # ==== 1) Exchange netflows: BTC/ETH native + 2) Stablecoin netflows ====
    if transfers_path is not None:
        tcols = ["ts", "chain", "token", "from_address", "to_address", "amount", "amount_usd"]
        tr = _read_parquet(transfers_path, tcols)
        tr = _ensure_ts_index(tr, "ts")
        tr = tr.loc[ts_start:ts_end]

        # Flag CEX direction
        tr["from_is_cex"] = tr["from_address"].str.lower().map(lambda a: _is_cex(a, labels))
        tr["to_is_cex"] = tr["to_address"].str.lower().map(lambda a: _is_cex(a, labels))

        # BTC/ETH native netflows (in - out)
        # Convention: rows for native should have token == "BTC" or "ETH" with amount in native units.
        def _netflow(token_sym: str) -> pd.Series:
            df = tr.query("token == @token_sym")
            inflow = df.loc[df["to_is_cex"], ["amount"]].rename(columns={"amount": "inflow"})
            outflow = df.loc[df["from_is_cex"], ["amount"]].rename(columns={"amount": "outflow"})
            inflow = _resample_sum(inflow, freq).rename(columns={"inflow": "inflow"})
            outflow = _resample_sum(outflow, freq).rename(columns={"outflow": "outflow"})
            nf = inflow.join(outflow, how="outer")
            return (nf["inflow"].fillna(0) - nf["outflow"].fillna(0)).rename(f"ex_netflow_{token_sym.lower()}")

        try:
            out["ex_netflow_btc"] = _align_index(_netflow("BTC").to_frame(), idx)["ex_netflow_btc"]
        except Exception:
            out["ex_netflow_btc"] = np.nan
        try:
            out["ex_netflow_eth"] = _align_index(_netflow("ETH").to_frame(), idx)["ex_netflow_eth"]
        except Exception:
            out["ex_netflow_eth"] = np.nan

        # Stablecoin USD netflows (sum of tokens)
        st = tr[tr["token"].isin(stable_tokens)]
        st_in = st.loc[st["to_is_cex"], ["amount_usd"]]
        st_out = st.loc[st["from_is_cex"], ["amount_usd"]]
        st_in = _resample_sum(st_in, freq).rename(columns={"amount_usd": "in_usd"})
        st_out = _resample_sum(st_out, freq).rename(columns={"amount_usd": "out_usd"})
        st_nf = st_in.join(st_out, how="outer")
        out["stables_netflow_cex_usd"] = _align_index(
            (st_nf["in_usd"].fillna(0) - st_nf["out_usd"].fillna(0)).to_frame(), idx
        )["in_usd"].fillna(0) - _align_index(st_nf["out_usd"].to_frame(), idx)["out_usd"].fillna(0)

        # 5) Whale bursts (≥ $1M touching CEX)
        whale = st[(st["amount_usd"] >= 1_000_000) & (st[["from_is_cex", "to_is_cex"]].any(axis=1))]
        whale_cnt = _resample_sum(whale.assign(one=1)[["one"]], freq).rename(columns={"one": "whale_cex_tx_count_gt_1m"})
        whale_usd = _resample_sum(whale[["amount_usd"]], freq).rename(columns={"amount_usd": "whale_cex_tx_usd_gt_1m"})
        out = out.join(_align_index(whale_cnt, idx)).join(_align_index(whale_usd, idx))

        # 6) Address activity breadth
        # Unique active addresses per hour (senders U receivers)
        active = (
            tr.assign(hour=lambda d: d.index.floor(freq))
              .groupby("hour")
              .apply(lambda g: pd.Series({
                  "active_addresses_hourly": pd.unique(pd.concat([g["from_address"], g["to_address"]])).size
              }))
              .sort_index()
        )
        out = out.join(_align_index(active, idx))

        # New addresses per hour (needs first_seen). If not provided, approximate by first occurrence in window.
        if address_first_seen_path is not None:
            fs = _read_parquet(address_first_seen_path, ["address", "first_ts"])
            fs["first_ts"] = pd.to_datetime(fs["first_ts"], utc=True).dt.floor(freq)
            new_counts = fs[(fs["first_ts"] >= idx[0]) & (fs["first_ts"] < idx[-1] + pd.Timedelta(freq))] \
                .groupby("first_ts").size().rename("new_addresses_hourly").to_frame()
            new_counts.index.name = None
            out = out.join(_align_index(new_counts, idx))
        else:
            # Approximate: first time seen within the requested window only
            seen = set()
            new_per_hour = []
            for t in idx:
                g = tr.loc[t:t + pd.Timedelta(freq) - pd.Timedelta("1ns"), ["from_address", "to_address"]]
                addrs = pd.unique(pd.concat([g["from_address"], g["to_address"]])).tolist()
                new_now = sum(1 for a in addrs if a not in seen)
                seen.update(addrs)
                new_per_hour.append(new_now)
            out["new_addresses_hourly"] = new_per_hour

    else:
        # No transfers → fill feature columns with NaN
        out["ex_netflow_btc"] = np.nan
        out["ex_netflow_eth"] = np.nan
        out["stables_netflow_cex_usd"] = np.nan
        out["whale_cex_tx_count_gt_1m"] = np.nan
        out["whale_cex_tx_usd_gt_1m"] = np.nan
        out["active_addresses_hourly"] = np.nan
        out["new_addresses_hourly"] = np.nan

    # ==== 3) DEX swap intensity & price impact ====
    if uniswap_swaps_path is not None:
        scol = ["ts", "amount_usd", "mid_before", "mid_after"]
        sw = _read_parquet(uniswap_swaps_path, scol)
        sw = _ensure_ts_index(sw, "ts")
        sw = sw.loc[ts_start:ts_end]
        sw["impact_bps"] = (sw["mid_after"] - sw["mid_before"]).abs() / sw["mid_before"].replace(0, np.nan) * 1e4
        notional = _resample_sum(sw[["amount_usd"]], freq).rename(columns={"amount_usd": "dex_swap_notional_usd"})
        impact = _resample_mean(sw[["impact_bps"]], freq).rename(columns={"impact_bps": "dex_price_impact_bps"})
        out = out.join(_align_index(notional, idx)).join(_align_index(impact, idx))
    else:
        out["dex_swap_notional_usd"] = np.nan
        out["dex_price_impact_bps"] = np.nan

    # ==== 4) Gas/fee pressure ====
    if eth_blocks_path is not None:
        bcol = ["ts_block", "basefee_gwei", "priority_fee_gwei", "gas_used", "gas_limit"]
        bl = _read_parquet(eth_blocks_path, bcol)
        bl = _ensure_ts_index(bl, "ts_block")
        bl = bl.loc[ts_start:ts_end]
        bl["util"] = bl["gas_used"] / bl["gas_limit"].replace(0, np.nan)
        basefee = _resample_median(bl[["basefee_gwei"]], freq)
        tip_p90 = _resample_quantile(bl[["priority_fee_gwei"]], 0.90, freq).rename(columns={"priority_fee_gwei": "gas_tip_p90_gwei"})
        pct_full = (
            bl.assign(full=lambda d: (d["util"] >= 0.95).astype(int))[["full"]]
              .resample(freq).mean().rename(columns={"full": "pct_blocks_full"})
        )
        out = out.join(_align_index(basefee, idx)).join(_align_index(tip_p90, idx)).join(_align_index(pct_full, idx))
    else:
        out["basefee_gwei"] = np.nan
        out["gas_tip_p90_gwei"] = np.nan
        out["pct_blocks_full"] = np.nan

    # ==== Z-scores (7-day) for a few key pressure series ====
    for col in ["ex_netflow_btc", "ex_netflow_eth", "stables_netflow_cex_usd",
                "dex_swap_notional_usd", "dex_price_impact_bps",
                "whale_cex_tx_count_gt_1m", "active_addresses_hourly",
                "basefee_gwei", "pct_blocks_full"]:
        if col in out.columns:
            out[f"{col}_z7d"] = _zscore_rolling(out[col])

    return out.sort_index()
