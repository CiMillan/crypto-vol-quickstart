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
import argparse, json
import numpy as np
import pandas as pd
from pathlib import Path

# %%
def max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())

# %%
def backtest(pred_df: pd.DataFrame, ret_series: pd.Series, fee_bps: float = 1.0):
    df = pred_df.copy()
    df = df.join(ret_series, how="left").dropna(subset=["y_pred", "ret"])

    eps = 1e-8
    tgt_vol = df["y_pred"].rolling(50).median().bfill()
    w = (tgt_vol.median() / (df["y_pred"] + eps)).clip(0, 5.0)

    gross_ret = w * df["ret"]
    turn = w.diff().abs().fillna(0.0)
    fees = turn * (fee_bps / 10000.0)
    strat_ret = gross_ret - fees

    ann = np.sqrt(252 * 24 * 12)  # 5m bars
    sharpe = float(ann * strat_ret.mean() / (strat_ret.std() + 1e-12))
    sortino = float(ann * strat_ret.mean() / (strat_ret[strat_ret < 0].std() + 1e-12))

    stats = {
        "sharpe": sharpe,
        "sortino": sortino,
        "mdd": max_drawdown(strat_ret),
        "avg_turnover": float(turn.mean()),
        "n_obs": int(len(strat_ret)),
    }
    detail = pd.DataFrame({"w": w, "gross_ret": gross_ret, "fees": fees, "strat_ret": strat_ret})
    return stats, detail

# %%
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="predictions.parquet from modeling")
    ap.add_argument("--features", required=True, help="processed features parquet (for real returns)")
    ap.add_argument("--retcol", default="ret_5")
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="p_xgb")
    ap.add_argument("--fee_bps", type=float, default=1.0)
    args = ap.parse_args()

    pred = pd.read_parquet(args.pred).sort_values("ts")      # 'ts' written by modeling
    feat = pd.read_parquet(args.features).sort_index()       # features indexed by timestamp

    merged = pred.set_index("ts").join(feat[[args.retcol]], how="left")
    merged = merged.rename(columns={args.retcol: "ret"})
    merged["y_pred"] = merged[args.model]

    stats, detail = backtest(merged, merged["ret"], fee_bps=args.fee_bps)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(stats, f, indent=2)
    detail.to_parquet(args.output.replace(".json", "_detail.parquet"))
    print(json.dumps(stats, indent=2))

# %%
if __name__ == "__main__":
    main()
