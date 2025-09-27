#!/usr/bin/env python3
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
"""
Hedge MVP CLI (backtest + paper rebalancer + funding snapshot + XGB vol trigger)

Examples:
  # backtest with plots
  python -m src.hedge_mvp.cli backtest --symbol BTC/USDT --perp BTC/USDT:USDT --timeframe 1h --limit 1500 --plot

  # funding rate snapshot
  python -m src.hedge_mvp.cli funding --perp BTC/USDT:USDT --limit 200

  # paper rebalancer (testnet, dry-run logs)
  python -m src.hedge_mvp.cli paper --symbol BTC/USDT --perp BTC/USDT:USDT --timeframe 1h --limit 1000 --rebalance hourly --iterations 3 --sleep 60

  # paper rebalancer + XGB scaling (requires xgboost)
  python -m src.hedge_mvp.cli paper --symbol BTC/USDT --perp BTC/USDT:USDT --timeframe 1h --limit 1500 --xgb --rebalance hourly --iterations 3 --sleep 60
"""
import argparse, os, json, datetime as dt, time
import pandas as pd
import numpy as np
import ccxt

# %%
from .core import (
    ensure_dir, fetch_ohlcv, fetch_funding_rates, align_close, compute_log_returns,
    estimate_ols_hedge_ratio, backtest_static_hedge, sharpe_ratio, max_drawdown_from_returns,
    infer_periods_per_year, plot_series, plot_cumlogret, HedgeResults,
    build_ml_vol_features, train_xgb_vol_model, predict_next_vol, scale_hedge_ratio,
    init_binance, intended_rebalance_log, realized_vol
)

# %% [markdown]
# ---------- subcommands ----------

# %%
def cmd_backtest(args):
    ex = ccxt.binance()
    spot_df = fetch_ohlcv(ex, args.symbol, timeframe=args.timeframe, limit=args.limit)
    perp_df = fetch_ohlcv(ex, args.perp, timeframe=args.timeframe, limit=args.limit)
    spot_close, perp_close = align_close(spot_df, perp_df)
    r_spot = compute_log_returns(spot_close)
    r_perp = compute_log_returns(perp_close)

    beta = estimate_ols_hedge_ratio(r_spot, r_perp)
    r_spot_al, r_hedged = backtest_static_hedge(r_spot, r_perp, beta)

    var_spot = float(r_spot_al.var())
    var_hedged = float(r_hedged.var())
    variance_reduction = 1.0 - (var_hedged/var_spot) if var_spot>0 else 0.0
    periods = infer_periods_per_year(args.timeframe)
    sr_spot = sharpe_ratio(r_spot_al, periods)
    sr_hedged = sharpe_ratio(r_hedged, periods)
    mdd_spot = max_drawdown_from_returns(r_spot_al)
    mdd_hedged = max_drawdown_from_returns(r_hedged)

    # outdir
    sym = args.symbol.replace("/","").replace(":","")
    ts = dt.datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    outdir = os.path.join("runs", sym, "hedge_mvp", ts)
    ensure_dir(outdir)

    # save
    pd.DataFrame({"spot_close": spot_close, "perp_close": perp_close}).to_csv(os.path.join(outdir, "prices.csv"))
    pd.DataFrame({"r_spot": r_spot_al, "r_hedged": r_hedged}).to_csv(os.path.join(outdir, "returns.csv"))
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump({
            "timeframe": args.timeframe, "samples": int(len(r_spot_al)),
            "hedge_ratio": float(beta),
            "variance_spot": var_spot, "variance_hedged": var_hedged,
            "variance_reduction": variance_reduction,
            "sharpe_spot": sr_spot, "sharpe_hedged": sr_hedged,
            "maxdd_spot": mdd_spot, "maxdd_hedged": mdd_hedged
        }, f, indent=2)

    if args.plot:
        plot_series(spot_close, perp_close, outdir)
        plot_cumlogret(r_spot_al, r_hedged, outdir)

    print("=== Backtest Summary ===")
    print(f"Outdir: {outdir}")
    print(f"Hedge Ratio (β): {beta:.4f}")
    print(f"Variance Reduction: {variance_reduction:.2%}")
    print(f"Sharpe Spot/Hedged: {sr_spot:.3f} / {sr_hedged:.3f}")
    print(f"MaxDD  Spot/Hedged: {mdd_spot:.2%} / {mdd_hedged:.2%}")

# %%
def cmd_funding(args):
    ex = ccxt.binance()
    df = fetch_funding_rates(ex, args.perp, limit=args.limit)
    sym = args.perp.replace("/","").replace(":","")
    ts = dt.datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    outdir = os.path.join("runs", sym, "hedge_mvp", ts)
    ensure_dir(outdir)
    path = os.path.join(outdir, "funding_rates.csv")
    df.to_csv(path)
    print(f"Saved funding rates to: {path} (rows={len(df)})")

# %%
def cmd_paper(args):
    # testnet futures client (dry-run unless --execute)
    ex = init_binance(
        testnet=True,
        api_key=os.getenv("BINANCE_TESTNET_API_KEY"),
        secret=os.getenv("BINANCE_TESTNET_SECRET"),
    )

    sym = args.symbol.replace("/","").replace(":","")
    ts_root = dt.datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    outroot = os.path.join("runs", sym, "hedge_mvp", ts_root)
    ensure_dir(outroot)
    log_path = os.path.join(outroot, "paper_rebalance_log.jsonl")

    # initial data & model
    spot_df = fetch_ohlcv(ccxt.binance(), args.symbol, timeframe=args.timeframe, limit=args.limit)
    perp_df = fetch_ohlcv(ccxt.binance(), args.perp, timeframe=args.timeframe, limit=args.limit)
    spot_close, perp_close = align_close(spot_df, perp_df)
    r_spot = compute_log_returns(spot_close)
    r_perp = compute_log_returns(perp_close)
    beta = estimate_ols_hedge_ratio(r_spot, r_perp)

    model = None
    features = None
    vol_low = 0.0
    vol_high = 0.0
    if args.xgb:
        df_feat = build_ml_vol_features(r_spot, 48)
        if len(df_feat) > 200:
            model, features = train_xgb_vol_model(df_feat)
            # Calibrate bounds from recent realized vol
            recent_vol = df_feat["target_vol"].tail(500)
            vol_low = float(np.nanpercentile(recent_vol, 20))
            vol_high = float(np.nanpercentile(recent_vol, 80))
        else:
            print("Not enough data to train XGB; running without ML scaling.")

    # simple loop: recompute beta each iteration, scale with ML if enabled
    for i in range(int(args.iterations)):
        # refresh data window
        spot_df = fetch_ohlcv(ccxt.binance(), args.symbol, timeframe=args.timeframe, limit=args.limit)
        perp_df = fetch_ohlcv(ccxt.binance(), args.perp, timeframe=args.timeframe, limit=args.limit)
        spot_close, perp_close = align_close(spot_df, perp_df)
        r_spot = compute_log_returns(spot_close)
        r_perp = compute_log_returns(perp_close)
        beta = estimate_ols_hedge_ratio(r_spot, r_perp)

        scaled_beta = beta
        pred_vol = None
        if args.xgb and model is not None:
            df_feat = build_ml_vol_features(r_spot, 48)
            latest_row = df_feat.iloc[-1]
            pred_vol = predict_next_vol(model, features, latest_row)
            scaled_beta = scale_hedge_ratio(beta, pred_vol, vol_low, vol_high,
                                            scale_min=args.scale_min, scale_max=args.scale_max)

        # In a real system, we would read current perp position and compute delta to target.
        # Here we only LOG the intended order (dry-run) unless --execute is passed.
        payload = {
            "ts": dt.datetime.utcnow().isoformat(),
            "symbol_spot": args.symbol,
            "symbol_perp": args.perp,
            "timeframe": args.timeframe,
            "beta": float(beta),
            "scaled_beta": float(scaled_beta),
            "pred_vol": None if pred_vol is None else float(pred_vol),
            "note": "dry-run (no order placed)" if not args.execute else "execute requested (not implemented)"
        }
        intended_rebalance_log(log_path, payload)
        print(f"[{i+1}/{args.iterations}] logged target hedge β={beta:.4f}, scaled={scaled_beta:.4f}  -> {log_path}")

        # sleep until next rebalance tick
        time.sleep(float(args.sleep))

# %%
def main():
    ap = argparse.ArgumentParser(description="Hedge MVP CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # backtest
    b = sub.add_parser("backtest")
    b.add_argument("--symbol", default="BTC/USDT")
    b.add_argument("--perp", default="BTC/USDT:USDT")
    b.add_argument("--timeframe", default="1h")
    b.add_argument("--limit", type=int, default=1500)
    b.add_argument("--plot", action="store_true")
    b.set_defaults(func=cmd_backtest)

    # funding snapshot
    f = sub.add_parser("funding")
    f.add_argument("--perp", default="BTC/USDT:USDT")
    f.add_argument("--limit", type=int, default=200)
    f.set_defaults(func=cmd_funding)

    # paper rebalancer
    p = sub.add_parser("paper")
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--perp", default="BTC/USDT:USDT")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--limit", type=int, default=1500)
    p.add_argument("--rebalance", choices=["hourly","daily"], default="hourly")
    p.add_argument("--iterations", type=int, default=3, help="how many rebalance cycles to run")
    p.add_argument("--sleep", type=float, default=60, help="seconds between iterations")
    p.add_argument("--xgb", action="store_true", help="enable XGBoost vol trigger to scale hedge")
    p.add_argument("--scale_min", type=float, default=0.3)
    p.add_argument("--scale_max", type=float, default=1.2)
    p.add_argument("--execute", action="store_true", help="PLACE ORDERS (not implemented yet) – for now only logs")
    p.set_defaults(func=cmd_paper)

    args = ap.parse_args()
    args.func(args)

# %%
if __name__ == "__main__":
    main()
