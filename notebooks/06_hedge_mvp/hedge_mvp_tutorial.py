# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     notebook_metadata_filter: jupytext,text_representation,kernelspec
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: crypto-vol (.venv)
#     language: python
#     name: crypto-vol
# ---

# %% [markdown]
# # Hedge MVP — Step-by-step Tutorial (Spot + Perp, OLS Hedge, Funding, ML Scaling)
#
# **Goal:** running each function in isolation, checking the output, and visualizing as much as possible. This notebook uses your module `src/hedge_mvp/core.py` as the single source of truth.
#
# **Covered:**
# 1. Fetch spot & perp OHLCV (`fetch_ohlcv`)
# 2. Align series (`align_close`)
# 3. Log returns (`compute_log_returns`)
# 4. Hedge ratio via OLS (`estimate_ols_hedge_ratio`)
# 5. Static hedge backtest (`backtest_static_hedge`) + metrics (variance reduction, Sharpe, MaxDD)
# 6. Funding rates (`fetch_funding_rates`) quick QA
# 7. ML features (`build_ml_vol_features`), optional XGB vol prediction, and `scale_hedge_ratio`
# 8. Save artifacts to `runs/.../hedge_mvp_tutorial/<TS>/`

# %% [markdown]
# ## 00) Version diagnostic

# %%
import sys, subprocess, textwrap
print("Python:", sys.version)
print("Executable:", sys.executable)

def _pip_freeze_top(n=10):
    out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    print("\nTop of pip freeze:\n", "\n".join(out.splitlines()[:n]))
try:
    import numpy, pandas, matplotlib, statsmodels, ccxt
    print("numpy:", numpy.__version__)
    print("pandas:", pandas.__version__)
    print("matplotlib:", matplotlib.__version__)
    import statsmodels.api as sm; print("statsmodels:", sm.__version__)
    import ccxt; print("ccxt:", ccxt.__version__)
except Exception as e:
    print("Import check failed:", e)
    _pip_freeze_top()

# %%
import os, sys, subprocess, time

# 1) Speed up matplotlib import (must be set BEFORE importing matplotlib)
os.environ.setdefault("MPLBACKEND", "Agg")       # non-GUI backend
os.environ.setdefault("MPLCONFIGDIR", ".mpl")    # writable cache in repo

t0 = time.perf_counter()
print("Python:", sys.version.split()[0])
print("Executable:", sys.executable)

def t(msg):
    print(f"[+{time.perf_counter()-t0:5.2f}s] {msg}")

try:
    t("importing numpy ...");      import numpy as np;            t(f"numpy {np.__version__} ok")
    t("importing pandas ...");     import pandas as pd;           t(f"pandas {pd.__version__} ok")
    t("importing matplotlib ..."); import matplotlib;             t(f"matplotlib {matplotlib.__version__} ok")
    t("importing statsmodels ...");import statsmodels.api as sm;  t(f"statsmodels {sm.__version__} ok")
    t("importing ccxt ...");       import ccxt;                   t(f"ccxt {ccxt.__version__} ok")
except Exception as e:
    print("Import check failed:", repr(e))
    # MUCH faster than 'pip freeze':
    try:
        out = subprocess.check_output([sys.executable, "-m", "pip", "list", "--format=freeze"], text=True, timeout=10)
        print("\nTop of pip list:\n", "\n".join(out.splitlines()[:10]))
    except Exception as ee:
        print("pip list also failed:", repr(ee))


# %% [markdown]
# ## 0) Setup & sanity checks

# %%
import os, json, math, datetime as dt
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

# Libs for runtime display
import importlib, sys
from pathlib import Path

# Project imports — our core functions live here
from src.hedge_mvp.core import (
    ensure_dir, fetch_ohlcv, fetch_funding_rates, align_close, compute_log_returns,
    estimate_ols_hedge_ratio, backtest_static_hedge, sharpe_ratio, max_drawdown_from_returns,
    infer_periods_per_year, plot_series, plot_cumlogret,
    build_ml_vol_features, train_xgb_vol_model, predict_next_vol, scale_hedge_ratio,
)

# Basic display opts
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

# Versions (helpful for reproducibility)
import ccxt, statsmodels
print("Python:", sys.version.split()[0])
print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("matplotlib:", plt.matplotlib.__version__)
print("ccxt:", ccxt.__version__)
print("statsmodels:", statsmodels.__version__)

# Try XGBoost (optional)
try:
    import xgboost
    HAS_XGB = True
    print("xgboost:", xgboost.__version__)
except Exception:
    HAS_XGB = False
    print("xgboost: NOT INSTALLED (ML step will be skipped)")

# %% [markdown]
# ## 1) Configuration
# Pick symbols and timeframe. We start with BTC/USDT and 1h bars for quick iteration.

# %%
SYMBOL_SPOT = "BTC/USDT"
SYMBOL_PERP = "BTC/USDT:USDT"   # Binance USD-M perp routing via ccxt.binance()
TIMEFRAME   = "1h"
LIMIT       = 1500              # ~62 days of hourly data

# Where tutorial artifacts will be saved
RUNS_SYM = SYMBOL_SPOT.replace("/","").replace(":","")
TS_STR   = dt.datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
OUTDIR   = os.path.join("runs", RUNS_SYM, "hedge_mvp_tutorial", TS_STR)
ensure_dir(OUTDIR)
OUTDIR

# %% [markdown]
# ## 2) Fetch Spot OHLCV (`fetch_ohlcv`)
# - Pulls candles via `ccxt.binance().fetch_ohlcv`
# - Returns a DataFrame indexed by UTC timestamps with columns: open, high, low, close, volume

# %%
import ccxt
ex = ccxt.binance()
spot_df = fetch_ohlcv(ex, SYMBOL_SPOT, timeframe=TIMEFRAME, limit=LIMIT)
print("spot_df shape:", spot_df.shape)
display(spot_df.head(3))
display(spot_df.tail(3))
print("Index tz-aware:", spot_df.index.tz is not None)
print("Time span:", spot_df.index.min(), "→", spot_df.index.max())
print("Any NA? ", spot_df.isna().any().to_dict())

# Visualize spot close
plt.figure()
spot_df["close"].plot()
plt.title(f"{SYMBOL_SPOT} Close ({TIMEFRAME})")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 3) Fetch Perp OHLCV (`fetch_ohlcv`) and quick QA

# %%
perp_df = fetch_ohlcv(ex, SYMBOL_PERP, timeframe=TIMEFRAME, limit=LIMIT)
print("perp_df shape:", perp_df.shape)
display(perp_df.head(3))
display(perp_df.tail(3))
print("Index tz-aware:", perp_df.index.tz is not None)
print("Time span:", perp_df.index.min(), "→", perp_df.index.max())
print("Any NA? ", perp_df.isna().any().to_dict())

# Visualize perp close
plt.figure()
perp_df["close"].plot()
plt.title(f"{SYMBOL_PERP} Close ({TIMEFRAME})")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 4) Align series (`align_close`)
# - Ensures both series share the same timestamps.
# - We verify lengths & equality of indices.

# %%
spot_close, perp_close = align_close(spot_df, perp_df)
print("Aligned lengths:", len(spot_close), len(perp_close))
print("Same index?", spot_close.index.equals(perp_close.index))

# Quick overlay plot
plt.figure()
spot_close.plot(label="Spot Close")
perp_close.plot(label="Perp Close")
plt.title("Aligned Close Prices")
plt.legend(); plt.tight_layout(); plt.show()

# %% [markdown]
# ## 5) Compute log returns (`compute_log_returns`)
# - We compute `r_t = log(P_t) - log(P_{t-1})` for spot & perp.
# - Inspect summary stats and simple distribution shape.

# %%
r_spot = compute_log_returns(spot_close)
r_perp = compute_log_returns(perp_close)

# Sanity: aligned after diff
aligned = pd.concat([r_spot.rename("r_spot"), r_perp.rename("r_perp")], axis=1, join="inner").dropna()
display(aligned.describe(percentiles=[0.01,0.05,0.95,0.99]))

# Histograms (separate figures)
plt.figure()
aligned["r_spot"].hist(bins=50)
plt.title("Histogram: r_spot"); plt.tight_layout(); plt.show()

plt.figure()
aligned["r_perp"].hist(bins=50)
plt.title("Histogram: r_perp"); plt.tight_layout(); plt.show()

# Simple autocorrelation check for r_spot (lag-1..5)
acs = [aligned["r_spot"].autocorr(lag=k) for k in range(1,6)]
print("Autocorr r_spot lags 1..5:", [round(a,4) for a in acs])

# %% [markdown]
# ## 6) Estimate hedge ratio via OLS (`estimate_ols_hedge_ratio`)
# - Regression: spot returns ~ perp returns (with intercept)
# - β (the slope) is the **hedge ratio** to minimize variance of the hedged portfolio

# %%
beta = estimate_ols_hedge_ratio(r_spot, r_perp)
print("Estimated OLS hedge ratio (β):", round(beta, 6))

# (Optional) peek at regression summary by re-running manually
# import statsmodels.api as sm
# y = aligned["r_spot"].values
# X = sm.add_constant(aligned["r_perp"].values)
# model = sm.OLS(y, X).fit()
# print(model.summary())

# %% [markdown]
# ## 7) Backtest static hedge (`backtest_static_hedge`) + metrics
# - Construct hedged returns: `r_hedged = r_spot - β * r_perp`
# - Report variance reduction, Sharpe, Max Drawdown
# - Visualize cumulative **log** returns (additive over time)

# %%
r_sp_al, r_hedged = backtest_static_hedge(r_spot, r_perp, beta)

var_spot = float(r_sp_al.var())
var_hedged = float(r_hedged.var())
variance_reduction = 1.0 - (var_hedged/var_spot) if var_spot>0 else 0.0

periods = infer_periods_per_year(TIMEFRAME)
sr_spot = sharpe_ratio(r_sp_al, periods)
sr_hedged = sharpe_ratio(r_hedged, periods)
mdd_spot = max_drawdown_from_returns(r_sp_al)
mdd_hedged = max_drawdown_from_returns(r_hedged)

print("Variance (Spot):   ", f"{var_spot:.6e}")
print("Variance (Hedged): ", f"{var_hedged:.6e}")
print("Variance reduction:", f"{variance_reduction:.2%}")
print("Sharpe Spot/Hedged:", round(sr_spot,3), "/", round(sr_hedged,3))
print("MaxDD  Spot/Hedged:", f"{mdd_spot:.2%}", "/", f"{mdd_hedged:.2%}")

# Cumulative log returns
plt.figure()
r_sp_al.cumsum().plot(label="Spot log-return cum")
r_hedged.cumsum().plot(label="Hedged log-return cum")
plt.title("Cumulative Log Returns — Spot vs Hedged")
plt.legend(); plt.tight_layout(); plt.show()

# %% [markdown]
# ## 8) Funding rates (`fetch_funding_rates`)
# - Snapshot of recent perp funding; useful context for regimes & hedge costs.
# - Not all ccxt routes expose history; this returns an empty DF if unsupported.

# %%
fund_df = fetch_funding_rates(ex, SYMBOL_PERP, limit=200)
print("fund_df shape:", fund_df.shape)
display(fund_df.head(3))

# If it has a "fundingRate" column, quick plot:
if "fundingRate" in fund_df.columns:
    plt.figure()
    fund_df["fundingRate"].astype(float).plot()
    plt.title("Funding Rate (recent)")
    plt.tight_layout(); plt.show()

# %% [markdown]
# ## 9) ML features + optional XGBoost vol prediction & hedge scaling
# - Build simple volatility features (`build_ml_vol_features`)
# - If xgboost is installed, train a tiny model to predict next-step vol proxy
# - Map predicted vol to a scale in [scale_min, scale_max] and apply to β

# %%
feat_df = build_ml_vol_features(r_sp_al, 48)
print("Feature DF shape:", feat_df.shape)
display(feat_df.head(5))

scale_min, scale_max = 0.3, 1.2
pred_vol = None
scaled_beta = beta

if HAS_XGB and len(feat_df) > 200:
    model, features = train_xgb_vol_model(feat_df)
    latest_row = feat_df.iloc[-1]
    pred_vol = predict_next_vol(model, features, latest_row)
    # Calibrate rough bounds from recent target vols
    recent = feat_df["target_vol"].tail(500)
    vol_low  = float(np.nanpercentile(recent, 20)) if recent.size>0 else 0.0
    vol_high = float(np.nanpercentile(recent, 80)) if recent.size>0 else 1.0
    scaled_beta = scale_hedge_ratio(beta, pred_vol, vol_low, vol_high, scale_min=scale_min, scale_max=scale_max)
    print("Predicted next-step vol:", None if pred_vol is None else round(float(pred_vol), 6))
    print("Scaled β:", round(scaled_beta, 6))
else:
    print("Skipping ML scaling (xgboost missing or not enough rows).")

# Visualize the scaling effect across a grid of hypothetical vols
if HAS_XGB:
    grid = np.linspace(0.0, float(feat_df["target_vol"].quantile(0.99) if "target_vol" in feat_df else 0.01), 50)
    scaled_vals = []
    vol_low  = float(np.nanpercentile(feat_df["target_vol"], 20)) if "target_vol" in feat_df else 0.0
    vol_high = float(np.nanpercentile(feat_df["target_vol"], 80)) if "target_vol" in feat_df else 1.0
    for v in grid:
        scaled_vals.append(scale_hedge_ratio(beta, v, vol_low, vol_high, scale_min, scale_max))
    plt.figure()
    pd.Series(scaled_vals, index=grid).plot()
    plt.title("Hedge scaling vs hypothetical vol")
    plt.tight_layout(); plt.show()

# %% [markdown]
# ## 10) Save artifacts to disk
# - Prices & returns CSVs
# - Metrics JSON
# - Plots PNG (already displayed above)
# - Funding snapshot (if any)

# %%
# Prices
pd.DataFrame({"spot_close": spot_close, "perp_close": perp_close}).to_csv(os.path.join(OUTDIR, "prices.csv"))
# Returns
pd.DataFrame({"r_spot": r_sp_al, "r_hedged": r_hedged}).to_csv(os.path.join(OUTDIR, "returns.csv"))
# Metrics
with open(os.path.join(OUTDIR, "metrics.json"), "w") as f:
    json.dump({
        "timeframe": TIMEFRAME, "samples": int(len(r_sp_al)),
        "hedge_ratio": float(beta),
        "scaled_beta": float(scaled_beta),
        "variance_spot": float(var_spot), "variance_hedged": float(var_hedged),
        "variance_reduction": float(variance_reduction),
        "sharpe_spot": float(sr_spot), "sharpe_hedged": float(sr_hedged),
        "maxdd_spot": float(mdd_spot), "maxdd_hedged": float(mdd_hedged)
    }, f, indent=2)

# Funding (optional)
if isinstance(fund_df, pd.DataFrame) and not fund_df.empty:
    fund_df.to_csv(os.path.join(OUTDIR, "funding_rates.csv"))

print("Saved artifacts to:", OUTDIR)

# %% [markdown]
# ## 11) Where to go next
# - Use `make hedge-paper` for a **dry-run rebalancer** that logs intended hedge ratios per cycle.
# - Later: wire **testnet orders** (USD-M perps) with strict guardrails (min notional, leverage caps, kill-switch).
# - For the PhD: expand ML pipeline (XGBoost baseline → HAR-RV/MIDAS → LSTM/Transformer), add **regime filters**, and run **Diebold–Mariano** tests.
