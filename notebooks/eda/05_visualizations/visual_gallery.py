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
#       jupytext_version: 1.17.3
# ---

# %% [markdown]
# # Visualization Gallery
# Common charts to explore a new asset quickly:
# - Price & returns
# - Bollinger bands (20-period)
# - Rolling volatility (annualized)
# - Rolling Sharpe (annualized)
# - Drawdowns
# - Return autocorrelation (ACF)
# - Return distribution (hist + normal overlay)

# %%
import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
DATA_RAW = os.path.join(ROOT, "data", "raw")
REPORT_FIGS = os.path.join(ROOT, "reports", "figures")
os.makedirs(REPORT_FIGS, exist_ok=True)

# %% [markdown]
# ## 1) Load OHLCV (CSV/Parquet) from `data/raw/`
# Expected columns: `timestamp`, `close` (others optional)

# %%
files = [f for f in os.listdir(DATA_RAW) if f.lower().endswith((".csv",".parquet"))]
if not files:
    raise FileNotFoundError("No CSV/Parquet in data/raw. Drop OHLCV first.")
path = os.path.join(DATA_RAW, sorted(files)[0])
if path.endswith(".csv"):
    df = pd.read_csv(path, parse_dates=["timestamp"])
else:
    df = pd.read_parquet(path)
df = df.sort_values("timestamp").reset_index(drop=True)
if "close" not in df.columns:
    raise ValueError("Column 'close' not found.")

# Ensure tz-naive timestamps
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)

# %% [markdown]
# ## 2) Basic returns & annualization helper

# %%
# Log returns at native sampling
df["logret"] = np.log(df["close"]).diff()

# Infer periods-per-year from median sampling interval
dts = df["timestamp"].diff().dt.total_seconds()
median_sec = float(dts.dropna().median()) if dts.notna().any() else 60.0
sec_per_year = 365 * 24 * 3600
periods_per_year = max(1.0, sec_per_year / max(1.0, median_sec))
ann_sqrt = math.sqrt(periods_per_year)

# Windows ~ 1 week & 1 month in native periods
win_week = max(5, int(periods_per_year / 52))
win_month = max(20, int(periods_per_year / 12))

# %% [markdown]
# ## 3) Price

# %%
plt.figure(figsize=(11,4))
plt.plot(df["timestamp"], df["close"])
plt.title("Price")
plt.xlabel("Time"); plt.ylabel("Close")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_FIGS, "viz_price.png"))

# %% [markdown]
# ## 4) Returns (log)

# %%
plt.figure(figsize=(11,3.5))
plt.plot(df["timestamp"], df["logret"])
plt.title("Log Returns")
plt.xlabel("Time"); plt.ylabel("logret")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_FIGS, "viz_returns.png"))

# %% [markdown]
# ## 5) Bollinger Bands (20-period on price)

# %%
period = 20
ma = df["close"].rolling(period).mean()
sd = df["close"].rolling(period).std()
upper = ma + 2 * sd
lower = ma - 2 * sd

plt.figure(figsize=(11,4))
plt.plot(df["timestamp"], df["close"], label="Close")
plt.plot(df["timestamp"], ma, label=f"MA{period}")
plt.plot(df["timestamp"], upper, label="Upper")
plt.plot(df["timestamp"], lower, label="Lower")
plt.title("Bollinger Bands")
plt.xlabel("Time"); plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(REPORT_FIGS, "viz_bollinger.png"))

# %% [markdown]
# ## 6) Rolling Volatility (annualized) on returns

# %%
roll = df["logret"].rolling(win_month).std() * ann_sqrt
plt.figure(figsize=(11,3.5))
plt.plot(df["timestamp"], roll)
plt.title(f"Rolling Volatility (~1m window, annualized), window={win_month}")
plt.xlabel("Time"); plt.ylabel("Vol (ann)")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_FIGS, "viz_rolling_vol.png"))

# %% [markdown]
# ## 7) Rolling Sharpe (mean/std * sqrt(periods/year), rf=0)

# %%
ret_mean = df["logret"].rolling(win_month).mean()
ret_std = df["logret"].rolling(win_month).std()
rolling_sharpe = (ret_mean / ret_std) * ann_sqrt

plt.figure(figsize=(11,3.5))
plt.plot(df["timestamp"], rolling_sharpe)
plt.axhline(0, linestyle="--")
plt.title(f"Rolling Sharpe (rf=0), window={win_month}")
plt.xlabel("Time"); plt.ylabel("Sharpe (ann)")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_FIGS, "viz_rolling_sharpe.png"))

# %% [markdown]
# ## 8) Drawdowns (from cumulative log returns)

# %%
cum_logret = df["logret"].fillna(0).cumsum()
eq = np.exp(cum_logret)  # equity curve (normalized)
rolling_max = np.maximum.accumulate(eq)
drawdown = eq / rolling_max - 1.0

plt.figure(figsize=(11,3.5))
plt.plot(df["timestamp"], drawdown)
plt.title("Drawdown")
plt.xlabel("Time"); plt.ylabel("Drawdown")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_FIGS, "viz_drawdown.png"))

# %% [markdown]
# ## 9) Return ACF (first 50 lags)

# %%
lags = 50
acf_vals = [df["logret"].autocorr(lag=i) for i in range(lags+1)]
plt.figure(figsize=(10,3.5))
markerline, stemlines, baseline = plt.stem(range(lags+1), acf_vals, use_line_collection=True)
plt.title("ACF of Log Returns")
plt.xlabel("Lag"); plt.ylabel("Autocorr")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_FIGS, "viz_returns_acf.png"))

# %% [markdown]
# ## 10) Return distribution (hist + normal overlay)

# %%
rets = df["logret"].dropna()
mu, sd = rets.mean(), rets.std()

plt.figure(figsize=(8,4))
# Histogram
n, bins, _ = plt.hist(rets, bins=80, density=True, alpha=0.6)
# Normal overlay
x = np.linspace(rets.quantile(0.001), rets.quantile(0.999), 400)
pdf = norm.pdf(x, mu, sd if sd > 0 else 1e-9)
plt.plot(x, pdf)
plt.title("Return Distribution (with Normal overlay)")
plt.xlabel("logret"); plt.ylabel("density")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_FIGS, "viz_return_hist.png"))

print("Figures saved to:", REPORT_FIGS)
