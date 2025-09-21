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
# ## Plot defaults (project-wide)
# This cell ensures consistent Matplotlib styling and date axes.
# %%
# Make 'configs' importable from notebooks (.ipynb or .py)
import sys, os
from pathlib import Path
try:
    HERE = Path(__file__).parent
except NameError:
    HERE = Path.cwd()
ROOT = (HERE / "../../..").resolve()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from configs.plots.mpl_defaults import use_mpl_defaults, format_date_axis
use_mpl_defaults()

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
# # Returns & Realized Volatility (RV) + HAR-RV Prep
# This notebook builds basic returns, daily realized volatility, and HAR-RV regressors.
# Works with OHLCV data in `data/raw/` (CSV or Parquet) with at least: `timestamp`, `close`.

# %%
import os, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
DATA_RAW = os.path.join(ROOT, "data", "raw")
DATA_INTERIM = os.path.join(ROOT, "data", "interim")
DATA_PROCESSED = os.path.join(ROOT, "data", "processed")
REPORT_FIGS = os.path.join(ROOT, "reports", "figures")

os.makedirs(DATA_INTERIM, exist_ok=True)
os.makedirs(DATA_PROCESSED, exist_ok=True)
os.makedirs(REPORT_FIGS, exist_ok=True)

# %% [markdown]
# ## 1) Load data (first file in `data/raw/`)
# Supports CSV (expects `timestamp` parse) or Parquet. Ensures sorted by time.

# %%
cands = [f for f in os.listdir(DATA_RAW) if f.lower().endswith((".csv",".parquet"))]
if not cands:
    raise FileNotFoundError("Drop OHLCV into data/raw (CSV or Parquet with 'timestamp','close').")
path = os.path.join(DATA_RAW, cands[0])

if path.endswith(".csv"):
    df = pd.read_csv(path, parse_dates=["timestamp"])
else:
    df = pd.read_parquet(path)
df = df.sort_values("timestamp").reset_index(drop=True)

# Ensure tz-naive for resampling compatibility
if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
else:
    raise ValueError("Column 'timestamp' must be datetime-like.")

df = df[["timestamp","close"]].dropna()

# %% [markdown]
# ## 2) Intraday returns and daily Realized Volatility (RV)
# - Compute log returns at native frequency
# - Aggregate to daily RV = sqrt( sum intraday r_t^2 ) and annualize with sqrt(365)

# %%
df["logret"] = np.log(df["close"]).diff()
df["date"] = df["timestamp"].dt.date

# Daily realized variance and volatility
rv_daily = (
    df.dropna(subset=["logret"])
      .groupby("date")["logret"]
      .apply(lambda x: np.sqrt((x**2).sum()))  # daily RV (not annualized)
      .to_frame(name="RV")
      .reset_index()
)

# Annualize daily RV (sqrt(365) factor). Note: RV here is already sqrt of variance.
rv_daily["RV_ann"] = rv_daily["RV"] * math.sqrt(365)

# %% [markdown]
# ## 3) Rolling stats & sanity checks
# Create rolling (5d, 22d) means of RV and simple close-to-close daily vol proxy.

# %%
# Close-to-close daily returns (from last obs each day)
daily_close = df.set_index("timestamp")["close"].resample("1D").last().dropna()
cc_ret = np.log(daily_close).diff().dropna()
cc_vol_ann = cc_ret.rolling(22).std() * math.sqrt(252)

rv_series = rv_daily.set_index(pd.to_datetime(rv_daily["date"]))["RV_ann"]
rv_df = pd.DataFrame({
    "RV_ann": rv_series,
    "RV_5d": rv_series.rolling(5).mean(),
    "RV_22d": rv_series.rolling(22).mean(),
    "CC_vol_ann_22d": cc_vol_ann.reindex(rv_series.index)
}).dropna()

# %% [markdown]
# ## 4) HAR-RV regressors (Corsi, 2009 style)
# RV_t = β0 + βd * RV_{t-1} + βw * mean(RV_{t-5..t-1}) + βm * mean(RV_{t-22..t-1}) + ε_t

# %%
har = pd.DataFrame(index=rv_series.index)
har["RV_t"] = rv_series
har["RV_d"] = rv_series.shift(1)
har["RV_w"] = rv_series.rolling(5).mean().shift(1)
har["RV_m"] = rv_series.rolling(22).mean().shift(1)
har = har.dropna().copy()

# Simple OLS via numpy (avoids extra deps)
X = np.column_stack([
    np.ones(len(har)),
    har["RV_d"].values,
    har["RV_w"].values,
    har["RV_m"].values
])
y = har["RV_t"].values
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
har["RV_hat"] = X @ beta

coef = dict(beta0=beta[0], beta_d=beta[1], beta_w=beta[2], beta_m=beta[3])

# %% [markdown]
# ## 5) Quick visuals

# %%
plt.figure(figsize=(10,4))
plt.plot(rv_df.index, rv_df["RV_ann"])
plt.title("Daily Realized Volatility (annualized)")
plt.xlabel("Date"); plt.ylabel("RV_ann")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_FIGS, "rv_daily_ann.png"))

plt.figure(figsize=(10,4))
plt.plot(har.index, har["RV_t"], label="RV_t")
plt.plot(har.index, har["RV_hat"], label="HAR fit")
plt.title("HAR-RV: Actual vs Fitted")
plt.xlabel("Date"); plt.ylabel("RV (annualized)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(REPORT_FIGS, "har_rv_fit.png"))

# %% [markdown]
# ## 6) Save outputs
# - `data/interim/rv_daily.parquet`: daily RV (ann.)
# - `data/processed/har_rv_features.parquet`: features + fitted values

# %%
rv_daily_out = os.path.join(DATA_INTERIM, "rv_daily.parquet")
har_out = os.path.join(DATA_PROCESSED, "har_rv_features.parquet")
rv_daily.assign(date=pd.to_datetime(rv_daily["date"])).to_parquet(rv_daily_out, index=False)
har.reset_index().rename(columns={"index":"date"}).to_parquet(har_out, index=False)

print("Saved:", rv_daily_out)
print("Saved:", har_out)
print("HAR coefficients:", coef)
