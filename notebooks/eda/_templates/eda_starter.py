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
#     display_name: Python 3.11 (.venv)
#     language: python
#     name: crypto-vol-quickstart-311
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
#     notebook_metadata_filter: jupytext,text_representation,kernelspec
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3.11 (.venv)
#     language: python
#     name: crypto-vol-quickstart-311
# ---

# %% [markdown]
# # EDA Starter Notebook
# Quick checks to get familiar with the dataset structure and basic return/volatility views.

# %%
pip install jupytext

# %% [markdown]
# hello world

# %%
import os, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# project paths
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
DATA_RAW = os.path.join(ROOT, "data", "raw")
DATA_INTERIM = os.path.join(ROOT, "data", "interim")
DATA_PROCESSED = os.path.join(ROOT, "data", "processed")
REPORT_FIGS = os.path.join(ROOT, "reports", "figures")

os.makedirs(REPORT_FIGS, exist_ok=True)

# %% [markdown]
# ## 1) Load a sample file
# Drop a CSV/Parquet into `data/raw/` (e.g., OHLCV for BTC/ETH).
# Expected columns: timestamp, open, high, low, close, volume

# %%
# Example: auto-pick first CSV in data/raw
candidates = [f for f in os.listdir(DATA_RAW) if f.lower().endswith((".csv",".parquet"))]
if not candidates:
    raise FileNotFoundError("No CSV or Parquet found in data/raw. Drop a file first.")

path = os.path.join(DATA_RAW, candidates[0])
if path.endswith(".csv"):
    df = pd.read_csv(path, parse_dates=["timestamp"])
else:
    df = pd.read_parquet(path)

df = df.sort_values("timestamp").reset_index(drop=True)
df.head()

# %% [markdown]
# ## 2) Basic health checks

# %%
summary = {
    "n_rows": len(df),
    "n_cols": df.shape[1],
    "time_span": [df["timestamp"].min(), df["timestamp"].max()],
    "null_counts": df.isna().sum().to_dict(),
    "dtypes": df.dtypes.astype(str).to_dict(),
    "duplicates": int(df.duplicated(subset=["timestamp"]).sum()),
}
summary

# %% [markdown]
# ## 3) Compute returns & rolling volatility proxy

# %%
df["ret"] = df["close"].pct_change()
window = 30
ann_factor = math.sqrt(365)
df["vol_30d"] = df["ret"].rolling(window).std() * ann_factor
df[["timestamp","close","ret","vol_30d"]].tail()

# %% [markdown]
# ## 4) Quick plots

# %%
plt.figure(figsize=(10,4))
plt.plot(df["timestamp"], df["close"])
plt.title("Price"); plt.xlabel("Time"); plt.ylabel("Close")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_FIGS, "price.png"))

plt.figure(figsize=(10,4))
plt.plot(df["timestamp"], df["ret"])
plt.title("Returns"); plt.xlabel("Time"); plt.ylabel("Return")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_FIGS, "returns.png"))

plt.figure(figsize=(10,4))
plt.plot(df["timestamp"], df["vol_30d"])
plt.title("Rolling Volatility (30d, annualized)")
plt.xlabel("Time"); plt.ylabel("Vol")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_FIGS, "vol_30d.png"))

# %% [markdown]
# ## 5) Save interim snapshot

# %%
df.to_parquet(os.path.join(DATA_INTERIM, "sample_with_returns.parquet"), index=False)
print("Saved interim snapshot to data/interim/sample_with_returns.parquet")
