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
# ---

# %% [markdown]
# # Data Quality Checks (OHLCV)
# - Schema summary (types, null %, unique counts)
# - Missingness & duplicates (rows, timestamps)
# - Time gaps vs. expected interval
# - OHLCV sanity (high/low bounds, negatives)
# - Return outliers (z-scores)
# - Saves a JSON report + CSVs + figures to reports/

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

# %%
import json, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORT_FIGS = ROOT / "reports" / "figures"
REPORT_TBLS = ROOT / "reports" / "tables"
REPORT_FIGS.mkdir(parents=True, exist_ok=True)
REPORT_TBLS.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1) Load data (prefers `data/raw`, falls back to `data/processed`)
# Supports CSV (expects `timestamp`) and Parquet.

# %%
cands = [f for f in os.listdir(DATA_RAW) if f.lower().endswith((".csv",".parquet"))] if DATA_RAW.exists() else []
src = DATA_RAW if cands else DATA_PROCESSED
files = [f for f in os.listdir(src) if f.lower().endswith((".csv",".parquet"))]
if not files:
    raise FileNotFoundError("No CSV/Parquet found in data/raw or data/processed.")
path = (src / sorted(files)[0])
if str(path).endswith(".csv"):
    df = pd.read_csv(path, parse_dates=["timestamp"], infer_datetime_format=True)
else:
    df = pd.read_parquet(path)
if "timestamp" not in df.columns:
    raise ValueError("Expected a 'timestamp' column.")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
df = df.sort_values("timestamp").reset_index(drop=True)
print("Loaded:", path, "shape:", df.shape)

# Try to coerce common OHLCV columns to numeric
for c in ["open","high","low","close","volume"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# %% [markdown]
# ## 2) Schema summary

# %%
schema = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "non_null": df.notna().sum(),
    "nulls": df.isna().sum(),
})
schema["null_pct"] = (schema["nulls"] / len(df) * 100).round(3)
schema["nunique"] = df.nunique(dropna=True)
schema.reset_index(names="column", inplace=True)
schema_path = REPORT_TBLS / "dq_schema.csv"
schema.to_csv(schema_path, index=False)
schema.head(20)

# %% [markdown]
# ## 3) Duplicates & timestamp health

# %%
dup_rows = int(df.duplicated().sum())
dup_ts = int(df.duplicated(subset=["timestamp"]).sum())
print("Duplicate rows:", dup_rows, " Duplicate timestamps:", dup_ts)

# %% [markdown]
# ## 4) Time gaps vs expected sampling
# Estimate median interval; list the top gaps and approximate missing bars.

# %%
dsec = df["timestamp"].diff().dt.total_seconds()
median_sec = float(dsec.dropna().median())
expected = max(1.0, median_sec)  # seconds
gaps = dsec[dsec > expected * 1.5].sort_values(ascending=False)
gap_tbl = pd.DataFrame({
    "timestamp": df.loc[gaps.index, "timestamp"].astype(str),
    "gap_seconds": gaps.astype(float),
    "missing_intervals_est": ((gaps / expected) - 1).round(2)
})
gap_csv = REPORT_TBLS / "dq_time_gaps.csv"
gap_tbl.to_csv(gap_csv, index=False)
gap_tbl.head(10)

# %% [markdown]
# ## 5) OHLCV sanity checks
# - No negatives for prices; volume ≥ 0
# - high ≥ max(open,close,low) and low ≤ min(open,close,low)

# %%
issues = {}
if set(["open","high","low","close"]).issubset(df.columns):
    o,h,l,c = df["open"], df["high"], df["low"], df["close"]
    issues["neg_price_rows"] = int(((o<0)|(h<0)|(l<0)|(c<0)).sum())
    issues["high_bound_viol"] = int((h < pd.concat([o,l,c], axis=1).max(axis=1)).sum())
    issues["low_bound_viol"]  = int((l > pd.concat([o,l,c], axis=1).min(axis=1)).sum())
if "volume" in df.columns:
    issues["neg_volume_rows"] = int((df["volume"] < 0).sum())
issues

# %% [markdown]
# ## 6) Return outliers (z-scores on log returns)

# %%
df["logret"] = np.log(df["close"]).diff() if "close" in df.columns else np.nan
mu, sd = df["logret"].mean(), df["logret"].std()
z = (df["logret"] - mu) / (sd if sd and sd>0 else 1.0)
out_idx = z.abs().sort_values(ascending=False).head(20).index
outliers = df.loc[out_idx, ["timestamp","close","logret"]].assign(z=z.loc[out_idx].values).sort_values("timestamp")
out_csv = REPORT_TBLS / "dq_return_outliers.csv"
outliers.to_csv(out_csv, index=False)
outliers.head(10)

# %% [markdown]
# ## 7) Missingness bar chart

# %%
null_counts = df.isna().sum().sort_values(ascending=False)
plt.figure(figsize=(10,4))
plt.bar(null_counts.index[:40], null_counts.values[:40])
plt.xticks(rotation=75, ha="right")
plt.title("Null counts by column (top 40)")
plt.tight_layout()
fig_nulls = REPORT_FIGS / "dq_null_counts.png"
plt.savefig(fig_nulls)
print("Saved:", fig_nulls)

# %% [markdown]
# ## 8) Report summary (JSON)

# %%
report = {
    "dataset": os.path.basename(path),
    "rows": int(len(df)),
    "cols": int(df.shape[1]),
    "time_start": df["timestamp"].min().isoformat() if len(df) else None,
    "time_end": df["timestamp"].max().isoformat() if len(df) else None,
    "median_step_seconds": median_sec,
    "duplicate_rows": dup_rows,
    "duplicate_timestamps": dup_ts,
    "gap_csv": str(gap_csv),
    "schema_csv": str(schema_path),
    "outliers_csv": str(out_csv),
    "nulls_fig": str(fig_nulls),
    "ohlcv_issues": issues
}
rep_json = REPORT_TBLS / "dq_report.json"
with open(rep_json, "w") as f:
    json.dump(report, f, indent=2)
print("Saved report:", rep_json)
