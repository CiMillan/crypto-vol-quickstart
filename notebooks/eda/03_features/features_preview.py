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
# # Features Preview + Leakage Checks
# Quick look at processed features, basic target wiring, feature importance, and simple leakage checks.

# %%
import os, re, math, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
DATA_PROCESSED = os.path.join(ROOT, "data", "processed")
REPORT_FIGS = os.path.join(ROOT, "reports", "figures")
os.makedirs(REPORT_FIGS, exist_ok=True)

pd.set_option("display.max_columns", 120)

# %% [markdown]
# ## 1) Load processed feature set
# Loads the **first** `.parquet` in `data/processed` (adjust if you want a specific file).

# %%
parqs = [f for f in os.listdir(DATA_PROCESSED) if f.endswith(".parquet")]
if not parqs:
    raise FileNotFoundError("No parquet in data/processed. Run a pipeline (e.g., make quick-features).")
path = os.path.join(DATA_PROCESSED, sorted(parqs)[0])
df = pd.read_parquet(path)
print("Loaded:", path, "shape:", df.shape)

# %% [markdown]
# ## 2) Identify time index, target, and features
# Heuristics: `timestamp` (or `date`) for time, and common target names like `y`, `target`, `vol_target`, `ret_hXX`.

# %%
# Time index
time_col = None
for c in ["timestamp", "date", "time"]:
    if c in df.columns:
        time_col = c
        break
if time_col is None:
    raise ValueError("No timestamp/date column found. Add one of ['timestamp','date','time'].")

df[time_col] = pd.to_datetime(df[time_col], utc=False)
df = df.sort_values(time_col).reset_index(drop=True)

# Target column
target_candidates = [c for c in df.columns if re.search(r"^(y|target|vol_?target|ret_h\d+)$", c)]
target_col = target_candidates[0] if target_candidates else None

# If no explicit target, fall back to forward abs-return over horizon=12 (approx)
if target_col is None:
    ret_cols = [c for c in df.columns if c.startswith("ret_")]
    if ret_cols:
        base = ret_cols[0]
        df["y_proxy"] = df[base].rolling(12).apply(lambda x: np.sqrt(np.sum(np.square(x))), raw=True)
        target_col = "y_proxy"
        print("No explicit target found → using proxy:", target_col)
    else:
        raise ValueError("No target found and no return cols to build proxy. Please add a target.")

# Feature set: exclude time & any known labels
exclude = {time_col, target_col}
meta_like = {c for c in df.columns if c.lower() in {"symbol","asset"}}
features = [c for c in df.columns if c not in exclude.union(meta_like)]

print("time_col:", time_col)
print("target_col:", target_col)
print("n_features:", len(features))

# %% [markdown]
# ## 3) Basic sanity & leakage checks
# - Nulls
# - Duplicate columns
# - Exact leaks (features identical to target)
# - Suspicious names (e.g., *future*, *target*, *pred*)

# %%
nulls = df[features+[target_col]].isna().sum().sort_values(ascending=False)
dupe_cols = []
seen = {}
for c in features:
    sig = (df[c].astype("float64").fillna(-1234567.89).values.tobytes())
    if sig in seen:
        dupe_cols.append((c, seen[sig]))
    else:
        seen[sig] = c

exact_leaks = [c for c in features if df[c].equals(df[target_col])]
suspect_by_name = [c for c in features if re.search(r"(future|lead|target|pred|label)", c, re.I)]

print("Top nulls:\n", nulls.head(10))
print("Duplicate feature pairs:", dupe_cols[:5])
print("Exact leaks:", exact_leaks)
print("Suspicious names:", suspect_by_name[:10])

# %% [markdown]
# ## 4) Train/valid split (time-based) + quick model
# Use a small RF just to get rough feature importances (XGB optional but RF is dependency-light).

# %%
clean_cols = [c for c in features if c not in set(x for x,_ in dupe_cols)]
X = df[clean_cols].select_dtypes(include=[np.number]).fillna(0.0).values
y = df[target_col].values

# Last 20% as validation
split_idx = int(len(df) * 0.8)
X_tr, X_va = X[:split_idx], X[split_idx:]
y_tr, y_va = y[:split_idx], y[split_idx:]

rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)
pred = rf.predict(X_va)

rmse = mean_squared_error(y_va, pred, squared=False)
r2 = r2_score(y_va, pred)
print("Valid RMSE:", rmse, "R2:", r2)

# %% [markdown]
# ## 5) Feature importance & quick plots

# %%
imp = pd.Series(rf.feature_importances_, index=df[clean_cols].select_dtypes(include=[np.number]).columns)
imp = imp.sort_values(ascending=False)

plt.figure(figsize=(9,5))
plt.bar(imp.index[:25], imp.values[:25])
plt.xticks(rotation=75, ha="right")
plt.title("Top 25 Feature Importances (RF)")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_FIGS, "feature_importance_rf_top25.png"))

plt.figure(figsize=(9,4))
plt.plot(df[time_col].iloc[split_idx:], y_va, label="actual")
plt.plot(df[time_col].iloc[split_idx:], pred, label="pred")
plt.title("Validation: Actual vs Pred (time series)")
plt.xlabel("Time"); plt.ylabel(target_col)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(REPORT_FIGS, "valid_actual_vs_pred.png"))

# %% [markdown]
# ## 6) Save quick report (JSON)
# Includes basic stats and the top importances.

# %%
report = {
    "dataset": os.path.basename(path),
    "rows": int(df.shape[0]),
    "cols": int(df.shape[1]),
    "time_start": df[time_col].min().isoformat(),
    "time_end": df[time_col].max().isoformat(),
    "target": target_col,
    "valid_rmse": float(rmse),
    "valid_r2": float(r2),
    "exact_leaks": exact_leaks,
    "suspect_by_name": suspect_by_name[:20],
    "top_importances": imp.head(30).to_dict(),
}
out_json = os.path.join(REPORT_FIGS, "features_preview_report.json")
with open(out_json, "w") as f:
    json.dump(report, f, indent=2)
print("Saved report:", out_json)
