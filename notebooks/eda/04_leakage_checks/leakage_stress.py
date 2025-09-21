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
# # Leakage Stress Tests
# Tools to detect potential target leakage and time-split issues.
# - ACF of target (autocorrelation)
# - Lead/lag correlation scan: features vs. shifted target (future info?)
# - Random split vs. time-based split comparison (red flag if random >> time)
# - Simple equality/duplication checks

# %%
import os, re, json, math, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
DATA_PROCESSED = os.path.join(ROOT, "data", "processed")
REPORT_FIGS = os.path.join(ROOT, "reports", "figures")
os.makedirs(REPORT_FIGS, exist_ok=True)

pd.set_option("display.max_columns", 120)

# %% [markdown]
# ## 1) Load a processed dataset
# Picks the **first** parquet under `data/processed/`. Adjust path if needed.

# %%
parqs = [f for f in os.listdir(DATA_PROCESSED) if f.endswith(".parquet")]
if not parqs:
    raise FileNotFoundError("No parquet in data/processed. Run features pipeline first.")
path = os.path.join(DATA_PROCESSED, sorted(parqs)[0])
df = pd.read_parquet(path)
print("Loaded:", path, "shape:", df.shape)

# %% [markdown]
# ## 2) Identify time, target, features

# %%
# Time index
time_col = None
for c in ["timestamp", "date", "time"]:
    if c in df.columns:
        time_col = c
        break
if time_col is None:
    raise ValueError("No time column found. Add 'timestamp' or 'date'.")

df[time_col] = pd.to_datetime(df[time_col], utc=False)
df = df.sort_values(time_col).reset_index(drop=True)

# Target column heuristics
target_candidates = [c for c in df.columns if re.search(r"^(y|target|vol_?target|ret_h\\d+)$", c)]
target_col = target_candidates[0] if target_candidates else None

# Fallback: proxy target via forward realized vol of a return column
if target_col is None:
    ret_cols = [c for c in df.columns if c.startswith("ret_")]
    if not ret_cols:
        raise ValueError("No explicit target or return columns to build proxy.")
    base = ret_cols[0]
    horizon = 12
    df["y_proxy"] = df[base].rolling(horizon).apply(lambda x: np.sqrt(np.sum(np.square(x))), raw=True)
    target_col = "y_proxy"
    print("Using proxy target:", target_col)

# Feature set
exclude = {time_col, target_col}
meta_like = {c for c in df.columns if c.lower() in {"symbol","asset"}}
features = [c for c in df.columns if c not in exclude.union(meta_like)]

print("time_col:", time_col)
print("target_col:", target_col)
print("n_features:", len(features))

# Keep numeric features only
num_features = df[features].select_dtypes(include=[np.number]).columns.tolist()
X_full = df[num_features]
y_full = df[target_col]

# %% [markdown]
# ## 3) Target autocorrelation (ACF)
# High autocorrelation is common in vol series; useful context for leakage tests.

# %%
lags = 60
acf_vals = []
for L in range(lags + 1):
    acf_vals.append(y_full.autocorr(lag=L))

plt.figure(figsize=(10,4))
plt.stem(range(lags + 1), acf_vals, use_line_collection=True)
plt.title(f"Target ACF (0..{lags})")
plt.xlabel("Lag"); plt.ylabel("Autocorr")
plt.tight_layout()
acf_png = os.path.join(REPORT_FIGS, "leakage_target_acf.png")
plt.savefig(acf_png)
print("Saved ACF plot:", acf_png)

# %% [markdown]
# ## 4) Lead/Lag correlation scan
# If a feature is more correlated with a **future** target (negative shift), it may contain lookahead info.

# %%
lead_lags = [-10, -5, -1, 0, 1, 5, 10]
lead_lag_report = {}
y = y_full.copy()

for f in num_features:
    s = df[f]
    corr_by_shift = {}
    for k in lead_lags:
        # Shift target by k: negative k => compare feature with future target
        corr = s.corr(y.shift(k))
        corr_by_shift[int(k)] = None if pd.isna(corr) else float(corr)
    # Best (absolute) correlation and where it occurs
    best_shift = max(corr_by_shift, key=lambda kk: abs(corr_by_shift[kk]) if corr_by_shift[kk] is not None else -1)
    lead_lag_report[f] = {
        "corr_by_shift": corr_by_shift,
        "best_shift": int(best_shift),
        "best_corr": corr_by_shift[best_shift]
    }

# Flag suspicious: best shift is negative (aligns with **future** target)
suspicious_features = [f for f, v in lead_lag_report.items() if v["best_shift"] < 0 and v["best_corr"] is not None and abs(v["best_corr"]) > 0.1]
print("Suspicious (future-aligned) features:", suspicious_features[:20])

# %% [markdown]
# ## 5) Equality/duplicate checks (hard leaks)

# %%
exact_equal = [f for f in num_features if df[f].equals(y_full)]
dupe_pairs = []
seen = {}
for c in num_features:
    sig = (df[c].astype("float64").fillna(-1234567.89).values.tobytes())
    if sig in seen:
        dupe_pairs.append((c, seen[sig]))
    else:
        seen[sig] = c

print("Exact equals to target:", exact_equal)
print("Duplicate feature pairs (first 5):", dupe_pairs[:5])

# %% [markdown]
# ## 6) Random split vs. time split (red-flag gap)
# If performance on a random split is much higher than on a proper time split, leakage or temporal dependency issues are likely.

# %%
# Prepare matrices
X_num = X_full.fillna(0.0).values
y_num = y_full.values

# Chronological split (last 20% for validation)
split_idx = int(len(df) * 0.8)
X_tr, X_va = X_num[:split_idx], X_num[split_idx:]
y_tr, y_va = y_num[:split_idx], y_num[split_idx:]

rf_time = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
rf_time.fit(X_tr, y_tr)
pred_time = rf_time.predict(X_va)
rmse_time = mean_squared_error(y_va, pred_time, squared=False)
r2_time = r2_score(y_va, pred_time)

# Random split (same size) for comparison
rng = np.random.default_rng(42)
perm = rng.permutation(len(df))
cut = int(len(df) * 0.8)
idx_tr, idx_va = perm[:cut], perm[cut:]
rf_rand = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
rf_rand.fit(X_num[idx_tr], y_num[idx_tr])
pred_rand = rf_rand.predict(X_num[idx_va])
rmse_rand = mean_squared_error(y_num[idx_va], pred_rand, squared=False)
r2_rand = r2_score(y_num[idx_va], pred_rand)

print(f"Time-split  RMSE={rmse_time:.4f} R2={r2_time:.3f}")
print(f"Random-split RMSE={rmse_rand:.4f} R2={r2_rand:.3f}")

gap_r2 = float(r2_rand - r2_time)

# %% [markdown]
# ## 7) Save report

# %%
report = {
    "dataset": os.path.basename(path),
    "rows": int(df.shape[0]),
    "cols": int(df.shape[1]),
    "time_start": df[time_col].min().isoformat(),
    "time_end": df[time_col].max().isoformat(),
    "target": target_col,
    "acf_top_lags": {str(i): float(acf_vals[i]) for i in range(min(10, len(acf_vals)))},
    "suspicious_future_aligned_features": suspicious_features[:50],
    "lead_lag_scan": {f: v for f, v in list(lead_lag_report.items())[:200]},  # cap size
    "exact_equals_to_target": exact_equal,
    "duplicate_feature_pairs": dupe_pairs[:50],
    "time_split": {"rmse": float(rmse_time), "r2": float(r2_time)},
    "random_split": {"rmse": float(rmse_rand), "r2": float(r2_rand)},
    "r2_gap_random_minus_time": gap_r2
}
out_json = os.path.join(REPORT_FIGS, "leakage_checks_report.json")
with open(out_json, "w") as f:
    json.dump(report, f, indent=2)
print("Saved report:", out_json)
