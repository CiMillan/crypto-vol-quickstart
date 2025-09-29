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
#     display_name: crypto-vol (.venv)
#     language: python
#     name: crypto-vol
# ---

# %%
# Resolve repository root (no data loading here)
from pathlib import Path
import os

ROOT = Path(os.environ.get("PROJECT_ROOT", Path.cwd())).resolve()
os.environ["PROJECT_ROOT"] = str(ROOT)  # ensure downstream cells see it
print("ROOT =", ROOT)

# Quick existence check (prints True/False)
for p in [
    ROOT / "data/raw/binance_spot_BTCUSDT_5m.parquet",
    ROOT / "data/raw/binance_funding_BTCUSDT.parquet",
    ROOT / "data/processed/BTCUSDT_5m.parquet",
]:
    print(p, "exists?", p.exists())


# %% [markdown]
# # 📚 Notebook Index
#
# - [Cell 1 — Goal & Success Criteria](#c1)
# - [Cell 2 — Imports, Config, Run Metadata](#c2)
# - [Cell 3 — Load Data & Sanity Checks](#c3)
# - [Cell 4 — Resample & Log Returns](#c4)
# - [Cell 5 — Align Hedge Instrument & Funding](#c5)
# - [Cell 6 — OLS β (static & rolling)](#c6)
# - [Cell 7 — Volatility-Scaled β](#c7)
# - [Cell 8 — Hedged Returns + Fees + Funding](#c8)
# - [Cell 9 — Metrics (VR, Sharpe, MDD, Turnover)](#c9)
# - [Cell 10 — Visuals](#c10)
# - [Cell 11 — Paper Rebalancer JSONL Log](#c11)
# - [Cell 12 — Next Steps & Deliverables](#c12)
#
# 🧭 Quick tip: after a run, jump to **Artifacts** with the code cell below for direct file paths (plots, CSVs, metrics).
#

# %% [markdown]
# ### Tutorial vs MVP — cell-by-cell
#
# | #  | Topic               | Tutorial (first notebook)                     | MVP (new notebook)                                      | Status                   | What changed / why it matters                                                       |
# | -- | ------------------- | --------------------------------------------- | ------------------------------------------------------- | ------------------------ | ----------------------------------------------------------------------------------- |
# | 1  | Goal & framing      | “Project goal / what this proves” intro       | **Cell 1 — Project goal & success criteria**            | **Overlap**              | Same intent; MVP states success metrics up front (variance reduction, Sharpe, NRR). |
# | 2  | Setup               | Basic imports + config                        | **Cell 2 — Imports, config, run metadata**              | **Enhanced**             | Adds timestamped `RUN_DIR` + `metadata.json` → reproducibility/audit.               |
# | 3  | Data quality        | Load OHLCV, basic checks, tz aware            | **Cell 3 — Load raw & sanity checks**                   | **Overlap**              | Same checks; MVP persists CSV snapshots for audit.                                  |
# | 4  | Returns prep        | Resample + compute returns; quick peek/plot   | **Cell 4 — Resample & log returns**                     | **Overlap**              | Same math; MVP standardizes helper and prints spans.                                |
# | 5  | Alignment + funding | (Often implicit / light)                      | **Cell 5 — Align spot vs hedge & merge funding**        | **New focus**            | Explicit funding merge (8h → bar forward-fill); correlation sanity.                 |
# | 6  | Hedge ratio (OLS)   | OLS ( \beta ) baseline; optional intercept    | **Cell 6 — OLS ( \beta ) (static + rolling option)**    | **Enhanced**             | Adds rolling ( \beta_t ) path + train/test split for static fit.                    |
# | 7  | Vol-aware scaling   | Optional scaling idea                         | **Cell 7 — Volatility-scaled ( \beta )**                | **Overlap/Refined**      | Formalizes target-vol scheme, clips extremes, documents rationale.                  |
# | 8  | Economics           | Hedge PnL; fees/funding mentioned but lighter | **Cell 8 — Hedged returns + fees + funding**            | **Enhanced**             | Clear formulas; prorates funding vs 8h; explicit fee model via turnover.            |
# | 9  | Metrics             | Basic stats / visuals                         | **Cell 9 — VR, Sharpe, MDD, turnover**                  | **Enhanced**             | Writes `metrics.json`; adds turnover as fee proxy; consistent annualization.        |
# | 10 | Visuals             | Spot/hedged plots; some rolling stats         | **Cell 10 — Cum returns, rolling vol, beta vs funding** | **Overlap/Expanded**     | Adds funding vs hedge overlay and saved PNGs for reports.                           |
# | 11 | Paper run log       | Dry-run intent logging (educational)          | **Cell 11 — JSONL paper rebalancer**                    | **Overlap/Stronger ops** | Structured `paper_rebalance_log.jsonl` with full decomposition fields.              |
# | 12 | Next steps          | Where to go next (paper → testnet → research) | **Cell 12 — Same, with deliverables**                   | **Overlap/Expanded**     | Adds deliverables checklist & reporting outputs for stakeholders.                   |
#
# **Quick legend**
#
# * **Overlap** = same concept, similar implementation.
# * **Enhanced** = same concept but with more robust, production-ish outputs (artifacts, metrics, logs).
# * **New focus** = additional step emphasized in MVP for institutional economics (funding integration, auditability).

# %% [markdown]
# ## 🗂️ 0B — Artifact Index (code)

# %%
from pathlib import Path
import json, os

# If you inserted this at the very top, RUN_DIR might not exist yet.
# This will still run later; re-run this cell AFTER the notebook finishes to refresh links.

def human(p: Path):
    return str(p) if p.exists() else "(pending)"

try:
    run_dir = RUN_DIR  # defined in Cell 2 after you run it
except NameError:
    # Fallback: try to guess most recent run folder
    base = Path("runs") / "BTCUSDT" / "hedge_mvp"
    if base.exists():
        candidates = sorted([d for d in base.iterdir() if d.is_dir()], reverse=True)
        run_dir = candidates[0] if candidates else base
    else:
        run_dir = base

artifacts = {
    "Run directory": run_dir,
    "Snapshots": {
        "spot_snapshot.csv": run_dir / "spot_snapshot.csv",
        "funding_snapshot.csv": run_dir / "funding_snapshot.csv",
        "prices_merged.csv": run_dir / "prices_merged.csv",
        "beta_series.csv": run_dir / "beta_series.csv",
        "returns_decomposed.csv": run_dir / "returns_decomposed.csv",
        "perp_snapshot.csv": run_dir / "perp_snapshot.csv",
    },
    "Plots": {
        "cumlogret.png": run_dir / "cumlogret.png",
        "rolling_vol.png": run_dir / "rolling_vol.png",
        "beta_vs_funding.png": run_dir / "beta_vs_funding.png",
        "variance_components.png": run_dir / "variance_components.png",
    },
    "Logs & Metrics": {
        "paper_rebalance_log.jsonl": run_dir / "paper_rebalance_log.jsonl",
        "metrics.json": run_dir / "metrics.json",
        "metadata.json": run_dir / "metadata.json",
    }
}

print("📦 ARTIFACT INDEX\n")
print("Run directory:", human(artifacts["Run directory"]), "\n")

def print_block(title, dct):
    print(f"— {title} —")
    for k, v in dct.items():
        print(f"  {k:28}  {human(v)}")
    print()

print_block("Snapshots", artifacts["Snapshots"])
print_block("Plots", artifacts["Plots"])
print_block("Logs & Metrics", artifacts["Logs & Metrics"])

# Mini metrics preview (if available)
m_path = artifacts["Logs & Metrics"]["metrics.json"]
if isinstance(m_path, Path) and m_path.exists():
    with open(m_path) as f:
        m = json.load(f)
    print("📊 METRICS SNAPSHOT")
    for k in ["variance_reduction_pct", "sharpe_spot", "sharpe_net", "mdd_spot", "mdd_net", "turnover"]:
        if k in m:
            print(f"  {k:24} {m[k]}")
else:
    print("📊 METRICS SNAPSHOT\n  (metrics.json pending — run the full notebook, then re-run this cell)")

# Quick QC for paper log (rows, incidents, span), if present
log_path = artifacts["Logs & Metrics"]["paper_rebalance_log.jsonl"]
if isinstance(log_path, Path) and log_path.exists():
    n_rows = 0
    n_inc = 0
    first_ts = last_ts = None
    try:
        with open(log_path) as f:
            for line in f:
                n_rows += 1
                rec = json.loads(line)
                ts = rec.get("ts")
                if ts:
                    first_ts = first_ts or ts
                    last_ts = ts
                if rec.get("incident"):
                    n_inc += 1
        print("\n🧾 PAPER LOG QC")
        print(f"  rows: {n_rows} | incidents: {n_inc} | span: {first_ts} → {last_ts}")
    except Exception as e:
        print("\n🧾 PAPER LOG QC")
        print("  (could not parse log)", e)



# %% [markdown]
# ## 🔬 Deep-dive analysis of this run
#
# ### 1) What the artifacts say (economics & risk)
#
# - **Variance reduction (VR): 92.42%**
#
# Your hedge removes almost all spot variance. That’s exactly what we expect with BTC spot ↔ BTC-USDT perps at 1-hour bars and β≈1.
#
# - **Sharpe: 1.18 → 1.24 (uplift)**
#
# Despite fees and and an explicit **exit cost** (≈ −20 bp in the last hour), the hedged series still **improves Sharpe**. That means the **tiny residual drift + funding carry** outweigh your frictions over the sample.
#     
# - **Max drawdown: −32.3% → −8.1%**
#
# Stakeholder catnip. It visually matches your plots: the hedged curve is much smoother and shallow.
#     
# - **Turnover (Σ|Δβ|): 56.08** over ~15k hours
#
#   
# That’s **~0.37% of notional per hour** on average. With your fee knobs (taker ~4 bp + optional 2 bp slippage on the perp leg), **ongoing fee drag is microscopic**; the only meaningful fee was the **entry/exit** (charged once).
#     
# - **Funding mapping & sign: correct**
# - 
# You’re using **per-hour funding** (```funding\_rate\_hr``` = 8h/8). With a **short-perp** hedge and **positive** funding, you **receive funding** (small positive ```r\_fund``` each hour). The tail shows ~3e-6 → 7e-6 per hour, i.e., **micro-bps** that add up over time.
#     
# **Hedge stability:**
#
# ```beta\_exec``` ≈ 0.996–0.997, tiny ```|Δβ|``` → **low churn**, **low fee drag**, **stable risk**. Your rolling β / scaled β pipeline is behaving: down-only scaling didn’t engage much because realized vol < target.
#     
#
# ### 2) What’s working (and why)
#
# *   **Risk is basically solved** at this horizon: spot & perp co-move so tightly that β≈1 kills variance.
#     
# *   **Economics are clean**: small positive funding in many hours, negligible ongoing fees, explicit one-offs well accounted.
#     
# *   **Auditability**: CSVs, metrics.json, plot PNGs, and a **JSONL log** (with cadence flags) mean you can replay anything.
#     
#
# ### 3) What to keep an eye on
#
# *   **One-off fees** (entry/exit) can dominate very short slices; report steady-state vs inclusive metrics.
#     
# *   **Funding regimes** change. If funding turns **negative** persistently, being short perp becomes a **carry cost**—we’ll cover how to adapt below.
#     
# *   **Venue specifics**: fee tier, rebates (maker vs taker), and borrow rates if you ever short spot instead of shorting perps.

# %% [markdown]
# ## 🧭 Project goal & framing (what this notebook proves)
#
# This notebook demonstrates a **risk-managed BTC hedge** that aims to **preserve yield while reducing volatility**. We’ll:
# - (1) **validate data quality**,
# - (2) estimate a **hedge ratio** β,
# - (3) optionally **scale β** using a volatility forecast,
# - (4) compute **net hedged returns** with **fees & funding**, and
# - (5) produce **clear, decision-ready charts and metrics**.
#
# **Key formulas**
# *   Log return:
# $$
# r_t = \ln\!\left(\frac{P_t}{P_{t-1}}\right)
# $$
# *   OLS hedge (baseline):
# $$
# r^{\text{spot}}_t = \alpha + \beta \, r^{\text{fut}}_t + \varepsilon_t
# $$
# *   Net hedged return:
# $$
# r^{\text{net}}_t
# =
# r^{\text{spot}}_t
# - \beta^{(\cdot)}_t \, r^{\text{fut}}_t
# + r^{\text{fund}}_t
# + r^{\text{fees}}_t
# $$
#
# **Takeaway:** a reader should finish with **evidence** that the hedge **reduces variance** and **retains net return** after costs.

# %% [markdown]
# ## 📦 Imports, config, and run metadata
#
# We load **config** (symbols, timeframe, paths) and **set run metadata** (timestamped output folder under runs/BTCUSDT/…). This ensures results are **reproducible** and **auditable** across runs.
#
# **Why this matters:** clean config → **consistent experiments**; timestamped outputs → **traceability** for comparisons.

# %%
import os, json, math, time, shutil
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Resolve project root (picked up from Makefile; falls back to CWD) ---
ROOT = Path(os.environ.get("PROJECT_ROOT", Path.cwd())).resolve()
os.environ["PROJECT_ROOT"] = str(ROOT)  # ensure downstream cells see it
print("ROOT:", ROOT)

# --- Config ---
SYMBOL_SPOT = "BTCUSDT"
TIMEFRAME   = "1h"  # working horizon for hedge cycles

RAW_DIR   = ROOT / "data" / "raw"
PROC_DIR  = ROOT / "data" / "processed"
RUNS_DIR  = ROOT / "runs" / SYMBOL_SPOT / "hedge_mvp"

# --- Input files (prefer RAW; fallback to PROCESSED for spot) ---
PATH_SPOT_PARQUET    = RAW_DIR / "binance_spot_BTCUSDT_5m.parquet"
PATH_FUNDING_PARQUET = RAW_DIR / "binance_funding_BTCUSDT.parquet"

if not PATH_SPOT_PARQUET.exists():
    alt = PROC_DIR / "BTCUSDT_5m.parquet"
    if alt.exists():
        print(f"⚠️ RAW spot parquet not found; using processed file: {alt}")
        PATH_SPOT_PARQUET = alt
    else:
        raise FileNotFoundError(
            "Missing spot parquet files:\n"
            f" - {PATH_SPOT_PARQUET}\n"
            f" - {alt}\n"
            "Run your downloader to create the raw file, or place the processed file, then re-run."
        )

if not PATH_FUNDING_PARQUET.exists():
    raise FileNotFoundError(f"Missing funding parquet file: {PATH_FUNDING_PARQUET}")

# --- Perp parquet path (simple, like spot/funding) ---
# Prefer RAW; try a few common filenames. Then fallback to PROCESSED.
CANDIDATES_PERP_RAW = [
    RAW_DIR / "binance_perp_BTCUSDTUSDT_5m.parquet",
    RAW_DIR / "binance_perp_BTCUSDT_5m.parquet",
    RAW_DIR / "binance_usdm_BTCUSDT_5m.parquet",
    RAW_DIR / "binance_futures_BTCUSDT_5m.parquet",
]
CANDIDATES_PERP_PROC = [
    PROC_DIR / "BTCUSDT_perp_5m.parquet",
    PROC_DIR / "BTCUSDT_perp.parquet",
]

PATH_PERP_PARQUET = None
for p in CANDIDATES_PERP_RAW + CANDIDATES_PERP_PROC:
    if p.exists():
        PATH_PERP_PARQUET = p
        break

if PATH_PERP_PARQUET is None:
    raise FileNotFoundError(
        "Missing perp parquet for BTCUSDT.\n"
        "Place one of these files:\n" +
        "\n".join(f"  - {p}" for p in CANDIDATES_PERP_RAW + CANDIDATES_PERP_PROC)
    )

# --- Economics / ops params ---
FEE_RATE    = 0.0004         # taker-equivalent per rebalance (conservative)
TARGET_VOL  = 0.5 / math.sqrt(252*24)  # target per-hour vol (50%/yr as example)
ROLL_VOL_W  = 30             # bars for realized vol
TRAIN_SPLIT = 0.6            # fraction for static beta fit

# --- Annualization factor based on TIMEFRAME ---
if TIMEFRAME.endswith("h"):
    try:
        hours = float(TIMEFRAME[:-1]) or 1.0
    except ValueError:
        hours = 1.0
    ANN_FACTOR = (24 / hours) * 365
elif TIMEFRAME.endswith("m"):
    minutes = float(TIMEFRAME[:-1]) or 1.0
    ANN_FACTOR = (60 / minutes) * 24 * 365
else:
    ANN_FACTOR = 24 * 365  # default

# --- Run metadata / output folder ---
ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
RUN_DIR = RUNS_DIR / ts
RUN_DIR.mkdir(parents=True, exist_ok=True)

meta = {
    "symbol_spot": SYMBOL_SPOT,
    "timeframe": TIMEFRAME,
    "created_utc": ts,
    "paths": {
        "root": str(ROOT),
        "spot_parquet": str(PATH_SPOT_PARQUET),
        "funding_parquet": str(PATH_FUNDING_PARQUET),
        "perp_parquet": str(PATH_PERP_PARQUET),
        "run_dir": str(RUN_DIR),
    },
    "params": {
        "fee_rate": FEE_RATE,
        "target_vol_per_bar": TARGET_VOL,
        "roll_vol_window": ROLL_VOL_W,
        "ann_factor": ANN_FACTOR,
        "train_split": TRAIN_SPLIT,
    },
}
with open(RUN_DIR / "metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print("Run directory:", RUN_DIR)
print("Spot parquet:", PATH_SPOT_PARQUET)
print("Funding parquet:", PATH_FUNDING_PARQUET)
print("Perp parquet:", PATH_PERP_PARQUET)


# %% [markdown]
# ## 🧪 Load raw data & sanity checks
#
# We load:
# - **Spot OHLCV** (e.g., data/raw/binance\_spot\_BTCUSDT\_5m.parquet)
# - **Funding rates** (e.g., data/raw/binance\_funding\_BTCUSDT.parquet)
#
# We then **assert**:
# - **Index timezone** is set and aligned (ideally **UTC**).
# - **No missing values** in key columns (price, funding).
# - **Continuous time** for the chosen sampling (no gaps beyond tolerance).
#
# **Why this matters:** models are only as good as **clean, correctly-timed data** — otherwise results become **misleading**.

# %%
def _ensure_utc_index(df, idx="timestamp"):
    if df.index.name != idx and idx in df.columns:
        df = df.set_index(idx)
    if df.index.tz is None:
        df.index = pd.to_datetime(df.index, utc=True)
    else:
        df.index = df.index.tz_convert("UTC")
    return df.sort_index()

# Spot OHLCV (expects columns like: open, high, low, close, volume)
spot = pd.read_parquet(PATH_SPOT_PARQUET, engine="pyarrow")
spot = _ensure_utc_index(spot)

# Funding (expects columns: timestamp, fundingRate; cadence ~8h)
fund = pd.read_parquet(PATH_FUNDING_PARQUET, engine="pyarrow")
fund = _ensure_utc_index(fund)

# Perp OHLCV (expects columns like: open, high, low, close, volume)
# NOTE: PATH_PERP_PARQUET should be defined in your Config cell (as suggested earlier)
perp = pd.read_parquet(PATH_PERP_PARQUET, engine="pyarrow")
perp = _ensure_utc_index(perp)

# Spans / tz
print("Spot:",    spot.shape, "Span:", spot.index.min(), "→", spot.index.max(), "UTC tz:", spot.index.tz)
print("Funding:", fund.shape, "Span:", fund.index.min(), "→", fund.index.max(), "UTC tz:", fund.index.tz)
print("Perp:",    perp.shape, "Span:", perp.index.min(), "→", perp.index.max(), "UTC tz:", perp.index.tz)

# Simple sanity checks
assert spot["close"].isna().sum() == 0, "NA in spot close"
assert fund.isna().sum().sum() == 0, "NA in funding data"
assert perp["close"].isna().sum() == 0, "NA in perp close"

# Persist snapshots for audit
spot.to_csv(   RUN_DIR / "spot_snapshot.csv")
fund.to_csv(   RUN_DIR / "funding_snapshot.csv")
perp.to_csv(   RUN_DIR / "perp_snapshot.csv")


# %% [markdown]
# ### 1) Critical analysis of results
#
# **Spot data**
#
# *   Shape (180,853, 5) → ~180k rows, 5 OHLCV fields. Matches expectation for ~21 months of 1-minute data.
#     
# *   Span: **Jan 1, 2024 → Sep 19, 2025 (23:00)**.
#     
# *   Clean: no missing closes, UTC tz-aware, sorted.
#     
#
# **Funding data**
#
# *   Shape (1,884, 1) → only fundingRate.
#     
# *   Cadence: ~3/day → 627 days × 3 = 1,881 expected, we see 1,884. Perfect match (includes edge settlements).
#     
# *   Span: **Jan 1, 2024 → Sep 19, 2025 (16:00)**.
#     
# *   End slightly earlier than spot, as expected (funding settles in fixed blocks).
#     
#
# **Perp data**
#
# *   Shape (183,360, 5) → 183k rows, 5 OHLCV fields.
#     
# *   Span: **Jan 1, 2024 → Sep 28, 2025 (15:55)**. This **extends beyond spot/funding by 9 days**.
#     
# *   Coverage is consistent with continuous 1-minute perps, ~183k minutes = ~127 days equivalent, which matches 21 months of uninterrupted trading.
#     
# *   Clean: no missing closes, UTC tz-aware.
#     
#
# **Overlap / mismatches**
#
# *   Spot ends **Sep 19**, perp continues to **Sep 28**.
#     
# *   Funding ends **Sep 19**, so funding < spot < perp in terms of length.
#     
# *   This creates three zones:
#     
#     1.  **Jan 2024 → Sep 19 2025 16:00** → full overlap across all three.
#         
#     2.  **Sep 19 16:00 → Sep 19 23:00** → spot + perp only.
#         
#     3.  **Sep 19 23:00 → Sep 28 15:55** → perp only.
#         
#
# This means for backtests with funding included, you must truncate at **Sep 19 16:00**. For pure spot–perp hedge tests, you can extend until **Sep 19 23:00** (spot end). Beyond that, only perp continues.
#
# ### 2) Comparison with expected results
#
# *   **Spot**: as expected for 1m OHLCV.
#     
# *   **Funding**: perfectly consistent with 8h cadence, slightly shorter tail (common).
#     
# *   **Perp**: shape and span are correct for 1m Binance perps.
#     
# *   The fact that perps extend 9 days beyond spot is **expected** because perpetuals trade continuously and are often easier to query in real-time.
#     
#
# No anomalies detected in alignment, except that the three series end at slightly different times.
#
# ### 3) Suggestions for improvement
#
# *   **Span harmonization**:
#     
#     *   If hedge tests require funding, trim all datasets to funding’s end: **Sep 19, 16:00**.
#         
#     *   If only spot vs. perp hedge (ignoring funding), trim to spot’s end: **Sep 19, 23:00**.
#         
#     *   Avoid extending analysis into the “perp-only” region.
#         
# *   **Continuity checks**:
#     
#     *   Confirm perp has no missing minutes: perp.resample("1min").size().value\_counts().
#         
#     *   Same for spot (already noted earlier).
#         
# *   **Cross-instrument alignment**:
#     
#     *   Spot and perp are from different feeds (spot vs. USD-M perpetuals). Small timing drifts (off by a second/minute) can occur. Worth checking with a left-join on timestamps.
#         
# *   **Future prep**:
#     
#     *   When merging, resample spot/perp to 1h and forward-fill funding.
#         
#     *   Consider logging the last aligned timestamp of each dataset inside your run folder for reproducibility.
#         
#
# ### 4) ✅ Good to go / ⚠️ Potential issues
#
# **✅ Good to go**:
#
# *   All three datasets are healthy, UTC-aligned, no NaNs in closes.
#     
# *   Funding cadence correct, perp extension normal.
#     
# *   Ready for hedging pipeline with real perp values.
#     
#
# **⚠️ Potential issues**:
#
# *   Small end-date mismatches must be handled explicitly to avoid biased results.
#     
# *   Need to check continuity (no missing minutes) in both spot and perp.
#     
# *   When merging spot + perp + funding, **choose the cutoff carefully** (Sep 19 16:00 for funding-aware analysis, Sep 19 23:00 for pure hedge tests).

# %% [markdown]
# ## ⏱️ Resampling & returns (definitions)
#
# We resample to the working horizon (e.g., 5-minute or 1-hour) and compute log returns.
#
# **Definitions:**
#
# Log return over interval $t$:
# $$
# r_t = \ln\!\left(\frac{P_t}{P_{t-1}}\right)
# $$
#
# Realized variance (window $W$):
# $$
# \widehat{\sigma}^2_t = \sum_{i=t-W+1}^{t} r_i^{2}
# $$
#
# **Why log returns?** They **add over time**, simplify math, and handle compounding naturally.
#

# %%
def resample_close(df, timeframe="1h"):
    # df must have a DatetimeIndex (UTC is fine)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("resample_close: df.index must be a DatetimeIndex")
    out = df[["close"]].resample(timeframe).last()
    out = out.dropna()
    return out
    
print("Index tz:", spot.index.tz)
print("Span:", spot.index.min(), "→", spot.index.max())
print("Rows:", len(spot))


# %%
spot_h = resample_close(spot, TIMEFRAME)
spot_h["r_spot"] = np.log(spot_h["close"]).diff()
spot_h = spot_h.dropna()

print(spot_h[["close","r_spot"]].head())
print("Bars:", len(spot_h))


# %%
# --- 1) Perp: resample + hourly log-returns ---
perp_h = resample_close(perp, TIMEFRAME).rename(columns={"close": "close_perp"})
perp_h["r_perp"] = np.log(perp_h["close_perp"]).diff()
perp_h = perp_h.dropna()

# --- 2) Merge spot & perp on the hourly grid (inner to keep only common hours) ---
spot_h2 = spot_h.rename(columns={"close": "close_spot"})  # from your previous cell
merged = spot_h2.join(perp_h, how="inner")

# --- 3) Funding onto the same hourly grid ---
# Funding is recorded every 8h. We’ll create:
#  - funding_discrete_hr: 0 on non-settlement hours; equals the fundingRate at settlement hour
#  - funding_prorated_hr: fundingRate/8 applied to each hour until the next settlement (carry view)

# First, get the last observed fundingRate at the settlement hour on an hourly grid
fund_h_last = fund[["fundingRate"]].resample(TIMEFRAME).last()  # NaN except on settlement hours

# Discrete model: non-settlement hours = 0
fund_discrete = fund_h_last.fillna(0.0).rename(columns={"fundingRate": "funding_discrete_hr"})

# Prorated model: forward-fill the last known settlement rate across the next 8 hours, divided by 8
fund_ffill = fund_h_last.ffill()
fund_prorated = (fund_ffill / 8.0).rename(columns={"fundingRate": "funding_prorated_hr"})

# Join funding into merged hourly dataframe
merged = merged.join(fund_discrete, how="left").join(fund_prorated, how="left")

# --- 4) Trim span depending on whether you want funding in-sample ---
# If you will use funding in the backtest, trim to the last funding timestamp to avoid lookahead.
last_fund_ts = fund.index.max().floor(TIMEFRAME)
merged_funding_span = merged.loc[:last_fund_ts].copy()

# If you only hedge spot vs perp (no funding terms), you can use the full overlap:
last_spot_ts = spot.index.max().floor(TIMEFRAME)
merged_spot_perp_span = merged.loc[:last_spot_ts].copy()

# --- 5) Final sanity checks & quick peek ---
def _qc(df, name):
    assert df[["r_spot", "r_perp"]].isna().sum().sum() == 0, f"NaNs in returns for {name}"
    print(
        f"{name}: rows={len(df)}, span={df.index.min()} → {df.index.max()}, "
        f"any NA funding cols? ",
        df[["funding_discrete_hr","funding_prorated_hr"]].isna().any().to_dict()
    )
    display(df.head(3))
    display(df.tail(3))

_qc(merged_funding_span, "merged_funding_span (for tests that include funding)")
_qc(merged_spot_perp_span, "merged_spot_perp_span (spot↔perp only)")


# %% [markdown]
# ### 1) Critical analysis of results
#
# **merged\_funding\_span (with funding)**
#
# *   **Rows = 15,064** covering **2024-01-01 01:00 → 2025-09-19 16:00**.
#     
# *   Both funding columns (```funding\_discrete\_hr```, ```funding\_prorated\_hr```) show **no NaNs** — the trimming at the funding horizon worked.
#     
# *   Early rows: spot and perp prices close together (≈42.6k), returns aligned and plausible.
#     
# *   Last row (Sep 19, 16:00): funding shows a discrete settlement (```0.000053```) and prorated share (```0.000007```), which matches Binance’s ~0.01% funding (per 8h → per hour).
#     
# *   Spot vs. perp returns track each other tightly (```-0.003313``` vs ```-0.003352``` at 14:00, then +0.001764 vs +0.001751 at 16:00).
#     
#
# **merged\_spot\_perp\_span (spot↔perp only)**
#
# *   **Rows = 15,071** covering **2024-01-01 01:00 → 2025-09-19 23:00**.
#     
# *   Includes the 7 extra hours after funding stops. As expected, funding columns are **NaN** in the tail (21:00–23:00).
#     
# *   Returns continue to look consistent and close between spot/perp.
#     
#
# ### 2) Comparison with expected results
#
# *   **Row counts**:
#     
#     *   With funding: 15,064 hours.
#         
#     *   Without funding: 15,071 hours.→ The 7-hour difference matches exactly the observed cutoff (```Sep 19 16:00``` vs ```Sep 19 23:00```).
#         
# *   **Return magnitudes**: consistent with BTC’s hourly volatility. Spot and perp track each other with minimal slippage — exactly what we expect in a liquid market like BTC/USDT.
#     
# *   **Funding**:
#     
#     *   Discrete series only non-zero at settlement hours.
#         
#     *   Prorated smooths into ~0.000003–0.000007 per hour. Both forms align with theoretical expectation.
#         
#
# ### 3) Suggestions for improvement
#
# *   **Explicit choice**: Keep two dataframes but be very clear in the code which one you use depending on the test:
#     
#     *   Hedge ratio + PnL including funding → ```merged\_funding\_span```.
#         
#     *   Hedge ratio only (spot vs perp) → ```merged\_spot\_perp\_span```.
#  
# *   **Continuity check**:
#         
# ```assert merged\_funding\_span.asfreq("1h").index.equals(merged\_funding\_span.index)```
#
# This guarantees there are no missing hourly bars inside the trimmed set.
#     
# *   **Funding diagnostics**: Plot funding\_discrete vs funding\_prorated distributions — should center near zero, tails around ±0.01% (Binance max ±0.75% daily).
#     
# *   **Basis check**: Plot ```close\_perp – close\_spot``` over time. This reveals whether perps trade rich/cheap vs spot, and how funding compensates.
#     
#
# ### 4) ✅ Good to go / ⚠️ Potential issues
#
# **✅ Good to go**
#
# *   Both merged datasets are clean, UTC, with aligned returns.
#     
# *   Funding is forward-filled and behaves as expected.
#     
# *   Returns are stable and consistent.
#     
#
# **⚠️ Potential issues**
#
# *   The last 7 hours (Sep 19 16–23h) can sneak in funding-less data if you’re not careful. Be explicit about your cutoff.
#     
# *   Minor risk of missing hours (should confirm with ```.asfreq()``` check).
#     
# *   Spot vs perp microstructure gaps (<1bp/hour) may exist but are likely negligible at hourly resolution.
#     
#
# ### 5) 📈 Interpretation for trading impact
#
# *   Having both **funding-aware** and **funding-agnostic** datasets lets you separate pure hedge variance reduction (β estimation) from **carry PnL attribution**.
#     
# *   Because spot and perp returns are tightly matched, most residual risk will come from **funding** and occasional dislocations (basis moves).
#     
# *   Choosing discrete vs prorated funding representation changes how you model PnL smoothness:
#     
#     *   Discrete = matches exchange statements.
#         
#     *   Prorated = smoother hourly attribution, easier for regressions and forecasting.

# %% [markdown]
# ## 🔗 Align spot vs. hedge instrument & merge funding
#
# We align the **spot BTC** series with the hedge instrument (**BTC-USDT perpetual futures**). Because both reference the same underlying, returns are **highly correlated** — this is exactly what makes perps a suitable hedge.
#
# We also merge the **funding rate** onto the same hourly grid. Binance quotes funding every **8h**; we forward-fill to hourly bars and store both:
#
# *   **funding\_rate\_8h** (as posted by the exchange),
#     
# *   **funding\_rate\_hr** (8h rate divided by 8, for hourly PnL attribution).
#     
#
# **Key checks when using perps**:
#
# *   **Timestamp alignment** — no lookahead between spot and perp series.
#     
# *   **Return correlation** — should be very high (close to 1); if not, the hedge will be weak.
#     
# *   **Funding cadence** — confirm the 8h → hourly mapping is correct.
#     
# *   **Basis sanity** — perp and spot prices should be nearly identical, with deviations explained by funding over time.
#     
#
# Because we now have **real perp prices**, there is no need to approximate with spot returns. Hedge ratio estimation will therefore reflect the true micro-basis dynamics between spot and perps.
#
# **📈 Trading impact**
#
# With spot and perp returns nearly identical, a hedge ratio β ≈ 1 delivers almost complete variance reduction. That means the **main risk driver shifts away from price moves** and into:
#
# *   **Funding costs** — the periodic payments that keep perps tethered to spot.
#     
# *   **Basis drifts** — short-term deviations between spot and perp that may not be fully explained by funding.
#     
#
# In practice, this setup highlights that a delta-neutral strategy isn’t about eliminating BTC risk (that’s solved by the hedge), but about **managing the cash flows from funding and execution frictions**.

# %%
import statsmodels.api as sm

# --- 1) Resample perp closes to TIMEFRAME and compute returns ---
perp_h = resample_close(perp, TIMEFRAME).rename(columns={"close": "close_perp"})
perp_h["r_fut"] = np.log(perp_h["close_perp"]).diff()
perp_h = perp_h.dropna()

# --- 2) (Optional) rename spot close for clarity; keep r_spot as-is ---
spot_h = spot_h.rename(columns={"close": "close_spot"})

# --- 3) Funding: resample to TIMEFRAME and forward-fill ---
fund_h = fund.resample(TIMEFRAME).last()
fund_h["funding_rate_8h"] = (fund_h["fundingRate"] if "fundingRate" in fund_h else fund_h.iloc[:, 0]).astype(float)
fund_h["funding_rate_8h"] = fund_h["funding_rate_8h"].ffill()     # piecewise-constant in 8h blocks
fund_h["funding_rate_hr"] = fund_h["funding_rate_8h"] / 8.0       # per-hour attribution

# --- 4) Merge (inner join to align spot, perp, funding) ---
df = (
    spot_h[["close_spot", "r_spot"]]
    .join(perp_h[["close_perp", "r_fut"]], how="inner")
    .join(fund_h[["funding_rate_8h", "funding_rate_hr"]], how="left")
)

# Backfill leading funding gaps at the very start if needed
if df["funding_rate_8h"].isna().any():
    df[["funding_rate_8h","funding_rate_hr"]] = df[["funding_rate_8h","funding_rate_hr"]].bfill()

# Sanity: ensure r_fut != r_spot (otherwise you’re still using proxy)
if np.allclose(df["r_spot"].values, df["r_fut"].values, rtol=0, atol=0):
    raise RuntimeError("r_fut equals r_spot — check that you loaded real perp data, not the proxy.")

# Save snapshot
df.to_csv(RUN_DIR / "prices_merged.csv")
print("Merged shape:", df.shape)
print(df.head(8))

# --- 5) Trim spans ---
# With funding (cut at last funding timestamp)
df_funding_span = df.loc[:fund.index.max().floor(TIMEFRAME)].copy()

# Spot↔perp only (cut at last spot timestamp)
df_spot_perp_span = df.loc[:spot.index.max().floor(TIMEFRAME)].copy()

# --- 6) Quick checks ---
assert df_funding_span.asfreq("1h").index.equals(df_funding_span.index)
print("corr(r_spot, r_fut):", df_funding_span["r_spot"].corr(df_funding_span["r_fut"]))

# --- 7) OLS regression helper ---
def estimate_beta(df, y_col="r_spot", x_col="r_fut", add_const=True, label=""):
    y = df[y_col]
    X = df[[x_col]]
    if add_const:
        X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    print(f"\n=== OLS Hedge Ratio (dataset: {label}) ===")
    print(model.summary().tables[1])  # coefficients table
    return model

# --- 8) Run regression on both datasets ---
model_with_funding = estimate_beta(df_funding_span, x_col="r_fut", label="with funding span")
model_spot_perp_only = estimate_beta(df_spot_perp_span, x_col="r_fut", label="spot↔perp span")

# --- 9) Extract betas ---
beta_with_funding = model_with_funding.params["r_fut"]
beta_spot_perp_only = model_spot_perp_only.params["r_fut"]

print(f"\nHedge ratio β (with funding span): {beta_with_funding:.6f}")
print(f"Hedge ratio β (spot↔perp only):     {beta_spot_perp_only:.6f}")


# %% [markdown]
# ### 1) Critical analysis of results
#
# **Merged grid & funding units**
#
# *   ```Merged shape: (15071, 6)``` matches the **spot↔perp span** (through 2025-09-19 23:00 UTC).
#     
# *   Funding is present in **both units**: ```funding\_rate\_8h``` (exchange’s 8h print) and ```funding\_rate\_hr = 8h/8```. Early hours show a steady 0.000374 (8h) → 0.000047 (per hour), then a step to 0.000272 → 0.000034 — exactly what we expect from Binance’s piecewise-constant funding.
#     
#
# **Return alignment & co-movement**
#
# *   ```corr(r\_spot, r\_fut) = 0.99969``` → extremely high, but **not identical** (good sign we’re using real perps, not a proxy).
#     
# *   OLS on **with-funding span** and **spot↔perp span** both give:
#     
#     *   **β ≈ 1.00012** (slightly > 1), meaning perps under-react by a hair vs. spot; you need ~0.012% _more_ perp exposure than a 1:1 to minimize variance.
#         
#     *   **α ≈ 0** (insignificant), as expected for two nearly identical BTC instruments.
#         
#     *   Enormous t-stats (≈ 4900) reflect near-perfect co-movement, not a modeling issue.
#         
#
# **Tails & spans**
#
# *   Funding-trimmed sample ends **2025-09-19 16:00**; spot↔perp sample extends to **23:00** with funding NaNs (correctly excluded in the funding analysis).
#     
#
# ### 2) Comparison with expected results
#
# *   **β ~ 1** is the textbook outcome for BTC spot vs BTC-USDT perps at hourly frequency.
#     
# *   **Correlation ~0.9997** is consistent for a highly liquid pair with tight basis.
#     
# *   Funding magnitudes (~0.03–0.04% per 8h) are typical near neutral; stepwise behavior matches the 8h cadence.
#     
#
# ### 3) Suggestions for improvement
#
# *   **Hedge effectiveness (variance reduction):** compute ```var(r\_spot)``` vs ```var(r\_spot − β·r\_fut)``` and report % reduction. Also check **Sharpe** of the hedged leg.
#     
# *   **Robust SEs:** the residuals can be serially correlated at 1h; use **Newey–West (HAC)** SEs for inference (β point estimate won’t change much).
#     
# *   **Rolling stability:** estimate a **rolling β** (e.g., 30d/60d windows). Stable β ~1 supports a static hedge; drift suggests rolling recalibration.
#     
# *   **Basis diagnostics:** plot ```close\_perp − close\_spot``` and compare to cumulated funding; the two should co-move over medium horizons.
#     
# *   **Cutoffs:** keep using the **funding-trimmed sample** when attributing PnL with funding; use the **spot↔perp sample** for pure hedge tests.
#     
#
# ### 4) ✅ Good to go / ⚠️ Potential issues
#
# **✅ Good to go**
#
# *   Clean hourly grid, correct funding units (8h and per-hour), real perp returns confirmed (r\_spot ≠ r\_fut), β ≈ 1 with α ≈ 0.
#     
#
# **⚠️ Potential issues**
#
# *   Don’t **double-count** funding: if you aggregate hourly PnL, use funding\_rate\_hr; if you work at 8h boundaries, use funding\_rate\_8h.
#     
# *   Extremely high correlation means tiny numerical differences dominate; be careful with **float precision** in PnL attribution and logging.
#     
#
# ### 5) 📈 Interpretation for trading impact
#
# *   A **β just above 1** implies you need a touch more short-perp notional than spot to minimize variance. At hourly cadence, this is expected to be very close to 1.
#     
# *   With such tight co-movement, the **residual risk** in a delta-neutral book comes mostly from **funding** and occasional basis shifts, not from price divergence.
#     
# *   This setup is ready for a production-style test of **variance reduction**, **carry capture**, and **execution frictions** (fees, slippage, min notional).

# %% [markdown]
# ## 📏 Estimate hedge ratio 𝛽 via OLS (static & optional rolling)
#
# We estimate the optimal hedge ratio $\beta$ with an OLS regression of spot returns on futures returns (no intercept or with a small $\alpha$):
#
# $$
# r^{\text{spot}}_t = \alpha + \beta \, r^{\text{fut}}_t + \varepsilon_t
# $$
#
# The hedge position scales with $\beta$: long $1$ unit of spot, short $\beta$ units of futures (sign may flip depending on instrument convention).
#
# **Interpretation:**
# - $\beta \approx 1$ $\;\rightarrow\;$ near one-to-one hedge.  
# - $\beta < 1$ $\;\rightarrow\;$ partial hedge; often prudent when costs exist.  
# - **Stability**: We compute a static hedge ratio $\beta$ on a **training window**, and (optionally) a **rolling** $\beta_t$.

# %%
def ols_beta(y, x):
    # y = a + b x + e  → solve for [a,b] via least squares
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = coef
    return a, b

n = len(df)
split = int(n * TRAIN_SPLIT)
train = df.iloc[:split].copy()
test  = df.iloc[split:].copy()

alpha_hat, beta_hat = ols_beta(train["r_spot"].values, train["r_fut"].values)
print(f"Static OLS: alpha={alpha_hat:.6g}, beta={beta_hat:.4f}")

# Optional: rolling beta for adaptivity
ROLL_WIN = 200  # bars; adjust as needed
betas_roll = []
x = df["r_fut"].values
y = df["r_spot"].values
for i in range(len(df)):
    if i < ROLL_WIN:
        betas_roll.append(np.nan)
    else:
        a, b = ols_beta(y[i-ROLL_WIN+1:i+1], x[i-ROLL_WIN+1:i+1])
        betas_roll.append(b)

df["beta_static"] = beta_hat
df["beta_roll"]   = pd.Series(betas_roll, index=df.index)

# Choose which beta to use downstream
USE_ROLLING_BETA = True
df["beta"] = df["beta_roll"] if USE_ROLLING_BETA else df["beta_static"]

df[["r_spot","r_fut","beta","beta_static","beta_roll"]].to_csv(RUN_DIR / "beta_series.csv")
df.tail(3)


# %%
import numpy as np
import matplotlib.pyplot as plt

# --- 1) Rolling vs static β plot ---
plt.figure()
plt.plot(df.index, df["beta_roll"], label="Rolling β (200h window)")
plt.axhline(df["beta_static"].iloc[0], color="red", linestyle="--", label=f"Static β = {beta_hat:.4f}")
plt.title("Rolling vs Static Hedge Ratio β")
plt.xlabel("Time")
plt.ylabel("β")
plt.legend()
plt.grid(True)

# --- 2) Hedged returns (static vs rolling β) ---
r_hedged_static = df["r_spot"] - df["beta_static"] * df["r_fut"]
r_hedged_roll   = df["r_spot"] - df["beta_roll"].fillna(df["beta_static"]) * df["r_fut"]

# --- 3) Variance reduction (whole sample) ---
vr_static = 1 - (r_hedged_static.var(ddof=1) / df["r_spot"].var(ddof=1))
vr_roll   = 1 - (r_hedged_roll.var(ddof=1)   / df["r_spot"].var(ddof=1))

print(f"\nVariance reduction (whole sample):")
print(f"  Static β : {vr_static*100:.4f}%")
print(f"  Rolling β: {vr_roll*100:.4f}%")

# --- 4) Sharpe ratio comparison ---
def sharpe_annualized(ret, hours_per_year=24*365):
    mu = ret.mean()
    sd = ret.std(ddof=1)
    return (mu / sd) * np.sqrt(hours_per_year)

sr_unhedged = sharpe_annualized(df["r_spot"])
sr_static   = sharpe_annualized(r_hedged_static)
sr_roll     = sharpe_annualized(r_hedged_roll)

print(f"\nSharpe ratio (annualized, hourly returns):")
print(f"  Unhedged (spot):     {sr_unhedged:.3f}")
print(f"  Hedged (static β):   {sr_static:.3f}")
print(f"  Hedged (rolling β):  {sr_roll:.3f}")

# --- 5) Rolling variance comparison (static vs rolling hedge) ---
WIN = 500  # ~21 days of hourly data
roll_var_spot   = df["r_spot"].rolling(WIN).var()
roll_var_static = r_hedged_static.rolling(WIN).var()
roll_var_roll   = r_hedged_roll.rolling(WIN).var()

plt.figure()
plt.plot(roll_var_spot.index, roll_var_spot, label="Unhedged var (spot)")
plt.plot(roll_var_static.index, roll_var_static, label="Hedged var (static β)")
plt.plot(roll_var_roll.index, roll_var_roll, label="Hedged var (rolling β)")
plt.title(f"Rolling Variance (window={WIN}h)")
plt.xlabel("Time")
plt.ylabel("Variance")
plt.legend()
plt.grid(True)

# %% [markdown]
# ## 🎚️ Volatility-aware scaling of β (optional, keeps risk steady)
#
# To tame **over-hedging in turbulence** and avoid **levering up in calm markets**, we scale the hedge ratio using a recent volatility signal (e.g., rolling realized vol).
#
# **Rule (down-only):**
# $$
# \beta^{\text{scaled}}_t
# =
# \beta \times \min\!\left(1,\;
# \frac{\widehat{\sigma}^{\text{target}}}{\widehat{\sigma}^{\text{recent}}_t + \epsilon}
# \right)
# $$
#
# - $ \widehat{\sigma}^{\text{recent}}_t $ = **per-bar** realized volatility over the last ```ROLL_VOL_W``` bars.
#
# - $ \widehat{\sigma}^{\text{target}} $ = **per-bar** target volatility (consistent with your timeframe).
#
# - $ \epsilon $ is a small constant to avoid division by zero.
#
# **Why it matters**: keeps **risk steadier**, often improving **Sharpe** and **drawdown**, while also **reducing churn** (smaller notional when markets are noisy).
#
# **Practical notes**
#
# - Down-only: if $ \widehat{\sigma}^{\text{recent}}_t < \widehat{\sigma}^{\text{target}} $, the scale is 1 (no leverage-up).
#
# - Warm-up: the first ```ROLL_VOL_W−1``` bars have undefined (NaN) vol; drop or forward-fill if needed.
#
# - Tuning: if scaling rarely engages (average scale $\approx 1$), either lower the target or shorten the window.
#
# - Interpretation: if recent vol is 3× the target, the scale becomes $\approx 1/3$, shrinking the hedge accordingly.
#
# **Instrumented output**
#
# - We compute and store:
#
#     - rv (recent realized vol),
#     
#     - $ \text{scale} = \min!\left(1,; \dfrac{\widehat{\sigma}^{\text{target}}}{\widehat{\sigma}^{\text{recent}}_t + \epsilon} \right) $,
#     
#     - $ \beta_{\text{scaled}} = \beta \times \text{scale} $,
#     
#     - and we print the ```average scale factor``` to show whether scaling was active in this run.

# %%
eps = 1e-8

# Recent realized volatility per bar (same timeframe as df)
df["rv"] = df["r_spot"].rolling(ROLL_VOL_W).std()

# Down-only scale factor (cap at 1.0)
df["scale"] = np.minimum(1.0, TARGET_VOL / (df["rv"] + eps))

# Apply scaling and clip extremes for safety
df["beta_scaled"] = (df["beta"] * df["scale"]).clip(lower=-3.0, upper=3.0)

# Quick summary
print(f"Avg scale factor: {df['scale'].mean():.6f}")
df[["beta", "rv", "scale", "beta_scaled"]].tail(5)

# --- Optional refinements (leave commented unless you want them) ---
# # Deadband to avoid tiny rebalances (e.g., ignore <5 bps changes in scale)
# DEADBAND = 0.0005  # 5 bps on scale
# scale_raw = np.minimum(1.0, TARGET_VOL / (df["rv"] + eps))
# scale_smooth = scale_raw.where((scale_raw - 1.0).abs() > DEADBAND, 1.0)
# df["scale"] = scale_smooth
# df["beta_scaled"] = (df["beta"] * df["scale"]).clip(lower=-3.0, upper=3.0)

# # EWMA vol alternative (smoother than rolling std)
# # LAMBDA = 0.94
# # ret2 = (df["r_spot"]**2).to_numpy()
# # ewma_var = np.zeros_like(ret2)
# # for i, r2 in enumerate(ret2):
# #     ewma_var[i] = LAMBDA*ewma_var[i-1] + (1-LAMBDA)*r2 if i > 0 else r2
# # df["rv_ewma"] = np.sqrt(ewma_var)
# # df["scale"] = np.minimum(1.0, TARGET_VOL / (df["rv_ewma"] + eps))
# # df["beta_scaled"] = (df["beta"] * df["scale"]).clip(lower=-3.0, upper=3.0)



# %% [markdown]
# ### 1) Critical analysis
#
# *   **Down-only policy**: ```scale = min(1, TARGET\_VOL / rv)``` will **not lever up** when markets are calm. That’s operationally safer and reduces over-trading in quiet regimes.
#     
# *   **Current run**: your tail shows ```rv ≈ 0.00185``` (≈0.185%/h) while ```TARGET\_VOL ≈ 0.00643``` (50%/yr → ≈0.643%/h). Since ```rv < TARGET\_VOL```, ```scale = 1.0```, so ```beta\_scaled == beta```. The printed **Avg scale factor** should be ~1.0, confirming scaling is inactive in this sample.
#     
# *   **Stability**: retains the excellent β you estimated (≈1). When volatility spikes, ```scale < 1``` will automatically shrink hedge notional and cut fee/funding exposure.
#     
#
# ### 2) Comparison with expected results
#
# *   For **BTC hourly**, realized vol often sits below a 50%/yr target, so a down-only scaler frequently stays at **1.0** — exactly what you observed.
#     
# *   In stressed windows (e.g., large moves), ```rv``` rises; you’ll see ```scale < 1``` kick in, reducing hedge size and **stabilizing hedged variance** without re-estimating β.
#     
#
# ### 3) Suggestions for improvement
#
# *   **Make scaling visible**: keep the ```scale``` column (added above). Plot it alongside rolling volatility to explain when/why the hedge shrinks.
#     
# *   **Deadband** (optional, commented): avoids tiny, noisy rebalances that add fees without benefit.
#     
# *   **EWMA vol** (optional, commented): smoother and quicker to react than simple rolling std; consider for live runs.
#     
# *   **Target tuning**: if you want scaling to _actually_ engage more often, try a lower target (e.g., 30%/yr) or make it regime-dependent (e.g., lower in high-fee environments).
#     
#
# ### 4) ✅ Good to go / ⚠️ Potential issues
#
# **✅ Good to go**
#
# *   Works out-of-the-box, integrated with your downstream PnL cell (which imports ```beta\_scaled``` into ```df\_funding\_span```).
#     
# *   Safety clips on ```beta\_scaled``` prevent pathological notional.
#     
#
# **⚠️ Potential issues**
#
# *   **Warm-up NaNs**: first ```ROLL\_VOL\_W-1``` rows of ```rv``` are NaN. If those rows are used downstream, consider ```ffill()``` or drop the warm-up period.
#     
# *   **Parameter sensitivity**: scaling depends on ```ROLL\_VOL\_W``` and ```TARGET\_VOL```. Document defaults and rationale in the notebook header for reproducibility.
#     
#
# ### 5) 📈 Interpretation for trading impact
#
# *   When **scale = 1** (calm markets), your economics are unchanged: hedge behaves like the pure OLS β.
#     
# *   When **scale < 1** (turbulence), you trim hedge notional, which:
#     
#     *   Lowers **fee drag** (less turnover in notional),
#         
#     *   Shrinks **funding exposure** (β smaller → funding cash flows smaller),
#         
#     *   Trades off a bit more **tracking error** for **smoother risk**.
#         
# *   Net effect: a more **robust hedged PnL** across regimes, with clearer guardrails on risk and costs.

# %% [markdown]
# ## 🧮 Hedged returns, fees, and funding (economics)
#
# We combine spot returns, hedge PnL, trading fees, and funding to get the net hedged return.
#
# **Hedge PnL (per interval) when short futures:**
# $$
# r^{\text{hedge}}_t \approx -\,\beta^{(\cdot)}_t \, r^{\text{fut}}_t
# $$
#
# **Funding accrual (per interval length $\Delta t$ as fraction of 8h, if funding is quoted per 8h):**
# $$
# r^{\text{fund}}_t
# =
# \beta^{(\cdot)}_t \times \text{FundingRate}_t \times \frac{\Delta t}{8\text{h}}
# $$
#
# **Fees (maker/taker, both legs when rebalancing):**
# $$
# r^{\text{fees}}_t \approx -\,\text{fee\_rate} \times \left|\Delta \beta^{(\cdot)}_t\right|
# $$
#
# **Net hedged return:**
# $$
# r^{\text{net}}_t
# =
# r^{\text{spot}}_t
# + r^{\text{hedge}}_t
# + r^{\text{fund}}_t
# + r^{\text{fees}}_t
# $$
#
# **Note**: use instrument conventions (signs) carefully; verify with a **small backtest snippet** and **unit tests** on toy data.

# %%
# %% [markdown]
# ## 🧮 Hedged returns, fees, and funding (economics)
# Advanced Cell 8 — production-style accounting (with fee-sign fix)
# - Uses **df_funding_span** (trimmed to last funding print → no lookahead).
# - Funding column is **funding_rate_hr** (per-hour). If only 8h is present, derive hr = 8h / 8.
# - Sign convention: long 1 spot, short β perps → positive funding means **shorts pay** (negative PnL).
# - Saves decomposition to **runs/.../returns_decomposed.csv** (Artifact Index).
# - Syncs outputs back into **df** so later cells (metrics/plots) continue to work.

# %%
import numpy as np
import pandas as pd

# ========= CONFIG =========
DF = df_funding_span.copy()          # ← funding-trimmed sample (recommended for economics)
RET_SPOT = "r_spot"
RET_FUT  = "r_fut"
FUND_HR  = "funding_rate_hr"         # hourly funding rate (8h/8)

# Ensure funding_rate_hr exists (fallback if only 8h is present)
if FUND_HR not in DF.columns:
    if "funding_rate_8h" in DF.columns:
        DF[FUND_HR] = DF["funding_rate_8h"].astype(float) / 8.0
    elif "funding_rate" in DF.columns:
        # If 'funding_rate' is the 8h print, convert to hourly
        DF[FUND_HR] = DF["funding_rate"].astype(float) / 8.0
    else:
        raise KeyError("Neither 'funding_rate_hr' nor 'funding_rate_8h' (or 'funding_rate') found in DF.")

# Bring over beta series if they were computed on a different DataFrame (df)
for col in ["beta", "beta_scaled", "beta_roll", "beta_static"]:
    if col not in DF.columns and col in df.columns:
        DF[col] = df[col].reindex(DF.index)

# ========= Beta source selection (robust) =========
USE_ROLLING_BETA = True  # prefer rolling if available
if USE_ROLLING_BETA and "beta_roll" in DF.columns and DF["beta_roll"].notna().any():
    beta_base = DF["beta_roll"].fillna(method="ffill")
else:
    if "beta_static" in DF.columns:
        beta_base = DF["beta_static"]
    elif "beta" in DF.columns:
        beta_base = DF["beta"]
    else:
        raise KeyError("Missing 'beta_roll'/'beta_static'/'beta' — run the OLS β cell first.")

# If a scaled version exists, use it as execution beta; else use base
if "beta_scaled" in DF.columns:
    beta_exec = DF["beta_scaled"].where(DF["beta_scaled"].notna(), beta_base)
else:
    beta_exec = beta_base

DF["beta_exec"] = beta_exec.fillna(method="ffill").fillna(0.0)

# ========= Turnover for fees (|Δβ| each bar, per 1 spot notional) =========
DF["d_beta"] = DF["beta_exec"].diff().fillna(0.0)
turnover_perp = DF["d_beta"].abs()

# ========= Price PnL (hedged return) =========
# Sign convention: long 1 spot, short β perps
DF["r_hedge"] = - DF["beta_exec"] * DF[RET_FUT]

# ========= Funding PnL (hourly) =========
# Positive funding_rate_hr → longs pay shorts on Binance; with short-perp hedge, you RECEIVE when rate > 0
# If you want the "shorts pay" convention instead, flip the sign here.
DF["r_fund"] = + DF["beta_exec"] * DF[FUND_HR]

# ========= Execution costs (fees + slippage) =========
# Fees & slippage (fractions, not bps) — adjust per venue/account
FEE_PERP_TAKER = 0.0004     # 4 bps taker on perp rebalances
FEE_SPOT_TAKER = 0.0002     # 2 bps taker on spot (entry/exit only if enabled)
SLIPPAGE_PERP  = 0.0002     # 2 bps slippage on perp turnover (optional)
INCLUDE_SPOT_ENTRY_EXIT = True

# Make fee components POSITIVE "costs", then subtract when forming returns
fees_perp_cost = (FEE_PERP_TAKER + SLIPPAGE_PERP) * turnover_perp

fees_spot_cost = pd.Series(0.0, index=DF.index)
if INCLUDE_SPOT_ENTRY_EXIT and len(DF) > 1:
    fees_spot_cost.iloc[0]  = FEE_SPOT_TAKER * 1.0   # buy 1 spot at start (cost)
    fees_spot_cost.iloc[-1] = FEE_SPOT_TAKER * 1.0   # sell 1 spot at end (cost)

# Total execution costs as (negative) return
DF["r_fees"] = -(fees_perp_cost + fees_spot_cost)

# ========= Net hedged return =========
DF["r_net"] = DF[RET_SPOT] + DF["r_hedge"] + DF["r_fund"] + DF["r_fees"]

# ========= Save outputs (Artifact Index) =========
out_cols = [RET_SPOT, RET_FUT, "beta_exec", "d_beta", "r_hedge", "r_fund", "r_fees", "r_net", FUND_HR]
out = DF[out_cols].copy()
out.to_csv(RUN_DIR / "returns_decomposed.csv")
print("Saved decomposition →", RUN_DIR / "returns_decomposed.csv")

# Quick sanity print
print(out.tail(3))

# ========= Sync outputs back into 'df' for downstream cells (metrics/plots) =========
cols_to_copy = ["beta_exec","d_beta","r_hedge","r_fund","r_fees","r_net", FUND_HR]
common_idx = df.index.intersection(DF.index)
df.loc[common_idx, cols_to_copy] = DF.loc[common_idx, cols_to_copy].values

# ========= Optional sanity checks =========
# No NaNs in core outputs on funding span
assert not DF[["r_spot","r_fut","r_net"]].isna().any().any(), "NaNs in core PnL columns on DF."

# Fees should be <= 0 aside from harmless float noise
assert DF["r_fees"].max() <= 1e-9, "Fees look positive—check fee sign convention."

# Funding sign sanity (Binance convention: when FundingRate > 0, longs pay shorts → short-perp receives)
_bad = DF[(DF[FUND_HR] > 0) & (DF["beta_exec"] > 0) & (DF["r_fund"] < -1e-12)]
assert _bad.empty, "Funding sign looks inverted—check 'r_fund' sign vs exchange convention."


# %% [markdown]
# ### 1) Critical analysis
#
# *   **Hedge mechanics (last 3 funding-safe hours):**
#     
#     *   ```beta\_exec ≈ 0.997``` (near one-to-one).
#         
#     *   ```r\_hedge``` ≈ -β·r\_fut nearly cancels ```r\_spot```, leaving tiny residuals:
#         
#         *   ```14:00``` residual ≈ ```+2.9e-05```, ```15:00``` ≈ ```+5.8e-05```, ```16:00``` ≈ ```+1.8e-05```.
#             
# *   **Funding mapping & sign:**
#     
#     *   ```funding\_rate\_hr``` steps ```~3e-6 → 7e-6``` by ```16:00```, as expected.
#         
#     *   With a **short-perp** hedge and positive funding, ```r\_fund = +β·funding\_rate\_hr``` ⇒ **small positive carry**:
#         
#         *   ```14:00``` ```+3e-06```, ```15:00``` ```+3e-06```, ```16:00``` ```+7e-06```.
#             
# *   **Fees (execution costs):**
#     
#     *   Turnover fees from ```|Δβ|``` are **negligible** intra-run (```~3–5e-08```).
#         
#     *   **Spot exit fee** hits at ```16:00```: ```r\_fees ≈ -2.000182e-04``` (≈ **−20.0 bp**), now correctly **negative**.
#         
# *   **Net result per hour:**
#     
#     *   ```14:00``` ```r\_net ≈ +3.1e-05``` (≈ **+0.31 bp**)
#         
#     *   ```15:00``` ```r\_net ≈ +6.1e-05``` (≈ **+0.61 bp**)
#         
#     *   ```16:00``` ```r\_net ≈ -1.75e-04``` (≈ **−17.5 bp**) — dominated by the one-off **exit fee**.
#         
#
# ### 2) Comparison with expected results
#
# *   **β near 1** for BTC spot vs BTC-USDT perps — matches theory and your earlier OLS (```β ≈ 1.0001``` static; rolling ≈ ```0.996–0.997```).
#     
# *   **Hedge effectiveness:** residual price noise in the **few bp** per hour range — consistent with near-perfect spot/perp co-movement.
#     
# *   **Funding scale:** hourly carry in **single-digit micro-bps** (```3–7e-06```) aligns with small 8h funding divided by 8.
#     
# *   **Fee profile:** intra-run turnover costs microscopic; **entry/exit** dominates fee PnL — exactly what we expect in a static hedge demo.
#     
#
# ### 3) Suggestions for improvement
#
# *   **Split fee components:** keep ```r\_fees\_perp\_continuous``` (turnover) and ```r\_fees\_spot\_once``` (entry/exit) as separate columns; improves attribution and charts.
#     
# *   **Steady-state view:** report metrics both **including** and **excluding** entry/exit hours to separate ongoing economics from one-offs.
#     
# *   **Discrete funding check (optional):** compare your prorated hourly ```r\_fund``` to an **8h pulse** implementation to confirm equivalence around settlement times.
#     
# *   **Basis diagnostics:** plot ```close\_perp - close\_spot``` vs **cum funding**; medium-term co-movement validates funding/basis accounting.
#     
# *   **Assertions:** keep defensive checks (no positive fees; funding sign vs convention) to prevent silent regressions.
#     
#
# ### 4) ✅ Good to go / ⚠️ Potential issues
#
# **✅ Good to go**
#
# *   Correct **funding sign** for short-perp under positive funding; correct **fee sign** (exit cost negative).
#     
# *   Hedge neutralizes price moves; **residuals tiny** and plausible.
#     
# *   Artifacts saved (```returns\_decomposed.csv```) with all key fields for audit.
#     
#
# **⚠️ Potential issues**
#
# *   **One-off fees** can dominate short samples; interpret ```r\_net``` near start/end with care.
#     
# *   If you later change funding source/units, re-validate ```funding\_rate\_hr``` construction and sign conventions.
#     
# *   Precision: with residuals in the ```1e-5``` range, be mindful of float formatting and rounding in reports.
#     
#
# ### 5) 📈 Interpretation for trading impact
#
# *   The hedge delivers **near-total variance reduction** hour-to-hour; **economic PnL** is driven by:
#     
#     *   **Funding carry** (tiny but positive here), and
#         
#     *   **Execution costs**, especially **entry/exit**.
#         
# *   In steady state (excluding one-offs), expect ```r\_net``` ≈ **small positive** (residual + funding − minuscule turnover fees).
#     
# *   For production, the levers that matter are **funding regime**, **rebalance cadence** (turnover), and **fee tier**; β estimation is essentially solved for BTC at this horizon.

# %% [markdown]
# ## 📊 Performance metrics (variance reduction & risk-adjusted return)
#
# We summarize unhedged vs hedged:
#
# **Variance reduction (\%):**
# $$
# \text{VR} =
# 100 \times
# \left(
# 1 - \frac{\operatorname{Var}\!\left(r^{\text{net}}\right)}{\operatorname{Var}\!\left(r^{\text{spot}}\right)}
# \right)\%
# $$
#
# **Sharpe (per chosen annualization):**
# $$
# \text{Sharpe} =
# \frac{\mathbb{E}[r]}{\operatorname{Std}[r]} \times \sqrt{\text{annualization\_factor}}
# $$
#
# **Max drawdown:** computed on cumulative log returns.
#
# **Turnover (proxy for fee drag):**
# $$
# \text{Turnover} = \sum_t \left| \Delta \beta^{(\cdot)}_t \right|
# $$
#
# **Goal:** show **lower variance**, **competitive Sharpe**, and **controlled costs**.

# %%
def sharpe_ann(r, ann_factor=ANN_FACTOR):
    mu = np.nanmean(r)
    sd = np.nanstd(r, ddof=1)
    return (mu / sd) * math.sqrt(ann_factor) if sd > 0 else np.nan

def max_drawdown(cumret):
    roll_max = np.maximum.accumulate(cumret)
    dd = cumret/roll_max - 1.0
    return dd.min()

metrics = {}
var_spot = np.nanvar(df["r_spot"], ddof=1)
var_net  = np.nanvar(df["r_net"], ddof=1)
metrics["variance_reduction_pct"] = 100.0 * (1 - var_net/var_spot)
metrics["sharpe_spot"] = sharpe_ann(df["r_spot"].values)
metrics["sharpe_net"]  = sharpe_ann(df["r_net"].values)
metrics["turnover"]    = df["d_beta"].abs().sum()
cum_spot = np.exp(df["r_spot"].cumsum())
cum_net  = np.exp(df["r_net"].cumsum())
metrics["mdd_spot"] = float(max_drawdown(cum_spot))
metrics["mdd_net"]  = float(max_drawdown(cum_net))

with open(RUN_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

metrics


# %% [markdown]
# ### 1) Critical analysis
#
# *   **Variance reduction (VR):** ```92.42%``` — the hedge kills almost all spot variance. That magnitude is consistent with BTC spot vs BTC-USDT perps at 1h.
#     
# *   **Sharpe:** ```spot = 1.179``` → ```hedged = 1.242``` — a **clean uplift** despite fees and the one-off exit cost. Good sign that residuals + funding carry outweigh frictions.
#     
# *   **Turnover:** ```Σ|Δβ| = 56.08``` over the full run. With ~15k hourly bars, that’s ≈ **0.37% of notional per hour** on average — very low; fee drag from rebalances should be negligible (as seen in the prints).
#     
# *   **Drawdowns:** ```MDD\_spot = −32.31%``` → ```MDD\_net = −8.07%``` — **4× compression** in worst peak-to-trough. This is the most stakeholder-visible benefit.
#     
#
# ### 2) Comparison with expected results
#
# *   **VR in the 90–95% band** is typical when hedging BTC spot with BTC perps at hourly frequency and β≈1 — you’re right on target.
#     
# *   **Sharpe improvement** is expected when the hedge substantially reduces variance without destroying mean (funding carry small + residuals small).
#     
# *   **Turnover** being tiny matches the very stable rolling β (≈0.996–0.997), so the strategy isn’t churning.
#     
# *   **Drawdown compression** of ~24 p.p. is in line with the VR result and the nearly flat residual series.
#     
#
# ### 3) Suggestions for improvement
#
# *   **Report steady-state vs one-offs:** compute metrics **excluding the first/last rebalance hours** (where entry/exit fees hit) to show ongoing economics.
#     
# *   **Decompose performance:** add funding/fees summaries to ```metrics.json``` (e.g., annualized funding bp, total fee bp, average hourly fee).
#     
# *   **Robustness:** include **HAC (Newey–West)** SEs for β and a **rolling VR** plot (you already have code snippets) to demonstrate stability over time.
#     
# *   **Alt accounting cross-check:** compare **hourly-prorated** vs **8h-pulse** funding PnL; they should reconcile around settlement windows.
#     
# *   **Parameter sweeps:** quick sensitivity on ```ROLL\_VOL\_W```, ```TARGET\_VOL```, and fee assumptions (maker vs taker) to show conclusions are not parameter-fragile.
#     
#
# ### 4) ✅ Good to go / ⚠️ Potential issues
#
# **✅ Good to go**
#
# *   Results are **economically coherent**: huge VR, better Sharpe, small turnover, and tight residuals.
#     
# *   Fee/sign conventions are correct; artifacts saved (```metrics.json```, ```returns\_decomposed.csv```).
#     
#
# **⚠️ Potential issues**
#
# *   **Sample edges:** one-off entry/exit fees can dominate the last hour; call that out in reports or compute a steady-state slice.
#     
# *   **Unit consistency:** if funding source changes, re-verify ```funding\_rate\_hr``` construction (8h ÷ 8) and sign (short-perp receives when funding > 0 under Binance USD-M).
#     
#
# ### 5) 📈 Interpretation for trading impact
#
# *   The strategy delivers **institutional-grade risk compression** (VR ~92%, MDD ~−8%) while slightly **improving Sharpe** — ideal for “keep the yield, cut the noise.”
#     
# *   Ongoing PnL is driven by **funding carry** and **minuscule turnover fees**; **basis/β drift** contributes little at this horizon.
#     
# *   Operationally, focus on **fee tier** and **funding regime**; β estimation is essentially solved for BTC perps at 1h, so most gains come from execution quality and cost control.

# %% [markdown]
# ## 📈 Visuals (what to plot and why)
#
# Plot:
# - **Cumulative log returns**: unhedged vs hedged → visually show **drawdown compression**.
# - **Rolling volatility** (e.g., 30× interval) → hedged series should **stabilize**.
# - **Funding vs hedge size** → exposes **carry headwinds/tailwinds**.
# - **Bar chart** of **variance components** (spot var, hedge var, covariance) → intuitive hedge story.
#
# **Reading the plots:** the hedged curve should look **smoother**; spikes in funding aligned with **net return changes** explain economics to stakeholders.

# %%
# %% [markdown]
# ## 📈 Visuals (what to plot and why)
#
# **Plot:**
# - **Cumulative log returns** — unhedged vs hedged → shows **drawdown compression**.
# - **Rolling volatility** (e.g., 30× interval) — hedged series should **stabilize**.
# - **Funding vs hedge size** — exposes **carry headwinds/tailwinds**.
# - **Variance components bar chart** — intuitive hedge story:
#   - Var(spot), Var(β·fut), and the **cross term** −2·Cov(spot, β·fut), whose sum ≈ Var(spot − β·fut).
#
# **Reading the plots:** the hedged curve should look **smoother**; spikes in funding aligned with **net return changes**
# explain economics to stakeholders.

# %%
import numpy as np
import matplotlib.pyplot as plt

# 1) Cumulative log returns (exp to get growth of 1)
plt.figure()
plt.plot(np.exp(df["r_spot"].cumsum()), label="Unhedged (Spot)")
plt.plot(np.exp(df["r_net"].cumsum()),  label="Hedged (Net)")
plt.title("Cumulative Growth — Unhedged vs Hedged")
plt.legend()
plt.grid(True)
plt.savefig(RUN_DIR / "cumlogret.png", dpi=160, bbox_inches="tight")
plt.show()

# 2) Rolling volatility (annualized)
ROLL_PLOT_W = 30
plt.figure()
plt.plot(df["r_spot"].rolling(ROLL_PLOT_W).std()*math.sqrt(ANN_FACTOR), label="Spot vol (ann.)")
plt.plot(df["r_net"].rolling(ROLL_PLOT_W).std()*math.sqrt(ANN_FACTOR),  label="Hedged vol (ann.)")
plt.title(f"Rolling Volatility (window={ROLL_PLOT_W} bars)")
plt.legend()
plt.grid(True)
plt.savefig(RUN_DIR / "rolling_vol.png", dpi=160, bbox_inches="tight")
plt.show()

# 3) Funding vs hedge size (use per-hour funding if available)
fund_col = "funding_rate_hr" if "funding_rate_hr" in df.columns else ("funding_rate" if "funding_rate" in df.columns else None)
plt.figure()
ax1 = df["beta_exec"].plot(figsize=(9,4), label="Hedge size (beta_exec)")
if fund_col is None:
    raise KeyError("No funding column found: expected 'funding_rate_hr' or 'funding_rate'.")
ax2 = df[fund_col].plot(secondary_y=True, label=("Funding (per hour)" if fund_col=="funding_rate_hr" else "Funding (raw)"))
ax1.set_title("Hedge Size vs Funding Rate")
ax1.grid(True)
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")
plt.savefig(RUN_DIR / "beta_vs_funding.png", dpi=160, bbox_inches="tight")
plt.show()

# 4) Variance components bar chart
# Hedged variance identity: Var(r_spot - β·r_fut) = Var(r_spot) + Var(β·r_fut) - 2·Cov(r_spot, β·r_fut)
rs = df["r_spot"].dropna()
rf = (df["beta_exec"] * df["r_fut"]).dropna()
align_idx = rs.index.intersection(rf.index)
rs = rs.loc[align_idx]
rf = rf.loc[align_idx]

var_spot = rs.var(ddof=1)
var_beta_fut = rf.var(ddof=1)
cov_term = rs.cov(rf)
hedged_var = var_spot + var_beta_fut - 2*cov_term

components = {
    "Var(spot)": var_spot,
    "Var(β·fut)": var_beta_fut,
    "−2·Cov(spot, β·fut)": -2*cov_term,
    "Var(spot−β·fut) (check)": hedged_var,
}

plt.figure()
labels = list(components.keys())[:-1]  # exclude the check in the bars
vals = [components[k] for k in labels]
plt.bar(labels, vals)
plt.title("Variance Components (should sum to hedged variance)")
plt.ylabel("Variance")
plt.xticks(rotation=15)
plt.grid(True, axis="y", linestyle="--", alpha=0.4)
plt.savefig(RUN_DIR / "variance_components.png", dpi=160, bbox_inches="tight")
plt.show()

print("Saved plots →",
      RUN_DIR / "cumlogret.png", ",",
      RUN_DIR / "rolling_vol.png", ",",
      RUN_DIR / "beta_vs_funding.png", ",",
      RUN_DIR / "variance_components.png")



# %% [markdown]
# ### 1) Critical analysis
#
# *   **Cumulative growth (spot vs hedged):**
#
# *   The hedged curve is **much smoother** with visibly shallower pullbacks. This matches your metrics: VR ≈ **92%** and MDD compressed from **−32% → −8%**. Small upward drift in the hedged curve reflects tiny positive residuals + **funding carry**, net of minimal turnover fees (outside the exit hour).
#     
# *   **Rolling volatility (30-bar, annualized):**
#
# *   Hedged vol tracks **well below** spot vol across the sample and reacts less to spikes — exactly what a good hedge should do. The gap between the two series is fairly constant, supporting the near-constant β you estimated (≈1).
#     
# *   **Hedge size vs funding:**
#
# *   ```beta\_exec``` stays near **0.997** with **microscopic** adjustments — low turnover, low fee drag. Funding (per-hour) oscillates near zero with small **positive** prints in your tail; as expected, that yields a **small positive** ```r\_fund``` for the **short-perp** hedge. No obvious misalignment between funding steps and hedge size.
#     
# *   **Variance components bar chart:**
#     
#     *   ```Var(spot)``` is large.
#         
#     *   ```Var(β·fut)``` is similar magnitude (since β≈1).
#         
#     *   The **−2·Cov(spot, β·fut)** term is **strongly negative**, driving the bulk of risk reduction.
#         
#     *   The check (```Var(spot−β·fut)```) is tiny versus spot var, consistent with the ~**92%** VR figure; numerically, the components add up correctly within floating tolerance.
#         
#
# ### 2) Comparison with expected results
#
# *   For BTC spot vs BTC-USDT perps at **1h**, we expect:**(i)** near-perfect tracking (β≈1),**(ii)** large covariance cancellation,**(iii)** hedged vol well below spot, and**(iv)** cumulative hedged curve much smoother.All four appear clearly in your figures and match the computed metrics.
#     
#
# ### 3) Suggestions for improvement
#
# *   **Shaded regimes / markers:** add vertical lines at **funding settlement** hours (every 8h) to visually link funding steps to PnL inflections.
#     
# *   **Exclude one-offs in visuals:** optionally plot a **steady-state hedged curve** (excluding entry/exit hours) alongside the full hedged curve to separate ongoing economics from one-time fees.
#     
# *   **Add rolling VR panel:** plot ```1 − Var(hedged)/Var(spot)``` over a 30-day window to show **stability** of hedge effectiveness through time.
#     
# *   **Basis diagnostic (optional):** a small subplot of ```close\_perp − close\_spot``` vs cum-funding helps stakeholders see the **basis–funding** linkage.
#     
#
# ### 4) ✅ Good to go / ⚠️ Potential issues
#
# **✅ Good to go**
#
# *   Plots align with numbers: **huge** variance cut, **stable** β, **tiny** turnover, and funding behaving with the correct **sign and scale**.
#     
#
# **⚠️ Potential issues**
#
# *   If you switch exchanges or symbols later, re-check **funding units** (8h vs hourly) and **sign** conventions; the funding plot is the first place such errors show up.
#     
# *   On very short samples, the one-off **exit fee** can dominate the tail of the cumulative hedged curve — call it out in captions.

# %% [markdown]
# ## 🧾 Paper rebalancer log (audit trail)
#
# We run a **paper (simulation) rebalancer** that does not place orders. Each cycle it logs: timestamp, $\beta$, $\beta^{\text{scaled}}$, prices, **intended** order deltas, and snapshots metrics to ```runs/\ldots/paper\_rebalance\_log.jsonl```.
#
# **Why this matters:** you get an **operational timeline**. You can compare the **intended** actions with what actually happened in the market, quantify **slippage risk**, and prepare for **testnet**.
#
# **Checklist to review:**
# - **Timestamp cadence** (e.g., hourly) and **UTC alignment**.
# - **Position drift** vs **intended hedge size**.
# - **Funding & fee** attribution per cycle.
# - **Incidents** clearly flagged (e.g., data gaps, cadence irregularities, sanity-check failures).
#

# %%
# %% [markdown]
# ## 🧾 Paper rebalancer log (audit trail)
# We run a paper (simulation) rebalancer that **does not place orders**. Each cycle we log:
# timestamp (UTC ISO), β, β_scaled, β_exec, Δβ (intended order delta), returns decomposition
# (r_spot, r_fut, r_hedge, r_fund, r_fees, r_net), funding rate, and optional prices.
#
# **Why this matters:** this is your **operational timeline**. You can verify UTC cadence, inspect
# position drift vs intended hedge size, attribute funding/fee per cycle, and flag incidents before testnet.
#
# **Checklist to review in the log:**
# - Timestamp cadence (hourly) and UTC alignment
# - Position drift vs intended hedge size (β_exec vs β, Δβ)
# - Funding & fee attribution per cycle
# - Incidents: data gaps, missing inputs, unusual turnover

# %%
import json, math
import numpy as np
import pandas as pd

# --- Helpers for cadence / incident flags ---
def _bar_minutes_from_timeframe(tf: str) -> float:
    if tf.endswith("h"):
        return float(tf[:-1]) * 60.0
    if tf.endswith("m"):
        return float(tf[:-1])
    return 60.0  # default 1h

BAR_MIN = _bar_minutes_from_timeframe(TIMEFRAME)

# Pick funding column for logging (prefer per-hour)
fund_col = "funding_rate_hr" if "funding_rate_hr" in df.columns else ("funding_rate" if "funding_rate" in df.columns else None)
if fund_col is None:
    raise KeyError("No funding column found for logging: expected 'funding_rate_hr' or 'funding_rate'.")

# Optional prices if present (nice for audits)
price_spot_col = "close_spot" if "close_spot" in df.columns else None
price_perp_col = "close_perp" if "close_perp" in df.columns else None

log_path = RUN_DIR / "paper_rebalance_log.jsonl"
with open(log_path, "w") as f:
    prev_ts = None
    prev_beta = 0.0
    for ts_i, row in df.iterrows():
        # cadence / incident checks
        dt_flag = None
        if prev_ts is not None:
            gap_min = (ts_i - prev_ts).total_seconds() / 60.0
            # allow some slack (±50%)
            if not (0.5 * BAR_MIN <= gap_min <= 1.5 * BAR_MIN):
                dt_flag = f"irregular_cadence: {gap_min:.2f} min (expected ~{BAR_MIN:.2f})"

        # intended perp order delta (short β perps for hedge)
        d_beta = float(row["d_beta"])
        perp_order_delta = -d_beta  # increase short when β_exec increases

        rec = {
            "ts": ts_i.isoformat(),
            "symbol_spot": "BTC/USDT",
            "timeframe": TIMEFRAME,
            # Betas
            "beta": (float(row["beta"]) if "beta" in row and not math.isnan(row["beta"]) else None),
            "beta_scaled": (float(row["beta_scaled"]) if "beta_scaled" in row and not math.isnan(row["beta_scaled"]) else None),
            "beta_exec": float(row["beta_exec"]),
            "d_beta": d_beta,
            "perp_order_delta": perp_order_delta,  # sign reflects short-perp convention
            # Returns decomposition
            "r_spot": float(row["r_spot"]),
            "r_fut": float(row["r_fut"]),
            "r_hedge": float(row["r_hedge"]),
            "r_fund": float(row["r_fund"]),
            "r_fees": float(row["r_fees"]),
            "r_net": float(row["r_net"]),
            # Funding
            "funding_col": fund_col,
            "funding_rate": (float(row[fund_col]) if not math.isnan(row[fund_col]) else None),
            # Optional prices (if available)
            "price_spot": (float(row[price_spot_col]) if price_spot_col and not math.isnan(row[price_spot_col]) else None),
            "price_perp": (float(row[price_perp_col]) if price_perp_col and not math.isnan(row[price_perp_col]) else None),
            # Incidents
            "incident": dt_flag,
            "note": "dry-run (no order placed)"
        }
        f.write(json.dumps(rec) + "\n")
        prev_ts = ts_i
        prev_beta = float(row["beta_exec"])

print("Wrote:", log_path)



# %%
# Quick QC
import json
n_rows = n_inc = 0
first_ts = last_ts = None
with open(log_path) as f:
    for line in f:
        n_rows += 1
        rec = json.loads(line)
        first_ts = first_ts or rec["ts"]
        last_ts  = rec["ts"]
        if rec.get("incident"):
            n_inc += 1
print(f"Log rows: {n_rows} | Incidents: {n_inc} | Span: {first_ts} → {last_ts}")


# %% [markdown]
# ### 1) Critical analysis
#
# *   **What you just produced:** a **per-bar audit trail** with ```β```, ```β\_scaled```, ```β\_exec```, ```Δβ```, returns decomposition (```r\_\*```), and ```funding\_rate```, plus **cadence checks** and optional prices.
#     
# *   **Cadence guard:** the cell flags irregular spacing with ```incident="irregular\_cadence: …"``` if a gap deviates ~±50% from your ```TIMEFRAME```.
#     
# *   **Order intent:** ```perp\_order\_delta = -Δβ``` reflects the **short-perp** convention (increase short when ```β\_exec``` rises).
#     
#
# ### 2) Comparison with expected results
#
# *   The schema is **complete** for replay/forensics and aligns with your economics: you can reconstruct hedged PnL, turnover (fees), and funding per bar.
#     
# *   UTC ISO timestamps and symbol/timeframe fields make it easy to merge with external market snapshots if needed.
#     
#
# ### 3) Suggestions for improvement
#
# *   **Schema version & run id:** add ```schema\_version``` and ```run\_id``` (e.g., the ```RUN\_DIR``` timestamp) to each record for long-term compatibility.
#     
# *   **Explicit fee split:** log ```r\_fees\_perp\_continuous``` and ```r\_fees\_spot\_once``` separately to distinguish ongoing drag vs one-off entry/exit.
#     
# *   **Sequence number:** include a monotonically increasing ```bar\_seq``` to detect missing bars without parsing timestamps.
#     
# *   **Hash for integrity:** optionally add a rolling hash (e.g., SHA256 of prior line + current fields) to make tampering detectable.
#     
# *   **Lightweight summary:** after writing the log, print a tiny QC: number of rows, # of incidents, first/last timestamp.
#     
#
# ### 4) ✅ Good to go / ⚠️ Potential issues
#
# **✅ Good to go**
#
# *   Fields cover positions, PnL, funding, and timing — enough to **replay decisions** and attribute PnL cleanly.
#     
# *   Cadence flag helps catch silent data gaps.
#     
#
# **⚠️ Potential issues**
#
# *   If you change funding units or sign conventions later, **log both the value and the column name** (you already do via ```funding\_col```) and note the convention in the notebook header.
#     
# *   Very large notebooks can generate big JSONL files — consider **gzip** if you’re archiving runs.

# %% [markdown]
# ## 🧭 Where to go next (practice → prototype → research)
#
# 1.  **Dry-run rebalancer — make hedge-paper**
#
#    - **What it does:** simulates rebalances without placing orders; logs $\beta / \beta^{\text{scaled}}$, intended size, and expected hedged return.
#
#    - **Why it matters:** **verifies logic & timing** without risk; gives a **timeline** for variance reduction, Sharpe, and drawdowns.
#    - **Inspect:** **UTC alignment**, **hedge tightness**, **performance trend**, **sensitivity to fees/funding**.
#     
# 2.  **Safe exchange prototype — testnet with guardrails**
#
#    - **Testnet first:** connect to sandbox perps (USD-M).
#    - **Guardrails (non-negotiable):** **min notional checks**, **leverage caps**, **kill-switch**, **rate-limit/backoff**, **state checks pre/post**, **durable JSON logs**.
#    - **Graduate when:** **2+ weeks error-free**, metrics match paper, **no unintended guardrail triggers**, **clear funding/fee accounting**.
#     
# 3.  **Research roadmap (for PhD-grade evidence)**
#    - **Baselines:** OLS $\beta$ + simple volatility scaling.
#    - **Classical vol models:** **HAR-RV**, **MIDAS**.
#    - **Deep baselines:** **LSTM**, **Transformer** for longer context/regimes.
#    - **Regime filters:** **vol quantiles**, **funding sign/magnitude**, **trend** → **gate** or **scale** the hedge.
#    - **Evaluation:** **walk-forward**, losses for vol forecast (**MSE**, **QLIKE**, **MAE**), **Diebold–Mariano tests** for model comparisons.
#    - **Reporting:** ablations (**with/without funding**, **static vs scaled $\beta$**), timeframes, error bars, **multi-exchange robustness**.
#
#
# **Deliverables checklist:**
# - **Paper run outputs:** intended trades log, cycle metrics, cumulative plots.
# - **Testnet report:** incidents, guardrail triggers, per-cycle fees & funding, **PnL vs spot**.
# - **Model cards:** one-pager per model (features, window, metrics, limits).
# - **Evaluation appendix:** DM tables, QLIKE/MSE, regime-wise results.
# - **Repro pack:** ```runs/``` snapshots, ```metrics.json```, README to re-run.
#
# **Plain-English glossary:** 
# - **USD-M perps:** USDT-margined perpetual futures.
# - **Min notional:** smallest acceptable order size.
# - **Leverage cap:** hard limit to avoid liquidation risk.
# - **Kill-switch:** emergency off that flattens and halts.**HAR-RV / MIDAS:** classical volatility models with multi-horizon and mixed-frequency data.**Diebold–Mariano test:** checks if one forecast is **significantly** more accurate than another.
#
# **Final takeaway:** move **tutorial → paper trading → testnet with guardrails** before any real orders. For the PhD track, elevate with **strong baselines**, **regime logic**, and **formal comparisons** so your hedge is **statistically sound**, **economically aware** (fees & funding), and **operationally safe**.

# %% [markdown]
# 📑 Hedged Staking Prototype — Demo Report
# =========================================
#
# ### 🎯 Objective
#
# Demonstrate a **risk-managed BTC hedge** that preserves staking-like yield while reducing volatility.Goal: show **variance reduction** and **funding-aware net returns (NRR)** in a reproducible framework.
#
# ### 📦 Data & Setup
#
# *   **Source:** Binance spot + USDT-margined perpetuals (BTC/USDT).
#     
# *   **Frequency:** 5-min resampled to hourly cycles.
#     
# *   **Period:** \[insert dates of test run\].
#     
# *   **Logs:** Stored in /runs/BTCUSDT/hedge\_mvp/....
#     
#
# ### 🔑 Methodology
#
# 1. **Optimal hedge ratio ($\beta$) via OLS regression of spot vs futures returns:**
# $$
# r^{\text{spot}}_t = \alpha + \beta \, r^{\text{fut}}_t + \varepsilon_t
# $$
#
# 2. **Volatility scaling:** adjust $\beta$ using realized volatility to stabilize risk.
#
# 3. **Net hedged return (includes funding rates and fee drag):**
# $$
# r^{\text{net}}_t
# =
# r^{\text{spot}}_t
# - \beta \, r^{\text{fut}}_t
# + r^{\text{fund}}_t
# + r^{\text{fees}}_t
# $$
#
# 4. **Paper rebalancer:** produces structured JSON logs of intended trades, $\beta$, and metrics.
#
#
# ### 📊 Results Snapshot
#
# *   **Variance Reduction:** XX% vs unhedged spot.
#     
# *   **Sharpe Improvement:** from X.XX → Y.YY.
#     
# *   **Max Drawdown:** compressed from –A% → –B%.
#     
# *   **Funding Carry:** net +C% annualized contribution.
#     
# *   **Costs:** turnover-driven fees ≈ D% per annum.
#     
#
# ### 📈 Visual Evidence
#
# 1.  **Cumulative log returns:** smoother hedged curve, tighter drawdowns.
#     
# 2.  **Rolling volatility:** hedged series consistently lower.
#     
# 3.  **Funding vs hedge size:** explains net return swings.
#     
# 4.  **Variance decomposition chart:** shows covariance effect.
#     
#
# ### 🛡️ Guardrails for Prototype → Testnet
#
# *   **Min notional checks**
#     
# *   **Leverage caps**
#     
# *   **Kill-switch** (halts & flattens)
#     
# *   **API rate-limit compliance**
#     
# *   **State & balance verification**
#     
# *   **Durable JSON logging**
#     
#
# ### 📌 Takeaways
#
# *   **Proof of value:** The hedge **retains yield** while cutting risk — institutional clients gain **stable APR/NRR**.
#     
# *   **Operational readiness:** Logging + guardrails → safe migration to testnet.
#     
# *   **Research track:** extend with HAR-RV, MIDAS, LSTM, Transformer, regime filters; compare forecasts via **Diebold–Mariano tests**.
#     
#
# ### 🔗 P2P.org Alignment
#
# This prototype mirrors P2P’s mission: **maximize APR while enhancing security**.It demonstrates the exact skills for a **Data Analyst role**:
#
# *   Handling **large crypto datasets** (on/off-chain).
#     
# *   Monitoring **economic performance metrics**.
#     
# *   Producing **clear dashboards & reports**.
#     
# *   Translating findings into **strategic insights for staking infrastructure**.
