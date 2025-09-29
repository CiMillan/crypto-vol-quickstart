# Hedge MVP — Spot ↔ Perp Hedge with Funding

**Files**
- `hedge_MVP.ipynb` — main notebook
- `hedge_MVP.py` — jupytext pair (recommended for diffs/PRs)

**What it does**
- Loads **spot**, **funding**, and (optionally) **perp** prices  
- Resamples to **1h**, computes `r_spot`, `r_fut`  
- Aligns **funding** (8h → per-hour)  
- Estimates **OLS β** (static & rolling) and **vol-scaled β**  
- Computes **hedged returns** with **fees & funding**  
- Writes artifacts under `runs/BTCUSDT/hedge_mvp/<timestamp>/`

**Outputs (per run)**
- Snapshots: `spot_snapshot.csv`, `perp_snapshot.csv`, `funding_snapshot.csv`
- Merged & series: `prices_merged.csv`, `beta_series.csv`, `returns_decomposed.csv`
- Metrics: `metrics.json` (variance reduction, Sharpe, MDD, turnover)
- Plots: `cumlogret.png`, `rolling_vol.png`, `beta_vs_funding.png`, `variance_components.png`
- Audit trail: `paper_rebalance_log.jsonl` (dry-run rebalance intents per bar)

**Sample outcomes (reference)**
- Variance reduction ≈ **92%** @ 1h  
- **Sharpe** improves (hedged ≥ unhedged)  
- **MDD** compressed from ~−32% → ~−8%  
- **Low turnover** (small fee drag), funding correctly signed

## How to run (CLI)
```bash
python -m src.hedge_mvp.run \
  --spot data/raw/binance_spot_BTCUSDT_5m.parquet \
  --perp data/raw/binance_perp_BTCUSDT_5m.parquet \
  --funding data/raw/binance_funding_BTCUSDT.parquet \
  --timeframe 1h \
  --fees_bps 1 \
  --out runs/BTCUSDT/hedge_mvp
```

> Don’t have perps handy? Omit --perp to run spot+funding only.

Notes & QA
Use real perp returns (no proxies) for final results.

Funding alignment: 8h prints → allocate evenly across the next 8 hourly bars.

β estimation: report both static OLS and rolling OLS; log any NaN windows.

Fees: apply on rebalance trades only; log turnover.

Repro: every run writes a timestamped folder under runs/….

Makefile (optional)
makefile
Copy code
.PHONY: hedge-mvp
hedge-mvp:
\t@python -m src.hedge_mvp.run \\
\t  --spot data/raw/binance_spot_BTCUSDT_5m.parquet \\
\t  --perp data/raw/binance_perp_BTCUSDT_5m.parquet \\
\t  --funding data/raw/binance_funding_BTCUSDT.parquet \\
\t  --timeframe 1h \\
\t  --fees_bps 1 \\
\t  --out runs/BTCUSDT/hedge_mvp
