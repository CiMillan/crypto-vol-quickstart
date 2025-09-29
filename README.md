# Crypto Volatility Forecasting — Quickstart

**Production-grade scaffold** to **download market data**, **engineer features**, **train & compare models** (GARCH, XGBoost, LSTM, CNN-LSTM), and **backtest** a volatility-targeting strategy with walk-forward validation.  

*Focus assets:* BTCUSDT, ETHUSDT (spot + perp) · *Exchange:* Binance (public endpoints via CCXT + REST)

---

## Why this repo matters

- **Robust by design** — walk-forward validation, Diebold–Mariano tests, and economic significance (Sharpe, MDD, turnover).
- **Research ↔ production bridge** — notebooks paired with Python modules and a reproducible CLI/Makefile workflow.
- **Extensible signals** — plug-in **on-chain features** (entity-adjusted flows, DEX microstructure, gas) alongside derivatives/funding.

---

## Quick links (docs)

- **0) Setup** → See **[Setup](docs/00_setup.md)**  
- **1) Download data** → See **[Download data](docs/01_download_data.md)**  
- **2) Build feature sets & targets** → See **[Build feature sets & targets](docs/02_build_features_targets.md)**  
- **3) Walk-forward training & model comparison** → See **[Walk-forward training & model comparison](docs/03_walkforward_modeling.md)**  
- **4) Backtest — volatility targeting** → See **[Backtest — volatility targeting](docs/04_backtest_vol_targeting.md)**  
- **5) Hedge MVP — Spot↔Perp hedge with funding** → See **[Hedge MVP — Spot↔Perp hedge with funding](notebooks/06_hedge_mvp/README.md)**  
- **6) On-chain features** → See **[On-Chain Data Dictionary](data/processed/onchain/README.md)**

---

## Project highlights

- **Hedge MVP:** ~**92% variance reduction** at 1h; **Sharpe** improves (hedged ≥ unhedged); **MDD** compresses (~−32% → ~−8%); **low turnover** with fee drag logged.
- **Modeling:** clean **horse-race** between GARCH, XGBoost, LSTM (optionally CNN-LSTM), with **feature importance** and **DM tests**.
- **On-chain signals (MV set):** exchange/stablecoin net flows, Uniswap v2/v3 **swap intensity & impact**, **gas pressure**, **whale transfers**, **address breadth** — each with z-scored pressure variants.
- **Reproducibility:** deterministic resampling, versioned artifacts under `runs/`, and docs for schema/QA (decimals, label coverage, UTC clocks).

---

## Tech stack & skills showcased

- **Python:** pandas, numpy, statsmodels/arch, scikit-learn, xgboost, pytorch/keras (optional), matplotlib
- **Data:** CCXT (spot/perp), Binance funding (8h), parquet I/O; plug-in **Dune/BigQuery** exports for on-chain
- **Workflow:** Makefile + CLI modules, jupytext pairing, pre-commit hooks, repo-level docs
- **Methods:** OLS hedge ratio (static/rolling), vol-scaling, volatility targeting, regime slicing, DM tests

---

## Minimal usage (glance)

```bash
# 0) Setup → docs/00_setup.md
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1–4) End-to-end pipeline (details in docs/)
python -m src.data.ccxt_download --symbol BTC/USDT --timeframe 1m --since 2020-01-01 --out data/raw/binance_spot_BTCUSDT_1m.parquet
python -m src.features.make_features --spot data/raw/binance_spot_BTCUSDT_1m.parquet --funding data/raw/binance_funding_BTCUSDT.parquet --symbol BTCUSDT --timeframe 5m --out data/processed/BTCUSDT_5m.parquet
python -m src.modeling.run_experiments --data data/processed/BTCUSDT_5m.parquet --symbol BTCUSDT --horizon 12 --output runs/BTCUSDT
python -m src.backtest.vol_targeting --pred runs/BTCUSDT/predictions.parquet --retcol ret_5m --fee_bps 1 --output runs/BTCUSDT/backtest_BTCUSDT.json

# 5) Hedge MVP (docs in notebooks/06_hedge_mvp/README.md)
python -m src.hedge_mvp.run \
  --spot data/raw/binance_spot_BTCUSDT_5m.parquet \
  --perp data/raw/binance_perp_BTCUSDT_5m.parquet \
  --funding data/raw/binance_funding_BTCUSDT.parquet \
  --timeframe 1h --fees_bps 1 --out runs/BTCUSDT/hedge_mvp
```

---

## 7) Repository layout

```bash
crypto-vol-quickstart/
├── data/raw/               # raw downloads
├── data/processed/         # aligned features/targets (+ on-chain docs)
├── runs/                   # experiment & hedge outputs (timestamped)
├── notebooks/              # EDA & hedge notebooks (paired .py via jupytext)
├── src/                    # Python modules (data, features, modeling, backtest, hedge_mvp)
└── docs/                   # split README sections (0–4)
```

---

## 8) Notes

- Public endpoints only; no API keys required for the base pipeline.

- Exchange can be swapped (CCXT). Paths/symbols are configurable via CLI.

- Extend src/features/make_features.py to integrate on-chain once your Dune/BigQuery exports are ready.

---

## Maintainer

Built by Cíntia Millan — Data Scientist & PhD Candidate (NOVA IMS).

Focus: robust ML for crypto volatility forecasting, hedging, and economic validation.
