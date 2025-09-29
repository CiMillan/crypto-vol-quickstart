# Crypto Volatility Forecasting — Quickstart

A minimal, end-to-end scaffold to **download data**, **build features**, **train models** (GARCH, XGBoost, LSTM), and **backtest** a volatility-targeting strategy with walk-forward validation.

> Focus assets: BTCUSDT, ETHUSDT (spot + perp). Exchange: Binance (public endpoints via CCXT + REST).

## 0) Setup

> See **[Setup](docs/00_setup.md)**.
## 1) Download data

> See **[Download data](docs/01_download_data.md)**.
## 2) Build feature sets & targets

> See **[Build feature sets & targets](docs/02_build_features_targets.md)**.
## 3) Walk-forward training & model comparison

> See **[Walk-forward training & model comparison](docs/03_walkforward_modeling.md)**.
## 4) Backtest — volatility targeting

> See **[Backtest — volatility targeting](docs/04_backtest_vol_targeting.md)**.
## 5) Hedge MVP — Spot↔Perp hedge with funding

> **Hedge MVP docs:** See the **[Spot↔Perp Hedge with Funding (README)](notebooks/06_hedge_mvp/README.md)** for full usage, outputs, and QA notes.

## 6) On-chain features:

> See the **[MV On-Chain Data Dictionary](data/processed/onchain/README.md)** for schema and QA checks.


## 7) Project structure

```
crypto-vol-quickstart/
├── data/raw/               # raw downloads
├── data/processed/         # aligned features/targets
├── runs/                   # experiment & hedge outputs
├── notebooks/              # EDA & hedge notebooks (paired .py)
├── src/
│   ├── data/
│   ├── features/
│   ├── modeling/
│   ├── backtest/
│   └── hedge_mvp/
└── requirements.txt
```

**Notes**
- Uses only public endpoints. No API key required for these scripts.
- You can swap Binance with other CCXT exchanges easily.
- Symbols/paths in the README should match your files; adjust dates (e.g., --end) as needed.
- Extend `src/features/make_features.py` to add on-chain data when you obtain an API (e.g., Glassnode).

