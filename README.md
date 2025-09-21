# Crypto Volatility Forecasting — Quickstart

A minimal, end-to-end scaffold to **download data**, **build features**, **train models** (GARCH, XGBoost, LSTM), and **backtest** a volatility-targeting strategy with walk-forward validation.

> Focus assets: BTCUSDT, ETHUSDT (spot + perp). Exchange: Binance (public endpoints via CCXT + REST).

## 0) Setup

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 1) Download data

- Spot OHLCV via CCXT (Binance) — 1m to 1h aggregation
- Perp funding rates via Binance public endpoint
- (Optional) Open interest & futures klines via CCXT if available

```bash
python -m src.data.ccxt_download --symbol BTC/USDT --timeframe 1m --since 2020-01-01 --limit 1000 --out data/raw/binance_spot_BTCUSDT_1m.parquet
python -m src.data.ccxt_download --symbol ETH/USDT --timeframe 1m --since 2020-01-01 --limit 1000 --out data/raw/binance_spot_ETHUSDT_1m.parquet

python -m src.data.binance_funding --symbol BTCUSDT --start 2020-01-01 --end 2025-09-19 --out data/raw/binance_funding_BTCUSDT.parquet
python -m src.data.binance_funding --symbol ETHUSDT --start 2020-01-01 --end 2025-09-19 --out data/raw/binance_funding_ETHUSDT.parquet
```

## 2) Build feature sets & targets

```bash
python -m src.features.make_features --spot data/raw/binance_spot_BTCUSDT_1m.parquet --funding data/raw/binance_funding_BTCUSDT.parquet --symbol BTCUSDT --timeframe 5m --out data/processed/BTCUSDT_5m.parquet
python -m src.features.make_features --spot data/raw/binance_spot_ETHUSDT_1m.parquet --funding data/raw/binance_funding_ETHUSDT.parquet --symbol ETHUSDT --timeframe 5m --out data/processed/ETHUSDT_5m.parquet
```

Produces features (returns, realized vol, rolling stats, RSI, funding rates aligned) and **targets**: next-horizon realized volatility.

## 3) Walk-forward training & model comparison

```bash
python -m src.modeling.run_experiments --data data/processed/BTCUSDT_5m.parquet --symbol BTCUSDT --horizon 12 --output runs/BTCUSDT
python -m src.modeling.run_experiments --data data/processed/ETHUSDT_5m.parquet --symbol ETHUSDT --horizon 12 --output runs/ETHUSDT
```

Models compared: **GARCH(1,1)** (ARCH package), **XGBoost**, **LSTM**. Metrics: MASE, RMSE, R2. Includes **Diebold–Mariano** tests and **feature importance** (XGB).

## 4) Backtest — volatility targeting

```bash
python -m src.backtest.vol_targeting --pred runs/BTCUSDT/predictions.parquet --retcol ret_5m --fee_bps 1 --output runs/BTCUSDT/backtest_BTCUSDT.json
```

Computes Sharpe, Sortino, max drawdown, turnover, and regime-sliced results (k-means on vol regime).

## 5) Project structure

```
crypto-vol-quickstart/
├── data/raw/               # raw downloads
├── data/processed/         # aligned features/targets
├── runs/                   # experiment outputs
├── src/
│   ├── data/
│   ├── features/
│   ├── modeling/
│   └── backtest/
└── requirements.txt
```

**Notes**
- Uses only public endpoints. No API key required for these scripts.
- You can swap Binance with other CCXT exchanges easily.
- Extend `src/features/make_features.py` to add on-chain data when you obtain an API (e.g., Glassnode).

