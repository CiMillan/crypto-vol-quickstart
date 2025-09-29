# Crypto Volatility Forecasting — Quickstart

A minimal, end-to-end scaffold to **download data**, **build features**, **train models** (GARCH, XGBoost, LSTM), and **backtest** a volatility-targeting strategy with walk-forward validation.

> Focus assets: BTCUSDT, ETHUSDT (spot + perp). Exchange: Binance (public endpoints via CCXT + REST).

## 0) Setup

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
> This repo uses a pre-commit hook with jupytext to keep ```.ipynb``` and paired ```.py``` files in sync.

## 1) Download data

- Spot OHLCV via CCXT (Binance) — 1m (later resampled to 5m/1h)
- Perp funding rates via Binance public endpoint
- (Optional) Perp klines via CCXT if available

```bash
# BTC & ETH spot (1m)
python -m src.data.ccxt_download --symbol BTC/USDT --timeframe 1m --since 2020-01-01 --limit 1000 --out data/raw/binance_spot_BTCUSDT_1m.parquet
python -m src.data.ccxt_download --symbol ETH/USDT --timeframe 1m --since 2020-01-01 --limit 1000 --out data/raw/binance_spot_ETHUSDT_1m.parquet

# Funding (8h prints)
python -m src.data.binance_funding --symbol BTCUSDT --start 2020-01-01 --end 2025-09-19 --out data/raw/binance_funding_BTCUSDT.parquet
python -m src.data.binance_funding --symbol ETHUSDT --start 2020-01-01 --end 2025-09-19 --out data/raw/binance_funding_ETHUSDT.parquet
```
> If you also want perp candles, add a CLI for BTC/USDT perps (naming convention in your repo is `binance_perp_..._5m.parquet`). Make sure the symbol is correct (avoid `BTCUSDTUSDT` typos).

## 2) Build feature sets & targets

```bash
python -m src.features.make_features \
  --spot data/raw/binance_spot_BTCUSDT_1m.parquet \
  --funding data/raw/binance_funding_BTCUSDT.parquet \
  --symbol BTCUSDT \
  --timeframe 5m \
  --out data/processed/BTCUSDT_5m.parquet

python -m src.features.make_features \
  --spot data/raw/binance_spot_ETHUSDT_1m.parquet \
  --funding data/raw/binance_funding_ETHUSDT.parquet \
  --symbol ETHUSDT \
  --timeframe 5m \
  --out data/processed/ETHUSDT_5m.parquet
```

Produces features (returns, realized vol, rolling stats, RSI, funding rates aligned) and **targets**: next-horizon realized volatility.

## 3) Walk-forward training & model comparison

```bash
python -m src.modeling.run_experiments \
  --data data/processed/BTCUSDT_5m.parquet --symbol BTCUSDT --horizon 12 --output runs/BTCUSDT

python -m src.modeling.run_experiments \
  --data data/processed/ETHUSDT_5m.parquet --symbol ETHUSDT --horizon 12 --output runs/ETHUSDT
```

Models compared: **GARCH(1,1)** (ARCH package), **XGBoost**, **LSTM**. Metrics: MASE, RMSE, R2. Includes **Diebold–Mariano** tests and **feature importance** (XGB).

## 4) Backtest — volatility targeting

```bash
python -m src.backtest.vol_targeting \
  --pred runs/BTCUSDT/predictions.parquet \
  --retcol ret_5m \
  --fee_bps 1 \
  --output runs/BTCUSDT/backtest_BTCUSDT.json
```

Computes Sharpe, Sortino, MDD, turnover, and regime-sliced results (k-means on vol regime).

5) **Hedge MVP — Spot↔Perp hedge with funding (new)**

*   Notebook/script:
    
    *   notebooks/06\_hedge\_mvp/hedge\_MVP.ipynb
        
    *   notebooks/06\_hedge\_mvp/hedge\_MVP.py (paired by jupytext)
        
*   What it does:
    
    *   Loads **spot**, **funding**, and (optionally) **perp** prices
        
    *   Resamples to **1h**, computes r\_spot, r\_fut
        
    *   Merges **funding** (8h → per-hour)
        
    *   Estimates **OLS β** (static & rolling) and **vol-scaled β**
        
    *   Computes **hedged returns** with **fees & funding**
        
    *   Writes artifacts under runs/BTCUSDT/hedge\_mvp//:
        
        *   spot\_snapshot.csv, perp\_snapshot.csv, funding\_snapshot.csv
            
        *   prices\_merged.csv, beta\_series.csv, returns\_decomposed.csv
            
        *   metrics.json (variance reduction, Sharpe, MDD, turnover)
            
        *   Plots: cumlogret.png, rolling\_vol.png, beta\_vs\_funding.png, variance\_components.png
            
        *   paper\_rebalance\_log.jsonl (per-bar audit trail)
            
*   Outcomes from a sample run:
    
    *   **Variance reduction** ~ **92%** at 1h
        
    *   **Sharpe** improves (hedged ≥ unhedged)
        
    *   **MDD** compressed from ~−32% to ~−8%
        
    *   **Low turnover** (tiny fee drag), funding correctly signed
        

## 5) Project structure

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

