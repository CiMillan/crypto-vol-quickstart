# Download data


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

