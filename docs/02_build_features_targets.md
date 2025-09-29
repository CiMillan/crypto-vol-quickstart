# Build feature sets & targets


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

