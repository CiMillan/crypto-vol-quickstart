.PHONY: venv data features exp backtest

venv:
\tpython3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

data:
\tpython -m src.data.ccxt_download --symbol BTC/USDT --timeframe 1m --since 2020-01-01 --limit 1000 --out data/raw/binance_spot_BTCUSDT_1m.parquet && \
\tpython -m src.data.binance_funding --symbol BTCUSDT --start 2020-01-01 --end 2025-09-19 --out data/raw/binance_funding_BTCUSDT.parquet

features:
\tpython -m src.features.make_features --spot data/raw/binance_spot_BTCUSDT_1m.parquet --funding data/raw/binance_funding_BTCUSDT.parquet --symbol BTCUSDT --timeframe 5m --out data/processed/BTCUSDT_5m.parquet

exp:
\tpython -m src.modeling.run_experiments --data data/processed/BTCUSDT_5m.parquet --symbol BTCUSDT --horizon 12 --output runs/BTCUSDT

backtest:
\tpython -m src.backtest.vol_targeting --pred runs/BTCUSDT/predictions.parquet --retcol ret_5 --fee_bps 1 --output runs/BTCUSDT/backtest_BTCUSDT.json
