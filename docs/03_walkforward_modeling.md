# Walk-forward training & model comparison


```bash
python -m src.modeling.run_experiments \
  --data data/processed/BTCUSDT_5m.parquet --symbol BTCUSDT --horizon 12 --output runs/BTCUSDT

python -m src.modeling.run_experiments \
  --data data/processed/ETHUSDT_5m.parquet --symbol ETHUSDT --horizon 12 --output runs/ETHUSDT
```

Models compared: **GARCH(1,1)** (ARCH package), **XGBoost**, **LSTM**. Metrics: MASE, RMSE, R2. Includes **Diebold–Mariano** tests and **feature importance** (XGB).

