# Backtest — volatility targeting


```bash
python -m src.backtest.vol_targeting \
  --pred runs/BTCUSDT/predictions.parquet \
  --retcol ret_5m \
  --fee_bps 1 \
  --output runs/BTCUSDT/backtest_BTCUSDT.json
```

Computes Sharpe, Sortino, MDD, turnover, and regime-sliced results (k-means on vol regime).

