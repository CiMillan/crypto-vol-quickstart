# data/raw — Ingestion Drop-Zone

Put source datasets here (CSV or Parquet). The EDA notebooks expect at least **OHLCV** or a **processed features file** elsewhere.

## Accepted file types
- `.csv` (UTF-8). **Required column:** `timestamp`
- `.parquet` (PyArrow). **Required column:** `timestamp`

## Time & sampling
- `timestamp` should be **UTC** (ISO-8601 like `2025-09-20T12:34:56Z`) or tz-naive UTC equivalent.
- Bars must be **sorted** by time and ideally **unique per bar**.
- The notebooks infer the sampling interval from the median step; large gaps will be flagged.

## Naming convention (recommended)
<source>_<datatype>_<symbol>_<timeframe>.<ext>
Examples:
- binance_spot_BTCUSDT_5m.parquet
- binance_funding_BTCUSDT.csv
- ccxt_spot_ETHUSDT_1h.csv

## OHLCV schema (minimum)
column | type | notes
---|---|---
timestamp | datetime | UTC; sorted; unique if possible
open | float | ≥ 0
high | float | ≥ max(open, close, low)
low | float | ≤ min(open, close, high)
close | float | ≥ 0
volume | float | ≥ 0

## Funding (optional)
timestamp, symbol, funding

## Quick QA checklist
- No NaN timestamps; sorted; no duplicate timestamp rows
- No negative prices; volume ≥ 0
- Reasonable time steps (no huge unexpected gaps)
- Prefer Parquet for large files

## Notes
- Big/raw files are git-ignored by default.
- Notebooks read the first file found here unless you override the path.
