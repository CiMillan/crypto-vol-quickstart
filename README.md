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

Built by Cintia Millan — Data Scientist & PhD Candidate (NOVA IMS).

Focus: robust ML for crypto volatility forecasting, hedging, and economic validation.

## Dune Export — On-Chain Data

**Run all jobs (writes Parquet to `data/processed/onchain/`):**
```bash
# set your key once (or copy .env.example to .env)
export DUNE_API_KEY=... 
make dune-install
make dune-onchain-all START=2025-09-01T00:00:00Z END=2025-09-02T00:00:00Z
Run one job:

bash
Copy code
make dune-onchain-one JOB=uniswap_swaps START=2025-09-01T00:00:00Z END=2025-09-02T00:00:00Z

## Heads-up: Dune params & free tier
- In Dune UI, each saved query must define **TEXT** params `start` and `end` (ISO8601).
- For large tables (raw transfers), the free tier can return **402 Payment Required** even on narrow windows. Prefer **server-side aggregations** (hourly netflows, swap notional) and download the small result sets.

## Pro tip: Warm cache workflow
If you can open Dune in the browser and a CLI job is blocked by credits:

1) Open the saved query in Dune with the **exact same** `start`/`end` you plan to pass from CLI.  
2) **Run it in the UI** — this pays the compute cost.  
3) Re-run your CLI; fetching results of a **cached** execution is often allowed (plan-dependent).

---

## Appendix: On-Chain Feature Set (Minimum-Viable) — Data Dictionary

**Scope**  
Hourly features aligned to UTC, designed for short-horizon (1h–1d) volatility forecasting for BTC/ETH.

### Inputs (standardized tables)
- **`transfers.parquet`** — token transfers (BTC/ETH L1 native & ERC-20).  
  **Columns:** `ts` (datetime UTC, floor to minute), `chain`, `token`, `from_address`, `to_address`, `amount_raw`, `decimals`, `amount`, `amount_usd`
- **`labels/exchanges.json`** — address → `{ entity: str, is_exchange: bool }`
- **`uniswap_swaps.parquet`** — DEX swaps (Uniswap v2/v3).  
  **Columns:** `ts`, `pool_address`, `base_token`, `quote_token`, `amount_usd`, `mid_before`, `mid_after`
- **`eth_blocks.parquet`** — per-block gas/fee metrics.  
  **Columns:** `ts_block`, `basefee_gwei`, `priority_fee_gwei`, `gas_used`, `gas_limit`
- *(Optional)* **`address_first_seen.parquet`** — first time we observed an address.  
  **Columns:** `address`, `first_ts`

> All monetary columns are **post-decimal** (human units). If you ingest raw logs, apply:  
> `amount = amount_raw / 10**decimals`.

### Features (output columns)

| Column                      | Description                                | Units | Construction (hourly)                                                                     |
|----------------------------|--------------------------------------------|-------|-------------------------------------------------------------------------------------------|
| `ex_netflow_btc`           | BTC net flow to CEX (in − out)             | BTC   | Sum transfers where CEX is receiver minus where CEX is sender (BTC chain)                 |
| `ex_netflow_eth`           | ETH net flow to CEX (in − out)             | ETH   | Same as above for ETH native (exclude ERC-20)                                             |
| `stables_netflow_cex_usd`  | Stablecoin net flow to CEX (USDT+USDC+DAI) | USD   | Sum USD of ERC-20 transfers **into** CEX minus **out**                                    |
| `dex_swap_notional_usd`    | Uniswap v2/v3 total traded notional        | USD   | Sum `amount_usd` over swaps per hour                                                      |
| `dex_price_impact_bps`     | Avg immediate price impact per swap (abs)  | bps   | `abs((mid_after - mid_before)/mid_before)*1e4`, then mean per hour                        |
| `gas_basefee_gwei`         | Median basefee within the hour             | gwei  | Median of `basefee_gwei` across blocks in hour                                            |
| `gas_tip_p90_gwei`         | 90th percentile priority fee               | gwei  | P90 of `priority_fee_gwei`                                                                |
| `pct_blocks_full`          | % blocks with utilization ≥ 95%            | %     | Share where `gas_used / gas_limit ≥ 0.95`                                                 |
| `whale_cex_tx_count_gt_1m` | Count of ≥$1M transfers touching a CEX     | count | Count (native/ERC-20) transfers `amount_usd ≥ 1e6` with CEX as sender or receiver         |
| `whale_cex_tx_usd_gt_1m`   | USD sum of those whale transfers           | USD   | Sum `amount_usd` for same filter                                                           |
| `active_addresses_hourly`  | Unique senders+receivers per hour          | count | `nunique(from ∪ to)`                                                                      |
| `new_addresses_hourly`     | Addresses seen for the first time          | count | Count addresses whose `first_ts` equals current hour                                       |
| `*_z7d`                    | 7-day z-score of any base series           | z     | `(x − mean_7d) / std_7d` (rolling, min periods = 24)                                      |

### Quality checks (must pass)
1. **Conservation sanity:** `inflows − outflows ≈ Δreserves` over long windows (where reserves available).  
2. **Decimals applied once:** No sudden ×10^N jumps.  
3. **Label coverage:** Report `% of volume with is_exchange` label per day.  
4. **No mixed clocks:** All inputs UTC, then resampled to exact `:00` hourly bins.
