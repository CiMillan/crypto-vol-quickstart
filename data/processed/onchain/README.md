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

| Column                         | Description                                   | Units | Construction (hourly)                                                                 |
|--------------------------------|-----------------------------------------------|-------|----------------------------------------------------------------------------------------|
| `ex_netflow_btc`               | BTC net flow to CEX (in − out)               | BTC   | Sum transfers where CEX is receiver minus where CEX is sender (BTC chain)             |
| `ex_netflow_eth`               | ETH net flow to CEX (in − out)               | ETH   | Same as above for ETH native (exclude ERC-20)                                         |
| `stables_netflow_cex_usd`      | Stablecoin net flow to CEX (USDT+USDC+DAI)   | USD   | Sum USD of ERC-20 transfers **into** CEX minus **out**                                |
| `dex_swap_notional_usd`        | Uniswap v2/v3 total traded notional          | USD   | Sum `amount_usd` over swaps per hour                                                  |
| `dex_price_impact_bps`         | Avg immediate price impact per swap (abs)    | bps   | `abs((mid_after - mid_before)/mid_before)*1e4`, then mean per hour                    |
| `gas_basefee_gwei`             | Median basefee within the hour               | gwei  | Median of `basefee_gwei` across blocks in hour                                        |
| `gas_tip_p90_gwei`             | 90th percentile priority fee                 | gwei  | P90 of `priority_fee_gwei`                                                            |
| `pct_blocks_full`              | % blocks with utilization ≥ 95%              | %     | Share where `gas_used / gas_limit ≥ 0.95`                                             |
| `whale_cex_tx_count_gt_1m`     | Count of ≥$1M transfers touching a CEX       | count | Count (native/ERC-20) transfers `amount_usd ≥ 1e6` with CEX as sender or receiver     |
| `whale_cex_tx_usd_gt_1m`       | USD sum of those whale transfers             | USD   | Sum `amount_usd` for same filter                                                       |
| `active_addresses_hourly`      | Unique senders+receivers per hour            | count | `nunique(from ∪ to)`                                                                  |
| `new_addresses_hourly`         | Addresses seen for the first time            | count | Count addresses whose `first_ts` equals current hour                                   |
| `*_z7d`                        | 7-day z-score of any base series             | z     | `(x − mean_7d) / std_7d` (rolling, min periods = 24)                                  |

### Quality checks (must pass)
1. **Conservation sanity:** `inflows − outflows ≈ Δreserves` over long windows (where reserves available).  
2. **Decimals applied once:** No sudden ×10^N jumps.  
3. **Label coverage:** Report `% of volume with is_exchange` label per day.  
4. **No mixed clocks:** All inputs UTC, then resampled to exact `:00` hourly bins.
