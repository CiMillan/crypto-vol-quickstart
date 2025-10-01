- **stables_cex_hourly** → based on `transfers_cex_flags.sql` with `{{granularity}}='hour'` and token filter to stablecoins (USDT/USDC/DAI) by contract list.
- **eth_cex_hourly** → based on `transfers_cex_flags.sql` with `asset_type='ETH'` and `{{granularity}}='hour'`.
- **dex_uniswap_hourly** → based on `uniswap_swaps.sql` with `{{granularity}}='hour'`, `{{uniswap_only}}='true'`.

-- Add token allowlists/denylists near the top of each SQL file as CTEs for reproducibility.
