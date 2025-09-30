-- {{start}} TEXT, {{end}} TEXT
WITH bounds AS (
  SELECT
    COALESCE(TRY(FROM_ISO8601_TIMESTAMP('{{start}}')),
             FROM_ISO8601_TIMESTAMP('2025-09-01T00:00:00Z')) AS ts_start,
    COALESCE(TRY(FROM_ISO8601_TIMESTAMP('{{end}}')),
             FROM_ISO8601_TIMESTAMP('2025-09-02T00:00:00Z')) AS ts_end
),
trades AS (
  SELECT
    DATE_TRUNC('minute', block_time)          AS ts,
    TO_HEX(project_contract_address)          AS pool_address_hex,  -- VARBINARY → hex
    amount_usd
  FROM dex.trades
  CROSS JOIN bounds b
  WHERE blockchain = 'ethereum'
    AND block_time >= b.ts_start
    AND block_time <  b.ts_end
    -- keep all DEXs OR narrow to Uniswap V3:
    -- AND (LOWER(project) LIKE 'uniswap%' OR (LOWER(project)='uniswap' AND (version='3' OR version='v3')))
)
SELECT
  ts,
  LOWER(pool_address_hex) AS pool_address,
  SUM(amount_usd)         AS usd_volume,
  COUNT(*)                AS n_trades
FROM trades
GROUP BY 1, 2
ORDER BY 1, 2
