-- {{start}} TEXT, {{end}} TEXT
WITH bounds AS (
  SELECT
    COALESCE(TRY(FROM_ISO8601_TIMESTAMP('{{start}}')),
             FROM_ISO8601_TIMESTAMP('2025-09-01T00:00:00Z')) AS ts_start,
    COALESCE(TRY(FROM_ISO8601_TIMESTAMP('{{end}}')),
             FROM_ISO8601_TIMESTAMP('2025-09-02T00:00:00Z')) AS ts_end
),
tx_agg AS (
  SELECT
    t.block_time AS ts_block,
    AVG(TRY_CAST(t.priority_fee_per_gas AS DOUBLE)) / 1e9 AS avg_priority_gwei
  FROM ethereum.transactions t
  CROSS JOIN bounds b
  WHERE t.block_time >= b.ts_start
    AND t.block_time <  b.ts_end
    AND t.priority_fee_per_gas IS NOT NULL
  GROUP BY 1
)
SELECT
  b."time"                                   AS ts_block,
  b.base_fee_per_gas / 1e9                   AS basefee_gwei,
  x.avg_priority_gwei                        AS priority_fee_gwei,
  b.gas_used,
  b.gas_limit,
  TRY_CAST(b.gas_used AS DOUBLE) / NULLIF(TRY_CAST(b.gas_limit AS DOUBLE),0) AS utilization,
  -- burn in ETH = base fee (wei) * gas_used / 1e18
  (TRY_CAST(b.base_fee_per_gas AS DOUBLE) * TRY_CAST(b.gas_used AS DOUBLE)) / 1e18 AS burn_eth
FROM ethereum.blocks b
LEFT JOIN tx_agg x ON x.ts_block = b."time"
CROSS JOIN bounds q
WHERE b."time" >= q.ts_start
  AND b."time" <  q.ts_end
ORDER BY ts_block;
