-- Description: Per-interval **Ethereum block gas stats** with base fee (gwei), est. priority fee (gwei), gas_used, gas_limit.
-- Params: {{start}}, {{end}}, optional {{granularity}} in ['minute','hour'] (defaults 'minute').
-- Notes: base_fee_per_gas and gas_price units are wei; divide by 1e9 for gwei.

WITH bounds AS (
  SELECT
    FROM_ISO8601_TIMESTAMP(COALESCE('{{start}}','1970-01-01T00:00:00Z')) AS start_ts,
    FROM_ISO8601_TIMESTAMP(COALESCE('{{end}}','2100-01-01T00:00:00Z'))   AS end_ts,
    COALESCE(NULLIF('{{granularity}}',''), 'minute')                       AS g
),
blk AS (
  SELECT
    date_trunc((SELECT CASE WHEN g='hour' THEN 'hour' ELSE 'minute' END FROM bounds), b."time") AS ts,
    AVG(b.base_fee_per_gas)/1e9 AS basefee_gwei,
    SUM(b.gas_used) AS gas_used,
    SUM(b.gas_limit) AS gas_limit
  FROM ethereum.blocks b
  WHERE b."time" >= (SELECT start_ts FROM bounds)
    AND b."time" <  (SELECT end_ts   FROM bounds)
  GROUP BY 1
),
txp AS (
  SELECT
    date_trunc((SELECT CASE WHEN g='hour' THEN 'hour' ELSE 'minute' END FROM bounds), t.block_time) AS ts,
    AVG(GREATEST(t.gas_price - t.base_fee_per_gas, 0))/1e9 AS priority_fee_gwei
  FROM ethereum.transactions t
  WHERE t.block_time >= (SELECT start_ts FROM bounds)
    AND t.block_time <  (SELECT end_ts   FROM bounds)
  GROUP BY 1
)
SELECT
  b.ts,
  b.basefee_gwei,
  COALESCE(p.priority_fee_gwei, 0) AS priority_fee_gwei,
  b.gas_used,
  b.gas_limit
FROM blk b
LEFT JOIN txp p USING (ts)
ORDER BY b.ts;
