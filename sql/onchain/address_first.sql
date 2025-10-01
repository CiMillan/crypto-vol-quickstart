-- Description: First-seen timestamp per **address** (from and to) on Ethereum, hour-bucketed.
-- Params: {{start}}, {{end}}.
-- Notes: Keep addresses VARBINARY; convert to hex only at the end.

WITH bounds AS (
  SELECT
    FROM_ISO8601_TIMESTAMP(COALESCE('{{start}}','1970-01-01T00:00:00Z')) AS start_ts,
    FROM_ISO8601_TIMESTAMP(COALESCE('{{end}}','2100-01-01T00:00:00Z'))   AS end_ts
),
flat AS (
  SELECT t.block_time AS ts, t."from" AS addr FROM ethereum.transactions t
  WHERE t.block_time >= (SELECT start_ts FROM bounds) AND t.block_time < (SELECT end_ts FROM bounds)
  UNION ALL
  SELECT t.block_time AS ts, t."to"   AS addr FROM ethereum.transactions t
  WHERE t.block_time >= (SELECT start_ts FROM bounds) AND t.block_time < (SELECT end_ts FROM bounds)
),
firsts AS (
  SELECT addr, MIN(ts) AS first_seen
  FROM flat
  WHERE addr IS NOT NULL
  GROUP BY 1
)
SELECT
  date_trunc('hour', first_seen) AS first_seen_hour,
  TO_HEX(addr) AS address
FROM firsts
ORDER BY first_seen_hour;
