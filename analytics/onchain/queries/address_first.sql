-- {{start}} TEXT, {{end}} TEXT
WITH bounds AS (
  SELECT
    COALESCE(TRY(FROM_ISO8601_TIMESTAMP('{{start}}')),
             FROM_ISO8601_TIMESTAMP('2025-09-01T00:00:00Z')) AS ts_start,
    COALESCE(TRY(FROM_ISO8601_TIMESTAMP('{{end}}')),
             FROM_ISO8601_TIMESTAMP('2025-09-02T00:00:00Z')) AS ts_end
),
firsts AS (
  SELECT "from" AS addr, MIN(block_time) AS first_ts
  FROM ethereum.transactions
  GROUP BY 1
  UNION ALL
  SELECT "to"   AS addr, MIN(block_time) AS first_ts
  FROM ethereum.transactions
  GROUP BY 1
),
dedup AS (
  SELECT addr, MIN(first_ts) AS first_ts
  FROM firsts
  GROUP BY 1
)
SELECT
  LOWER(TO_HEX(addr))       AS address,        -- stringify at the edge
  DATE_TRUNC('hour', first_ts) AS first_ts
FROM dedup
CROSS JOIN bounds b
WHERE first_ts >= b.ts_start
  AND first_ts <  b.ts_end
ORDER BY first_ts
