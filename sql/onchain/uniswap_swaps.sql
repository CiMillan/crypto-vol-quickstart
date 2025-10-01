-- Description: **DEX trades volume** over time. Defaults to Uniswap-only; switch flags to include all DEXs.
-- Params: {{start}}, {{end}}, optional {{granularity}} in ['minute','hour','day'] (defaults 'hour'), optional {{uniswap_only}} in ['true','false'].
-- Source: dex.trades (standardized across many protocols in Dune).

WITH bounds AS (
  SELECT
    FROM_ISO8601_TIMESTAMP(COALESCE('{{start}}','1970-01-01T00:00:00Z')) AS start_ts,
    FROM_ISO8601_TIMESTAMP(COALESCE('{{end}}','2100-01-01T00:00:00Z'))   AS end_ts,
    COALESCE(NULLIF('{{granularity}}',''), 'hour')                        AS g,
    COALESCE(NULLIF('{{uniswap_only}}',''), 'true')                       AS uo
),
filtered AS (
  SELECT
    d.block_time,
    d.amount_usd,
    d.project,
    d.version,
    d.project_contract_address
  FROM dex.trades d
  WHERE d.block_time >= (SELECT start_ts FROM bounds)
    AND d.block_time <  (SELECT end_ts   FROM bounds)
    AND (
      (SELECT uo FROM bounds) = 'false'
      OR lower(d.project) LIKE 'uniswap%'
      OR lower(d.project) = 'uniswap' OR d.version IN ('v2','v3','3')
    )
)
SELECT
  date_trunc((SELECT g FROM bounds), block_time) AS ts,
  SUM(amount_usd) AS volume_usd,
  any_value(project) AS any_project,
  any_value(version) AS any_version,
  TO_HEX(any_value(project_contract_address)) AS any_pool
FROM filtered
GROUP BY 1
ORDER BY 1;
