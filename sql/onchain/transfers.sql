-- Description: Minute-/hour-level **ETH native + ERC-20** transfer flows on Ethereum.
-- Params (string): {{start}}, {{end}}, optional {{granularity}} in ['minute','hour'] (defaults to 'minute').
-- Notes:
--   * Keep addresses as VARBINARY until final SELECT; only TO_HEX at the end.
--   * Dune schema variations for ERC-20:
--       - Newer: erc20_ethereum.evt_Transfer  (preferred)
--       - Older:  erc20."ERC20_evt_Transfer"

WITH bounds AS (
  SELECT
    FROM_ISO8601_TIMESTAMP(COALESCE('{{start}}','1970-01-01T00:00:00Z')) AS start_ts,
    FROM_ISO8601_TIMESTAMP(COALESCE('{{end}}','2100-01-01T00:00:00Z'))   AS end_ts,
    COALESCE(NULLIF('{{granularity}}',''), 'minute')                       AS g
),
-- Native ETH value transfers (from regular transactions)
native AS (
  SELECT
    date_trunc(CASE WHEN (SELECT g FROM bounds) = 'hour' THEN 'hour' ELSE 'minute' END, tx.block_time) AS ts,
    tx."from"   AS from_addr,
    tx."to"     AS to_addr,
    tx.value     AS value_wei,
    CAST(NULL AS VARBINARY) AS token_contract
  FROM ethereum.transactions tx
  WHERE tx.block_time >= (SELECT start_ts FROM bounds)
    AND tx.block_time <  (SELECT end_ts   FROM bounds)
    AND tx.value > 0
),
-- ERC-20 transfers
erc20 AS (
  SELECT
    date_trunc(CASE WHEN (SELECT g FROM bounds) = 'hour' THEN 'hour' ELSE 'minute' END, t.evt_block_time) AS ts,
    t."from"       AS from_addr,
    t."to"         AS to_addr,
    t.value         AS value_raw,
    t.contract_address AS token_contract
  FROM erc20_ethereum.evt_Transfer t
  -- If your Dune uses the legacy schema, switch to: erc20."ERC20_evt_Transfer" t and columns evt_block_time/contract_address
  WHERE t.evt_block_time >= (SELECT start_ts FROM bounds)
    AND t.evt_block_time <  (SELECT end_ts   FROM bounds)
),
-- Union in a common schema; keep value columns separate to avoid confusion
u AS (
  SELECT ts, from_addr, to_addr, token_contract, CAST(value_wei AS DOUBLE) AS amount_raw, 'ETH'  AS asset_type FROM native
  UNION ALL
  SELECT ts, from_addr, to_addr, token_contract, CAST(value_raw AS DOUBLE) AS amount_raw, 'ERC20' AS asset_type FROM erc20
)
SELECT
  ts,
  TO_HEX(from_addr) AS from_address,
  TO_HEX(to_addr)   AS to_address,
  CASE WHEN asset_type = 'ETH' THEN 'ETH' ELSE TO_HEX(token_contract) END AS asset_id,
  asset_type,
  amount_raw
FROM u
ORDER BY ts;
