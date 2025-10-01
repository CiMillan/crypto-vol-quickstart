-- Description: Same as transfers + joins **CEX labels** and classifies flow_type.
-- Params: {{start}}, {{end}}, optional {{granularity}} in ['minute','hour'] (defaults 'hour' is typical for dashboards).
-- Label sources on Dune: labels.addresses (per-address tags) and labels.owner_addresses (owner strings like 'binance', 'coinbase').

WITH bounds AS (
  SELECT
    FROM_ISO8601_TIMESTAMP(COALESCE('{{start}}','1970-01-01T00:00:00Z')) AS start_ts,
    FROM_ISO8601_TIMESTAMP(COALESCE('{{end}}','2100-01-01T00:00:00Z'))   AS end_ts,
    COALESCE(NULLIF('{{granularity}}',''), 'hour')                        AS g
),
base AS (
  SELECT
    date_trunc((SELECT CASE WHEN g='hour' THEN 'hour' ELSE 'minute' END FROM bounds), COALESCE(tx.block_time, t.evt_block_time)) AS ts,
    COALESCE(tx."from", t."from") AS from_addr,
    COALESCE(tx."to",   t."to")   AS to_addr,
    CASE WHEN tx.value IS NOT NULL THEN 'ETH' ELSE 'ERC20' END AS asset_type,
    CASE WHEN tx.value IS NOT NULL THEN tx.value ELSE t.value END AS amount_raw,
    CASE WHEN tx.value IS NOT NULL THEN CAST(NULL AS VARBINARY) ELSE t.contract_address END AS token_contract
  FROM ethereum.transactions tx
  FULL OUTER JOIN erc20_ethereum.evt_Transfer t
    ON FALSE -- disjoint sources, we just want a vertical union via FULL + COALESCE
  WHERE (COALESCE(tx.block_time, t.evt_block_time)) >= (SELECT start_ts FROM bounds)
    AND (COALESCE(tx.block_time, t.evt_block_time)) <  (SELECT end_ts   FROM bounds)
    AND (tx.value > 0 OR t.value > 0)
),
from_lab AS (
  SELECT addr.address AS address, any_value(owner_name) AS owner, max_by(label, updated_at) AS label
  FROM labels.owner_addresses oa
  JOIN labels.addresses addr ON oa.owner = addr.owner
  GROUP BY 1
),
addr_tags AS (
  SELECT address, owner, label,
         (CASE WHEN lower(coalesce(owner,'')) LIKE '%binance%' OR lower(coalesce(label,'')) LIKE '%exchange%' THEN 1 ELSE 0 END) AS is_cex
  FROM from_lab
),
joined AS (
  SELECT
    b.ts,
    b.from_addr,
    b.to_addr,
    b.asset_type,
    b.amount_raw,
    b.token_contract,
    COALESCE(f.is_cex, 0) AS from_is_cex,
    COALESCE(tg.is_cex, 0) AS to_is_cex
  FROM base b
  LEFT JOIN addr_tags f ON b.from_addr = f.address
  LEFT JOIN addr_tags tg ON b.to_addr = tg.address
)
SELECT
  ts,
  TO_HEX(from_addr) AS from_address,
  TO_HEX(to_addr)   AS to_address,
  CASE WHEN asset_type='ETH' THEN 'ETH' ELSE TO_HEX(token_contract) END AS asset_id,
  asset_type,
  amount_raw,
  from_is_cex,
  to_is_cex,
  CASE
    WHEN from_is_cex=1 AND to_is_cex=1 THEN 'cex_to_cex'
    WHEN from_is_cex=1 AND to_is_cex=0 THEN 'cex_to_user'
    WHEN from_is_cex=0 AND to_is_cex=1 THEN 'user_to_cex'
    ELSE 'user_to_user'
  END AS flow_type,
  CASE WHEN token_contract = 0x0000000000000000000000000000000000000000 THEN 1 ELSE 0 END AS involves_zero_address -- marks possible mint/burn patterns on some tokens
FROM joined
ORDER BY ts;
