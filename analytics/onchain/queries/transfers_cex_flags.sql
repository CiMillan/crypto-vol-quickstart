-- {{start}} TEXT, {{end}} TEXT

WITH bounds AS (
  SELECT
    COALESCE(TRY(FROM_ISO8601_TIMESTAMP('{{start}}')), FROM_ISO8601_TIMESTAMP('2025-09-01T00:00:00Z')) AS ts_start,
    COALESCE(TRY(FROM_ISO8601_TIMESTAMP('{{end}}')),   FROM_ISO8601_TIMESTAMP('2025-09-02T00:00:00Z')) AS ts_end
),

-- CEX addresses from curated labels (address is VARBINARY)
cex AS (
  SELECT address AS addr_bin
  FROM labels.addresses
  WHERE blockchain = 'ethereum'
    AND (
      LOWER(CAST(category AS VARCHAR)) = 'cex' OR
      LOWER(CAST(category AS VARCHAR)) LIKE '%exchange%'
    )
),

-- Stablecoin ERC-20 transfers (USDT/USDC/DAI) — keep VARBINARY until the end
st AS (
  SELECT
    DATE_TRUNC('minute', t.evt_block_time) AS ts,
    'ETH' AS chain,
    CASE t.contract_address
      WHEN FROM_HEX('dac17f958d2ee523a2206206994597c13d831ec7') THEN 'USDT'
      WHEN FROM_HEX('a0b86991c6218b36c1d19d4a2e9eb0ce3606eb48') THEN 'USDC'
      WHEN FROM_HEX('6b175474e89094c44da98b954eedeac495271d0f') THEN 'DAI'
      ELSE 'OTHER'
    END AS token,
    t."from" AS from_raw,
    t."to"   AS to_raw,
    TRY_CAST(t.value AS DOUBLE) / POWER(
      10,
      CASE t.contract_address
        WHEN FROM_HEX('dac17f958d2ee523a2206206994597c13d831ec7') THEN 6
        WHEN FROM_HEX('a0b86991c6218b36c1d19d4a2e9eb0ce3606eb48') THEN 6
        WHEN FROM_HEX('6b175474e89094c44da98b954eedeac495271d0f') THEN 18
        ELSE 18
      END
    ) AS amount,
    CASE t.contract_address
      WHEN FROM_HEX('dac17f958d2ee523a2206206994597c13d831ec7') THEN 6
      WHEN FROM_HEX('a0b86991c6218b36c1d19d4a2e9eb0ce3606eb48') THEN 6
      WHEN FROM_HEX('6b175474e89094c44da98b954eedeac495271d0f') THEN 18
      ELSE 18
    END AS decimals
  FROM erc20_ethereum.evt_Transfer t
  CROSS JOIN bounds b
  WHERE t.evt_block_time >= b.ts_start
    AND t.evt_block_time <  b.ts_end
    AND t.contract_address IN (
      FROM_HEX('dac17f958d2ee523a2206206994597c13d831ec7'), -- USDT
      FROM_HEX('a0b86991c6218b36c1d19d4a2e9eb0ce3606eb48'), -- USDC
      FROM_HEX('6b175474e89094c44da98b954eedeac495271d0f')  -- DAI
    )
),

-- Native ETH transfers
eth_native AS (
  SELECT
    DATE_TRUNC('minute', tx.block_time) AS ts,
    'ETH' AS chain,
    'ETH' AS token,
    tx."from" AS from_raw,
    tx."to"   AS to_raw,
    TRY_CAST(tx.value AS DOUBLE) / 1e18 AS amount,
    18 AS decimals
  FROM ethereum.transactions tx
  CROSS JOIN bounds b
  WHERE tx.block_time >= b.ts_start
    AND tx.block_time <  b.ts_end
    AND tx.value > 0
),

u AS (
  SELECT * FROM st
  UNION ALL
  SELECT * FROM eth_native
)

SELECT
  ts,
  chain,
  token,
  LOWER(TO_HEX(from_raw)) AS from_address,  -- stringify for output only
  LOWER(TO_HEX(to_raw))   AS to_address,
  amount,
  decimals,
  CASE WHEN c1.addr_bin IS NOT NULL THEN 1 ELSE 0 END AS from_is_cex,
  CASE WHEN c2.addr_bin IS NOT NULL THEN 1 ELSE 0 END AS to_is_cex
FROM u
LEFT JOIN cex c1 ON u.from_raw = c1.addr_bin   -- join on VARBINARY
LEFT JOIN cex c2 ON u.to_raw   = c2.addr_bin
ORDER BY ts
