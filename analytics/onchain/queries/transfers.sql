-- {{start}} TEXT, {{end}} TEXT   (you can leave them blank; this query has safe fallbacks)

WITH bounds AS (
  SELECT
    COALESCE(TRY(FROM_ISO8601_TIMESTAMP('{{start}}')), FROM_ISO8601_TIMESTAMP('2025-09-01T00:00:00Z')) AS ts_start,
    COALESCE(TRY(FROM_ISO8601_TIMESTAMP('{{end}}')),   FROM_ISO8601_TIMESTAMP('2025-09-02T00:00:00Z')) AS ts_end
),
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
    LOWER(TO_HEX(t."from")) AS from_address,
    LOWER(TO_HEX(t."to"))   AS to_address,
    TRY_CAST(t.value AS DOUBLE)
      / POWER(
          10,
          CASE t.contract_address
            WHEN FROM_HEX('dac17f958d2ee523a2206206994597c13d831ec7') THEN 6   -- USDT
            WHEN FROM_HEX('a0b86991c6218b36c1d19d4a2e9eb0ce3606eb48') THEN 6   -- USDC
            WHEN FROM_HEX('6b175474e89094c44da98b954eedeac495271d0f') THEN 18  -- DAI
            ELSE 18
          END
        ) AS amount,
    CASE t.contract_address
      WHEN FROM_HEX('dac17f958d2ee523a2206206994597c13d831ec7') THEN 6
      WHEN FROM_HEX('a0b86991c6218b36c1d4a2e9eb0ce3606eb48') THEN 6
      WHEN FROM_HEX('6b175474e89094c44da98b954eedeac495271d0f') THEN 18
      ELSE 18
    END AS decimals,
    CAST(NULL AS DOUBLE) AS amount_usd
  FROM erc20_ethereum.evt_Transfer t
  CROSS JOIN bounds b
  WHERE t.evt_block_time >= b.ts_start
    AND t.evt_block_time <  b.ts_end
    AND t.contract_address IN (
      FROM_HEX('dac17f958d2ee523a2206206994597c13d831ec7'),
      FROM_HEX('a0b86991c6218b36c1d19d4a2e9eb0ce3606eb48'),
      FROM_HEX('6b175474e89094c44da98b954eedeac495271d0f')
    )
),
eth_native AS (
  SELECT
    DATE_TRUNC('minute', tx.block_time) AS ts,
    'ETH' AS chain,
    'ETH' AS token,
    LOWER(TO_HEX(tx."from")) AS from_address,
    LOWER(TO_HEX(tx."to"))   AS to_address,
    TRY_CAST(tx.value AS DOUBLE) / 1e18 AS amount,
    18 AS decimals,
    CAST(NULL AS DOUBLE) AS amount_usd
  FROM ethereum.transactions tx
  CROSS JOIN bounds b
  WHERE tx.block_time >= b.ts_start
    AND tx.block_time <  b.ts_end
    AND tx.value > 0
)
SELECT ts, chain, token, from_address, to_address, amount, decimals, amount_usd
FROM st
UNION ALL
SELECT ts, chain, token, from_address, to_address, amount, decimals, amount_usd
FROM eth_native
ORDER BY ts
