# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     notebook_metadata_filter: jupytext,text_representation,kernelspec
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3.11 (.venv)
#     language: python
#     name: crypto-vol-quickstart-311
# ---

# %% [markdown]
# 1️⃣ **Core Idea**
#
# You will **trade spot BTC (or ETH)** and hedge it with **CME or Binance perpetual futures**.
#
# **Academic Side**: test hedge effectiveness across regimes, estimate optimal hedge ratios, compare static vs. ML-based dynamic hedges.
#
# **Practical Side**: deploy an automated pipeline that adjusts your futures hedge daily/hourly, based on your model’s volatility signals.

# %% [markdown]
# 2️⃣ **Data Pipeline (via APIs)**
#
# You need **low-latency, historical + live data** from both spot and futures markets.
#
# **Binance (free & deep liquidity)**
#
# ```ccxt``` Python library → pull **spot BTC/ETH** OHLCV data (1m/5m/daily).
#
# ```ccxt``` also supports **perpetual futures** endpoints (funding rates, open interest, perp prices).
#
# Funding rates = signal for hedging pressure (research + trading feature).
#
# **CME Futures (research grade, not free real-time)**
#
# Historical data from Quandl/Nasdaq Data Link.
#
# For live trading: skip CME, focus on Binance perps first (they trade 24/7).
#
# **Implementation step:**

# %%
pip install ccxt pandas numpy

# %%
import ccxt
import pandas as pd

binance = ccxt.binance()
# Spot BTC
spot = binance.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=1000)
spot_df = pd.DataFrame(spot, columns=['ts','open','high','low','close','vol'])

# Futures BTC-PERP
fut = binance.fetch_ohlcv('BTC/USDT:USDT', timeframe='1h', limit=1000)
fut_df = pd.DataFrame(fut, columns=['ts','open','high','low','close','vol'])

# %% [markdown]
# 3️⃣ **Hedge Ratio Calculation**
#
# Optimal hedge ratio (OHR) = **minimizes variance of hedged portfolio**.
#
# Classic = **OLS regression**:

# %% [markdown]
# $$
# r_{t}^{\text{spot}} \;=\; \alpha \;+\; \beta \, r_{t}^{\text{futures}} \;+\; \epsilon_{t}
# $$
#

# %% [markdown]
# - $r_{t}^{\text{spot}}$: return of the asset you own (the “thing you’re protecting”).
#
# - $r_{t}^{\text{futures}}$: return of the hedge instrument (like BTC futures).
#
# - $\beta$: slope coefficient $\;\rightarrow\;$ this is the hedge ratio.
#
# - $\alpha$: intercept (often close to 0).
#
# - $\epsilon_{t}$: leftover “noise” the hedge can’t explain.
#
# 👉 The logic: if futures returns strongly move with spot returns, then a certain multiple of futures (𝛽) will “cancel out” the spot risk.
#
#
# Dynamic = **rolling regression** or **ML forecast-based ratio**.
#
# **Implementation step**:

# %%
import statsmodels.api as sm

y = spot_df['close'].pct_change().dropna()
x = fut_df['close'].pct_change().dropna()
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
hedge_ratio = model.params[1]
print("Optimal hedge ratio:", hedge_ratio)

# %% [markdown]
# 1. **What is a hedge?**
#
# Imagine you **own something risky** (say Bitcoin). Its price jumps up and down a lot.
#
# You don’t want to suffer big losses if the price suddenly drops.
#
# To protect yourself, you “hedge” — which usually means taking an opposite position in a related market (like futures contracts).
#
# 👉 The idea: losses in one side (spot) are offset by gains in the hedge (futures).
#
# 2. **What is the hedge ratio?**
#
# The hedge ratio tells you how big your hedge should be.
#
# If you own 1 BTC, do you hedge with 1 BTC worth of futures? Or 0.7 BTC worth? Or 1.2 BTC worth?
#
# That proportion is the hedge ratio.
#
# 3. **Optimal Hedge Ratio (OHR)**
#
# The OHR is the 𝛽 that minimizes the variance (ups and downs) of the combined portfolio.
#
# That’s why we run the regression: OLS slope gives the best linear “fit” between spot and futures returns.

# %% [markdown]
# 4️⃣ **Backtesting Framework**
#
# You’ll want to simulate hedging:
#
# - Start with spot long only.
#
# - Add futures short with OHR.
#
# - Compare variance & drawdowns.
#
# Metric:
#
# - **Variance reduction %**
#
# - **Sharpe ratio improvement**
#
# This is your PhD Study 1 empirical test and your P&L check before going live.

# %% [markdown]
# 5️⃣ **Execution Layer (Live Trading)**
#
# To make money now, you need execution.
#
# **Binance API** (via ```ccxt```) allows **create_order** for spot & futures.
#
# Start with **paper trading mode** (binance testnet) until system stable.
#
# **Implementation step**:

# %%
# Example: place short futures hedge
binance.set_sandbox_mode(True)  # testnet
binance.create_order('BTC/USDT:USDT', type='market', side='sell', amount=0.1)

# %% [markdown]
# 6️⃣ **Add Dynamic Hedging (PhD–Practical Bridge)**
#
# Static hedge = always OHR.
# Dynamic hedge = adjust based on volatility forecast.
#
# Train ML models (XGBoost baseline → later LSTM/Transformers) to forecast next-day volatility.
#
# Rule:
#
# If vol forecast > threshold → increase hedge ratio.
#
# If vol forecast low → reduce/skip hedge.
#
# This is **directly your PhD contribution**: proving ML adds value vs. static hedge.

# %% [markdown]
# 7️⃣ **Automation & Monitoring**
#
# Schedule pipeline: ```cron``` / ```Airflow``` / ```GitHub Actions```.
#
# Logging: write trades & hedge ratios to CSV + Google Sheets (for quick sanity check).
#
# Alerts: Slack/Zapier webhook for failed trades or funding rate anomalies.

# %%
