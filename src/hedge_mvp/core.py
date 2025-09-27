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
# ---

# %%
from dataclasses import dataclass
from typing import Tuple, Optional
import math, os, time, json, datetime as dt
import numpy as np
import pandas as pd
import ccxt
import statsmodels.api as sm
import matplotlib.pyplot as plt

# %%
# Optional ML
try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

# %%
@dataclass
class HedgeResults:
    hedge_ratio: float
    var_spot: float
    var_hedged: float
    variance_reduction: float
    sharpe_spot: float
    sharpe_hedged: float
    maxdd_spot: float
    maxdd_hedged: float

# %%
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# %%
def fetch_ohlcv(exchange: ccxt.binance, market: str, timeframe: str = "1h", limit: int = 1000) -> pd.DataFrame:
    data = exchange.fetch_ohlcv(market, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts").sort_index()
    return df

# %%
def fetch_funding_rates(exchange: ccxt.binance, symbol_perp: str, limit: int = 200) -> pd.DataFrame:
    """Fetch recent funding rate history if supported by ccxt; fallback empty df."""
    if hasattr(exchange, "fetch_funding_rate_history"):
        try:
            rows = exchange.fetch_funding_rate_history(symbol_perp, limit=limit) or []
            if rows:
                df = pd.DataFrame(rows)
                if "datetime" in df.columns:
                    df["ts"] = pd.to_datetime(df["datetime"], utc=True)
                elif "timestamp" in df.columns:
                    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                return df.set_index("ts").sort_index()
        except Exception:
            pass
    return pd.DataFrame(columns=["fundingRate"])

# %%
def align_close(spot_df: pd.DataFrame, perp_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    df = pd.DataFrame({"spot": spot_df["close"], "perp": perp_df["close"]}).dropna()
    return df["spot"], df["perp"]

# %%
def compute_log_returns(series: pd.Series) -> pd.Series:
    return np.log(series).diff().dropna()

# %%
def estimate_ols_hedge_ratio(r_spot: pd.Series, r_perp: pd.Series) -> float:
    aligned = pd.concat([r_spot, r_perp], axis=1, join="inner").dropna()
    y = aligned.iloc[:,0].values
    x = aligned.iloc[:,1].values
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    return float(model.params[1])

# %%
def backtest_static_hedge(r_spot: pd.Series, r_perp: pd.Series, beta: float) -> Tuple[pd.Series, pd.Series]:
    aligned = pd.concat([r_spot.rename("spot"), r_perp.rename("perp")], axis=1, join="inner").dropna()
    r_spot_al = aligned["spot"]
    r_perp_al = aligned["perp"]
    r_hedged = r_spot_al - beta * r_perp_al
    return r_spot_al, r_hedged

# %%
def max_drawdown_from_returns(returns: pd.Series) -> float:
    equity = (1.0 + returns).cumprod()
    rolling_max = equity.cummax()
    dd = equity / rolling_max - 1.0
    return float(dd.min())

# %%
def sharpe_ratio(returns: pd.Series, periods_per_year: int) -> float:
    if returns.empty or returns.std() == 0:
        return 0.0
    return float((returns.mean() / returns.std()) * math.sqrt(periods_per_year))

# %%
def infer_periods_per_year(timeframe: str) -> int:
    mapping = {"1m":365*24*60, "5m":365*24*12, "15m":365*24*4, "30m":365*24*2, "1h":365*24, "4h":365*6, "1d":365}
    return mapping.get(timeframe, 365*24)

# %%
# ---------- plotting (single-plot, default colors) ----------
def plot_series(spot_close: pd.Series, perp_close: pd.Series, outdir: str):
    ensure_dir(outdir)
    plt.figure()
    spot_close.plot(label="Spot Close")
    perp_close.plot(label="Perp Close")
    plt.title("Spot vs Perp Close")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "prices.png"))
    plt.close()

# %%
def plot_cumlogret(r_spot: pd.Series, r_hedged: pd.Series, outdir: str):
    ensure_dir(outdir)
    plt.figure()
    r_spot.cumsum().plot(label="Spot log-return cum")
    r_hedged.cumsum().plot(label="Hedged log-return cum")
    plt.title("Cumulative Log Returns")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "cumlogret.png"))
    plt.close()

# %%
# ---------- simple realized vol + XGB model ----------
def realized_vol(returns: pd.Series, window: int = 24) -> pd.Series:
    """Rolling realized vol (std of returns) annualized by sqrt(periods). Returns same frequency."""
    if returns.empty:
        return returns
    rv = returns.rolling(window).std()
    return rv

# %%
def build_ml_vol_features(returns: pd.Series, window: int = 48) -> pd.DataFrame:
    df = pd.DataFrame({"r": returns})
    df["abs_r"] = df["r"].abs()
    df["rv_24"] = realized_vol(returns, 24)
    df["rv_48"] = realized_vol(returns, 48)
    df["lag1_abs"] = df["abs_r"].shift(1)
    df["lag2_abs"] = df["abs_r"].shift(2)
    df["lag3_abs"] = df["abs_r"].shift(3)
    df = df.dropna()
    # target = next window vol (e.g., rv_24 forward)
    df["target_vol"] = df["rv_24"].shift(-1)
    return df.dropna()

# %%
def train_xgb_vol_model(df: pd.DataFrame):
    if not _HAS_XGB:
        raise RuntimeError("xgboost not installed; add it to requirements.txt")
    features = ["abs_r","rv_24","rv_48","lag1_abs","lag2_abs","lag3_abs"]
    X = df[features].values
    y = df["target_vol"].values
    # very small model for speed; tune later
    model = XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=42
    )
    model.fit(X, y)
    return model, features

# %%
def predict_next_vol(model, features: list, latest_row: pd.Series) -> Optional[float]:
    try:
        x = latest_row[features].values.reshape(1, -1)
        return float(model.predict(x)[0])
    except Exception:
        return None

# %%
def scale_hedge_ratio(beta: float, pred_vol: Optional[float],
                      vol_low: float, vol_high: float,
                      scale_min: float = 0.3, scale_max: float = 1.2) -> float:
    """
    Map predicted vol to a scaling in [scale_min, scale_max].
    If no prediction, return beta.
    """
    if pred_vol is None or not np.isfinite(pred_vol):
        return beta
    z = (pred_vol - vol_low) / max(1e-9, (vol_high - vol_low))
    z = min(max(z, 0.0), 1.0)
    return beta * (scale_min + z * (scale_max - scale_min))

# %%
# ---------- paper-trade helper (dry-run by default) ----------
def init_binance(testnet: bool = True, api_key: Optional[str] = None, secret: Optional[str] = None):
    """
    Initialize binance client. For testnet trading, pass testnet=True and your testnet keys via env.
    """
    ex = ccxt.binanceusdm()  # USD-M futures for perps
    if testnet and hasattr(ex, "set_sandbox_mode"):
        ex.set_sandbox_mode(True)
    if api_key and secret:
        ex.apiKey = api_key
        ex.secret = secret
    return ex

# %%
def intended_rebalance_log(path: str, payload: dict):
    ensure_dir(os.path.dirname(path))
    with open(path, "a") as f:
        f.write(json.dumps(payload) + "\n")

