import argparse
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from ..utils.io import read_parquet, write_parquet

def resample_ohlc(df, rule='5T'):
    o = df['open'].resample(rule).first()
    h = df['high'].resample(rule).max()
    l = df['low'].resample(rule).min()
    c = df['close'].resample(rule).last()
    v = df['volume'].resample(rule).sum()
    out = pd.concat({'open':o,'high':h,'low':l,'close':c,'volume':v}, axis=1).dropna()
    return out

def realized_vol(returns, window):
    return returns.rolling(window).std() * np.sqrt(returns.freq.n / pd.Timedelta('1D'))

def make_basic_features(ohlc):
    ohlc = ohlc.copy()
    ohlc['ret_1'] = np.log(ohlc['close']).diff()
    ohlc['ret_5'] = ohlc['ret_1'].rolling(5).sum()
    ohlc['rv_5'] = ohlc['ret_1'].rolling(5).std() * np.sqrt(5)
    ohlc['rv_12'] = ohlc['ret_1'].rolling(12).std() * np.sqrt(12)
    ohlc['rsi_14'] = ta_rsi(ohlc['close'], 14)
    ohlc['sma_20'] = ohlc['close'].rolling(20).mean()
    ohlc['ema_20'] = ohlc['close'].ewm(span=20, adjust=False).mean()
    return ohlc

def ta_rsi(close, period=14):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=(period-1), adjust=False).mean()
    ma_down = down.ewm(com=(period-1), adjust=False).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def align_funding(features, funding):
    # Funding is ~8-hour cadence; forward-fill to bar frequency
    funding = funding.reindex(features.index).ffill()
    features = features.join(funding, how='left')
    return features

def add_target(features, horizon=12):
    # Target: next-horizon realized volatility (sum of future returns std)
    ret1 = features['ret_1']
    fut_ret = ret1.shift(-1).rolling(horizon).sum()
    target = ret1.shift(-1).rolling(horizon).std() * np.sqrt(horizon)
    features['target_rv'] = target
    return features

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spot', required=True)
    ap.add_argument('--funding', required=True)
    ap.add_argument('--symbol', required=True)
    ap.add_argument('--timeframe', default='5m')  # 5T pandas alias
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    spot = read_parquet(args.spot)
    spot = spot.tz_convert('UTC')
    rule = {'1m':'1T','3m':'3T','5m':'5T','15m':'15T','1h':'1H'}.get(args.timeframe, '5T')
    spot = resample_ohlc(spot, rule=rule)

    funding = read_parquet(args.funding)

    feat = make_basic_features(spot)
    feat = align_funding(feat, funding)
    feat = add_target(feat, horizon=12 if rule=='5T' else 12)
    feat['symbol'] = args.symbol
    feat = feat.dropna()
    write_parquet(feat, args.out)
    print(f'Features written: {args.out}, rows={len(feat):,}')

if __name__ == '__main__':
    main()
