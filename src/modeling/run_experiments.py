import argparse, json
import numpy as np, pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from ..utils.io import read_parquet
from pathlib import Path

def mase(y_true, y_pred, y_naive):
    denom = np.mean(np.abs(y_true[1:] - y_true[:-1]))
    return np.mean(np.abs(y_true - y_pred)) / (denom + 1e-12)

def diebold_mariano(e1, e2, h=1):
    # Simple DM test (Harvey et al. small-sample correction omitted for brevity)
    d = e1**2 - e2**2
    dbar = d.mean()
    s = d.std(ddof=1) / np.sqrt(len(d))
    dm = dbar / (s + 1e-12)
    return float(dm)

def garch_forecast(close, horizon):
    ret = 100 * np.log(close).diff().dropna()
    am = arch_model(ret, vol='Garch', p=1, q=1, rescale=False)
    res = am.fit(disp='off')
    f = res.forecast(horizon=horizon, reindex=False).variance.values[-1, :]
    # map variance to per-step std; naive alignment
    return np.sqrt(f / 100**2).mean()

def build_matrix(df):
    y = df['target_rv'].values
    X = df[['ret_1','ret_5','rv_5','rv_12','rsi_14','sma_20','ema_20','fundingRate']].values
    return X, y

def run_wfv(df, horizon, outdir):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    X, y = build_matrix(df)
    tscv = TimeSeriesSplit(n_splits=5)
    preds = []
    rows = []
    for i, (tr, te) in enumerate(tscv.split(X)):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]

        # XGBoost
        xgb = XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method='hist'
        )
        xgb.fit(Xtr, ytr)
        p_xgb = xgb.predict(Xte)

        # LSTM (very small, unoptimized baseline) — optional import torch
        try:
            import torch, torch.nn as nn
            class LSTMReg(nn.Module):
                def __init__(self, nfeat):
                    super().__init__()
                    self.lstm = nn.LSTM(nfeat, 32, batch_first=True)
                    self.fc = nn.Linear(32, 1)
                def forward(self, x):
                    o,_ = self.lstm(x)
                    return self.fc(o[:,-1,:])
            # reshape to sequences of len=12
            L=12
            def mkseq(XA, yA):
                xs, ys = [], []
                for j in range(L, len(XA)):
                    xs.append(XA[j-L:j])
                    ys.append(yA[j])
                return np.array(xs), np.array(ys)
            Xtr_s, ytr_s = mkseq(Xtr, ytr); Xte_s, yte_s = mkseq(Xte, yte)
            device = 'cpu'
            model = LSTMReg(X.shape[1]).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            lossf = nn.MSELoss()
            TX = torch.tensor(Xtr_s, dtype=torch.float32)
            Ty = torch.tensor(ytr_s, dtype=torch.float32).unsqueeze(-1)
            for epoch in range(20):
                model.train(); opt.zero_grad()
                out = model(TX)
                loss = lossf(out, Ty)
                loss.backward(); opt.step()
            model.eval()
            PX = torch.tensor(Xte_s, dtype=torch.float32)
            P = model(PX).detach().cpu().numpy().ravel()
            # align to test indices
            p_lstm = np.concatenate([np.full(len(yte)-len(P), np.nan), P])
        except Exception as e:
            p_lstm = np.full(len(yte), np.nan)

        # GARCH baseline (re-estimate each fold, crude)
        # map close series in the test span
        close = df['close'].values[te]
        try:
            garch_pred = np.array([garch_forecast(df['close'].iloc[tr], horizon)] * len(yte))
        except Exception:
            garch_pred = np.full(len(yte), np.nan)

        # naive baseline
        y_naive = np.roll(yte, 1); y_naive[0] = ytr[-1]

        for name, pred in [('XGB', p_xgb), ('LSTM', p_lstm), ('GARCH', garch_pred), ('Naive', y_naive)]:
            rmse = float(np.sqrt(mean_squared_error(yte[~np.isnan(pred)], pred[~np.isnan(pred)])))
            m_mase = float(mase(yte[~np.isnan(pred)], pred[~np.isnan(pred)], y_naive[~np.isnan(pred)]))
            rows.append({'fold': i+1, 'model': name, 'rmse': rmse, 'mase': m_mase, 'n': int(np.sum(~np.isnan(pred)))})
        preds.append(pd.DataFrame({'idx': te, 'y_true': yte, 'p_xgb': p_xgb, 'p_lstm': p_lstm, 'p_garch': garch_pred}))

    resdf = pd.DataFrame(rows)
    pred_df = pd.concat(preds, ignore_index=True)
    resdf.to_csv(f'{outdir}/metrics.csv', index=False)
    pred_df.to_parquet(f'{outdir}/predictions.parquet', index=False)
    print(resdf.groupby('model')[['rmse','mase']].mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--symbol', required=True)
    ap.add_argument('--horizon', type=int, default=12)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    df = read_parquet(args.data)
    run_wfv(df, args.horizon, args.output)

if __name__ == '__main__':
    main()
