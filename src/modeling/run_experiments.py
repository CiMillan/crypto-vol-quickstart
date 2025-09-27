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
import argparse
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from arch import arch_model
from ..utils.io import read_parquet

# %%
def mase(y_true, y_pred):
    if len(y_true) < 2: return np.nan
    denom = np.mean(np.abs(y_true[1:] - y_true[:-1])) + 1e-12
    return np.mean(np.abs(y_true - y_pred)) / denom

# %%
def safe_metrics(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    n = int(mask.sum())
    if n == 0: return np.nan, np.nan, n
    yt, yp = y_true[mask], y_pred[mask]
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    return rmse, float(mase(yt, yp)), n

# %%
def garch_forecast(close, horizon):
    ret = 100 * np.log(close).diff().dropna()
    if len(ret) < 50: return np.nan
    am = arch_model(ret, vol='Garch', p=1, q=1, rescale=False)
    try:
        res = am.fit(disp='off')
        f = res.forecast(horizon=horizon, reindex=False).variance.values[-1, :]
        return float(np.sqrt(np.mean(f)) / 100.0)
    except Exception:
        return np.nan

# %%
def build_matrix(df):
    cols = ['ret_1','ret_5','rv_5','rv_12','rsi_14','sma_20','ema_20','fundingRate']
    cols = [c for c in cols if c in df.columns]
    return df[cols].values, df['target_rv'].values

# %%
def run_wfv(df, horizon, outdir, enable_lstm=False):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    X, y = build_matrix(df)
    tscv = TimeSeriesSplit(n_splits=5)
    preds_all, rows = [], []

    for i, (tr, te) in enumerate(tscv.split(X)):
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]

        # 1) XGBoost
        xgb = XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method='hist'
        )
        xgb.fit(Xtr, ytr)
        p_xgb = xgb.predict(Xte)

        # 2) LSTM (optional; only import torch if explicitly enabled)
        p_lstm = np.full(len(yte), np.nan)
        if enable_lstm:
            try:
                import torch, torch.nn as nn
                class LSTMReg(nn.Module):
                    def __init__(self, nfeat):
                        super().__init__()
                        self.lstm = nn.LSTM(nfeat, 32, batch_first=True)
                        self.fc = nn.Linear(32, 1)
                    def forward(self, x):
                        o,_ = self.lstm(x); return self.fc(o[:,-1,:])
                L=12
                def mkseq(XA, yA):
                    xs, ys = [], []
                    for j in range(L, len(XA)):
                        xs.append(XA[j-L:j]); ys.append(yA[j])
                    return np.array(xs), np.array(ys)
                Xtr_s, ytr_s = mkseq(Xtr, ytr); Xte_s, yte_s = mkseq(Xte, yte)
                if len(Xte_s):
                    model = LSTMReg(X.shape[1])
                    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
                    lossf = nn.MSELoss()
                    TX = torch.tensor(Xtr_s, dtype=torch.float32)
                    Ty = torch.tensor(ytr_s, dtype=torch.float32).unsqueeze(-1)
                    for _ in range(15):
                        model.train(); opt.zero_grad()
                        out = model(TX); loss = lossf(out, Ty)
                        loss.backward(); opt.step()
                    model.eval(); PX = torch.tensor(Xte_s, dtype=torch.float32)
                    P = model(PX).detach().cpu().numpy().ravel()
                    p_lstm = np.concatenate([np.full(len(yte)-len(P), np.nan), P])
            except BaseException:
                p_lstm = np.full(len(yte), np.nan)

        # 3) GARCH (broadcast one-step vol across fold)
        garch_pred = np.array([garch_forecast(df['close'].iloc[tr], horizon)] * len(yte), dtype=float)

        # metrics
        for name, pred in [('XGB', p_xgb), ('LSTM', p_lstm), ('GARCH', garch_pred)]:
            rmse, m_mase, n = safe_metrics(yte, pred)
            rows.append({'fold': i+1, 'model': name, 'rmse': rmse, 'mase': m_mase, 'n': n})

        # keep timestamps for return merge
        ts = df.index[te]
        preds_all.append(pd.DataFrame({'idx': te, 'ts': ts, 'y_true': yte,
                                       'p_xgb': p_xgb, 'p_lstm': p_lstm, 'p_garch': garch_pred}))

    resdf = pd.DataFrame(rows)
    pred_df = pd.concat(preds_all, ignore_index=True)
    resdf.to_csv(f'{outdir}/metrics.csv', index=False)
    pred_df.to_parquet(f'{outdir}/predictions.parquet', index=False)
    print("Average by model:"); print(resdf.groupby('model')[['rmse','mase']].mean())

# %%
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--symbol', required=True)
    ap.add_argument('--horizon', type=int, default=12)
    ap.add_argument('--output', required=True)
    ap.add_argument('--enable-lstm', action='store_true')
    args = ap.parse_args()
    df = read_parquet(args.data)
    run_wfv(df, args.horizon, args.output, enable_lstm=args.enable_lstm)

# %%
if __name__ == '__main__':
    main()
