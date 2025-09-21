import argparse, json
import numpy as np, pandas as pd
from pathlib import Path

def backtest(pred_df, retcol='ret_1', fee_bps=1.0):
    df = pred_df.copy()
    # Position proportional to 1 / predicted volatility (target vol scaling)
    eps = 1e-8
    tgt_vol = df['y_pred'].rolling(50).median().fillna(method='bfill')
    w = (tgt_vol.median() / (df['y_pred'] + eps)).clip(0, 5.0)  # cap leverage
    gross_ret = w * df[retcol]
    # fees from turnover
    turn = np.abs(w.diff()).fillna(0.0)
    fees = turn * (fee_bps/10000.0)
    strat_ret = gross_ret - fees
    out = {
        'sharpe': float(np.sqrt(252*24*12) * strat_ret.mean() / (strat_ret.std()+1e-12)),
        'sortino': float(np.sqrt(252*24*12) * strat_ret.mean() / (strat_ret[strat_ret<0].std()+1e-12)),
        'mdd': float(((1+strat_ret).cumprod().cummax() / (1+strat_ret).cumprod() - 1).min()),
        'avg_turnover': float(turn.mean()),
    }
    return out, pd.DataFrame({'w': w, 'gross_ret': gross_ret, 'fees': fees, 'strat_ret': strat_ret})

def main():
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True, help='predictions.parquet from modeling')
    ap.add_argument('--retcol', default='ret_5')
    ap.add_argument('--output', required=True)
    ap.add_argument('--model', default='p_xgb')
    ap.add_argument('--fee_bps', type=float, default=1.0)
    args = ap.parse_args()

    pred = pd.read_parquet(args.pred).sort_values('idx')
    # align to returns (this assumes your processed data exists with same index)
    # For simplicity, we treat y_true as realized vol and build synthetic returns:
    # In your research, **replace with actual asset returns aligned to prediction timestamps**.
    np.random.seed(0)
    synthetic_ret = pd.Series(np.random.normal(0, 0.0005, size=len(pred)))
    pred['ret_5'] = synthetic_ret.values
    pred['y_pred'] = pred[args.model]

    stats, detail = backtest(pred, retcol=args.retcol, fee_bps=args.fee_bps)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(stats, f, indent=2)
    detail.to_parquet(args.output.replace('.json', '_detail.parquet'))
    print(json.dumps(stats, indent=2))

if __name__ == '__main__':
    main()
