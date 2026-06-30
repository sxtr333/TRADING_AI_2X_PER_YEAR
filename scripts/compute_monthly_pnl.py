#!/usr/bin/env python3
import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--equity-csv', required=True)
    ap.add_argument('--out-csv', required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.equity_csv)
    if df.empty:
        raise SystemExit('equity csv is empty')
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values('timestamp')
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month

    rows = []
    for (y, m), g in df.groupby(['year', 'month']):
        g = g.sort_values('timestamp')
        start = float(g.iloc[0]['equity'])
        end = float(g.iloc[-1]['equity'])
        pnl = end - start
        pct = (pnl / start * 100.0) if start != 0 else 0.0
        rows.append({'year': y, 'month': m, 'equity_start': start, 'equity_end': end, 'pnl': pnl, 'pnl_pct': pct})

    out = pd.DataFrame(rows).sort_values(['year', 'month'])
    out.to_csv(args.out_csv, index=False)

if __name__ == '__main__':
    main()
