#!/usr/bin/env python3
"""
Merge auxiliary datasets (Bybit + Binance) into a unified aux parquet.
Preference order: Bybit for open_interest / funding / ratios; Binance for liq/cvd/vol if missing.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def load_df(path: Path) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["timestamp"])
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, parse_dates=["timestamp"])
    else:
        df = pd.read_parquet(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"Missing timestamp in {path}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp")


def merge_aux(bybit: pd.DataFrame, binance: pd.DataFrame) -> pd.DataFrame:
    # Outer merge on timestamp then fill missing from binance
    if bybit.empty and binance.empty:
        return pd.DataFrame(columns=["timestamp"])
    if bybit.empty:
        return binance
    if binance.empty:
        return bybit

    merged = pd.merge_asof(
        bybit.sort_values("timestamp"),
        binance.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta("2H"),
        suffixes=("_bybit", "_bin"),
    )

    def pick(col: str, default=0.0):
        a = merged.get(f"{col}_bybit")
        # if no suffix was applied (column existed only in binance), keep original name
        b = merged.get(f"{col}_bin")
        if b is None and col in merged.columns:
            b = merged[col]
        if a is None and b is None:
            return pd.Series(default, index=merged.index, dtype="float64")
        if a is None:
            return b.fillna(default)
        if b is None:
            return a.fillna(default)
        out = a.copy()
        out = out.where(out.notna(), b)
        return out.fillna(default)

    out = pd.DataFrame({"timestamp": merged["timestamp"]})
    # Prefer Bybit for OI/funding/ratios
    for col in ["open_interest", "open_interest_value", "funding_rate", "buy_sell_ratio", "taker_long_short_vol_ratio", "toptrader_long_short_ratio", "long_short_ratio"]:
        out[col] = pick(col, default=0.0)
    # Prefer Binance for liq/cvd/volumes if missing
    for col in ["liq_long", "liq_short", "tick_buy_volume", "tick_sell_volume", "delta", "cvd", "basis", "ob_imb_01", "ob_imb_025", "ob_imb_05", "ob_imb_1"]:
        out[col] = pick(col, default=0.0)

    out = out.sort_values("timestamp").ffill().fillna(0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bybit", help="Bybit aux parquet/csv")
    ap.add_argument("--binance", help="Binance aux parquet/csv")
    ap.add_argument("--out", default="data/BTCUSDT_1h_aux_merged.parquet")
    args = ap.parse_args()

    bybit = load_df(Path(args.bybit)) if args.bybit else pd.DataFrame(columns=["timestamp"])
    binance = load_df(Path(args.binance)) if args.binance else pd.DataFrame(columns=["timestamp"])

    merged = merge_aux(bybit, binance)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.out, index=False)
    print(f"Saved merged aux to {args.out} rows={len(merged)}")


if __name__ == "__main__":
    main()
