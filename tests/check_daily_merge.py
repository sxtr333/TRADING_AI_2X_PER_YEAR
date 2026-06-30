#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quick sanity check for daily->15m merge anti-leakage.
Prints a window around UTC midnight and asserts daily features
change only on day boundaries.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from build_features import merge_daily_features


def _load_df(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "timestamp" not in df.columns:
        raise ValueError("features parquet must include 'timestamp'")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)

def _daily_cols_from_path(path: Path) -> list[str]:
    if path.suffix.lower() == ".csv":
        d = pd.read_csv(path)
    else:
        d = pd.read_parquet(path)
    date_col = None
    for c in ("date", "Date", "timestamp", "Timestamp", "observation_date"):
        if c in d.columns:
            date_col = c
            break
    if date_col is None and "Unnamed: 0" in d.columns:
        date_col = "Unnamed: 0"
    if date_col and date_col in d.columns:
        d = d.drop(columns=[date_col])
    cols = d.select_dtypes(include=["number", "bool"]).columns.tolist()
    return cols

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="15m parquet (OHLCV + features)")
    ap.add_argument("--macro-daily-path", default=None)
    ap.add_argument("--fed-daily-path", default=None)
    ap.add_argument("--inst-daily-path", default=None)
    ap.add_argument("--days", type=int, default=3, help="Number of days to print around midnights")
    ap.add_argument("--date", default=None, help="Optional YYYY-MM-DD (UTC) to anchor checks")
    args = ap.parse_args()

    df = _load_df(Path(args.features))
    daily_paths = [p for p in [args.macro_daily_path, args.fed_daily_path, args.inst_daily_path] if p]
    if daily_paths:
        df = merge_daily_features(df, daily_paths, join_on="timestamp", shift_days=1, ffill_daily=True)

    daily_cols: list[str] = []
    for p in daily_paths:
        daily_cols.extend(_daily_cols_from_path(Path(p)))
    daily_cols = sorted(set(daily_cols))
    if "daily_missing_any" in df.columns:
        daily_cols.append("daily_missing_any")
    if not daily_cols:
        print("[warn] no daily columns detected; skipping checks")
        return

    if args.date:
        anchor = pd.Timestamp(args.date, tz="UTC").normalize()
    else:
        anchor = df["timestamp"].iloc[-1].normalize()

    start = anchor - pd.Timedelta(days=args.days)
    end = anchor + pd.Timedelta(days=1)

    window = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()
    window["date_key"] = window["timestamp"].dt.floor("D")

    # print rows around midnight
    midnights = pd.date_range(start, end, freq="D", tz="UTC")
    for dt in midnights:
        seg = window[(window["timestamp"] >= dt - pd.Timedelta(hours=1)) & (window["timestamp"] <= dt + pd.Timedelta(hours=1))]
        if seg.empty:
            continue
        print(f"\n=== Around {dt} UTC ===")
        cols = ["timestamp", "close"] + daily_cols[:8]
        print(seg[cols].head(12).to_string(index=False))

    # validation: constant within day
    for c in daily_cols:
        nunique_max = int(window.groupby("date_key")[c].nunique(dropna=False).max())
        assert nunique_max <= 1, f"{c} varies within a day (max nunique={nunique_max})"

    if "daily_missing_any" in df.columns:
        miss_rate = float(df["daily_missing_any"].mean())
        print(f"daily_missing_any share: {miss_rate:.4f}")
    print("\n[ok] daily merge checks passed")


if __name__ == "__main__":
    main()
