"""
Downloader/aggregator for Bybit public trade archives (https://public.bybit.com/trading/BTCUSDT/).
- Grabs daily .csv.gz trade files for BTCUSDT and aggregates to OHLCV klines for given timeframes.
- Avoids REST rate limits; much faster for bulk history.
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import requests

BASE_ARCHIVE = "https://public.bybit.com/trading"

# pandas resample frequency aliases
FREQ_MAP: Dict[str, str] = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "10m": "10min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1H",
    "2h": "2H",
    "4h": "4H",
    "6h": "6H",
    "12h": "12H",
    "1d": "1D",
    "1w": "1W",
}


def daterange(start_date: dt.date, end_date: dt.date) -> Iterable[dt.date]:
    """Inclusive date range."""
    cur = start_date
    while cur <= end_date:
        yield cur
        cur += dt.timedelta(days=1)


def download_day(symbol: str, day: dt.date) -> pd.DataFrame:
    """
    Download one day's trade CSV.GZ from public bucket and return DataFrame.
    Columns: timestamp,symbol,side,size,price,tickDirection,trdMatchID,grossValue,homeNotional,foreignNotional
    """
    url = f"{BASE_ARCHIVE}/{symbol}/{symbol}{day.strftime('%Y-%m-%d')}.csv.gz"
    resp = requests.get(url, stream=True, timeout=30)
    if resp.status_code == 404:
        print(f"{day}: not found (404), skipping.")
        return pd.DataFrame()
    resp.raise_for_status()
    buf = io.BytesIO(resp.content)
    df = pd.read_csv(
        buf,
        compression="gzip",
        dtype={
            "timestamp": float,
            "symbol": str,
            "side": str,
            "size": float,
            "price": float,
            "tickDirection": str,
            "trdMatchID": str,
            "grossValue": float,
            "homeNotional": float,
            "foreignNotional": float,
        },
    )
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return df


def agg_ohlcv(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Aggregate trades to OHLCV for a given pandas frequency.
    Volume uses 'size' (contracts/homeNotional), turnover uses 'foreignNotional'.
    """
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "turnover", "vwap"])
    res = (
        df.set_index("timestamp")
        .resample(freq)
        .agg(
            open=("price", "first"),
            high=("price", "max"),
            low=("price", "min"),
            close=("price", "last"),
            volume=("size", "sum"),
            turnover=("foreignNotional", "sum"),
        )
        .dropna(subset=["open", "high", "low", "close"])
    )
    res["vwap"] = res["turnover"] / res["volume"].replace(0, pd.NA)
    res = res.reset_index()
    return res


def update_parquet(out_path: Path, new_df: pd.DataFrame) -> None:
    """
    Append new data to Parquet, de-duplicate by timestamp.
    """
    if new_df.empty:
        return
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)
    print(f"Saved {len(combined)} rows to {out_path}")


def process_day(symbol: str, day: dt.date, timeframes: List[str], out_dir: Path) -> None:
    trades = download_day(symbol, day)
    if trades.empty:
        return
    for tf in timeframes:
        if tf not in FREQ_MAP:
            print(f"Unknown timeframe {tf}, skipping.")
            continue
        freq = FREQ_MAP[tf]
        ohlcv = agg_ohlcv(trades, freq)
        out_path = out_dir / f"{symbol}_{tf}.parquet"
        update_parquet(out_path, ohlcv)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Bybit public trades and aggregate to klines.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Symbol, e.g., BTCUSDT")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD, inclusive)")
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=["1m", "5m", "10m", "15m", "30m", "1h", "4h", "1d", "1w"],
        help="Timeframes to aggregate",
    )
    parser.add_argument("--out", default="data", help="Output directory for Parquet files")
    args = parser.parse_args()

    start_date = dt.date.fromisoformat(args.start)
    end_date = dt.date.fromisoformat(args.end)
    out_dir = Path(args.out)

    for day in daterange(start_date, end_date):
        print(f"Processing {day}...")
        try:
            process_day(args.symbol, day, args.timeframes, out_dir)
        except Exception as e:
            print(f"Error on {day}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
