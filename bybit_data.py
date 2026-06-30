"""
Bybit public data downloader for BTCUSDT perpetual.
- Fetches OHLCV klines for multiple timeframes and stores them to Parquet.
- Extend later with OI/funding/book features; start with stable kline endpoint.
"""

from __future__ import annotations

import argparse
import datetime as dt
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

BASE_URL = "https://api.bybit.com"


TIMEFRAME_MAP: Dict[str, str] = {
    "1m": "1",
    "3m": "3",
    "5m": "5",
    "10m": "10",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "2h": "120",
    "4h": "240",
    "6h": "360",
    "12h": "720",
    "1d": "D",
    "1w": "W",
}


def parse_time(ts: str) -> int:
    """
    Parse ISO-like datetime string to milliseconds since epoch (UTC).
    Accepts "YYYY-MM-DD" or full ISO with timezone; assumes UTC if none provided.
    """
    dt_obj = dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if dt_obj.tzinfo is None:
        dt_obj = dt_obj.replace(tzinfo=dt.timezone.utc)
    return int(dt_obj.timestamp() * 1000)


def fetch_kline_batch(
    symbol: str,
    bybit_interval: str,
    start_ms: int,
    end_ms: int,
    category: str = "linear",
    limit: int = 1000,
    timeout: int = 10,
    retries: int = 5,
    backoff: float = 1.0,
) -> List[Dict]:
    """
    Single request to Bybit v5 market kline with retry on transient errors.
    """
    params = {
        "category": category,
        "symbol": symbol,
        "interval": bybit_interval,
        "start": start_ms,
        "end": end_ms,
        "limit": limit,
    }
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.get(f"{BASE_URL}/v5/market/kline", params=params, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            if data.get("retCode") != 0:
                raise RuntimeError(f"Bybit error {data.get('retCode')}: {data.get('retMsg')}")
            return data["result"]["list"]
        except Exception as e:
            last_err = e
            sleep = backoff * (2 ** attempt)
            print(f"Request failed (attempt {attempt+1}/{retries}): {e}. Sleeping {sleep:.1f}s")
            time.sleep(sleep)
    raise RuntimeError(f"Failed after retries: {last_err}")


def kline_list_to_df(rows: List[List[str]], interval_ms: int) -> pd.DataFrame:
    """
    Convert raw list to DataFrame; fields: start, open, high, low, close, volume, turnover.
    """
    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
    df = pd.DataFrame(
        rows,
        columns=["start", "open", "high", "low", "close", "volume", "turnover"],
    )
    df["timestamp"] = pd.to_datetime(df["start"].astype(int), unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.drop(columns=["start"]).sort_values("timestamp")
    # Add placeholder VWAP from turnover/volume if volume > 0.
    df["vwap"] = df["turnover"] / df["volume"].replace(0, pd.NA)
    df["interval_ms"] = interval_ms
    return df


def fetch_kline_range(
    symbol: str,
    timeframe: str,
    start_ms: int,
    end_ms: int,
    category: str = "linear",
    sleep_secs: float = 0.2,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Paginate through the kline endpoint and return a DataFrame covering [start_ms, end_ms].
    """
    if timeframe not in TIMEFRAME_MAP:
        raise ValueError(f"Unsupported timeframe {timeframe}")
    bybit_interval = TIMEFRAME_MAP[timeframe]
    interval_ms = _interval_to_ms(bybit_interval)

    cursor = start_ms
    all_rows: List[List[str]] = []

    while cursor <= end_ms:
        batch = fetch_kline_batch(symbol, bybit_interval, cursor, end_ms, category=category)
        if not batch:
            break
        all_rows.extend(batch)
        last_ts = int(batch[-1][0])
        if verbose:
            print(f"{timeframe}: fetched {len(batch)} rows up to {dt.datetime.utcfromtimestamp(last_ts/1000)}")
        cursor = last_ts + interval_ms
        time.sleep(sleep_secs)

    df = kline_list_to_df(all_rows, interval_ms)
    # Trim to range in case of overshoot
    return df[(df["timestamp"].astype("int64") // 10**6 >= start_ms) & (df["timestamp"].astype("int64") // 10**6 <= end_ms)]


def update_parquet(out_path: Path, new_df: pd.DataFrame) -> None:
    """
    Append new data to Parquet, de-duplicate by timestamp.
    """
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    else:
        combined = new_df.sort_values("timestamp")
    save_parquet(combined, out_path)


def _interval_to_ms(bybit_interval: str) -> int:
    if bybit_interval == "D":
        return 24 * 60 * 60 * 1000
    if bybit_interval == "W":
        return 7 * 24 * 60 * 60 * 1000
    if bybit_interval == "M":
        return 30 * 24 * 60 * 60 * 1000
    return int(bybit_interval) * 60 * 1000


def save_parquet(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")


def download_timeframes(
    symbol: str,
    timeframes: List[str],
    start_ms: int,
    end_ms: int,
    out_dir: Path,
    category: str = "linear",
    chunk_days: int = 30,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_ms = chunk_days * 24 * 60 * 60 * 1000

    for tf in timeframes:
        bybit_interval = TIMEFRAME_MAP[tf]
        interval_ms = _interval_to_ms(bybit_interval)
        out_path = out_dir / f"{symbol}_{tf}.parquet"

        # Resume from existing file if present, but also backfill earlier gaps.
        effective_start = start_ms
        if out_path.exists():
            try:
                existing = pd.read_parquet(out_path)
                if not existing.empty:
                    min_ts = existing["timestamp"].min().to_pydatetime().replace(tzinfo=dt.timezone.utc)
                    max_ts = existing["timestamp"].max().to_pydatetime().replace(tzinfo=dt.timezone.utc)
                    # If we already have all data through end_ms, skip.
                    if int(max_ts.timestamp() * 1000) >= end_ms - interval_ms:
                        print(f"{tf}: existing file already covers requested range, skipping.")
                        continue
                    # If existing starts after requested start, backfill from requested start.
                    if int(min_ts.timestamp() * 1000) > start_ms:
                        effective_start = start_ms
                        print(f"{tf}: existing data starts later; backfilling from {dt.datetime.fromtimestamp(start_ms/1000, tz=dt.timezone.utc)}")
                    else:
                        effective_start = max(effective_start, int(max_ts.timestamp() * 1000) + interval_ms)
                        print(f"{tf}: resuming from {max_ts} (ms {effective_start})")
            except Exception as e:
                print(f"{tf}: could not read existing file for resume ({e}), downloading full range.")

        cursor_start = effective_start
        while cursor_start < end_ms:
            cursor_end = min(end_ms, cursor_start + chunk_ms)
            print(f"{tf}: chunk {dt.datetime.fromtimestamp(cursor_start/1000, tz=dt.timezone.utc)} -> {dt.datetime.fromtimestamp(cursor_end/1000, tz=dt.timezone.utc)}")
            df_chunk = fetch_kline_range(symbol, tf, cursor_start, cursor_end, category=category)
            if df_chunk.empty:
                print(f"{tf}: no data for chunk, stopping.")
                break
            update_parquet(out_path, df_chunk)
            last_ts = df_chunk["timestamp"].max().to_pydatetime().replace(tzinfo=dt.timezone.utc)
            print(f"{tf}: saved chunk up to {last_ts}, total rows now {len(pd.read_parquet(out_path))}")
            cursor_start = int(last_ts.timestamp() * 1000) + interval_ms


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Bybit kline data to Parquet.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Symbol, e.g., BTCUSDT")
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=["1m", "5m", "10m", "15m", "30m", "1h", "4h", "1d", "1w"],
        help="Timeframes to fetch",
    )
    parser.add_argument("--start", required=True, help="Start time (ISO, e.g., 2023-01-01)")
    parser.add_argument("--end", required=True, help="End time (ISO, e.g., 2024-01-01)")
    parser.add_argument("--out", default="data", help="Output directory")
    parser.add_argument("--category", default="linear", help="Bybit category (linear/inverse/spot)")
    parser.add_argument("--chunk-days", type=int, default=30, help="Download in chunks of N days to allow incremental saves")
    args = parser.parse_args()

    start_ms = parse_time(args.start)
    end_ms = parse_time(args.end)

    out_dir = Path(args.out)
    download_timeframes(
        args.symbol,
        args.timeframes,
        start_ms,
        end_ms,
        out_dir,
        category=args.category,
        chunk_days=args.chunk_days,
    )


if __name__ == "__main__":
    main()
