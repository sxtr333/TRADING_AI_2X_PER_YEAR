"""
Fetch BitMEX public liquidation data (XBTUSD) and aggregate to hourly liq_long/liq_short.

Data source: https://public.bitmex.com/data/liquidations/
Files are NDJSON per day (UTC). We will download a date range, parse, and aggregate.

Usage:
  python model6/scripts/fetch_bitmex_liquidations.py \
    --start 2023-01-01 --end 2023-01-10 \
    --raw-out model6/data/liquidations/bitmex_liq_raw.parquet \
    --agg-out model6/data/liquidations/bitmex_liq_1h.parquet

Notes:
- Symbol in files is XBTUSD (inverse contract). Notional is abs(size * price).
- side: "Sell" indicates long liquidation, "Buy" indicates short liquidation.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import io
import json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import requests


BASE_URL = "https://public.bitmex.com/data/liquidations/"


def daterange(start: dt.date, end: dt.date):
    cur = start
    while cur < end:
        yield cur
        cur += dt.timedelta(days=1)


def fetch_day(day: dt.date) -> List[Dict[str, Any]]:
    fname = f"{day.isoformat()}.json.gz"
    url = BASE_URL + fname
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        print(f"[warn] {day} got {r.status_code}, skipping")
        return []
    with gzip.GzipFile(fileobj=io.BytesIO(r.content)) as f:
        lines = f.read().decode("utf-8").splitlines()
    events = []
    for line in lines:
        try:
            ev = json.loads(line)
            events.append(ev)
        except json.JSONDecodeError:
            continue
    return events


def to_dataframe(events: List[Dict[str, Any]]) -> pd.DataFrame:
    if not events:
        return pd.DataFrame()
    df = pd.DataFrame(events)
    # Columns include: timestamp, side, price, leavesQty, symbol
    df = df[df.get("symbol") == "XBTUSD"]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    price = pd.to_numeric(df.get("price"), errors="coerce")
    size = pd.to_numeric(df.get("leavesQty"), errors="coerce").abs()
    df["notional"] = (price * size).fillna(0.0)
    df["side"] = df["side"].str.lower()
    return df[["timestamp", "side", "notional"]]


def agg_hourly(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df["liq_long"] = df.apply(lambda r: r["notional"] if r["side"] == "sell" else 0.0, axis=1)
    df["liq_short"] = df.apply(lambda r: r["notional"] if r["side"] == "buy" else 0.0, axis=1)
    hourly = (
        df[["timestamp", "liq_long", "liq_short"]]
        .set_index("timestamp")
        .resample("1h")
        .sum()
        .reset_index()
    )
    return hourly


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="Start date YYYY-MM-DD (UTC)")
    ap.add_argument("--end", required=True, help="End date YYYY-MM-DD (UTC, exclusive)")
    ap.add_argument("--raw-out", default="model6/data/liquidations/bitmex_liq_raw.parquet")
    ap.add_argument("--agg-out", default="model6/data/liquidations/bitmex_liq_1h.parquet")
    args = ap.parse_args()

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    all_events: List[Dict[str, Any]] = []
    for day in daterange(start, end):
        evs = fetch_day(day)
        if evs:
            all_events.extend(evs)
    df = to_dataframe(all_events)
    Path(args.raw_out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.raw_out, index=False)
    hourly = agg_hourly(df)
    hourly.to_parquet(args.agg_out, index=False)
    print(f"Saved raw to {args.raw_out} ({len(df)} rows) and hourly to {args.agg_out} ({len(hourly)} rows)")


if __name__ == "__main__":
    main()
