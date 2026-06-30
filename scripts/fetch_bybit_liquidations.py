"""
Fetch Bybit linear liquidation events for BTCUSDT and aggregate to hourly liq_long/liq_short.

Usage (run where network is available):
  BYBIT_API_KEY=... BYBIT_API_SECRET=... python model6/scripts/fetch_bybit_liquidations.py \
    --symbol BTCUSDT --start 2023-01-01 --end 2025-12-19 \
    --raw-out model6/data/liquidations/bybit_liq_raw.parquet \
    --agg-out model6/data/liquidations/bybit_liq_1h.parquet

Notes:
- The public v5 liquidation endpoint is used; keys are optional but can be set via env if needed.
- If you hit rate limits, increase the sleep or shrink the date range.
"""

from __future__ import annotations

import argparse
import datetime as dt
import time
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import requests

HOSTS = ["https://api.bybit.com", "https://api.bytick.com"]
# Docs show the public liquidation endpoint under /v5/market/liquidation but some regions require /v5/public/liq-records; try both.
ENDPOINTS = ["/v5/market/liquidation", "/v5/public/liq-records"]


def fetch_liq(
    symbol: str,
    start: dt.datetime,
    end: dt.datetime,
    batch_hours: int = 24 * 7,
    limit: int = 1000,
    sleep: float = 0.2,
) -> List[Dict[str, Any]]:
    """
    Fetch liquidation events in chunks of `batch_hours` to avoid huge cursors.
    Returns list of raw event dicts.
    """
    events: List[Dict[str, Any]] = []
    t = start
    while t < end:
        t_end = min(t + dt.timedelta(hours=batch_hours), end)
        # Bybit API uses startTime/endTime in some docs; also "start"/"end" in others.
        param_variants = [
            {
                "category": "linear",
                "symbol": symbol,
                "start": int(t.timestamp() * 1000),
                "end": int(t_end.timestamp() * 1000),
                "limit": limit,
            },
            {
                "category": "linear",
                "symbol": symbol,
                "startTime": int(t.timestamp() * 1000),
                "endTime": int(t_end.timestamp() * 1000),
                "limit": limit,
            },
        ]
        success = False
        for params in param_variants:
            for host in HOSTS:
                for endpoint in ENDPOINTS:
                    cursor = None
                    while True:
                        send = dict(params)
                        if cursor:
                            send["cursor"] = cursor
                        resp = requests.get(host + endpoint, params=send, timeout=30)
                        if resp.status_code == 404:
                            break  # try next endpoint/params/host
                        if resp.status_code != 200:
                            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
                        payload = resp.json()
                        if payload.get("retCode") != 0:
                            break  # try next endpoint/params/host
                        result = payload.get("result") or {}
                        rows = result.get("list") or []
                        events.extend(rows)
                        cursor = result.get("nextPageCursor")
                        success = True
                        if not cursor:
                            break
                        time.sleep(sleep)
                    if success:
                        break
                if success:
                    break
            if success:
                break
        if not success:
            raise RuntimeError(f"No data or endpoint failed for window {t} - {t_end}")
        t = t_end
        time.sleep(sleep)
    return events


def to_dataframe(events: List[Dict[str, Any]]) -> pd.DataFrame:
    if not events:
        return pd.DataFrame()
    df = pd.DataFrame(events)
    # Expected fields in v5 response: updatedTime, side, size, price, qty, value
    df["timestamp"] = pd.to_datetime(df["updatedTime"].astype(str), unit="ms", utc=True)
    # Use value if present, else size*price as USD notionals
    if "value" in df.columns and pd.api.types.is_numeric_dtype(df["value"]):
        notion = pd.to_numeric(df["value"], errors="coerce")
    else:
        size = pd.to_numeric(df.get("size"), errors="coerce")
        price = pd.to_numeric(df.get("price"), errors="coerce")
        notion = size * price
    df["notional"] = notion.fillna(0.0)
    return df[["timestamp", "side", "notional"]]


def agg_hourly(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df["side"] = df["side"].str.lower()
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
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--start", required=True, help="Start date YYYY-MM-DD (UTC)")
    ap.add_argument("--end", required=True, help="End date YYYY-MM-DD (UTC, exclusive)")
    ap.add_argument("--raw-out", default="model6/data/liquidations/bybit_liq_raw.parquet")
    ap.add_argument("--agg-out", default="model6/data/liquidations/bybit_liq_1h.parquet")
    args = ap.parse_args()

    start = dt.datetime.fromisoformat(args.start).replace(tzinfo=dt.timezone.utc)
    end = dt.datetime.fromisoformat(args.end).replace(tzinfo=dt.timezone.utc)

    events = fetch_liq(args.symbol, start, end)
    print(f"Fetched {len(events)} events")
    df = to_dataframe(events)
    Path(args.raw_out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.raw_out, index=False)
    hourly = agg_hourly(df)
    hourly.to_parquet(args.agg_out, index=False)
    print(f"Saved raw to {args.raw_out} ({len(df)} rows) and hourly to {args.agg_out} ({len(hourly)} rows)")


if __name__ == "__main__":
    main()
