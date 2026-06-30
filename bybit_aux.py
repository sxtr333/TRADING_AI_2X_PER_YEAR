"""
Bybit auxiliary data downloader: open interest, funding history, account long/short ratio.
- Outputs a Parquet with timestamp and columns expected by build_features.py:
  open_interest, funding_rate, liq_long, liq_short, buy_sell_ratio.
- Uses Bybit public v5 endpoints (no API key needed).
- Liquidation history is not available via Bybit public REST; we fill liq_long/liq_short with zeros.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

BASE_URL = "https://api.bybit.com"


def parse_time(ts: str) -> int:
    """ISO string -> milliseconds UTC."""
    dt_obj = dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if dt_obj.tzinfo is None:
        dt_obj = dt_obj.replace(tzinfo=dt.timezone.utc)
    return int(dt_obj.timestamp() * 1000)


def paginate(url: str, params: Dict[str, str], cursor_field: str = "nextPageCursor", ts_field: Optional[str] = None, start_ms: Optional[int] = None) -> List[Dict]:
    """
    Generic paginator for Bybit v5 endpoints that use nextPageCursor.
    Returns the raw list items across all pages.
    """
    items: List[Dict] = []
    cursor: Optional[str] = None
    while True:
        p = dict(params)
        if cursor:
            p["cursor"] = cursor
        r = requests.get(url, params=p, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("retCode") != 0:
            raise RuntimeError(f"Bybit error {data.get('retCode')}: {data.get('retMsg')}")
        page_items = data["result"].get("list") or []
        items.extend(page_items)
        if ts_field and start_ms is not None and page_items:
            try:
                min_ts = min(int(it[ts_field]) for it in page_items)
                if min_ts < start_ms:
                    break
            except Exception:
                pass
        cursor = data["result"].get(cursor_field)
        if not cursor:
            break
    return items


def fetch_open_interest(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """
    Fetch open interest time series via /v5/market/open-interest (public).
    The endpoint always paginates from newest to oldest using nextPageCursor.
    """
    params = {
        "category": "linear",
        "symbol": symbol,
        "intervalTime": interval,
        "limit": 200,
    }
    items = paginate(f"{BASE_URL}/v5/market/open-interest", params, ts_field="timestamp", start_ms=start_ms)
    if not items:
        return pd.DataFrame(columns=["timestamp", "open_interest"])
    rows = [
        {
            "timestamp": pd.to_datetime(int(it["timestamp"]), unit="ms", utc=True),
            "open_interest": float(it["openInterest"]),
        }
        for it in items
    ]
    df = pd.DataFrame(rows).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df = df[(df["timestamp"].astype("int64") // 10**6 >= start_ms) & (df["timestamp"].astype("int64") // 10**6 <= end_ms)]
    return df


def fetch_funding(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """
    Funding history via /v5/market/funding/history (public).
    """
    params = {
        "category": "linear",
        "symbol": symbol,
        "limit": 200,
    }
    items = paginate(f"{BASE_URL}/v5/market/funding/history", params, ts_field="fundingRateTimestamp", start_ms=start_ms)
    if not items:
        return pd.DataFrame(columns=["timestamp", "funding_rate"])
    rows = []
    for it in items:
        ts_val = int(it["fundingRateTimestamp"])
        if ts_val < start_ms or ts_val > end_ms:
            continue
        rows.append(
            {
                "timestamp": pd.to_datetime(ts_val, unit="ms", utc=True),
                "funding_rate": float(it["fundingRate"]),
            }
        )
    df = pd.DataFrame(rows).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    return df


def fetch_account_ratio(symbol: str, period: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """
    Long/short account ratio via /v5/market/account-ratio.
    period options include 5min/15min/30min/1h/4h/1d.
    """
    params = {
        "category": "linear",
        "symbol": symbol,
        "period": period,
        "limit": 200,
    }
    items = paginate(f"{BASE_URL}/v5/market/account-ratio", params, ts_field="timestamp", start_ms=start_ms)
    if not items:
        return pd.DataFrame(columns=["timestamp", "buy_sell_ratio"])
    rows = []
    for it in items:
        ts = pd.to_datetime(int(it["timestamp"]), unit="ms", utc=True)
        buy = float(it["buyRatio"])
        sell = float(it["sellRatio"])
        denom = sell if sell != 0 else 1e-9
        rows.append({"timestamp": ts, "buy_sell_ratio": buy / denom})
    df = pd.DataFrame(rows).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df = df[(df["timestamp"].astype("int64") // 10**6 >= start_ms) & (df["timestamp"].astype("int64") // 10**6 <= end_ms)]
    return df


def merge_aux(oi: pd.DataFrame, funding: pd.DataFrame, ratio: pd.DataFrame) -> pd.DataFrame:
    df = None
    for part in [oi, funding, ratio]:
        if part is None or part.empty:
            continue
        df = part if df is None else df.merge(part, on="timestamp", how="outer")
    if df is None:
        return pd.DataFrame(columns=["timestamp", "open_interest", "funding_rate", "liq_long", "liq_short", "buy_sell_ratio"])
    df = df.sort_values("timestamp").ffill()
    # add missing columns expected by build_features
    for col, default in [
        ("liq_long", 0.0),
        ("liq_short", 0.0),
        ("buy_sell_ratio", 1.0),
    ]:
        if col not in df.columns:
            df[col] = default
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Bybit OI/funding/liquidation as aux features.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Symbol, e.g., BTCUSDT")
    parser.add_argument("--start", required=True, help="Start time ISO")
    parser.add_argument("--end", required=True, help="End time ISO")
    parser.add_argument("--oi-interval", default="5min", help="Open interest interval (5min/15min/30min/1h/4h/1d)")
    parser.add_argument("--ratio-period", default="1h", help="Account ratio period (matches dataset TF)")
    parser.add_argument("--out", default="data/aux.parquet", help="Output Parquet")
    args = parser.parse_args()

    start_ms = parse_time(args.start)
    end_ms = parse_time(args.end)

    oi = fetch_open_interest(args.symbol, args.oi_interval, start_ms, end_ms)
    funding = fetch_funding(args.symbol, start_ms, end_ms)
    ratio = fetch_account_ratio(args.symbol, args.ratio_period, start_ms, end_ms)

    aux = merge_aux(oi, funding, ratio)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    aux.to_parquet(out_path, index=False)
    print(f"Saved aux data to {out_path} (rows={len(aux)})")


if __name__ == "__main__":
    main()
