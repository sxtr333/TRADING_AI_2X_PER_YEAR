"""
Build hourly auxiliary features for BTCUSDT from Binance Vision daily archives (public, no API key):
- perp klines 1h (USDT-M)
- spot klines 1h (for basis)
- fundingRate (daily zipped)
- liquidation (daily zipped) -> liq_long/liq_short by hour
- tick_buy_volume, tick_sell_volume, delta, cvd from taker buy/sell quote volumes in klines

Open interest and orderbook imbalance are left as 0.0 (not available in vision archives).
"""

from __future__ import annotations

import argparse
import io
import time
import zipfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests

BASE = "https://data.binance.vision/data"


def daterange_days(start: str, end: str) -> List[str]:
    days = pd.date_range(start=start, end=end, freq="D", tz="UTC")
    return [d.strftime("%Y-%m-%d") for d in days]


def cache_path(cache: Path, url: str) -> Path:
    # Preserve relative URL path inside cache dir.
    rel = url.replace(BASE + "/", "")
    rel = rel.replace("/", "_")
    return cache / rel


def download(url: str, cache: Path, retries: int = 3, backoff: float = 1.5) -> bytes:
    cache.mkdir(parents=True, exist_ok=True)
    cpath = cache_path(cache, url)
    if cpath.exists() and cpath.stat().st_size > 0:
        return cpath.read_bytes()

    last_err = None
    for i in range(retries):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            cpath.write_bytes(r.content)
            return r.content
        except Exception as e:
            last_err = e
            time.sleep(backoff * (i + 1))
    raise RuntimeError(f"Failed to download {url}: {last_err}")


def read_csv_from_zip(content: bytes, names=None, usecols=None, header=0):
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        csv_name = [f for f in z.namelist() if f.lower().endswith(".csv")]
        if not csv_name:
            raise ValueError("No CSV in zip")
        with z.open(csv_name[0]) as f:
            return pd.read_csv(f, header=header, names=names, usecols=usecols)


def read_kline_zip(content: bytes):
    cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]
    df = read_csv_from_zip(content, names=cols, header=None)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df


def read_funding_zip(content: bytes):
    df = read_csv_from_zip(content, header=0)
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "funding_rate"])
    df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    return df[["timestamp", "fundingRate"]].rename(columns={"fundingRate": "funding_rate"})


def read_liq_zip(content: bytes):
    df = read_csv_from_zip(content, header=0)
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "liq_long", "liq_short"])
    df["timestamp"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df["notional"] = df["origQty"] * df["price"]
    df["liq_long"] = np.where(df["side"].str.lower() == "sell", df["notional"], 0.0)
    df["liq_short"] = np.where(df["side"].str.lower() == "buy", df["notional"], 0.0)
    return df[["timestamp", "liq_long", "liq_short"]]


def fetch_daily(symbol: str, day: str, market: str, kind: str, interval: str, cache: Path):
    # market: "futures/um" or "spot"
    if kind == "kline":
        url = f"{BASE}/{market}/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{day}.zip"
    elif kind == "funding":
        url = f"{BASE}/{market}/daily/fundingRate/{symbol}/{symbol}-fundingRate-{day}.zip"
    elif kind == "liq":
        url = f"{BASE}/{market}/daily/liquidation/{symbol}/{symbol}-liquidation-{day}.zip"
    else:
        raise ValueError("kind must be kline/funding/liq")
    return download(url, cache=cache)


def build_aux(start: str, end: str, cache: Path) -> pd.DataFrame:
    days = daterange_days(start, end)
    perp_list = []
    spot_list = []
    fund_list = []
    liq_list = []
    errors: List[Tuple[str, str]] = []

    for d in days:
        # perp klines 1h
        try:
            content = fetch_daily("BTCUSDT", d, "futures/um", "kline", "1h", cache)
            perp_list.append(read_kline_zip(content))
        except Exception as e:
            errors.append((d, f"perp kline: {e}"))
        # spot klines 1h
        try:
            content = fetch_daily("BTCUSDT", d, "spot", "kline", "1h", cache)
            spot_list.append(read_kline_zip(content))
        except Exception as e:
            errors.append((d, f"spot kline: {e}"))
        # funding
        try:
            content = fetch_daily("BTCUSDT", d, "futures/um", "funding", "fundingRate", cache)
            fund_list.append(read_funding_zip(content))
        except Exception as e:
            errors.append((d, f"funding: {e}"))
        # liquidation
        try:
            content = fetch_daily("BTCUSDT", d, "futures/um", "liq", "liquidation", cache)
            liq_list.append(read_liq_zip(content))
        except Exception as e:
            errors.append((d, f"liq: {e}"))

    perp = pd.concat(perp_list) if perp_list else pd.DataFrame(columns=["timestamp", "close", "quote_volume", "taker_buy_quote"])
    spot = pd.concat(spot_list) if spot_list else pd.DataFrame(columns=["timestamp", "close"])
    funding = pd.concat(fund_list) if fund_list else pd.DataFrame(columns=["timestamp", "funding_rate"])
    liq = pd.concat(liq_list) if liq_list else pd.DataFrame(columns=["timestamp", "liq_long", "liq_short"])

    # Resample hourly
    perp_hour = perp[["timestamp", "close", "quote_volume", "taker_buy_quote"]].copy()
    perp_hour["timestamp"] = pd.to_datetime(perp_hour["timestamp"], utc=True)
    perp_hour["tick_buy_volume"] = perp_hour["taker_buy_quote"]
    perp_hour["tick_sell_volume"] = perp_hour["quote_volume"] - perp_hour["taker_buy_quote"]
    perp_hour = (
        perp_hour.set_index("timestamp")
        .resample("1h")
        .agg({"close": "last", "tick_buy_volume": "sum", "tick_sell_volume": "sum"})
        .reset_index()
    )
    perp_hour["delta"] = perp_hour["tick_buy_volume"] - perp_hour["tick_sell_volume"]
    perp_hour["cvd"] = perp_hour["delta"].cumsum()

    if not spot.empty:
        spot_hour = spot[["timestamp", "close"]].copy()
        spot_hour["timestamp"] = pd.to_datetime(spot_hour["timestamp"], utc=True)
        spot_hour = spot_hour.rename(columns={"close": "spot_close"}).set_index("timestamp").resample("1h").last()
        merged = perp_hour.set_index("timestamp").join(spot_hour, how="left").ffill().reset_index()
        merged["basis"] = (merged["close"] - merged["spot_close"]) / merged["spot_close"]
    else:
        merged = perp_hour.copy()
        merged["basis"] = 0.0

    if not funding.empty:
        funding = funding.copy()
        funding["timestamp"] = pd.to_datetime(funding["timestamp"], utc=True)
        funding_hour = funding.set_index("timestamp").resample("1h").ffill().reset_index()
    else:
        funding_hour = merged[["timestamp"]].copy()
        funding_hour["funding_rate"] = 0.0

    if not liq.empty:
        liq = liq.copy()
        liq["timestamp"] = pd.to_datetime(liq["timestamp"], utc=True)
        liq_hour = liq.set_index("timestamp").resample("1h").sum().reset_index()
    else:
        liq_hour = merged[["timestamp"]].copy()
        liq_hour["liq_long"] = 0.0
        liq_hour["liq_short"] = 0.0

    aux = merged.merge(funding_hour, on="timestamp", how="left").merge(liq_hour, on="timestamp", how="left")
    aux = aux.sort_values("timestamp").ffill().fillna(0.0)
    aux["open_interest"] = 0.0
    for col in ["ob_imb_01", "ob_imb_025", "ob_imb_05", "ob_imb_1"]:
        aux[col] = 0.0

    keep = [
        "timestamp",
        "open_interest",
        "funding_rate",
        "liq_long",
        "liq_short",
        "tick_buy_volume",
        "tick_sell_volume",
        "delta",
        "cvd",
        "basis",
        "ob_imb_01",
        "ob_imb_025",
        "ob_imb_05",
        "ob_imb_1",
    ]
    if errors:
        print(f"[warn] {len(errors)} download errors. Sample: {errors[:5]}")
    return aux[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="ISO start UTC")
    ap.add_argument("--end", required=True, help="ISO end UTC")
    ap.add_argument("--out", default="data/BTCUSDT_1h_aux_binance.parquet")
    args = ap.parse_args()
    aux = build_aux(args.start, args.end, Path("data/binance_cache"))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    aux.to_parquet(args.out, index=False)
    print(f"Saved aux {len(aux)} rows to {args.out}")


if __name__ == "__main__":
    main()
