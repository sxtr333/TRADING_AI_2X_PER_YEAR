#!/usr/bin/env python3
"""Build signal outcome dataset using coin move vs BTC move from entry.

For each signal:
- anchor at first 15m candle with open_time >= signal timestamp
- compute favorable/adverse/signed move for the signal coin
- compute the same for BTC on the same horizon
- compute alpha = coin_move - beta * btc_move
- derive a simple good/bad label from alpha + drawdown constraints
"""

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

FUT_KLINES = "https://fapi.binance.com/fapi/v1/klines"
SPOT_KLINES = "https://api.binance.com/api/v3/klines"


def http_json(url: str, timeout: int = 25, retries: int = 3, backoff: float = 0.5) -> Any:
    last_err: Optional[Exception] = None
    for i in range(max(1, retries)):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "model6-signal-alpha/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8", errors="ignore"))
        except Exception as e:
            last_err = e
            if i + 1 < retries:
                time.sleep(backoff * (2**i))
    raise RuntimeError(f"http_json failed for {url}: {last_err}")


def fetch_klines_page(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1500,
) -> List[List[Any]]:
    q = urllib.parse.urlencode(
        {
            "symbol": symbol,
            "interval": interval,
            "startTime": int(start_ms),
            "endTime": int(end_ms),
            "limit": int(limit),
        }
    )
    try:
        return http_json(f"{FUT_KLINES}?{q}")
    except Exception:
        return http_json(f"{SPOT_KLINES}?{q}")


def fetch_klines_range(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    rows = fetch_klines_page(symbol=symbol, interval=interval, start_ms=start_ms, end_ms=end_ms, limit=1500)
    if not isinstance(rows, list) or len(rows) == 0:
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close"])
    out = [
        {
            "open_time": int(r[0]),
            "open": float(r[1]),
            "high": float(r[2]),
            "low": float(r[3]),
            "close": float(r[4]),
        }
        for r in rows
    ]
    df = pd.DataFrame(out).sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
    return df


def normalize_symbol(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip().upper()
    if not s:
        return None
    if s.endswith("USDT"):
        return s
    if s.endswith("USD"):
        return s + "T"
    if s.isalpha() and len(s) <= 15:
        return s + "USDT"
    return None


def normalize_direction(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in {"long", "buy", "bull", "bullish"}:
        return "long"
    if s in {"short", "sell", "bear", "bearish"}:
        return "short"
    return None


def _safe_pct(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or abs(a) < 1e-12:
        return np.nan
    return (b / a - 1.0) * 100.0


def compute_moves(df: pd.DataFrame, start_idx: int, end_idx: int, direction: str) -> Dict[str, float]:
    entry = float(df.iloc[start_idx]["close"])
    wnd = df.iloc[start_idx : end_idx + 1]
    hi = float(np.nanmax(wnd["high"].to_numpy(dtype=np.float64)))
    lo = float(np.nanmin(wnd["low"].to_numpy(dtype=np.float64)))
    close_end = float(df.iloc[end_idx]["close"])

    if direction == "long":
        fav = _safe_pct(entry, hi)
        adv = _safe_pct(lo, entry)
        signed = _safe_pct(entry, close_end)
    else:
        fav = _safe_pct(lo, entry)
        adv = _safe_pct(entry, hi)
        signed = _safe_pct(close_end, entry)

    if np.isfinite(fav):
        fav = max(0.0, float(fav))
    if np.isfinite(adv):
        adv = max(0.0, float(adv))
    return {
        "entry_price": entry,
        "favorable_pct": float(fav) if np.isfinite(fav) else np.nan,
        "adverse_pct": float(adv) if np.isfinite(adv) else np.nan,
        "signed_ret_pct": float(signed) if np.isfinite(signed) else np.nan,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build coin-vs-BTC alpha move dataset from signal entries")
    ap.add_argument("--signals", default="data/telegram/signals_with_quality_strict_s3.parquet")
    ap.add_argument("--interval", default="15m")
    ap.add_argument("--bar-minutes", type=int, default=15)
    ap.add_argument("--horizons-hours", default="24,48,72")
    ap.add_argument(
        "--cache-bucket-hours",
        type=int,
        default=6,
        help="Reuse one kline fetch for signals in the same symbol/time bucket",
    )
    ap.add_argument("--beta", type=float, default=1.0, help="BTC beta used for alpha move")
    ap.add_argument("--good-alpha-fav-min", type=float, default=0.5, help="min favorable alpha pct to mark good")
    ap.add_argument("--good-alpha-signed-min", type=float, default=0.0, help="min signed alpha pct to mark good")
    ap.add_argument("--bad-adverse-max", type=float, default=3.0, help="max adverse move pct to mark good")
    ap.add_argument("--output", default="data/telegram/signals_move_vs_btc_alpha_strict_s3.parquet")
    ap.add_argument("--summary-json", default="reports/signals_move_vs_btc_alpha_strict_s3_summary.json")
    args = ap.parse_args()

    horizons = sorted({int(x.strip()) for x in str(args.horizons_hours).split(",") if x.strip()})
    horizons = [h for h in horizons if h > 0]
    if not horizons:
        raise ValueError("horizons-hours must contain at least one positive integer")
    max_h = max(horizons)
    bars_per_hour = max(1, int(round(60 / int(args.bar_minutes))))

    src = pd.read_parquet(args.signals).copy()
    src["timestamp_utc"] = pd.to_datetime(src["timestamp_utc"], utc=True, errors="coerce")
    src["symbol_norm"] = src["symbol"].map(normalize_symbol)
    src["direction_norm"] = src["direction"].map(normalize_direction)
    src = src[src["timestamp_utc"].notna() & src["symbol_norm"].notna() & src["direction_norm"].notna()].copy()
    src = src.sort_values("timestamp_utc").reset_index(drop=True)
    if src.empty:
        raise RuntimeError("No valid rows after timestamp/symbol/direction normalization")

    out = src.copy()
    out["status"] = "ok"
    out["status_reason"] = ""
    out["anchor_ts"] = pd.Series([pd.NaT] * len(out), dtype="datetime64[ns, UTC]")

    for h in horizons:
        out[f"coin_fav_h{h}_pct"] = np.nan
        out[f"coin_adv_h{h}_pct"] = np.nan
        out[f"coin_signed_h{h}_pct"] = np.nan
        out[f"btc_fav_h{h}_pct"] = np.nan
        out[f"btc_adv_h{h}_pct"] = np.nan
        out[f"btc_signed_h{h}_pct"] = np.nan
        out[f"alpha_fav_h{h}_pct"] = np.nan
        out[f"alpha_signed_h{h}_pct"] = np.nan
        out[f"label_h{h}"] = "skip"

    cache: Dict[Tuple[str, int], pd.DataFrame] = {}
    bucket_ms = max(1, int(args.cache_bucket_hours)) * 60 * 60 * 1000
    fetch_span_ms = max_h * 60 * 60 * 1000 + max(1, int(args.cache_bucket_hours)) * 60 * 60 * 1000
    total = len(out)
    for i, row in out.iterrows():
        sym = str(row["symbol_norm"])
        d = str(row["direction_norm"])
        ts = pd.Timestamp(row["timestamp_utc"])
        t_ms = int(ts.timestamp() * 1000)
        bucket_start_ms = (t_ms // bucket_ms) * bucket_ms
        end_ms = bucket_start_ms + fetch_span_ms

        key_sym = (sym, bucket_start_ms)
        key_btc = ("BTCUSDT", bucket_start_ms)
        if key_sym not in cache:
            cache[key_sym] = fetch_klines_range(sym, args.interval, bucket_start_ms, end_ms)
        if key_btc not in cache:
            cache[key_btc] = fetch_klines_range("BTCUSDT", args.interval, bucket_start_ms, end_ms)
        cdf = cache[key_sym]
        bdf = cache[key_btc]

        if cdf.empty:
            out.at[i, "status"] = "skip"
            out.at[i, "status_reason"] = "no_symbol_klines"
            continue
        if bdf.empty:
            out.at[i, "status"] = "skip"
            out.at[i, "status_reason"] = "no_btc_klines"
            continue

        c_open = cdf["open_time"].to_numpy(dtype=np.int64)
        b_open = bdf["open_time"].to_numpy(dtype=np.int64)
        c_idx = int(np.searchsorted(c_open, t_ms, side="left"))
        b_idx = int(np.searchsorted(b_open, t_ms, side="left"))
        if c_idx >= len(cdf) or b_idx >= len(bdf):
            out.at[i, "status"] = "skip"
            out.at[i, "status_reason"] = "anchor_not_found"
            continue

        anchor_ts = pd.to_datetime(int(cdf.iloc[c_idx]["open_time"]), unit="ms", utc=True)
        out.at[i, "anchor_ts"] = anchor_ts

        for h in horizons:
            bars = h * bars_per_hour
            c_end = c_idx + bars
            b_end = b_idx + bars
            if c_end >= len(cdf) or b_end >= len(bdf):
                out.at[i, f"label_h{h}"] = "skip"
                continue
            cm = compute_moves(cdf, c_idx, c_end, d)
            bm = compute_moves(bdf, b_idx, b_end, d)

            out.at[i, f"coin_fav_h{h}_pct"] = cm["favorable_pct"]
            out.at[i, f"coin_adv_h{h}_pct"] = cm["adverse_pct"]
            out.at[i, f"coin_signed_h{h}_pct"] = cm["signed_ret_pct"]
            out.at[i, f"btc_fav_h{h}_pct"] = bm["favorable_pct"]
            out.at[i, f"btc_adv_h{h}_pct"] = bm["adverse_pct"]
            out.at[i, f"btc_signed_h{h}_pct"] = bm["signed_ret_pct"]

            a_fav = np.nan
            a_signed = np.nan
            if np.isfinite(cm["favorable_pct"]) and np.isfinite(bm["favorable_pct"]):
                a_fav = float(cm["favorable_pct"] - float(args.beta) * bm["favorable_pct"])
            if np.isfinite(cm["signed_ret_pct"]) and np.isfinite(bm["signed_ret_pct"]):
                a_signed = float(cm["signed_ret_pct"] - float(args.beta) * bm["signed_ret_pct"])
            out.at[i, f"alpha_fav_h{h}_pct"] = a_fav
            out.at[i, f"alpha_signed_h{h}_pct"] = a_signed

            if (
                np.isfinite(a_fav)
                and np.isfinite(a_signed)
                and np.isfinite(cm["adverse_pct"])
                and a_fav >= float(args.good_alpha_fav_min)
                and a_signed >= float(args.good_alpha_signed_min)
                and cm["adverse_pct"] <= float(args.bad_adverse_max)
            ):
                out.at[i, f"label_h{h}"] = "good"
            else:
                out.at[i, f"label_h{h}"] = "bad"

        if (i + 1) % 100 == 0 or i + 1 == total:
            print(f"[progress] {i + 1}/{total}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    summary: Dict[str, Any] = {
        "rows_total": int(len(out)),
        "rows_ok_status": int((out["status"] == "ok").sum()),
        "rows_skip_status": int((out["status"] != "ok").sum()),
        "status_reason_counts": out["status_reason"].value_counts(dropna=False).to_dict(),
        "horizons_hours": horizons,
        "beta": float(args.beta),
        "label_rule": {
            "good_alpha_fav_min": float(args.good_alpha_fav_min),
            "good_alpha_signed_min": float(args.good_alpha_signed_min),
            "bad_adverse_max": float(args.bad_adverse_max),
        },
        "output_file": str(out_path),
    }
    for h in horizons:
        m = out[f"label_h{h}"] != "skip"
        summary[f"h{h}_rows_with_label"] = int(m.sum())
        summary[f"h{h}_label_counts"] = out.loc[m, f"label_h{h}"].value_counts(dropna=False).to_dict()
        for c in [f"alpha_fav_h{h}_pct", f"alpha_signed_h{h}_pct", f"coin_adv_h{h}_pct", f"coin_fav_h{h}_pct"]:
            s = pd.to_numeric(out.loc[m, c], errors="coerce")
            summary[f"{c}_mean"] = float(s.mean()) if s.notna().any() else np.nan
            summary[f"{c}_p50"] = float(s.quantile(0.5)) if s.notna().any() else np.nan
            summary[f"{c}_p75"] = float(s.quantile(0.75)) if s.notna().any() else np.nan

    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] saved: {out_path}")
    print(f"[ok] summary: {summary_path}")


if __name__ == "__main__":
    main()
