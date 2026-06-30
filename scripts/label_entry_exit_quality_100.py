#!/usr/bin/env python3
import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
from collections import Counter
from datetime import timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


INTERVAL = "15m"
INTERVAL_MS = 15 * 60 * 1000


def norm_symbol(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().upper().replace("#", "").replace("$", "")
    s = s.replace("/", "").replace("-", "").replace("_", "")
    if not s:
        return None
    if s.endswith("PERP"):
        s = s[: -4] + "USDT"
    if s.endswith("USDTP"):
        s = s[:-1]
    if s in {"BTC", "ETH", "BNB", "XRP", "SOL", "ADA", "DOGE", "DOT", "LINK", "LTC"}:
        s = s + "USDT"
    if not any(s.endswith(q) for q in ("USDT", "USDC", "BUSD", "USD")):
        if len(s) <= 12:
            s = s + "USDT"
    # Drop obvious non-ticker garbage from text parsing.
    bad = {"LONGUSDT", "SHORTUSDT", "BINANCEUSDT", "SIGNALUSDT", "ENTRYUSDT"}
    if s in bad:
        return None
    if s == "LITUSDT":
        s = "LTCUSDT"
    return s


def norm_direction(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in {"long", "buy", "bull", "bullish", "up"}:
        return "long"
    if s in {"short", "sell", "bear", "bearish", "down"}:
        return "short"
    return None


def http_json(url: str, timeout: int = 20) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "model6-quality-pilot/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="ignore"))


def fetch_klines(symbol: str, start_ms: int, end_ms: int, limit: int = 1500) -> List[List[Any]]:
    q = urllib.parse.urlencode(
        {
            "symbol": symbol,
            "interval": INTERVAL,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": limit,
        }
    )
    futures_url = f"https://fapi.binance.com/fapi/v1/klines?{q}"
    spot_url = f"https://api.binance.com/api/v3/klines?{q}"

    try:
        return http_json(futures_url)
    except Exception:
        return http_json(spot_url)


def pick_entry_index(kl: List[List[Any]], ts_ms: int) -> Optional[int]:
    for i, row in enumerate(kl):
        open_time = int(row[0])
        if open_time >= ts_ms:
            return i
    return None


def to_float(x: Any) -> float:
    return float(x)


def eval_quality(kl: List[List[Any]], idx0: int, direction: str, max_bars: int = 96) -> Dict[str, Any]:
    sign = 1.0 if direction == "long" else -1.0
    p0 = to_float(kl[idx0][4])  # close
    end = min(len(kl) - 1, idx0 + max_bars)
    if end <= idx0:
        return {"error": "insufficient_future_bars"}

    closes = [to_float(kl[i][4]) for i in range(idx0, end + 1)]
    highs = [to_float(kl[i][2]) for i in range(idx0, end + 1)]
    lows = [to_float(kl[i][3]) for i in range(idx0, end + 1)]

    def signed_ret(px: float) -> float:
        return sign * (px / p0 - 1.0)

    # Horizon returns
    bars_by_h = {"1h": 4, "4h": 16, "12h": 48, "24h": 96}
    ret = {}
    for h, b in bars_by_h.items():
        j = min(len(closes) - 1, b)
        ret[h] = signed_ret(closes[j])

    signed_path = [signed_ret(px) for px in closes]
    best_signed = max(signed_path)
    worst_signed = min(signed_path)
    best_idx = signed_path.index(best_signed)

    if direction == "long":
        max_fav = max((h / p0 - 1.0) for h in highs)
        max_adv = max((1.0 - l / p0) for l in lows)
    else:
        max_fav = max((p0 / l - 1.0) for l in lows)
        max_adv = max((h / p0 - 1.0) for h in highs)

    # Simple entry-quality label
    # good: 4h return >= 0.7% and adverse <= 1.5%
    # bad: 4h return <= -0.7% or adverse >= 2.0%
    r4 = ret["4h"]
    if r4 >= 0.007 and max_adv <= 0.015:
        label = "good"
    elif r4 <= -0.007 or max_adv >= 0.02:
        label = "bad"
    else:
        label = "neutral"

    return {
        "entry_price": p0,
        "ret_1h": ret["1h"],
        "ret_4h": ret["4h"],
        "ret_12h": ret["12h"],
        "ret_24h": ret["24h"],
        "max_favorable_move": max_fav,
        "max_adverse_move": max_adv,
        "best_hold_bars": best_idx,
        "best_signed_return": best_signed,
        "worst_signed_return": worst_signed,
        "quality_label": label,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Pilot entry/exit quality labels on first 100 signal images")
    ap.add_argument("--signals", default="data/telegram/signals_raw.parquet")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--output-csv", default="reports/entry_exit_quality_pilot100.csv")
    ap.add_argument("--summary-json", default="reports/entry_exit_quality_pilot100_summary.json")
    ap.add_argument("--sleep-ms", type=int, default=50)
    ap.add_argument("--require-direction", action="store_true", default=True)
    ap.add_argument("--require-symbol", action="store_true", default=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.signals)
    df = df[df["photos"].fillna("").astype(str) != ""].copy()
    if args.require_symbol:
        df = df[df["symbol"].fillna("").astype(str).str.strip() != ""].copy()
    if args.require_direction:
        d = df["direction"].fillna("").astype(str).str.lower().str.strip()
        df = df[d.isin(["long", "short"])].copy()
    df = df.sort_values("timestamp_utc").head(args.limit).copy()

    out_rows: List[Dict[str, Any]] = []
    skip_reasons = Counter()
    label_counts = Counter()

    for _, r in df.iterrows():
        ts = pd.to_datetime(r["timestamp_utc"], utc=True, errors="coerce")
        photo_rel = str(r["photos"]).split(";")[0].strip()
        if photo_rel.endswith("_thumb.jpg"):
            photo_rel = photo_rel.replace("_thumb.jpg", ".jpg")
        elif photo_rel.endswith("_thumb.jpeg"):
            photo_rel = photo_rel.replace("_thumb.jpeg", ".jpeg")
        elif photo_rel.endswith("_thumb.png"):
            photo_rel = photo_rel.replace("_thumb.png", ".png")
        elif photo_rel.endswith("_thumb.webp"):
            photo_rel = photo_rel.replace("_thumb.webp", ".webp")
        photo_path = str(r["export_root"]).rstrip("/") + "/photos/" + photo_rel

        symbol = norm_symbol(r.get("symbol"))
        direction = norm_direction(r.get("direction"))

        base = {
            "message_id": r.get("message_id"),
            "timestamp_utc": str(r.get("timestamp_utc")),
            "photo_path": photo_path,
            "symbol": symbol or "",
            "direction": direction or "",
        }

        if ts is pd.NaT or ts is None:
            skip_reasons["bad_timestamp"] += 1
            out_rows.append({**base, "status": "skip", "skip_reason": "bad_timestamp"})
            continue
        if not symbol:
            skip_reasons["no_symbol"] += 1
            out_rows.append({**base, "status": "skip", "skip_reason": "no_symbol"})
            continue
        if not direction:
            skip_reasons["no_direction"] += 1
            out_rows.append({**base, "status": "skip", "skip_reason": "no_direction"})
            continue

        ts_ms = int(ts.tz_convert(timezone.utc).timestamp() * 1000)
        start_ms = ts_ms - 4 * INTERVAL_MS
        end_ms = ts_ms + 120 * INTERVAL_MS

        try:
            kl = fetch_klines(symbol, start_ms, end_ms)
            if not isinstance(kl, list) or len(kl) < 20:
                raise RuntimeError("not_enough_klines")
            idx0 = pick_entry_index(kl, ts_ms)
            if idx0 is None or idx0 >= len(kl) - 2:
                raise RuntimeError("entry_idx_not_found")
            q = eval_quality(kl, idx0, direction, max_bars=96)
            if "error" in q:
                raise RuntimeError(q["error"])

            label_counts[q["quality_label"]] += 1
            out_rows.append({**base, "status": "ok", "skip_reason": "", **q})
        except Exception as e:
            reason = str(e)[:120] or "market_data_error"
            skip_reasons[reason] += 1
            out_rows.append({**base, "status": "skip", "skip_reason": reason})

        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fn = sorted({k for row in out_rows for k in row.keys()})
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        w.writerows(out_rows)

    n = len(out_rows)
    ok = sum(1 for r in out_rows if r.get("status") == "ok")
    summary = {
        "input_limit": args.limit,
        "rows_total": n,
        "rows_ok": ok,
        "rows_skip": n - ok,
        "ok_rate": round(ok / n, 4) if n else 0.0,
        "quality_labels": dict(label_counts),
        "skip_reasons": dict(skip_reasons),
        "output_csv": str(out_csv),
    }
    Path(args.summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
