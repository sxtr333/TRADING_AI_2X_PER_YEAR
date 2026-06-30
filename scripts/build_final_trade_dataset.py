#!/usr/bin/env python3
import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.append(str(HERE))

from build_stable_alt_universe import normalize_symbol

FUT_KLINES = "https://fapi.binance.com/fapi/v1/klines"
SPOT_KLINES = "https://api.binance.com/api/v3/klines"
INTERVAL = "1d"
INTERVAL_MS = 24 * 60 * 60 * 1000


def http_json(url: str, timeout: int = 25) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "model6-final-dataset/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="ignore"))


def fetch_klines_page(symbol: str, start_ms: int, end_ms: int, limit: int = 1500) -> List[List[Any]]:
    q = urllib.parse.urlencode(
        {
            "symbol": symbol,
            "interval": INTERVAL,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": limit,
        }
    )
    try:
        return http_json(f"{FUT_KLINES}?{q}")
    except Exception:
        return http_json(f"{SPOT_KLINES}?{q}")


def fetch_klines_all(symbol: str, start_ms: int, end_ms: int) -> List[List[Any]]:
    out: List[List[Any]] = []
    cur = start_ms
    for _ in range(20):
        part = fetch_klines_page(symbol, cur, end_ms, limit=1500)
        if not isinstance(part, list) or len(part) == 0:
            break
        out.extend(part)
        last_open = int(part[-1][0])
        nxt = last_open + INTERVAL_MS
        if nxt <= cur:
            break
        cur = nxt
        if cur > end_ms:
            break
        if len(part) < 1500:
            break
    # de-dup by open time
    uniq = {}
    for r in out:
        uniq[int(r[0])] = r
    return [uniq[k] for k in sorted(uniq.keys())]


def norm_direction(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in {"long", "buy", "bull", "bullish", "up"}:
        return "long"
    if s in {"short", "sell", "bear", "bearish", "down"}:
        return "short"
    return None


LONG_RE = re.compile(r"\b(long|buy|лонг|в\s*лонг|вверх|рост|отскок)\b", re.IGNORECASE)
SHORT_RE = re.compile(r"\b(short|sell|шорт|в\s*шорт|вниз|паден|снижен)\b", re.IGNORECASE)


def infer_direction_from_text(text: Any) -> Optional[str]:
    if text is None:
        return None
    s = str(text).strip().lower()
    if not s:
        return None
    has_long = bool(LONG_RE.search(s))
    has_short = bool(SHORT_RE.search(s))
    if has_long and not has_short:
        return "long"
    if has_short and not has_long:
        return "short"
    return None


def kline_df(kl: List[List[Any]]) -> pd.DataFrame:
    rows = []
    for r in kl:
        rows.append(
            {
                "open_time": int(r[0]),
                "open": float(r[1]),
                "high": float(r[2]),
                "low": float(r[3]),
                "close": float(r[4]),
                "quote_vol": float(r[7]),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)


def pick_entry_idx(df: pd.DataFrame, ts_ms: int) -> Optional[int]:
    m = df.index[df["open_time"] >= ts_ms]
    if len(m) == 0:
        return None
    return int(m[0])


def eval_row(df: pd.DataFrame, idx0: int, direction: str, max_days: int = 14) -> Dict[str, Any]:
    sign = 1.0 if direction == "long" else -1.0
    p0 = float(df.loc[idx0, "close"])
    end = min(len(df) - 1, idx0 + max_days)
    if end <= idx0:
        return {"error": "insufficient_future_bars"}

    part = df.iloc[idx0 : end + 1].copy()

    def signed_ret(px: float) -> float:
        return sign * (px / p0 - 1.0)

    closes = part["close"].tolist()
    highs = part["high"].tolist()
    lows = part["low"].tolist()
    signed_path = [signed_ret(x) for x in closes]
    best_signed = max(signed_path)
    worst_signed = min(signed_path)
    best_idx = int(signed_path.index(best_signed))

    def at_day(d: int) -> float:
        j = min(len(closes) - 1, d)
        return signed_ret(closes[j])

    if direction == "long":
        max_fav = max((h / p0 - 1.0) for h in highs)
        max_adv = max((1.0 - l / p0) for l in lows)
    else:
        max_fav = max((p0 / l - 1.0) for l in lows)
        max_adv = max((h / p0 - 1.0) for h in highs)

    r3 = at_day(3)
    if r3 >= 0.02 and max_adv <= 0.05:
        label = "good"
    elif r3 <= -0.02 or max_adv >= 0.08:
        label = "bad"
    else:
        label = "neutral"

    return {
        "entry_price": p0,
        "ret_1d": at_day(1),
        "ret_3d": r3,
        "ret_7d": at_day(7),
        "ret_14d": at_day(14),
        "max_favorable_move_14d": max_fav,
        "max_adverse_move_14d": max_adv,
        "best_hold_days": best_idx,
        "best_signed_return_14d": best_signed,
        "worst_signed_return_14d": worst_signed,
        "quality_label": label,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build final trade-quality dataset from filtered universe")
    ap.add_argument("--signals", default="data/telegram/signals_raw_tradeable_universe_c015.parquet")
    ap.add_argument("--output", default="data/telegram/final_trade_dataset_c015.parquet")
    ap.add_argument("--summary-json", default="reports/final_trade_dataset_c015_summary.json")
    ap.add_argument("--rows-csv", default="reports/final_trade_dataset_c015_preview.csv")
    args = ap.parse_args()

    df = pd.read_parquet(args.signals).copy()
    df["symbol_norm"] = df["symbol"].map(normalize_symbol)
    df["direction_norm"] = df["direction"].map(norm_direction)
    df["direction_from_text"] = df["text"].map(infer_direction_from_text)
    df["direction_final"] = df["direction_norm"]
    mask_fill = df["direction_final"].isna()
    df.loc[mask_fill, "direction_final"] = df.loc[mask_fill, "direction_from_text"]
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    if "is_candidate" in df.columns:
        df["is_candidate"] = df["is_candidate"].fillna(False).astype(bool)
    else:
        df["is_candidate"] = False

    # Keep rows with valid minimum fields; rows with missing direction are handled as "skip".
    df = df[(df["symbol_norm"].notna()) & (df["timestamp_utc"].notna())].copy()
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    if df.empty:
        raise RuntimeError("No valid rows after symbol/direction/timestamp filtering.")

    t0 = int(df["timestamp_utc"].min().timestamp() * 1000) - 3 * INTERVAL_MS
    t1 = int(df["timestamp_utc"].max().timestamp() * 1000) + 20 * INTERVAL_MS

    symbols = sorted(df["symbol_norm"].dropna().unique().tolist())
    market: Dict[str, pd.DataFrame] = {}
    fetch_errors = {}
    for i, sym in enumerate(symbols, 1):
        try:
            kl = fetch_klines_all(sym, t0, t1)
            kdf = kline_df(kl)
            if len(kdf) < 30:
                fetch_errors[sym] = "not_enough_klines"
            else:
                market[sym] = kdf
        except Exception as e:
            fetch_errors[sym] = str(e)[:120]
        if i % 40 == 0:
            print(f"fetched {i}/{len(symbols)} symbols")

    out_rows: List[Dict[str, Any]] = []
    skip_reasons = Counter()
    labels = Counter()

    for _, r in df.iterrows():
        sym = r["symbol_norm"]
        d = r["direction_final"]
        ts = r["timestamp_utc"]
        base = {
            "message_id": r.get("message_id"),
            "timestamp_utc": str(ts),
            "symbol": sym,
            "direction": d,
            "direction_parsed": r.get("direction_norm"),
            "direction_from_text": r.get("direction_from_text"),
            "text": r.get("text", ""),
            "parse_confidence": r.get("parse_confidence"),
            "is_candidate": bool(r.get("is_candidate")),
        }
        if not base["is_candidate"] and d is None:
            skip_reasons["not_candidate"] += 1
            out_rows.append({**base, "status": "skip", "skip_reason": "not_candidate"})
            continue
        if d is None:
            skip_reasons["no_direction"] += 1
            out_rows.append({**base, "status": "skip", "skip_reason": "no_direction"})
            continue
        if sym not in market:
            reason = fetch_errors.get(sym, "no_market_data")
            skip_reasons[reason] += 1
            out_rows.append({**base, "status": "skip", "skip_reason": reason})
            continue
        kdf = market[sym]
        idx0 = pick_entry_idx(kdf, int(ts.timestamp() * 1000))
        if idx0 is None:
            skip_reasons["entry_idx_not_found"] += 1
            out_rows.append({**base, "status": "skip", "skip_reason": "entry_idx_not_found"})
            continue
        q = eval_row(kdf, idx0, d, max_days=14)
        if "error" in q:
            skip_reasons[q["error"]] += 1
            out_rows.append({**base, "status": "skip", "skip_reason": q["error"]})
            continue
        labels[q["quality_label"]] += 1
        out_rows.append({**base, "status": "ok", "skip_reason": "", **q})

    out_df = pd.DataFrame(out_rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    # Small preview CSV for quick manual review
    pv = Path(args.rows_csv)
    pv.parent.mkdir(parents=True, exist_ok=True)
    out_df.head(500).to_csv(pv, index=False)

    ok = int((out_df["status"] == "ok").sum()) if not out_df.empty else 0
    total = int(len(out_df))
    summary = {
        "source_signals": args.signals,
        "rows_input_after_min_fields": int(len(df)),
        "rows_total": total,
        "rows_ok": ok,
        "rows_skip": total - ok,
        "ok_rate": round(ok / total, 4) if total else 0.0,
        "quality_labels": dict(labels),
        "skip_reasons": dict(skip_reasons),
        "rows_with_direction_parsed": int(df["direction_norm"].notna().sum()),
        "rows_with_direction_text_inferred": int(df["direction_from_text"].notna().sum()),
        "rows_with_direction_final": int(df["direction_final"].notna().sum()),
        "rows_candidate_true": int(df["is_candidate"].sum()),
        "symbols_requested": len(symbols),
        "symbols_with_market_data": len(market),
        "symbols_failed_market_data": len(fetch_errors),
        "output": str(out_path),
        "preview_csv": str(pv),
    }
    Path(args.summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
