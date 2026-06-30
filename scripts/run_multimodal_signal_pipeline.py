#!/usr/bin/env python3
"""
Multimodal pipeline for Telegram trading posts:
1) Vision model extracts chart facts from image only.
2) Text model extracts intent/rules from post text only.
3) Rule-based fusion combines both without hallucinating missing levels.
"""

import argparse
import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

QUOTE_SUFFIXES = ("USDT", "USDC", "BUSD", "USD", "PERP")
COIN_NAME_MAP = {
    "BITCOIN": "BTC",
    "ETHEREUM": "ETH",
    "MONERO": "XMR",
    "RIPPLE": "XRP",
    "BINANCECOIN": "BNB",
    "KUSAMA": "KSM",
    "LITECOIN": "LTC",
    "DOGECOIN": "DOGE",
}


VISION_SYSTEM = (
    "You are a strict visual analyst of crypto chart screenshots. "
    "Extract only what is explicitly visible in the image. "
    "Never invent hidden text or price levels."
)

VISION_USER = (
    "Return JSON with schema: "
    "{"
    "\"symbol_on_chart\": string|null,"
    "\"timeframe_on_chart\": string|null,"
    "\"visual_bias\": \"bullish\"|\"bearish\"|\"neutral\"|\"unclear\","
    "\"has_arrow\": boolean,"
    "\"has_drawn_levels\": boolean,"
    "\"explicit_price_levels\": number[],"
    "\"image_contains_trade_setup\": boolean,"
    "\"observations\": string[],"
    "\"chart_patterns\": string[],"
    "\"risk_flags\": string[],"
    "\"ocr_fragments\": string[],"
    "\"confidence\": number|null"
    "}. "
    "Rules: explicit_price_levels must contain only clearly readable numbers from image. "
    "Keep observations/chart_patterns/risk_flags/ocr_fragments short and up to 3 items each."
)

TEXT_SYSTEM = (
    "You are a strict parser of trading post text. "
    "Extract only explicit instructions written in the text. "
    "Do not infer missing levels."
)

TEXT_USER_TEMPLATE = (
    "Parse this post text and return JSON with schema: "
    "{"
    "\"symbol_from_text\": string|null,"
    "\"direction_from_text\": \"long\"|\"short\"|null,"
    "\"entry_min\": number|null,"
    "\"entry_max\": number|null,"
    "\"stop\": number|null,"
    "\"tp\": number[],"
    "\"conditions\": string[],"
    "\"confidence\": number|null"
    "}. "
    "Text:\n"
)


def _to_str_list(v: Any) -> List[str]:
    if not isinstance(v, list):
        return []
    out: List[str] = []
    for x in v:
        s = str(x).strip()
        if s:
            out.append(s)
    return out


@dataclass
class RowIn:
    message_id: Any
    timestamp_utc: str
    text: str
    photo_path: str
    cap_symbol: str
    cap_direction: str
    cap_entry_min: Any
    cap_entry_max: Any
    cap_stop: Any
    cap_tp1: Any
    cap_tp2: Any
    cap_tp3: Any


def _clean_symbol(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().upper().replace("#", "").replace("$", "")
    s = s.replace(" ", "")
    s = s.replace("TETHERUS", "USDT")
    s = s.replace("TETHER", "USDT")
    s = s.replace("USDTP", "USDT")
    s = s.replace("/", "").replace("-", "").replace("_", "")
    s = re.sub(r"[^A-Z0-9]", "", s)
    if not s:
        return None

    # Map full coin names if present.
    for name, ticker in COIN_NAME_MAP.items():
        if s.startswith(name):
            tail = s[len(name) :]
            if tail in QUOTE_SUFFIXES:
                return ticker + tail
            s = ticker + tail
            break

    m = re.match(r"^([A-Z0-9]{2,15})(USDT|USDC|BUSD|USD|PERP)$", s)
    if m:
        return m.group(1) + m.group(2)

    # Cases like KSMUSDT from "KSM/TetherUS" may already be normalized above.
    if "USDT" in s and not s.endswith("USDT"):
        i = s.find("USDT")
        base = s[:i]
        if 2 <= len(base) <= 15:
            return base + "USDT"

    return s


def _symbol_base(sym: Optional[str]) -> Optional[str]:
    if not sym:
        return None
    s = _clean_symbol(sym)
    if not s:
        return None
    for q in QUOTE_SUFFIXES:
        if s.endswith(q) and len(s) > len(q):
            return s[: -len(q)]
    return s


def _clean_direction(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in {"long", "buy", "bull", "bullish", "up"}:
        return "long"
    if s in {"short", "sell", "bear", "bearish", "down"}:
        return "short"
    return None


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip().replace(",", ".")
    if not s:
        return None
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def _to_float_list(v: Any) -> List[float]:
    if isinstance(v, list):
        out: List[float] = []
        for x in v:
            fx = _to_float(x)
            if fx is not None:
                out.append(fx)
        return out
    return []


def _safe_json_parse(s: str) -> Dict[str, Any]:
    s = (s or "").strip()
    if not s:
        return {}
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        m = re.search(r"\{.*\}", s, flags=re.S)
        if not m:
            return {}
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}


def ollama_chat(
    model: str,
    messages: List[Dict[str, Any]],
    fmt_json: bool = True,
    timeout: int = 180,
    num_predict: int = 160,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0, "num_predict": num_predict},
    }
    if fmt_json:
        payload["format"] = "json"
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8", errors="ignore"))
    return data


def vision_extract(photo_path: str, model: str, timeout: int) -> Dict[str, Any]:
    with open(photo_path, "rb") as f:
        b64 = __import__("base64").b64encode(f.read()).decode("ascii")
    resp = ollama_chat(
        model=model,
        messages=[
            {"role": "system", "content": VISION_SYSTEM},
            {"role": "user", "content": VISION_USER, "images": [b64]},
        ],
        fmt_json=True,
        timeout=timeout,
        num_predict=420,
    )
    return _safe_json_parse(resp.get("message", {}).get("content", ""))


def text_extract(text: str, model: str, timeout: int) -> Dict[str, Any]:
    prompt = TEXT_USER_TEMPLATE + (text or "")[:12000]
    resp = ollama_chat(
        model=model,
        messages=[
            {"role": "system", "content": TEXT_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        fmt_json=True,
        timeout=timeout,
        num_predict=120,
    )
    return _safe_json_parse(resp.get("message", {}).get("content", ""))


def fuse(vision: Dict[str, Any], text: Dict[str, Any]) -> Dict[str, Any]:
    symbol_v = _clean_symbol(vision.get("symbol_on_chart"))
    symbol_t = _clean_symbol(text.get("symbol_from_text"))

    dir_t = _clean_direction(text.get("direction_from_text"))
    vb = str(vision.get("visual_bias") or "").lower()
    dir_v = "long" if vb == "bullish" else ("short" if vb == "bearish" else None)

    # Text has priority for actionable levels.
    entry_min = _to_float(text.get("entry_min"))
    entry_max = _to_float(text.get("entry_max"))
    stop = _to_float(text.get("stop"))
    tp = _to_float_list(text.get("tp"))

    symbol = symbol_t or symbol_v
    direction = dir_t or dir_v

    base_t = _symbol_base(symbol_t)
    base_v = _symbol_base(symbol_v)
    conflict_symbol = bool(base_t and base_v and base_t != base_v)
    conflict_direction = bool(dir_t and dir_v and dir_t != dir_v)
    observations = _to_str_list(vision.get("observations"))
    chart_patterns = _to_str_list(vision.get("chart_patterns"))
    risk_flags = _to_str_list(vision.get("risk_flags"))
    text_conditions = _to_str_list(text.get("conditions"))

    has_actionable_text_levels = any(
        x is not None for x in [entry_min, entry_max, stop]
    ) or len(tp) > 0

    agreement_score = 0.0
    if base_t and base_v:
        agreement_score += 0.4 if base_t == base_v else -0.4
    if dir_t and dir_v:
        agreement_score += 0.4 if dir_t == dir_v else -0.4
    if has_actionable_text_levels:
        agreement_score += 0.2
    agreement_score = max(-1.0, min(1.0, agreement_score))

    if symbol_t and symbol_v and not conflict_symbol:
        # Prefer richer symbol form when one side has quote and the other is base-only.
        if symbol_t == base_t and symbol_v != base_v:
            symbol = symbol_v
        elif symbol_v == base_v and symbol_t != base_t:
            symbol = symbol_t

    if conflict_symbol or conflict_direction:
        action = "needs_review"
        confidence = "low"
    elif has_actionable_text_levels and direction and symbol:
        action = "trade_signal"
        confidence = "high" if agreement_score >= 0.6 else "medium"
    elif direction or (vision.get("image_contains_trade_setup") is True):
        action = "watchlist"
        confidence = "medium"
    else:
        action = "skip"
        confidence = "low"

    rationale: List[str] = []
    if observations:
        rationale.extend(observations[:3])
    if text_conditions:
        rationale.extend(text_conditions[:2])
    if not rationale and chart_patterns:
        rationale.extend(chart_patterns[:2])

    return {
        "symbol": symbol,
        "direction": direction,
        "entry_min": entry_min,
        "entry_max": entry_max,
        "stop": stop,
        "tp": tp,
        "action": action,
        "confidence": confidence,
        "agreement_score": round(agreement_score, 3),
        "conflict_symbol": conflict_symbol,
        "conflict_direction": conflict_direction,
        "rationale": rationale,
        "risk_flags": risk_flags,
        "chart_patterns": chart_patterns,
    }


def _split_photo_refs(raw: Any) -> List[str]:
    if raw is None:
        return []
    s = str(raw).strip()
    if not s or s == "[]":
        return []
    refs = [x.strip() for x in s.split(";")]
    out: List[str] = []
    seen = set()
    for r in refs:
        if not r or r in seen:
            continue
        seen.add(r)
        out.append(r)
    return out


def build_rows(df: pd.DataFrame, all_photos: bool = False) -> List[RowIn]:
    out: List[RowIn] = []
    for _, r in df.iterrows():
        refs = _split_photo_refs(r.get("photos"))
        if not refs:
            continue
        if not all_photos:
            refs = refs[:1]
        root = str(r.get("export_root") or "").rstrip("/")
        for ref in refs:
            photo_path = root + "/photos/" + ref
            out.append(
                RowIn(
                    message_id=r.get("message_id"),
                    timestamp_utc=str(r.get("timestamp_utc") or ""),
                    text=str(r.get("text") or ""),
                    photo_path=photo_path,
                    cap_symbol=str(r.get("symbol") or ""),
                    cap_direction=str(r.get("direction") or ""),
                    cap_entry_min=r.get("entry_min"),
                    cap_entry_max=r.get("entry_max"),
                    cap_stop=r.get("stop"),
                    cap_tp1=r.get("tp1"),
                    cap_tp2=r.get("tp2"),
                    cap_tp3=r.get("tp3"),
                )
            )
    return out


def load_seen(path: Path) -> set:
    seen = set()
    if not path.exists():
        return seen
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            key = (str(obj.get("message_id")), str(obj.get("photo_path")))
            seen.add(key)
    return seen


def text_from_parsed_fields(row: RowIn) -> Dict[str, Any]:
    tp = []
    for x in (row.cap_tp1, row.cap_tp2, row.cap_tp3):
        fx = _to_float(x)
        if fx is not None:
            tp.append(fx)
    return {
        "symbol_from_text": _clean_symbol(row.cap_symbol),
        "direction_from_text": _clean_direction(row.cap_direction),
        "entry_min": _to_float(row.cap_entry_min),
        "entry_max": _to_float(row.cap_entry_max),
        "stop": _to_float(row.cap_stop),
        "tp": tp,
        "conditions": [],
        "confidence": 1.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run multimodal vision+text+fusion pipeline on Telegram exports")
    ap.add_argument("--signals", default="data/telegram/signals_raw.parquet")
    ap.add_argument("--vision-model", default="qwen2.5vl:7b")
    ap.add_argument("--text-model", default="qwen2.5:7b")
    ap.add_argument("--output-jsonl", default="reports/multimodal_signal_outputs.jsonl")
    ap.add_argument("--output-summary", default="reports/multimodal_signal_summary.json")
    ap.add_argument("--limit", type=int, default=0, help="0 means all")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--sleep-ms", type=int, default=0)
    ap.add_argument("--vision-timeout", type=int, default=240)
    ap.add_argument("--text-timeout", type=int, default=120)
    ap.add_argument("--progress-every", type=int, default=10)
    ap.add_argument("--text-mode", choices=["llm", "parsed"], default="llm")
    ap.add_argument("--all-photos", action="store_true", help="Process all photos in each message")
    ap.add_argument("--newest-first", action="store_true", help="Process newest messages first")
    args = ap.parse_args()

    signals_path = Path(args.signals)
    out_jsonl = Path(args.output_jsonl)
    out_summary = Path(args.output_summary)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(signals_path)
    if args.newest_first and "timestamp_utc" in df.columns:
        ts = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
        df = df.assign(_ts=ts).sort_values("_ts", ascending=False, na_position="last").drop(columns=["_ts"])

    rows = build_rows(df, all_photos=bool(args.all_photos))
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    seen = load_seen(out_jsonl) if args.resume else set()

    total = len(rows)
    done = 0
    processed = 0
    counts = {
        "trade_signal": 0,
        "watchlist": 0,
        "needs_review": 0,
        "skip": 0,
        "errors": 0,
    }

    fout = out_jsonl.open("a", encoding="utf-8")
    try:
        for row in rows:
            key = (str(row.message_id), row.photo_path)
            if key in seen:
                done += 1
                continue

            if not os.path.isfile(row.photo_path):
                counts["errors"] += 1
                rec = {
                    "message_id": row.message_id,
                    "timestamp_utc": row.timestamp_utc,
                    "photo_path": row.photo_path,
                    "error": "photo_not_found",
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                done += 1
                continue

            try:
                v = vision_extract(row.photo_path, args.vision_model, args.vision_timeout)
                if args.text_mode == "parsed":
                    t = text_from_parsed_fields(row)
                else:
                    t = text_extract(row.text, args.text_model, args.text_timeout)
                fz = fuse(v, t)
                counts[fz["action"]] = counts.get(fz["action"], 0) + 1

                rec = {
                    "message_id": row.message_id,
                    "timestamp_utc": row.timestamp_utc,
                    "photo_path": row.photo_path,
                    "caption_symbol": _clean_symbol(row.cap_symbol),
                    "caption_direction": _clean_direction(row.cap_direction),
                    "vision": v,
                    "text": t,
                    "fused": fz,
                }
            except Exception as e:
                counts["errors"] += 1
                rec = {
                    "message_id": row.message_id,
                    "timestamp_utc": row.timestamp_utc,
                    "photo_path": row.photo_path,
                    "error": str(e),
                }

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            done += 1
            processed += 1
            if args.progress_every > 0 and processed % args.progress_every == 0:
                print(f"processed {done}/{total}")

            if args.sleep_ms > 0:
                time.sleep(args.sleep_ms / 1000.0)
    finally:
        fout.close()

    summary = {
        "total_rows": total,
        "newly_processed": processed,
        "counts": counts,
        "vision_model": args.vision_model,
        "text_model": args.text_model,
        "text_mode": args.text_mode,
        "signals": str(signals_path),
        "output_jsonl": str(out_jsonl),
    }

    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
