#!/usr/bin/env python3
"""Extract candidate trading signals from Telegram HTML exports.

Pipeline steps:
1) Discover export roots (directories with messages*.html).
2) Parse message timestamp/text/photo refs from HTML.
3) Heuristically extract signal fields from text.
4) Save rows + summary.
"""

import argparse
import json
import os
import re
from dataclasses import dataclass, asdict
from datetime import timezone
from html import unescape
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


DATE_RE = re.compile(r'title="(\d{2})\.(\d{2})\.(\d{4})\s+(\d{2}):(\d{2}):(\d{2})')
MSG_START_RE = re.compile(r'<div class="message[^\"]*"\s+id="message(-?\d+)"')
PHOTO_RE = re.compile(r'href="photos/([^"]+)"')
TAG_RE = re.compile(r"<[^>]+>")

SYMBOL_RE = re.compile(r"\b([A-Z]{2,15}(?:USDT|USD|BTC|ETH|PERP)?)\b")
NUMBER_RE = re.compile(r"[-+]?\d+(?:[\.,]\d+)?")

LONG_WORDS = ("long", "лонг", "buy", "покуп", "вверх")
SHORT_WORDS = ("short", "шорт", "sell", "продаж", "вниз")
ENTRY_WORDS = ("entry", "вход", "enter", "набора", "покупк", "sell zone")
STOP_WORDS = ("stop", "sl", "стоп")
TP_WORDS = ("tp", "take", "тейк", "цель", "target")


@dataclass
class ParsedMessage:
    export_root: str
    message_file: str
    message_id: str
    timestamp_utc: Optional[str]
    text: str
    photos: List[str]


@dataclass
class SignalRow:
    export_root: str
    message_file: str
    message_id: str
    timestamp_utc: Optional[str]
    text: str
    photos: str
    symbol: Optional[str]
    direction: Optional[str]
    entry_min: Optional[float]
    entry_max: Optional[float]
    stop: Optional[float]
    tp1: Optional[float]
    tp2: Optional[float]
    tp3: Optional[float]
    parse_confidence: float
    is_candidate: bool


def _strip_html(value: str) -> str:
    value = value.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    value = TAG_RE.sub("", value)
    value = unescape(value)
    value = value.replace("\xa0", " ")
    return re.sub(r"\s+", " ", value).strip()


def _to_float(token: str) -> Optional[float]:
    token = token.strip()
    if not token:
        return None
    if "," in token and "." in token:
        token = token.replace(",", "")
    else:
        token = token.replace(",", ".")
    try:
        return float(token)
    except ValueError:
        return None


def _numbers_from_text(line: str) -> List[float]:
    nums: List[float] = []
    for t in NUMBER_RE.findall(line):
        v = _to_float(t)
        if v is not None:
            nums.append(v)
    return nums


def _line_has_any(line: str, words: Tuple[str, ...]) -> bool:
    l = line.lower()
    return any(w in l for w in words)


def _discover_export_roots(root: Path) -> List[Path]:
    found: List[Path] = []
    for d, _, files in os.walk(root):
        p = Path(d)
        if any(f.startswith("messages") and f.endswith(".html") for f in files):
            found.append(p)
    return sorted(set(found))


def _parse_one_html(export_root: Path, html_path: Path) -> List[ParsedMessage]:
    messages: List[ParsedMessage] = []

    current_id: Optional[str] = None
    current_ts: Optional[str] = None
    current_text_chunks: List[str] = []
    current_photos: List[str] = []

    in_text = False

    def flush() -> None:
        nonlocal current_id, current_ts, current_text_chunks, current_photos
        if current_id is None:
            return
        text = " ".join(x for x in current_text_chunks if x).strip()
        if text or current_photos:
            messages.append(
                ParsedMessage(
                    export_root=str(export_root),
                    message_file=html_path.name,
                    message_id=current_id,
                    timestamp_utc=current_ts,
                    text=text,
                    photos=current_photos[:],
                )
            )
        current_text_chunks = []
        current_photos = []

    with html_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()

            m_start = MSG_START_RE.search(line)
            if m_start:
                flush()
                current_id = m_start.group(1)
                in_text = False

            m_date = DATE_RE.search(line)
            if m_date:
                dd, mm, yyyy, hh, mi, ss = m_date.groups()
                # Telegram export includes timezone offset in source text,
                # but we normalize to naive UTC string for downstream alignment.
                dt = pd.Timestamp(
                    year=int(yyyy),
                    month=int(mm),
                    day=int(dd),
                    hour=int(hh),
                    minute=int(mi),
                    second=int(ss),
                    tz="UTC",
                )
                current_ts = dt.isoformat()

            if '<div class="text">' in line:
                in_text = True
                part = line.split('<div class="text">', 1)[1]
                if "</div>" in part:
                    piece = part.split("</div>", 1)[0]
                    current_text_chunks.append(_strip_html(piece))
                    in_text = False
                else:
                    current_text_chunks.append(_strip_html(part))
                continue

            if in_text:
                if "</div>" in line:
                    piece = line.split("</div>", 1)[0]
                    current_text_chunks.append(_strip_html(piece))
                    in_text = False
                else:
                    current_text_chunks.append(_strip_html(line))

            for photo in PHOTO_RE.findall(line):
                if "_thumb" in photo.lower():
                    continue
                current_photos.append(photo)

    flush()
    return messages


def _extract_signal_fields(text: str) -> Dict[str, object]:
    lines = [x.strip() for x in text.split("\n") if x.strip()]
    if not lines:
        lines = [text.strip()] if text.strip() else []

    symbol: Optional[str] = None
    direction: Optional[str] = None
    entry_min: Optional[float] = None
    entry_max: Optional[float] = None
    stop: Optional[float] = None
    tps: List[float] = []

    symbol_hits = SYMBOL_RE.findall(text)
    if symbol_hits:
        symbol = symbol_hits[0]

    lower_text = text.lower()
    if any(w in lower_text for w in LONG_WORDS):
        direction = "long"
    if any(w in lower_text for w in SHORT_WORDS):
        direction = "short" if direction is None else direction

    for line in lines:
        nums = _numbers_from_text(line)
        if not nums:
            continue

        if _line_has_any(line, ENTRY_WORDS):
            entry_min = min(nums)
            entry_max = max(nums)
            continue

        if _line_has_any(line, STOP_WORDS) and stop is None:
            stop = nums[0]
            continue

        if _line_has_any(line, TP_WORDS):
            for x in nums:
                tps.append(x)

    if entry_min is None or entry_max is None:
        # fallback: try to infer first range from generic lines
        nums = _numbers_from_text(text)
        if len(nums) >= 2:
            entry_min, entry_max = min(nums[:2]), max(nums[:2])

    tps = sorted(set(tps))
    tp1 = tps[0] if len(tps) >= 1 else None
    tp2 = tps[1] if len(tps) >= 2 else None
    tp3 = tps[2] if len(tps) >= 3 else None

    confidence = 0.0
    if direction:
        confidence += 0.35
    if entry_min is not None:
        confidence += 0.25
    if stop is not None:
        confidence += 0.2
    if tp1 is not None:
        confidence += 0.15
    if symbol is not None:
        confidence += 0.1
    confidence = float(min(confidence, 1.0))

    return {
        "symbol": symbol,
        "direction": direction,
        "entry_min": entry_min,
        "entry_max": entry_max,
        "stop": stop,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "parse_confidence": confidence,
        "is_candidate": confidence >= 0.5,
    }


def _save_df(df: pd.DataFrame, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".csv":
        df.to_csv(output, index=False)
        return output

    try:
        df.to_parquet(output, index=False)
        return output
    except Exception:
        csv_fallback = output.with_suffix(".csv")
        df.to_csv(csv_fallback, index=False)
        return csv_fallback


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract candidate signals from Telegram exports.")
    ap.add_argument("--input-root", required=True, help="Root directory with Telegram export folders")
    ap.add_argument(
        "--output",
        default="data/telegram/signals_raw.parquet",
        help="Output file (.parquet preferred; auto-fallback to .csv)",
    )
    ap.add_argument(
        "--summary-json",
        default="reports/signal_extraction_quality.json",
        help="Summary report JSON",
    )
    ap.add_argument(
        "--summary-csv",
        default="reports/signal_extraction_quality.csv",
        help="Summary report CSV (metric,value table)",
    )
    args = ap.parse_args()

    root = Path(args.input_root).resolve()
    out = Path(args.output)
    summary_path = Path(args.summary_json)
    summary_csv_path = Path(args.summary_csv)

    export_roots = _discover_export_roots(root)
    rows: List[SignalRow] = []

    for export_root in export_roots:
        html_files = sorted(x for x in export_root.iterdir() if x.name.startswith("messages") and x.suffix == ".html")
        for html_file in html_files:
            parsed = _parse_one_html(export_root, html_file)
            for msg in parsed:
                fields = _extract_signal_fields(msg.text)
                rows.append(
                    SignalRow(
                        export_root=msg.export_root,
                        message_file=msg.message_file,
                        message_id=msg.message_id,
                        timestamp_utc=msg.timestamp_utc,
                        text=msg.text,
                        photos=";".join(msg.photos),
                        symbol=fields["symbol"],
                        direction=fields["direction"],
                        entry_min=fields["entry_min"],
                        entry_max=fields["entry_max"],
                        stop=fields["stop"],
                        tp1=fields["tp1"],
                        tp2=fields["tp2"],
                        tp3=fields["tp3"],
                        parse_confidence=fields["parse_confidence"],
                        is_candidate=bool(fields["is_candidate"]),
                    )
                )

    df = pd.DataFrame([asdict(x) for x in rows])
    if not df.empty:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
        df = df.sort_values(["timestamp_utc", "export_root", "message_id"], na_position="last").reset_index(drop=True)

    saved_path = _save_df(df, out)

    summary = {
        "input_root": str(root),
        "export_roots": [str(x) for x in export_roots],
        "export_roots_count": len(export_roots),
        "messages_parsed": int(len(df)),
        "candidates_count": int(df["is_candidate"].sum()) if not df.empty else 0,
        "candidate_rate": float(df["is_candidate"].mean()) if not df.empty else 0.0,
        "avg_parse_confidence": float(df["parse_confidence"].mean()) if not df.empty else 0.0,
        "output_file": str(saved_path),
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "metric": list(summary.keys()),
            "value": [json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for v in summary.values()],
        }
    ).to_csv(summary_csv_path, index=False)

    print(f"[extract] export_roots={len(export_roots)}")
    print(f"[extract] messages={len(df)} candidates={summary['candidates_count']}")
    print(f"[extract] saved={saved_path}")
    print(f"[extract] summary={summary_path}")
    print(f"[extract] summary_csv={summary_csv_path}")


if __name__ == "__main__":
    main()
