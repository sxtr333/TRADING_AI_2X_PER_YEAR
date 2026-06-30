#!/usr/bin/env python3
import argparse
import base64
import csv
import json
import os
import re
import time
import urllib.request
from typing import Any, Dict, List, Optional


def to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    s = s.replace(",", ".")
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def norm_symbol(s: Any) -> str:
    if s is None:
        return ""
    x = str(s).strip().upper().replace("/", "").replace("-", "")
    x = re.sub(r"[^A-Z0-9]", "", x)
    return x


def norm_direction(s: Any) -> str:
    if s is None:
        return ""
    x = str(s).strip().lower()
    if x in {"long", "buy", "up", "bull", "bullish"}:
        return "long"
    if x in {"short", "sell", "down", "bear", "bearish"}:
        return "short"
    return ""


def ollama_chat(model: str, messages: List[Dict[str, Any]], fmt_json: bool = False, timeout: int = 180) -> str:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
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
    return data.get("message", {}).get("content", "")


def safe_json_loads(s: str) -> Dict[str, Any]:
    s = s.strip()
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


def extract_with_vlm(image_path: str, model: str) -> Dict[str, Any]:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")

    sys_prompt = (
        "You extract crypto trading signal info from chart screenshots. "
        "Return JSON only."
    )
    user_prompt = (
        "From this image, extract fields if visible: "
        "symbol, direction(long/short), entry_min, stop, tp1, timeframe. "
        "If missing, use null. "
        "JSON schema: "
        "{\"symbol\":string|null,\"direction\":\"long\"|\"short\"|null,"
        "\"entry_min\":number|null,\"stop\":number|null,\"tp1\":number|null,"
        "\"timeframe\":string|null,\"confidence\":number|null}."
    )
    content = ollama_chat(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt, "images": [b64]},
        ],
        fmt_json=True,
        timeout=240,
    )
    return safe_json_loads(content)


def refine_with_text_model(raw_obj: Dict[str, Any], model: str) -> Dict[str, Any]:
    prompt = (
        "Normalize this JSON trading signal extraction. Keep only keys: "
        "symbol,direction,entry_min,stop,tp1,timeframe,confidence. "
        "Rules: symbol uppercase no separators (e.g. BTCUSDT), direction only long/short/null, "
        "numeric fields are numbers or null. Return JSON only.\n\n"
        + json.dumps(raw_obj, ensure_ascii=False)
    )
    content = ollama_chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        fmt_json=True,
        timeout=120,
    )
    out = safe_json_loads(content)
    return out if out else raw_obj


def run(args: argparse.Namespace) -> None:
    rows: List[Dict[str, Any]] = []
    with open(args.input_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    out_rows: List[Dict[str, Any]] = []
    processed = 0
    for r in rows:
        photo = r.get("photo_path", "")
        if not photo or not os.path.isfile(photo):
            continue

        cap_symbol = norm_symbol(r.get("cap_symbol"))
        cap_direction = norm_direction(r.get("cap_direction"))

        try:
            raw = extract_with_vlm(photo, args.vlm_model)
            final = refine_with_text_model(raw, args.text_model) if args.use_refiner else raw
        except Exception as e:
            final = {"error": str(e)}

        vlm_symbol = norm_symbol(final.get("symbol"))
        vlm_direction = norm_direction(final.get("direction"))
        vlm_entry_min = to_float(final.get("entry_min"))
        vlm_stop = to_float(final.get("stop"))
        vlm_tp1 = to_float(final.get("tp1"))
        vlm_timeframe = str(final.get("timeframe") or "").strip()
        vlm_conf = to_float(final.get("confidence"))

        symbol_agree = int(bool(cap_symbol and vlm_symbol and cap_symbol == vlm_symbol))
        direction_agree = int(bool(cap_direction and vlm_direction and cap_direction == vlm_direction))

        out_rows.append(
            {
                "timestamp_utc": r.get("timestamp_utc", ""),
                "photo_path": photo,
                "cap_symbol": cap_symbol,
                "vlm_symbol": vlm_symbol,
                "symbol_agree": symbol_agree,
                "cap_direction": cap_direction,
                "vlm_direction": vlm_direction,
                "direction_agree": direction_agree,
                "cap_entry_min": r.get("cap_entry_min", ""),
                "vlm_entry_min": "" if vlm_entry_min is None else vlm_entry_min,
                "cap_stop": r.get("cap_stop", ""),
                "vlm_stop": "" if vlm_stop is None else vlm_stop,
                "cap_tp1": r.get("cap_tp1", ""),
                "vlm_tp1": "" if vlm_tp1 is None else vlm_tp1,
                "vlm_timeframe": vlm_timeframe,
                "vlm_confidence": "" if vlm_conf is None else vlm_conf,
                "raw_json": json.dumps(final, ensure_ascii=False),
            }
        )

        processed += 1
        if processed % 10 == 0:
            print(f"processed {processed}/{len(rows)}")

        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = list(out_rows[0].keys()) if out_rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    n = len(out_rows)
    cap_symbol_present = sum(1 for x in out_rows if x["cap_symbol"]) if n else 0
    vlm_symbol_present = sum(1 for x in out_rows if x["vlm_symbol"]) if n else 0
    cap_dir_present = sum(1 for x in out_rows if x["cap_direction"]) if n else 0
    vlm_dir_present = sum(1 for x in out_rows if x["vlm_direction"]) if n else 0
    both_symbol = [x for x in out_rows if x["cap_symbol"] and x["vlm_symbol"]]
    both_dir = [x for x in out_rows if x["cap_direction"] and x["vlm_direction"]]

    summary = {
        "sample_size": n,
        "caption_symbol_present_rate": round(cap_symbol_present / n, 4) if n else 0.0,
        "vlm_symbol_present_rate": round(vlm_symbol_present / n, 4) if n else 0.0,
        "symbol_agreement_rate_on_both": round(sum(x["symbol_agree"] for x in both_symbol) / len(both_symbol), 4)
        if both_symbol
        else 0.0,
        "caption_direction_present_rate": round(cap_dir_present / n, 4) if n else 0.0,
        "vlm_direction_present_rate": round(vlm_dir_present / n, 4) if n else 0.0,
        "direction_agreement_rate_on_both": round(sum(x["direction_agree"] for x in both_dir) / len(both_dir), 4)
        if both_dir
        else 0.0,
        "with_refiner": bool(args.use_refiner),
        "vlm_model": args.vlm_model,
        "text_model": args.text_model if args.use_refiner else None,
    }

    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-csv",
        default="/home/vitamind/my_project/model6/reports/ocr_qa_pilot_100.csv",
    )
    ap.add_argument(
        "--output-csv",
        default="/home/vitamind/my_project/model6/reports/vlm_qa_pilot_100.csv",
    )
    ap.add_argument(
        "--summary-json",
        default="/home/vitamind/my_project/model6/reports/vlm_qa_pilot_100_summary.json",
    )
    ap.add_argument("--vlm-model", default="qwen2.5vl:7b")
    ap.add_argument("--text-model", default="qwen2.5:7b")
    ap.add_argument("--use-refiner", action="store_true")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--sleep-ms", type=int, default=0)
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
