#!/usr/bin/env python3
"""Extract market-maker stop-hunt features from chart images via Ollama VLM.

Input: CSV or Parquet with at least `photo_path` column.
Output: Parquet with strict normalized MM fields + raw JSON/error.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = ROOT / "configs" / "mm_stop_hunt"
SYSTEM_PROMPT_PATH = CFG_DIR / "system_prompt.txt"
USER_PROMPT_PATH = CFG_DIR / "user_prompt_template.txt"


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _safe_json_loads(s: str) -> dict[str, Any]:
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


def _to_float01(v: Any) -> float:
    try:
        x = float(v)
    except Exception:
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in {"1", "true", "yes", "y"}


def _to_int_or_none(v: Any) -> int | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def _to_float_or_none(v: Any) -> float | None:
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


def _norm_enum(v: Any, allowed: set[str], default: str) -> str:
    s = str(v).strip().lower()
    return s if s in allowed else default


def _ollama_chat(model: str, messages: list[dict[str, Any]], timeout: int = 240) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": "json",
    }
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8", errors="ignore"))
    return data.get("message", {}).get("content", "")


def _extract_single(photo_path: str, model: str, system_prompt: str, user_prompt: str) -> tuple[dict[str, Any], str | None]:
    if not os.path.isfile(photo_path):
        return {}, "photo_file_missing"
    with open(photo_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    try:
        content = _ollama_chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt, "images": [b64]},
            ],
            timeout=300,
        )
        obj = _safe_json_loads(content)
        if not obj:
            return {}, "empty_or_invalid_json"
        return obj, None
    except Exception as e:
        return {}, f"ollama_error:{type(e).__name__}"


def _normalize_mm(raw: dict[str, Any]) -> dict[str, Any]:
    mm = {
        "mm_crowd_imbalance_side": _norm_enum(
            raw.get("crowd_imbalance_side"),
            {"long", "short", "balanced", "unknown"},
            "unknown",
        ),
        "mm_plan": _norm_enum(
            raw.get("mm_plan"),
            {"push_with_crowd", "sweep_against_crowd", "unclear"},
            "unclear",
        ),
        "mm_stop_hunt_long_prob": _to_float01(raw.get("stop_hunt_long_prob")),
        "mm_stop_hunt_short_prob": _to_float01(raw.get("stop_hunt_short_prob")),
        "mm_fake_reversal_prob": _to_float01(raw.get("fake_reversal_prob")),
        "mm_liquidity_sweep_side": _norm_enum(
            raw.get("liquidity_sweep_side"),
            {"above", "below", "both", "none", "unclear"},
            "unclear",
        ),
        "mm_reclaim_after_sweep": _to_bool(raw.get("reclaim_after_sweep")),
        "mm_invalid_level_price": _to_float_or_none(raw.get("invalid_level_price")),
        "mm_expected_move_horizon_candles": _to_int_or_none(raw.get("expected_move_horizon_candles")),
        "mm_confidence": _to_float01(raw.get("confidence")),
        "mm_reason_short": str(raw.get("reason_short") or "").strip()[:180],
    }
    return mm


def _read_input(path: str) -> pd.DataFrame:
    p = str(path).lower()
    if p.endswith(".parquet"):
        return pd.read_parquet(path)
    if p.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError("input must be .csv or .parquet")


def run(args: argparse.Namespace) -> None:
    df = _read_input(args.input)
    if "photo_path" not in df.columns:
        raise ValueError("input must contain 'photo_path' column")

    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()
    else:
        df = df.copy()

    system_prompt = _load_text(Path(args.system_prompt))
    user_prompt = _load_text(Path(args.user_prompt))

    rows: list[dict[str, Any]] = []
    total = len(df)
    for i, row in enumerate(df.itertuples(index=False), start=1):
        photo_path = str(getattr(row, "photo_path", "") or "")
        raw, err = _extract_single(
            photo_path=photo_path,
            model=args.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        mm = _normalize_mm(raw) if raw else {
            "mm_crowd_imbalance_side": "unknown",
            "mm_plan": "unclear",
            "mm_stop_hunt_long_prob": 0.0,
            "mm_stop_hunt_short_prob": 0.0,
            "mm_fake_reversal_prob": 0.0,
            "mm_liquidity_sweep_side": "unclear",
            "mm_reclaim_after_sweep": False,
            "mm_invalid_level_price": None,
            "mm_expected_move_horizon_candles": None,
            "mm_confidence": 0.0,
            "mm_reason_short": "",
        }
        out = {
            "photo_path": photo_path,
            "mm_error": err,
            "mm_raw_json": json.dumps(raw, ensure_ascii=False) if raw else "",
            **mm,
        }
        # Keep common identifiers if present.
        for key in ("timestamp_utc", "timestamp", "ts", "symbol", "channel", "message_id"):
            if hasattr(row, key):
                out[key] = getattr(row, key)
        rows.append(out)
        if i % max(1, args.log_every) == 0:
            print(f"[mm-stop-hunt] processed {i}/{total}")

    out_df = pd.DataFrame(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    summary = {
        "rows_input": int(total),
        "rows_output": int(len(out_df)),
        "rows_with_error": int(out_df["mm_error"].notna().sum()) if "mm_error" in out_df.columns else 0,
        "error_rate": float((out_df["mm_error"].notna().sum() / len(out_df)) if len(out_df) else 0.0),
        "model": args.model,
        "input": args.input,
        "output": args.output,
    }
    if args.summary_json:
        sp = Path(args.summary_json)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved summary: {sp}")
    print(f"saved parquet: {out_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract MM stop-hunt features from chart screenshots with Qwen2.5-VL.")
    ap.add_argument("--input", required=True, help="Input .csv or .parquet with photo_path column.")
    ap.add_argument("--output", default="data/telegram/vision_mm_stop_hunt.parquet")
    ap.add_argument("--summary-json", default="reports/vision_mm_stop_hunt_summary.json")
    ap.add_argument("--model", default="qwen2.5vl:7b")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--system-prompt", default=str(SYSTEM_PROMPT_PATH))
    ap.add_argument("--user-prompt", default=str(USER_PROMPT_PATH))
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
