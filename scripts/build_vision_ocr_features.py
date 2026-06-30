#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageStat


PRICE_RE = re.compile(r"(?<!\d)(\d{1,6}(?:[.,]\d{1,6})?)(?!\d)")


def first_photo_path(export_root: str, photos: str) -> str:
    first = str(photos or "").split(";")[0].strip()
    if not first:
        return ""
    return str(export_root).rstrip("/") + "/photos/" + first


def ocr_text(path: str, timeout_sec: int = 15) -> str:
    try:
        proc = subprocess.run(
            ["tesseract", path, "stdout", "--psm", "6"],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        return (proc.stdout or "").strip()
    except Exception:
        return ""


def image_feats(path: str) -> Dict[str, float]:
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return {
            "img_w": 0.0,
            "img_h": 0.0,
            "img_aspect": 0.0,
            "gray_mean": 0.0,
            "gray_std": 0.0,
            "edge_mean": 0.0,
            "edge_std": 0.0,
        }
    w, h = img.size
    g = img.convert("L")
    stat = ImageStat.Stat(g)
    edge = g.filter(ImageFilter.FIND_EDGES)
    estat = ImageStat.Stat(edge)
    return {
        "img_w": float(w),
        "img_h": float(h),
        "img_aspect": float(w / h) if h else 0.0,
        "gray_mean": float(stat.mean[0]) if stat.mean else 0.0,
        "gray_std": float(stat.stddev[0]) if stat.stddev else 0.0,
        "edge_mean": float(estat.mean[0]) if estat.mean else 0.0,
        "edge_std": float(estat.stddev[0]) if estat.stddev else 0.0,
    }


def text_feats(t: str) -> Dict[str, float]:
    s = (t or "").lower()
    prices = PRICE_RE.findall(s)
    digits = sum(ch.isdigit() for ch in s)
    letters = sum(ch.isalpha() for ch in s)
    return {
        "ocr_len": float(len(s)),
        "ocr_digit_count": float(digits),
        "ocr_letter_count": float(letters),
        "ocr_price_count": float(len(prices)),
        "ocr_has_usdt": float(1 if "usdt" in s else 0),
        "ocr_has_long": float(1 if ("long" in s or "лонг" in s) else 0),
        "ocr_has_short": float(1 if ("short" in s or "шорт" in s) else 0),
        "ocr_has_tp": float(1 if ("tp" in s or "тейк" in s) else 0),
        "ocr_has_sl": float(1 if ("sl" in s or "stop" in s or "стоп" in s) else 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build fast vision features (tesseract + image stats)")
    ap.add_argument("--signals", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--summary-json", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--progress-every", type=int, default=100)
    args = ap.parse_args()

    df = pd.read_parquet(args.signals).copy()
    df["photo_path"] = [
        first_photo_path(str(er), str(ph)) for er, ph in zip(df["export_root"].fillna(""), df["photos"].fillna(""))
    ]
    df = df[df["photo_path"].astype(str).str.len() > 0].copy()
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    rows: List[Dict[str, Any]] = []
    miss = 0
    for i, r in enumerate(df.itertuples(index=False), 1):
        p = str(getattr(r, "photo_path"))
        rec = {
            "message_id": str(getattr(r, "message_id")),
            "timestamp_utc": str(getattr(r, "timestamp_utc")),
            "photo_path": p,
        }
        if not Path(p).is_file():
            miss += 1
            rec.update({k: 0.0 for k in [
                "img_w", "img_h", "img_aspect", "gray_mean", "gray_std", "edge_mean", "edge_std",
                "ocr_len", "ocr_digit_count", "ocr_letter_count", "ocr_price_count",
                "ocr_has_usdt", "ocr_has_long", "ocr_has_short", "ocr_has_tp", "ocr_has_sl"
            ]})
            rec["photo_exists"] = 0.0
            rows.append(rec)
            continue

        rec["photo_exists"] = 1.0
        rec.update(image_feats(p))
        ocr = ocr_text(p)
        rec.update(text_feats(ocr))
        rows.append(rec)
        if args.progress_every > 0 and i % args.progress_every == 0:
            print(f"processed {i}/{len(df)}")

    out_df = pd.DataFrame(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    summary = {
        "signals": args.signals,
        "rows_input": int(len(df)),
        "rows_features": int(len(out_df)),
        "photo_missing": int(miss),
        "output": str(out_path),
    }
    sp = Path(args.summary_json)
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

