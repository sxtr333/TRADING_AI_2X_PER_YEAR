#!/usr/bin/env python3
"""Filter Telegram signal rows by hashtags and basic symbol blacklist."""

import argparse
import json
import re
from pathlib import Path
from typing import List

import pandas as pd

HASHTAG_RE = re.compile(r"(?i)#([a-z][a-z0-9_]{1,19})")

# Keep this list short and obvious; can be extended from CLI.
DEFAULT_BLACKLIST = {
    "PIPIN",
    "PIPI",
    "PEPE2",
    "BABYPEPE",
    "1000RATS",
    "1000FLOKI",
    "1000BONK",
    "1000PEPE",
}


def extract_tags(text: str) -> List[str]:
    vals = [m.upper() for m in HASHTAG_RE.findall(str(text or ""))]
    # preserve order, remove duplicates
    seen = set()
    out = []
    for v in vals:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def parse_blacklist(raw: str) -> set:
    extra = {x.strip().upper() for x in str(raw or "").split(",") if x.strip()}
    return set(DEFAULT_BLACKLIST) | extra


def main() -> None:
    ap = argparse.ArgumentParser(description="Filter Telegram rows by hashtag and photos")
    ap.add_argument("--input", required=True, help="Input parquet/csv from extract_signals_from_export")
    ap.add_argument("--output", required=True, help="Output parquet/csv")
    ap.add_argument("--summary-json", default="reports/hashtag_filter_summary.json")
    ap.add_argument(
        "--extra-blacklist",
        default="",
        help="Comma-separated extra symbols to drop (case-insensitive)",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    if in_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(in_path)
    else:
        df = pd.read_csv(in_path)

    req = {"text", "photos"}
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}")

    df = df.copy()
    df["hashtags"] = df["text"].map(extract_tags)
    df["hashtag_count"] = df["hashtags"].map(len)
    df["has_photo"] = df["photos"].fillna("").astype(str).str.strip().ne("")
    bl = parse_blacklist(args.extra_blacklist)

    # Keep posts that have at least one non-blacklisted hashtag.
    df["hashtags_kept"] = df["hashtags"].map(lambda xs: [x for x in xs if x not in bl])
    df["hashtags_dropped"] = df["hashtags"].map(lambda xs: [x for x in xs if x in bl])
    df["primary_tag"] = df["hashtags_kept"].map(lambda xs: xs[0] if xs else None)

    filtered = df[
        (df["has_photo"]) &
        (df["hashtag_count"] > 0) &
        (df["hashtags_kept"].map(len) > 0)
    ].copy()

    # If parser missed symbol, fill from primary hashtag.
    if "symbol" in filtered.columns:
        sym = filtered["symbol"].fillna("").astype(str).str.upper().str.strip()
        filtered["symbol"] = sym.where(sym.ne(""), filtered["primary_tag"])
    else:
        filtered["symbol"] = filtered["primary_tag"]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".csv":
        filtered.to_csv(out_path, index=False)
    else:
        filtered.to_parquet(out_path, index=False)

    top_tags = (
        pd.Series([t for row in filtered["hashtags_kept"] for t in row], dtype="object")
        .value_counts()
        .head(30)
        .to_dict()
    )
    summary = {
        "input_rows": int(len(df)),
        "output_rows": int(len(filtered)),
        "rows_with_photo": int(df["has_photo"].sum()),
        "rows_with_hashtag": int((df["hashtag_count"] > 0).sum()),
        "rows_dropped_blacklist_only": int(
            ((df["has_photo"]) & (df["hashtag_count"] > 0) & (df["hashtags_kept"].map(len) == 0)).sum()
        ),
        "blacklist": sorted(bl),
        "top_tags": top_tags,
        "output_file": str(out_path),
    }

    s_path = Path(args.summary_json)
    s_path.parent.mkdir(parents=True, exist_ok=True)
    s_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
