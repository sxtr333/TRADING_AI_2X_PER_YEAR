#!/usr/bin/env python3
"""
Merge and deduplicate news parquet files.
Dedup priority:
  1) url
  2) (title, published_at) for rows without url
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def _load_inputs(paths: List[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        if p.is_dir():
            files = sorted(p.glob("*.parquet"))
            for f in files:
                frames.append(pd.read_parquet(f))
        else:
            frames.append(pd.read_parquet(p))
    if not frames:
        raise FileNotFoundError("No parquet files found in inputs.")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="Input parquet files or directories")
    ap.add_argument("--output", required=True, help="Output parquet path")
    args = ap.parse_args()

    paths = [Path(p) for p in args.inputs]
    df = _load_inputs(paths)

    # Normalize columns
    for col in ("url", "title", "published_at"):
        if col not in df.columns:
            df[col] = None

    # Dedup by url first
    df["url"] = df["url"].astype("string")
    df = df.drop_duplicates(subset=["url"], keep="first")

    # For rows without url, dedup by title+published_at
    no_url = df["url"].isna() | (df["url"].str.len() == 0)
    if no_url.any():
        df_no_url = df[no_url].drop_duplicates(subset=["title", "published_at"], keep="first")
        df_has_url = df[~no_url]
        df = pd.concat([df_has_url, df_no_url], ignore_index=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    print(f"Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
