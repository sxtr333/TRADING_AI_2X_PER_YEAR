#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


NEWS_COLS = ["news_count", "news_sentiment", "news_shock", "news_votes", "news_missing"]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Merge legacy news aggregate columns into a fresher features parquet."
    )
    ap.add_argument("--base-features", required=True, help="Fresh features parquet.")
    ap.add_argument("--legacy-features", required=True, help="Older compatible features parquet with news aggregates.")
    ap.add_argument("--output", required=True, help="Output parquet path.")
    args = ap.parse_args()

    base_path = Path(args.base_features)
    legacy_path = Path(args.legacy_features)
    out_path = Path(args.output)

    base = pd.read_parquet(base_path)
    legacy = pd.read_parquet(legacy_path, columns=["timestamp", *NEWS_COLS])

    base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True)
    legacy["timestamp"] = pd.to_datetime(legacy["timestamp"], utc=True)

    merged = base.merge(legacy, on="timestamp", how="left", suffixes=("", "_legacy"))

    # If the fresh base already contains any of these fields, prefer existing non-null values.
    for col in NEWS_COLS:
        legacy_col = f"{col}_legacy"
        if legacy_col in merged.columns and col in merged.columns:
            merged[col] = merged[col].where(merged[col].notna(), merged[legacy_col])
            merged = merged.drop(columns=[legacy_col])
        elif legacy_col in merged.columns:
            merged = merged.rename(columns={legacy_col: col})

    # Tail beyond legacy coverage is a real missing-news regime.
    merged["news_count"] = merged["news_count"].fillna(0.0)
    merged["news_sentiment"] = merged["news_sentiment"].fillna(0.0)
    merged["news_shock"] = merged["news_shock"].fillna(0.0)
    merged["news_votes"] = merged["news_votes"].fillna(0.0)
    merged["news_missing"] = merged["news_missing"].fillna(1.0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)

    print(f"Saved: {out_path}")
    print(f"Rows: {len(merged)}")
    print(f"Range: {merged['timestamp'].min()} -> {merged['timestamp'].max()}")
    tail_missing = merged.loc[merged["news_missing"] >= 1.0, "timestamp"].min()
    print(f"First missing-news timestamp: {tail_missing}")


if __name__ == "__main__":
    main()
