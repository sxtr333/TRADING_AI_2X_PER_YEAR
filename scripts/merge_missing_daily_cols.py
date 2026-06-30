#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base parquet to enrich")
    ap.add_argument("--ref-csv", required=True, help="Reference CSV with full columns")
    ap.add_argument("--stats", required=True, help="Model stats .npz (to get feature_names)")
    ap.add_argument("--out", required=True, help="Output parquet")
    args = ap.parse_args()

    base_cols = set(pq.read_schema(args.base).names)

    stats = np.load(args.stats, allow_pickle=True)
    feat_names = stats["feature_names"]
    if isinstance(feat_names, np.ndarray):
        feat_names = feat_names.tolist()
    feat_names = [str(x) for x in feat_names]

    missing = [c for c in feat_names if c not in base_cols]
    if not missing:
        print("[ok] no missing columns")
        df = pd.read_parquet(args.base)
        df.to_parquet(args.out, index=False)
        return

    # Always include timestamp for merge
    usecols = ["timestamp"] + missing
    ref = pd.read_csv(args.ref_csv, usecols=usecols)
    ref["timestamp"] = pd.to_datetime(ref["timestamp"], utc=True)

    df = pd.read_parquet(args.base)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Merge missing columns
    df = df.merge(ref, on="timestamp", how="left")

    # Fill missing with 0 and mark daily_missing_any if present/missing
    miss_mask = df[missing].isna().any(axis=1)
    if "daily_missing_any" in missing:
        # daily_missing_any was imported; fill NaN with 1 for missing rows
        df["daily_missing_any"] = df["daily_missing_any"].fillna(0.0)
        df.loc[miss_mask, "daily_missing_any"] = 1.0
    else:
        # If daily_missing_any not in missing and not in base, create it
        if "daily_missing_any" not in df.columns:
            df["daily_missing_any"] = miss_mask.astype(float)

    df[missing] = df[missing].fillna(0.0)

    df.to_parquet(args.out, index=False)
    print(f"[ok] wrote {args.out} with {len(missing)} added columns")


if __name__ == "__main__":
    main()
