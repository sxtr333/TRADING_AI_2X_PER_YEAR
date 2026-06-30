#!/usr/bin/env python3
"""Build a training-ready dataset from labeled Telegram signals."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write_table(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
        return path
    try:
        df.to_parquet(path, index=False)
        return path
    except Exception:
        fallback = path.with_suffix(".csv")
        df.to_csv(fallback, index=False)
        return fallback


def _split_time(df: pd.DataFrame, train_frac: float, val_frac: float):
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train : n_train + n_val].copy()
    test = df.iloc[n_train + n_val :].copy()
    return train, val, test


def main() -> None:
    ap = argparse.ArgumentParser(description="Build final ML dataset from labeled signals")
    ap.add_argument("--labeled", required=True, help="signals_labeled parquet/csv")
    ap.add_argument("--output-dir", default="data/telegram/train_dataset")
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument(
        "--include-outcomes",
        default="win,loss",
        help="Comma-separated outcomes to keep. Example: win,loss,no_entry,insufficient,invalid",
    )
    ap.add_argument("--summary-json", default="reports/signal_dataset_summary.json")
    args = ap.parse_args()

    if args.train_frac <= 0 or args.val_frac < 0 or args.train_frac + args.val_frac >= 1:
        raise ValueError("Invalid split fractions")

    df = _read_table(Path(args.labeled))

    if "timestamp_utc" not in df.columns:
        raise ValueError("Expected timestamp_utc column")

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    keep_outcomes = [x.strip() for x in str(args.include_outcomes).split(",") if x.strip()]
    if keep_outcomes:
        df = df[df["outcome"].isin(keep_outcomes)].copy()
    if df.empty:
        raise ValueError(f"No rows left after outcome filter: {keep_outcomes}")

    # Basic engineered features.
    # Supervised target is available only for win/loss rows.
    # For intermediate QA datasets, target_win remains NaN.
    df["target_win"] = np.where(
        df["outcome"] == "win",
        1.0,
        np.where(df["outcome"] == "loss", 0.0, np.nan),
    )
    df["direction_long"] = (df["direction"].astype(str).str.lower() == "long").astype(np.int8)
    df["text_len"] = df["text"].astype(str).str.len()
    df["photo_count"] = df["photos"].fillna("").astype(str).str.split(";").map(lambda x: 0 if x == [""] else len([y for y in x if y]))

    entry_mid = (pd.to_numeric(df["entry_min"], errors="coerce") + pd.to_numeric(df["entry_max"], errors="coerce")) / 2.0
    stop = pd.to_numeric(df["stop"], errors="coerce")
    tp1 = pd.to_numeric(df["tp1"], errors="coerce")

    risk = (entry_mid - stop).abs()
    df["risk_abs"] = risk
    df["tp1_rr"] = np.where(risk > 0, (tp1 - entry_mid).abs() / risk, np.nan)
    df["entry_mid"] = entry_mid
    df["symbol"] = df.get("symbol", "UNKNOWN").fillna("UNKNOWN")

    feature_cols = [
        "timestamp_utc",
        "symbol",
        "direction_long",
        "parse_confidence",
        "text_len",
        "photo_count",
        "entry_mid",
        "risk_abs",
        "tp1_rr",
        "realized_rr",
        "target_win",
    ]

    model_df = df[feature_cols].copy()

    train_df, val_df, test_df = _split_time(model_df, args.train_frac, args.val_frac)

    out_dir = Path(args.output_dir)
    full_path = _write_table(model_df, out_dir / "full.parquet")
    train_path = _write_table(train_df, out_dir / "train.parquet")
    val_path = _write_table(val_df, out_dir / "val.parquet")
    test_path = _write_table(test_df, out_dir / "test.parquet")

    summary = {
        "rows_full": int(len(model_df)),
        "rows_train": int(len(train_df)),
        "rows_val": int(len(val_df)),
        "rows_test": int(len(test_df)),
        "outcome_counts": df["outcome"].value_counts(dropna=False).to_dict(),
        "labeled_target_rows": int(model_df["target_win"].notna().sum()),
        "win_rate_full": float(model_df["target_win"].dropna().mean()) if model_df["target_win"].notna().any() else float("nan"),
        "symbol_top10": model_df["symbol"].value_counts().head(10).to_dict(),
        "files": {
            "full": str(full_path),
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
        },
    }

    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[dataset] full={len(model_df)} train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print(f"[dataset] full_file={full_path}")
    print(f"[dataset] summary={summary_path}")


if __name__ == "__main__":
    main()
