#!/usr/bin/env python3
"""Train a robust quality-gate model (bad/neutral/good) with scikit-learn."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


def build_text(df: pd.DataFrame) -> pd.Series:
    symbol = df.get("symbol", "").fillna("").astype(str)
    direction = df.get("direction", "").fillna("").astype(str)
    text = df.get("text", "").fillna("").astype(str)
    return (symbol + " " + direction + " " + text).str.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Train quality gate model with TF-IDF + LogisticRegression")
    ap.add_argument("--data", required=True, help="Parquet with quality_label/timestamp_utc/text/symbol/direction")
    ap.add_argument("--model-out", required=True, help="Output .joblib file")
    ap.add_argument("--summary-json", required=True, help="Output summary json")
    ap.add_argument("--train-frac", type=float, default=0.75)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--max-features", type=int, default=20000)
    ap.add_argument("--ngram-max", type=int, default=2)
    ap.add_argument("--c", type=float, default=1.0)
    ap.add_argument("--max-iter", type=int, default=3000)
    args = ap.parse_args()

    if args.train_frac <= 0 or args.val_frac <= 0 or (args.train_frac + args.val_frac) >= 1:
        raise ValueError("Invalid split fractions")

    df = pd.read_parquet(args.data).copy()
    req = {"quality_label", "timestamp_utc"}
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}")

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df[df["timestamp_utc"].notna()].copy()

    labels = ["bad", "neutral", "good"]
    label_to_id = {k: i for i, k in enumerate(labels)}
    df = df[df["quality_label"].isin(labels)].copy()
    if df.empty:
        raise ValueError("No rows after quality_label filter")
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    df["text_in"] = build_text(df)
    df["direction_long"] = (df.get("direction", "").astype(str).str.lower() == "long").astype(np.float32)
    df["parse_confidence"] = pd.to_numeric(df.get("parse_confidence", 0.0), errors="coerce").fillna(0.0).astype(np.float32)
    df["text_len"] = df["text_in"].astype(str).str.len().astype(np.float32)
    df["is_candidate_f"] = pd.to_numeric(df.get("is_candidate", False), errors="coerce").fillna(0.0).astype(np.float32)
    df["y"] = df["quality_label"].map(label_to_id).astype(np.int32)

    n = len(df)
    n_train = int(n * args.train_frac)
    n_val = int(n * args.val_frac)
    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train : n_train + n_val].copy()
    test = df.iloc[n_train + n_val :].copy()
    if min(len(train), len(val), len(test)) < 20:
        raise ValueError("Split too small for stable training/eval")

    tfidf = TfidfVectorizer(
        max_features=int(args.max_features),
        ngram_range=(1, int(args.ngram_max)),
        lowercase=True,
    )

    x_text_train = tfidf.fit_transform(train["text_in"].astype(str))
    x_text_val = tfidf.transform(val["text_in"].astype(str))
    x_text_test = tfidf.transform(test["text_in"].astype(str))

    num_cols = ["direction_long", "parse_confidence", "text_len", "is_candidate_f"]
    x_num_train = csr_matrix(train[num_cols].to_numpy(dtype=np.float32))
    x_num_val = csr_matrix(val[num_cols].to_numpy(dtype=np.float32))
    x_num_test = csr_matrix(test[num_cols].to_numpy(dtype=np.float32))

    x_train = hstack([x_text_train, x_num_train], format="csr")
    x_val = hstack([x_text_val, x_num_val], format="csr")
    x_test = hstack([x_text_test, x_num_test], format="csr")

    y_train = train["y"].to_numpy(dtype=np.int32)
    y_val = val["y"].to_numpy(dtype=np.int32)
    y_test = test["y"].to_numpy(dtype=np.int32)

    clf = LogisticRegression(
        C=float(args.c),
        max_iter=int(args.max_iter),
        class_weight="balanced",
        solver="lbfgs",
        n_jobs=1,
        random_state=42,
    )
    clf.fit(x_train, y_train)

    val_pred = clf.predict(x_val)
    test_pred = clf.predict(x_test)
    conf = confusion_matrix(y_test, test_pred, labels=[0, 1, 2]).tolist()

    rep = classification_report(
        y_test,
        test_pred,
        labels=[0, 1, 2],
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_type": "tfidf_logreg_quality_gate",
            "labels": labels,
            "vectorizer": tfidf,
            "classifier": clf,
            "num_cols": num_cols,
        },
        model_out,
    )

    summary = {
        "data": args.data,
        "rows_total": int(len(df)),
        "split_rows": {"train": int(len(train)), "val": int(len(val)), "test": int(len(test))},
        "class_counts_total": df["quality_label"].value_counts(dropna=False).to_dict(),
        "val_acc": float((val_pred == y_val).mean()),
        "test_acc": float((test_pred == y_test).mean()),
        "confusion_matrix_rows_true_cols_pred_bad_neutral_good": conf,
        "classification_report_test": rep,
        "model_out": str(model_out),
    }

    s = Path(args.summary_json)
    s.parent.mkdir(parents=True, exist_ok=True)
    s.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
