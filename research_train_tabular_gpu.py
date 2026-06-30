#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)


EXCLUDE_PREFIXES = (
    "target_",
    "label_",
    "tb_",
    "y_true_",
    "y_pred_",
    "future_",
    "meta_label_",
    "meta_y_",
)
EXCLUDE_COLS = {
    "timestamp",
    "label_3cls",
    "label_3cls_vf",
    "label_3cls_vf2",
    "tb_label",
    "tb_tth",
    "y_true_cls",
    "y_pred_cls",
}


def parse_ts(s: str) -> pd.Timestamp:
    return pd.to_datetime(s, utc=True)


def pick_feature_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c in EXCLUDE_COLS:
            continue
        if any(c.startswith(p) for p in EXCLUDE_PREFIXES):
            continue
        if df[c].dtype.kind in "ifb":
            cols.append(c)
    if not cols:
        raise ValueError("No numeric features found.")
    return cols


def make_label(df: pd.DataFrame, target_col: str) -> np.ndarray:
    if target_col == "tb_label":
        y = df[target_col].astype(int).to_numpy()
        if set(np.unique(y).tolist()) - {0, 1, 2}:
            raise ValueError(f"Unexpected tb_label values: {sorted(np.unique(y).tolist())}")
        return y
    if target_col == "label_3cls":
        y = df[target_col].astype(int).to_numpy()
        return np.where(y < 0, 0, np.where(y > 0, 2, 1)).astype(np.int32)
    raise ValueError(f"Unsupported target_col={target_col}")


def build_splits(df: pd.DataFrame, seq_len: int, train_end: str, val_end: str, target_horizon: int):
    n = len(df)
    n_train = int(df.index[df["timestamp"] <= parse_ts(train_end)].max()) + 1
    val_end_idx = int(df.index[df["timestamp"] <= parse_ts(val_end)].max()) + 1

    min_end = seq_len - 1
    hgap = max(0, int(target_horizon))
    train_max = max(min_end, n_train - hgap)
    val_max = max(min_end, val_end_idx - hgap)
    test_max = max(min_end, n - hgap)

    train_idx = np.arange(min_end, train_max, dtype=np.int64)
    val_idx = np.arange(max(n_train, min_end), val_max, dtype=np.int64)
    test_idx = np.arange(max(val_end_idx, min_end), test_max, dtype=np.int64)

    if len(train_idx) < 100 or len(val_idx) < 50 or len(test_idx) < 50:
        raise ValueError(f"Split too small: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    return train_idx, val_idx, test_idx


def train_catboost(X_train, y_train, X_val, y_val):
    from catboost import CatBoostClassifier

    model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="MultiClass",
        task_type="GPU",
        devices="0",
        random_seed=42,
        iterations=2500,
        depth=8,
        learning_rate=0.03,
        l2_leaf_reg=8.0,
        bootstrap_type="Bernoulli",
        subsample=0.8,
        verbose=100,
        early_stopping_rounds=150,
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    return model


def train_xgboost(X_train, y_train, X_val, y_val):
    from xgboost import XGBClassifier

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        tree_method="hist",
        device="cuda",
        max_depth=8,
        n_estimators=2200,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=8.0,
        reg_alpha=0.0,
        min_child_weight=4,
        random_state=42,
        eval_metric="mlogloss",
        early_stopping_rounds=150,
        verbosity=1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
    booster_cfg = model.get_booster().save_config()
    if '"device":"cuda"' not in booster_cfg and '"device":"cuda:0"' not in booster_cfg:
        raise RuntimeError("XGBoost did not stay on CUDA device.")
    return model


def feature_importance_df(model_name: str, model, feature_names: list[str]) -> pd.DataFrame:
    if model_name == "catboost":
        vals = model.get_feature_importance()
    else:
        vals = model.feature_importances_
    imp = pd.DataFrame({"feature": feature_names, "importance": vals})
    return imp.sort_values("importance", ascending=False).reset_index(drop=True)


def evaluate(y_true: np.ndarray, proba: np.ndarray) -> dict:
    pred = np.argmax(proba, axis=1)
    labels = [0, 1, 2]
    report = classification_report(y_true, pred, labels=labels, output_dict=True, zero_division=0)
    metrics = {
        "acc": float(accuracy_score(y_true, pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted")),
        "logloss": float(log_loss(y_true, proba, labels=labels)),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=labels).tolist(),
        "report": report,
    }
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--target-col", default="tb_label", choices=["tb_label", "label_3cls"])
    ap.add_argument("--model", required=True, choices=["catboost", "xgboost"])
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--train-end", required=True)
    ap.add_argument("--val-end", required=True)
    ap.add_argument("--target-horizon", type=int, default=20)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.features)
    if "timestamp" not in df.columns:
        raise ValueError("timestamp column is required")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
    df = df.sort_values("timestamp").reset_index(drop=True)

    feature_cols = pick_feature_cols(df)
    y = make_label(df, args.target_col)
    train_idx, val_idx, test_idx = build_splits(df, args.seq_len, args.train_end, args.val_end, args.target_horizon)

    X = df[feature_cols].to_numpy(dtype=np.float32)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"[data] rows={len(df)} features={len(feature_cols)}")
    print(f"[split] train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    print(f"[label] counts_train={dict(zip(*np.unique(y_train, return_counts=True)))}")

    if args.model == "catboost":
        model = train_catboost(X_train, y_train, X_val, y_val)
        model_path = out_dir / "catboost_pattern_tb_label_3cls.cbm"
        model.save_model(str(model_path))
    else:
        model = train_xgboost(X_train, y_train, X_val, y_val)
        model_path = out_dir / "xgboost_pattern_tb_label_3cls.json"
        model.save_model(str(model_path))

    val_proba = model.predict_proba(X_val)
    test_proba = model.predict_proba(X_test)
    val_metrics = evaluate(y_val, val_proba)
    test_metrics = evaluate(y_test, test_proba)

    importance = feature_importance_df(args.model, model, feature_cols)
    importance_path = out_dir / f"{args.model}_feature_importance.csv"
    importance.to_csv(importance_path, index=False)

    metrics = {
        "model": args.model,
        "target_col": args.target_col,
        "features_path": args.features,
        "seq_len_guard": args.seq_len,
        "train_end": args.train_end,
        "val_end": args.val_end,
        "target_horizon": args.target_horizon,
        "n_features": len(feature_cols),
        "top_features": importance.head(20).to_dict(orient="records"),
        "val": val_metrics,
        "test": test_metrics,
    }
    metrics_path = out_dir / f"{args.model}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))

    print(json.dumps({
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "importance_path": str(importance_path),
        "test_macro_f1": test_metrics["macro_f1"],
        "test_balanced_acc": test_metrics["balanced_acc"],
        "test_acc": test_metrics["acc"],
        "test_logloss": test_metrics["logloss"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
