"""
model6/evaluate_keras.py

Оценка Keras-модели. Совместимо с train_keras.py выше:
- тот же порядок фич (default_feature_list)
- нормализация из --stats (ОЧЕНЬ важно, если модель обучалась на нормализованных данных)
- окна строятся так же как на обучении: window(seq_len+1) -> x[:-1], label at i+seq_len

Ключевые флаги:
- --cls-only: считать только классификацию
- по умолчанию оценивает ТОЛЬКО события (label_3cls != 0) — как и в обучении
- --include-neutral: считать up vs (down+neutral) (диагностика, но это другой таргет)
- --invert-auto: если ROC-AUC<0.5, делаем p_up := 1 - p_up
- --auto-threshold: подбираем порог по --metric
- --start: фильтруем результаты по timestamp (history до start всё равно используется)

Пример:
  python model6/evaluate_keras.py \
    --features model6/data/BTCUSDT_15m_features_tb.parquet \
    --model model6/model_15m_bin.keras \
    --stats model6/norm_stats_15m_bin.npz \
    --seq-len 256 --batch-size 512 \
    --cls-only --invert-auto --auto-threshold \
    --start "2025-08-01 15:15:00+00:00" \
    --out model6/metrics_15m_bin_TEST.csv
"""

from __future__ import annotations

import argparse
import math
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from trading_keras_core import (
    AttentionPooling,
    CLSToken,
    DropPath,
    LayerScale,
    SqueezeExcite,
    TemporalBlock,
    default_feature_list,
)


def _as_utc_timestamp_series(s: pd.Series) -> pd.Series:
    ts = pd.to_datetime(s, utc=True, errors="coerce")
    if ts.isna().any():
        bad = s[ts.isna()].head(5).tolist()
        raise ValueError(f"Failed to parse some timestamps (showing up to 5): {bad}")
    return ts


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC without sklearn. Returns nan if undefined."""
    y_true = np.asarray(y_true).astype(np.int32)
    y_score = np.asarray(y_score).astype(np.float64)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)

    # average ranks for ties
    sorted_scores = y_score[order]
    dif = np.diff(sorted_scores)
    tie_starts = np.where(dif != 0)[0] + 1
    tie_starts = np.r_[0, tie_starts, len(y_score)]
    for i in range(len(tie_starts) - 1):
        a, b = tie_starts[i], tie_starts[i + 1]
        if b - a > 1:
            avg = (a + 1 + b) / 2.0
            ranks[order[a:b]] = avg

    sum_ranks_pos = float(ranks[y_true == 1].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average precision (PR-AUC) without sklearn. Returns nan if undefined."""
    y_true = np.asarray(y_true).astype(np.int32)
    y_score = np.asarray(y_score).astype(np.float64)
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    cum_pos = np.cumsum(y_sorted == 1)
    idx = np.arange(1, len(y_true) + 1)
    precision = cum_pos / idx
    ap = float(precision[y_sorted == 1].sum() / n_pos)
    return ap


def spearmanr_fast(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman without scipy."""
    a = pd.Series(a).rank(method="average").to_numpy(dtype=np.float64)
    b = pd.Series(b).rank(method="average").to_numpy(dtype=np.float64)
    a -= a.mean()
    b -= b.mean()
    denom = float(np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()))
    if denom == 0:
        return float("nan")
    return float((a * b).sum() / denom)


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    y_true = np.asarray(y_true).astype(np.int32)
    y_pred = np.asarray(y_pred).astype(np.int32)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tn, fp, fn, tp


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0


def pick_threshold(p_up: np.ndarray, y_true: np.ndarray, metric: str = "f1_macro", num: int = 401) -> float:
    """Pick threshold by scanning quantiles of p_up distribution."""
    p = np.asarray(p_up, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.int32)

    thr_grid = np.unique(np.quantile(p, np.linspace(0.0, 1.0, num=num)))
    if len(thr_grid) == 0:
        return 0.5

    best_thr = float(thr_grid[len(thr_grid) // 2])
    best_score = -1e9

    for t in thr_grid:
        pred = (p >= t).astype(np.int32)
        tn, fp, fn, tp = confusion(y, pred)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        bacc = 0.5 * (tpr + tnr)
        acc = (tp + tn) / max(len(y), 1)
        f1_up = f1_from_counts(tp, fp, fn)
        f1_dn = f1_from_counts(tn, fn, fp)
        f1_macro = 0.5 * (f1_up + f1_dn)

        if metric == "f1_macro":
            score = f1_macro
        elif metric == "bacc":
            score = bacc
        elif metric == "acc":
            score = acc
        else:
            raise ValueError(f"Unknown metric for threshold search: {metric}")

        if score > best_score:
            best_score = score
            best_thr = float(t)

    return best_thr


def build_pred_dataset(feats: np.ndarray, seq_len: int, batch_size: int) -> tf.data.Dataset:
    """
    feats: (N, F)
    returns dataset yielding x: (B, seq_len, F) for N - seq_len samples.
    """
    feats_tf = tf.convert_to_tensor(feats, dtype=tf.float32)
    ds = tf.data.Dataset.from_tensor_slices(feats_tf)
    ds = ds.window(seq_len + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(seq_len + 1))
    ds = ds.map(lambda w: w[:-1], num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Keras model on prepared features parquet.")
    parser.add_argument("--features", required=True, help="Path to features parquet")
    parser.add_argument("--model", required=True, help="Path to saved model (.keras)")
    parser.add_argument("--stats", default=None, help="Path to norm_stats.npz saved by train_keras.py")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=512)

    parser.add_argument("--target-col", default="target_amp_abs")
    parser.add_argument("--label-col", default="label_3cls")

    parser.add_argument("--cls-only", action="store_true", help="Skip regression metrics")
    parser.add_argument("--include-neutral", action="store_true", help="Treat neutral as class 0 (up vs not-up).")

    parser.add_argument("--invert-auto", action="store_true", help="If ROC-AUC < 0.5, flip p_up := 1-p_up")
    parser.add_argument("--auto-threshold", action="store_true", help="Search best threshold for selected metric")
    parser.add_argument("--metric", choices=["f1_macro", "bacc", "acc"], default="f1_macro", help="Metric for threshold search")
    parser.add_argument("--start", default=None, help="Only keep outputs with timestamp >= this (ISO string)")

    parser.add_argument("--out", default="metrics.csv", help="Where to save per-row predictions csv")
    args = parser.parse_args()

    df = pd.read_parquet(args.features)
    if df.empty:
        raise ValueError("Features parquet is empty.")

    if "timestamp" not in df.columns:
        raise ValueError("No 'timestamp' column in parquet (needed for alignment/output).")

    df["timestamp"] = _as_utc_timestamp_series(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    feature_cols = list(default_feature_list())
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing[:10]} (total {len(missing)})")

    if args.target_col not in df.columns:
        raise ValueError(f"Target column '{args.target_col}' not found in parquet.")
    if args.label_col not in df.columns:
        raise ValueError(f"Label column '{args.label_col}' not found in parquet.")

    feats = df[feature_cols].to_numpy(dtype=np.float32)

    # Apply normalization if provided
    if args.stats:
        st = np.load(args.stats, allow_pickle=True)
        mean = st["mean"].astype(np.float32)
        std = st["std"].astype(np.float32)
        std = np.where(std == 0, 1.0, std).astype(np.float32)

        if "features" in st:
            saved = [str(x) for x in st["features"].tolist()]
            if saved != feature_cols:
                print("[WARN] Feature order in stats differs from current default_feature_list(). "
                      "If order mismatch — predictions will be wrong.")

        feats = (feats - mean.reshape(1, -1)) / std.reshape(1, -1)
    else:
        print("[WARN] --stats not provided. Если модель обучалась с нормализацией, результаты будут некорректны.")

    target = df[args.target_col].to_numpy(dtype=np.float32)
    label_3cls = df[args.label_col].to_numpy()

    # Same encoding as train_keras.py (binary):
    y_true_row = (label_3cls == 1).astype(np.int32)  # up=1 else 0
    w_row = (label_3cls != 0).astype(np.float32)      # events weight

    ts = df["timestamp"]
    full_ts_out = ts.iloc[args.seq_len :].reset_index(drop=True)
    full_y_true = y_true_row[args.seq_len :]
    full_w = w_row[args.seq_len :]
    full_target_out = target[args.seq_len :]

    if len(full_y_true) <= 0:
        raise ValueError("Not enough rows for given --seq-len.")

    # Start mask
    if args.start:
        start_ts = pd.to_datetime(args.start, utc=True, errors="raise")
        mask_start = (full_ts_out >= start_ts).to_numpy()
    else:
        mask_start = np.ones(len(full_ts_out), dtype=bool)

    # Predict on full sequence stream (then apply mask)
    ds = build_pred_dataset(feats, seq_len=args.seq_len, batch_size=args.batch_size)

    custom_objects = {
        "TemporalBlock": TemporalBlock,
        "SqueezeExcite": SqueezeExcite,
        "DropPath": DropPath,
        "LayerScale": LayerScale,
        "CLSToken": CLSToken,
        "AttentionPooling": AttentionPooling,
    }
    model = tf.keras.models.load_model(args.model, custom_objects=custom_objects, compile=False, safe_mode=False)
    preds = model.predict(ds, verbose=0)

    pred_price = None
    pred_cls = None

    if isinstance(preds, dict):
        pred_price = preds.get("price", None)
        pred_cls = preds.get("cls", None)
    elif isinstance(preds, (list, tuple)):
        if len(preds) >= 1:
            pred_price = preds[0]
        if len(preds) >= 2:
            pred_cls = preds[1]
    else:
        pred_price = preds

    if pred_price is not None:
        pred_price = np.asarray(pred_price).reshape(-1)
    if pred_cls is not None:
        pred_cls = np.asarray(pred_cls)

    n_expected = len(full_ts_out)
    if pred_price is not None and len(pred_price) != n_expected:
        raise ValueError(f"Pred length mismatch: pred_price={len(pred_price)} expected={n_expected}")
    if pred_cls is not None and pred_cls.shape[0] != n_expected:
        raise ValueError(f"Pred length mismatch: pred_cls={pred_cls.shape[0]} expected={n_expected}")

    # p_up
    if pred_cls is None:
        p_up = None
    else:
        if pred_cls.ndim == 2 and pred_cls.shape[1] >= 2:
            p_up = pred_cls[:, 1].astype(np.float64)
        else:
            p_up = pred_cls.reshape(-1).astype(np.float64)

    # Apply start mask
    full_ts_out = full_ts_out[mask_start].reset_index(drop=True)
    full_y_true = full_y_true[mask_start]
    full_w = full_w[mask_start]
    full_target_out = full_target_out[mask_start]
    if pred_price is not None:
        pred_price = pred_price[mask_start]
    if p_up is not None:
        p_up = p_up[mask_start]

    # Metrics mask:
    #  - default: only events (label_3cls != 0)
    #  - include-neutral: evaluate up vs not-up on ALL rows
    if args.include_neutral:
        mask_eval = np.ones_like(full_y_true, dtype=bool)
    else:
        mask_eval = full_w > 0

    ts_eval = full_ts_out[mask_eval].reset_index(drop=True)
    y_true = full_y_true[mask_eval]
    target_eval = full_target_out[mask_eval]
    pred_price_eval = pred_price[mask_eval] if pred_price is not None else None
    p_up_eval = p_up[mask_eval] if p_up is not None else None

    print(f"N= {len(y_true)}")
    if len(y_true) == 0:
        raise ValueError("After filtering there are 0 samples to evaluate.")

    # Regression (optional)
    if (not args.cls_only) and (pred_price_eval is not None):
        mae = float(np.mean(np.abs(pred_price_eval - target_eval)))
        mse = float(np.mean((pred_price_eval - target_eval) ** 2))
        var = float(np.var(target_eval))
        r2 = float("nan") if var == 0 else float(1.0 - mse / var)
        spear = spearmanr_fast(pred_price_eval, target_eval)
        print(f"REG: MAE={mae:.6f} MSE={mse:.6f} R2={r2:.4f} Spearman={spear:.4f}")

    # Classification
    if p_up_eval is not None:
        class_dist = {0: float((y_true == 0).mean()), 1: float((y_true == 1).mean())}
        print("class dist:", class_dist)
        print("baseline acc (majority)=", max(class_dist.values()))

        roc = roc_auc_score(y_true, p_up_eval)
        pr = average_precision(y_true, p_up_eval)

        if args.invert_auto and (not math.isnan(roc)):
            roc_inv = roc_auc_score(y_true, 1.0 - p_up_eval)
            if (not math.isnan(roc_inv)) and roc_inv > roc:
                print(f"[invert] ROC-AUC improved {roc:.4f} -> {roc_inv:.4f}, using p_up := 1-p_up")
                p_up_eval = 1.0 - p_up_eval
                roc = roc_inv
                pr = average_precision(y_true, p_up_eval)

        thr = 0.5
        if args.auto_threshold:
            thr = pick_threshold(p_up_eval, y_true, metric=args.metric)
            print(f"[auto-threshold] metric={args.metric} thr={thr:.6f}")

        y_pred = (p_up_eval >= thr).astype(np.int32)
        tn, fp, fn, tp = confusion(y_true, y_pred)

        acc = float((tp + tn) / len(y_true))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        bacc = float(0.5 * (tpr + tnr))
        prec_up = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec_up = tpr
        f1_up = f1_from_counts(tp, fp, fn)
        f1_dn = f1_from_counts(tn, fn, fp)
        f1_macro = float(0.5 * (f1_up + f1_dn))

        conf = np.abs(p_up_eval - 0.5)
        order_conf = np.argsort(-conf)
        k = max(1, int(0.10 * len(y_true)))
        hit_top10 = float((y_pred[order_conf[:k]] == y_true[order_conf[:k]]).mean())

        print(
            f"CLS: thr={thr:.6f} acc={acc:.4f} bacc={bacc:.4f} f1_macro={f1_macro:.4f} "
            f"prec_up={prec_up:.4f} rec_up={rec_up:.4f} PR_AUC={pr:.4f} ROC_AUC={roc:.4f} "
            f"hit_top10(conf)={hit_top10:.4f}"
        )
        print("confusion [ [tn,fp],[fn,tp] ] =")
        print(np.array([[tn, fp], [fn, tp]], dtype=np.int64))

        # Save per-row
        out_df = pd.DataFrame(
            {
                "timestamp": ts_eval.astype(str),
                "pred_amp": pred_price_eval if pred_price_eval is not None else np.nan,
                "target_amp": target_eval,
                "p_up": p_up_eval,
                "p_down": 1.0 - p_up_eval,
                "y_pred_cls": y_pred,
                "y_true_cls": y_true,
            }
        )
        out_df.to_csv(args.out, index=False)
        print(f"Saved predictions to {args.out}")

    else:
        out_df = pd.DataFrame(
            {
                "timestamp": ts_eval.astype(str),
                "pred_amp": pred_price_eval if pred_price_eval is not None else np.nan,
                "target_amp": target_eval,
            }
        )
        out_df.to_csv(args.out, index=False)
        print(f"Saved predictions to {args.out}")


if __name__ == "__main__":
    main()
