#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

# Ensure project root is on path when running from scripts/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from train_keras import apply_norm
from model_layers import RevIN, ITransformerBlock, TSMixerBlock, LastStep, DropPath


def load_stats(path: str) -> dict:
    s = np.load(path, allow_pickle=True)
    return {
        "feature_names": s["feature_names"].tolist(),
        "mean": s["mean"].astype(np.float32),
        "std": s["std"].astype(np.float32),
        "q_low": s["q_low"].astype(np.float32),
        "q_high": s["q_high"].astype(np.float32),
        "seq_len": int(s["seq_len"][0]) if "seq_len" in s else 256,
        "test_start_ts": str(s["test_start_ts"][0]) if "test_start_ts" in s else None,
    }


def build_dataset(X: np.ndarray, end_indices: np.ndarray, seq_len: int, batch_size: int):
    X_tf = tf.convert_to_tensor(X, dtype=tf.float32)
    end_indices = end_indices.astype(np.int64)
    ds = tf.data.Dataset.from_tensor_slices(end_indices)

    def map_fn(i):
        i = tf.cast(i, tf.int32)
        start = i - (seq_len - 1)
        x_seq = X_tf[start:i + 1]
        x_seq = tf.ensure_shape(x_seq, [seq_len, X.shape[1]])
        return x_seq

    ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds


def safe_smape(pred: np.ndarray, actual: np.ndarray) -> float:
    denom = np.abs(pred) + np.abs(actual)
    denom = np.where(denom < 1e-9, 1e-9, denom)
    return float(np.mean(2.0 * np.abs(pred - actual) / denom) * 100.0)

def _tail_stats(abs_err: np.ndarray) -> dict:
    if abs_err.size == 0:
        return {"p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {
        "p50": float(np.quantile(abs_err, 0.50)),
        "p90": float(np.quantile(abs_err, 0.90)),
        "p95": float(np.quantile(abs_err, 0.95)),
        "p99": float(np.quantile(abs_err, 0.99)),
        "max": float(np.max(abs_err)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--ensemble", required=True,
                    help="Comma-separated model:stats pairs (e.g. m1.keras:s1.npz,m2.keras:s2.npz)")
    ap.add_argument("--horizons", default="20,80,160")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--start", default=None, help="ISO start timestamp (UTC). If omitted, uses first stats test_start_ts.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    max_h = max(horizons)

    df = pd.read_parquet(args.features)
    if "timestamp" not in df.columns:
        raise ValueError("features parquet must include 'timestamp'")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    close = df["close"].to_numpy(np.float64)
    ts = df["timestamp"].to_numpy()

    pairs = [p.strip() for p in args.ensemble.split(",") if p.strip()]
    if not pairs:
        raise ValueError("No ensemble pairs provided.")

    # Start timestamp
    if args.start:
        start_ts = pd.to_datetime(args.start, utc=True)
    else:
        first_stats = load_stats(pairs[0].split(":", 1)[1])
        start_ts = pd.to_datetime(first_stats.get("test_start_ts") or df["timestamp"].iloc[0], utc=True)

    # Collect predictions per model
    model_preds = []
    end_indices_list = []

    custom_objects = {
        "RevIN": RevIN,
        "ITransformerBlock": ITransformerBlock,
        "TSMixerBlock": TSMixerBlock,
        "LastStep": LastStep,
        "DropPath": DropPath,
    }

    for pair in pairs:
        model_path, stats_path = pair.split(":", 1)
        stats = load_stats(stats_path)
        feature_names = stats["feature_names"]
        seq_len = stats["seq_len"]

        missing = [c for c in feature_names if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns for {model_path}: {missing}")

        X_raw = df[feature_names].to_numpy(np.float32)
        X = apply_norm(X_raw, stats)

        start_idx = int(np.searchsorted(df["timestamp"].to_numpy(), start_ts))
        start_idx = max(start_idx, seq_len - 1)
        end_idx = len(df) - max_h - 1
        if end_idx <= start_idx:
            raise ValueError("Not enough rows for evaluation window.")

        end_indices = np.arange(start_idx, end_idx + 1, dtype=np.int64)
        ds = build_dataset(X, end_indices, seq_len=seq_len, batch_size=args.batch_size)

        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False, safe_mode=False)
        preds = model.predict(ds, verbose=0)
        if not isinstance(preds, dict):
            raise ValueError(f"Model {model_path} must return dict with price_h{{h}} heads.")

        pred_by_h = {}
        for h in horizons:
            key = f"price_h{h}"
            if key in preds:
                arr = np.asarray(preds[key]).reshape(-1)
            elif "price" in preds and len(horizons) == 1:
                arr = np.asarray(preds["price"]).reshape(-1)
            else:
                raise ValueError(f"Missing prediction head {key} in {model_path}")
            if len(arr) != len(end_indices):
                raise ValueError(f"Pred length mismatch for {model_path} {key}")
            pred_by_h[h] = arr

        model_preds.append(pred_by_h)
        end_indices_list.append(end_indices)

    # Common indices for all models
    common = end_indices_list[0]
    for idx in end_indices_list[1:]:
        common = np.intersect1d(common, idx, assume_unique=False)

    if len(common) == 0:
        raise ValueError("No overlapping indices across ensemble models.")

    results = []
    for h in horizons:
        preds_at_common = []
        for pred_by_h, end_indices in zip(model_preds, end_indices_list):
            pos = np.searchsorted(end_indices, common)
            preds_at_common.append(pred_by_h[h][pos])

        pred_log = np.mean(np.stack(preds_at_common, axis=0), axis=0)

        valid = (common + h < len(df))
        valid &= np.isfinite(close[common]) & np.isfinite(close[common + h])
        if not np.any(valid):
            results.append({"horizon": h, "n": 0})
            continue

        i = common[valid]
        base = close[i]
        actual = close[i + h]
        pred_price = base * np.exp(pred_log[valid])

        err = pred_price - actual
        abs_err = np.abs(err)
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mape = float(np.mean(np.abs(err) / np.maximum(actual, 1e-9)) * 100.0)
        smape = safe_smape(pred_price, actual)
        bias_usd = float(np.mean(err))
        bias_pct = float(np.mean(err / np.maximum(actual, 1e-9)) * 100.0)
        tail = _tail_stats(abs_err)

        actual_dir = np.sign(actual - base)
        pred_dir = np.sign(pred_price - base)
        dir_acc = float(np.mean(actual_dir == pred_dir) * 100.0)

        results.append({
            "horizon": h,
            "n": int(len(i)),
            "mae_usd": mae,
            "rmse_usd": rmse,
            "mape_pct": mape,
            "smape_pct": smape,
            "bias_usd": bias_usd,
            "bias_pct": bias_pct,
            "tail_abs_p50": tail["p50"],
            "tail_abs_p90": tail["p90"],
            "tail_abs_p95": tail["p95"],
            "tail_abs_p99": tail["p99"],
            "tail_abs_max": tail["max"],
            "direction_acc_pct": dir_acc,
            "start_ts": str(start_ts),
        })

    out_df = pd.DataFrame(results)
    print(out_df.to_string(index=False))
    if args.out:
        out_df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
