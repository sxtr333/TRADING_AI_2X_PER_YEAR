#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_keras import apply_norm
from train_keras_v7 import TimePositionalEncoding
from model_layers import RevIN, TSMixerBlock, ITransformerBlock, LastStep, DropPath


def _unpack_obj(x):
    if isinstance(x, np.ndarray):
        if x.dtype == object and x.size == 1:
            return x[0]
        if x.size == 1:
            return x.item()
    return x


def _load_stats(path: str) -> dict:
    s = np.load(path, allow_pickle=True)
    return {k: _unpack_obj(s[k]) for k in s.files}


def _load_model(path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(
        path,
        custom_objects={
            "model6>RevIN": RevIN,
            "model6>TSMixerBlock": TSMixerBlock,
            "model6>ITransformerBlock": ITransformerBlock,
            "model6>LastStep": LastStep,
            "model6>DropPath": DropPath,
            "TimePositionalEncoding": TimePositionalEncoding,
            "DropPath": DropPath,
        },
        compile=False,
    )


def build_dataset(X: np.ndarray, end_indices: np.ndarray, seq_len: int, batch_size: int):
    # Use the built-in timeseries dataset to avoid generator edge cases.
    if len(end_indices) == 0:
        return None
    end_index = int(end_indices[-1])
    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=X,
        targets=None,
        sequence_length=seq_len,
        sequence_stride=1,
        sampling_rate=1,
        start_index=0,
        end_index=end_index,
        batch_size=batch_size,
        shuffle=False,
    )
    return ds


def _pred_for_horizon(model, stats, df, horizon: int, batch_size: int):
    feature_names = list(_unpack_obj(stats["feature_names"]))
    seq_len = int(_unpack_obj(stats["seq_len"]))
    X_raw = df[feature_names].to_numpy(dtype=np.float32)
    X = apply_norm(X_raw, stats)

    end_indices = np.arange(seq_len - 1, len(df) - horizon - 1, dtype=np.int64)
    ds = build_dataset(X, end_indices, seq_len=seq_len, batch_size=batch_size)
    if ds is None:
        return np.array([], dtype=np.float32), end_indices, 0.0, seq_len, feature_names
    preds = model.predict(ds, verbose=0)

    if isinstance(preds, (list, tuple)):
        raise ValueError("Expected dict outputs for multi-horizon model.")
    if not isinstance(preds, dict):
        preds = {"price": np.asarray(preds).reshape(-1)}

    key = f"price_h{horizon}"
    if key not in preds:
        raise ValueError(f"Missing head {key} in model outputs.")

    pred = np.asarray(preds[key]).reshape(-1)
    if pred.shape[0] != end_indices.shape[0]:
        m = min(pred.shape[0], end_indices.shape[0])
        pred = pred[:m]
        end_indices = end_indices[:m]
    scale_map = _unpack_obj(stats.get("price_head_scale")) or {}
    if key in scale_map:
        pred = pred * float(scale_map[key])

    sigma = float(scale_map.get(key, np.std(pred)))
    return pred, end_indices, sigma, seq_len, feature_names


def _calc_bias_shift(df, idx, horizon, pred_log):
    close = df["close"].to_numpy(dtype=np.float64)
    y = np.log(close[idx + horizon] / close[idx])
    err = pred_log - y
    return float(np.median(err))


@dataclass
class ModelSpec:
    name: str
    horizon: int
    direction: str
    model_path: str
    stats_path: str
    threshold: float


def main():
    ap = argparse.ArgumentParser(description="Build meta-label dataset from base model predictions.")
    ap.add_argument("--features", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--cost-bps", type=float, default=15.0, help="Round-trip cost in bps.")
    ap.add_argument("--only-signal", action="store_true",
                    help="Filter dataset to rows where at least one meta signal fires.")
    ap.add_argument("--model-dir", default="/home/vitamind/my_project/model6/new_models")
    ap.add_argument("--model-tag", default="2026-01-18_v7", help="Prefix for model subdirs, e.g. 2026-01-18_v7")
    ap.add_argument("--best-h20", default="/home/vitamind/my_project/model6/reports/backtest_v7_long_short_sweep_best.csv")
    ap.add_argument("--best-v2", default="/home/vitamind/my_project/model6/reports/backtest_v7_long_short_v2_sweep_best.csv")
    ap.add_argument("--bias-start", default=None, help="Optional bias calibration start (UTC).")
    ap.add_argument("--bias-end", default=None, help="Optional bias calibration end (UTC).")
    ap.add_argument("--no-bias-shift", action="store_true", help="Disable bias shift calibration.")
    args = ap.parse_args()

    df = pd.read_parquet(args.features)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    best_h20 = pd.read_csv(args.best_h20)
    best_v2 = pd.read_csv(args.best_v2)

    def mp(*parts):
        return str(Path(args.model_dir, *parts))

    def get_best(df_best, horizon, direction):
        row = df_best[(df_best.horizon == horizon) & (df_best.direction == direction)].iloc[0]
        return float(row["threshold"])

    specs = [
        ModelSpec(
            name="h20_long",
            horizon=20,
            direction="long",
            model_path=mp(f"{args.model_tag}_h20_long", "model_15m_itransformer_v7_h20_long.keras"),
            stats_path=mp(f"{args.model_tag}_h20_long", "norm_stats_v7_h20_long.npz"),
            threshold=get_best(best_h20, 20, "long"),
        ),
        ModelSpec(
            name="h20_short",
            horizon=20,
            direction="short",
            model_path=mp(f"{args.model_tag}_h20_short", "model_15m_itransformer_v7_h20_short.keras"),
            stats_path=mp(f"{args.model_tag}_h20_short", "norm_stats_v7_h20_short.npz"),
            threshold=get_best(best_h20, 20, "short"),
        ),
        ModelSpec(
            name="h80_short_v2",
            horizon=80,
            direction="short",
            model_path=mp(f"{args.model_tag}_h80_short_v2", "model_15m_itransformer_v7_h80_short_v2.keras"),
            stats_path=mp(f"{args.model_tag}_h80_short_v2", "norm_stats_v7_h80_short_v2.npz"),
            threshold=get_best(best_v2, 80, "short"),
        ),
        ModelSpec(
            name="h160_long_v2",
            horizon=160,
            direction="long",
            model_path=mp(f"{args.model_tag}_h160_long_v2", "model_15m_itransformer_v7_h160_long_v2.keras"),
            stats_path=mp(f"{args.model_tag}_h160_long_v2", "norm_stats_v7_h160_long_v2.npz"),
            threshold=get_best(best_v2, 160, "long"),
        ),
    ]

    cost = float(args.cost_bps) / 10000.0
    close = df["close"].to_numpy(dtype=np.float64)

    any_signal = np.zeros(len(df), dtype=bool)

    for spec in specs:
        stats = _load_stats(spec.stats_path)
        model = _load_model(spec.model_path)
        pred, idx, sigma, seq_len, feature_names = _pred_for_horizon(model, stats, df, spec.horizon, args.batch_size)

        ts = df["timestamp"].to_numpy()
        if not args.no_bias_shift:
            if args.bias_start or args.bias_end:
                bs = pd.to_datetime(args.bias_start, utc=True) if args.bias_start else pd.Timestamp.min.tz_localize("UTC")
                be = pd.to_datetime(args.bias_end, utc=True) if args.bias_end else pd.Timestamp.max.tz_localize("UTC")
                bias_mask = (ts[idx] >= bs) & (ts[idx] < be)
            else:
                val_start = pd.to_datetime(_unpack_obj(stats["val_start_ts"]), utc=True)
                test_start = pd.to_datetime(_unpack_obj(stats["test_start_ts"]), utc=True)
                bias_mask = (ts[idx] >= val_start) & (ts[idx] < test_start)
            if bias_mask.any():
                bias_shift = _calc_bias_shift(df, idx[bias_mask], spec.horizon, pred[bias_mask])
                pred = pred - bias_shift

        thr = spec.threshold + sigma * 0.0
        sig = np.zeros(len(df), dtype=np.float32)
        pred_full = np.full(len(df), np.nan, dtype=np.float32)
        y_full = np.full(len(df), np.nan, dtype=np.float32)
        label_full = np.zeros(len(df), dtype=np.int32)

        y = np.log(close[idx + spec.horizon] / close[idx])
        pred_full[idx] = pred
        y_full[idx] = y

        if spec.direction == "long":
            signal = pred > thr
            label = (y > cost).astype(np.int32)
        else:
            signal = pred < -thr
            label = (y < -cost).astype(np.int32)

        sig[idx] = signal.astype(np.float32)
        label_full[idx] = label
        any_signal |= (sig > 0)

        df[f"meta_pred_{spec.name}"] = pred_full
        df[f"meta_y_{spec.name}"] = y_full
        df[f"meta_signal_{spec.name}"] = sig
        df[f"meta_label_{spec.name}"] = label_full

    # Replace NaNs in meta feature columns to avoid training NaNs
    meta_cols = [c for c in df.columns if c.startswith("meta_pred_") or c.startswith("meta_y_") or c.startswith("meta_signal_")]
    if meta_cols:
        df[meta_cols] = df[meta_cols].fillna(0.0)
    # Remove leakage: meta_y_* is future return; keep column for audit but zero it out for modeling
    meta_y_cols = [c for c in df.columns if c.startswith("meta_y_")]
    if meta_y_cols:
        df[meta_y_cols] = 0.0

    if args.only_signal:
        df = df[any_signal].reset_index(drop=True)

    df.to_parquet(args.output, index=False)
    print("Saved:", args.output)
    print("Rows:", len(df))


if __name__ == "__main__":
    main()
