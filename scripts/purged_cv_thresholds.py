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
    if isinstance(x, np.ndarray) and x.dtype == object and x.size == 1:
        return x[0]
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
    def gen():
        for i in end_indices:
            s = i - seq_len + 1
            yield X[s : i + 1]

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=tf.TensorSpec(shape=(seq_len, X.shape[1]), dtype=tf.float32),
    )
    return ds.batch(batch_size)


def _pred_for_horizon(model, stats, df, horizon: int, batch_size: int):
    feature_names = list(_unpack_obj(stats["feature_names"]))
    seq_len = int(_unpack_obj(stats["seq_len"]))
    X_raw = df[feature_names].to_numpy(dtype=np.float32)
    X = apply_norm(X_raw, stats)

    end_indices = np.arange(seq_len - 1, len(df) - horizon - 1, dtype=np.int64)
    ds = build_dataset(X, end_indices, seq_len=seq_len, batch_size=batch_size)
    preds = model.predict(ds, verbose=0)

    if isinstance(preds, (list, tuple)):
        raise ValueError("Expected dict outputs for multi-horizon model.")
    if not isinstance(preds, dict):
        preds = {"price": np.asarray(preds).reshape(-1)}

    key = f"price_h{horizon}"
    if key not in preds:
        raise ValueError(f"Missing head {key} in model outputs.")

    pred = np.asarray(preds[key]).reshape(-1)
    scale_map = _unpack_obj(stats.get("price_head_scale")) or {}
    if key in scale_map:
        pred = pred * float(scale_map[key])

    sigma = float(scale_map.get(key, np.std(pred)))
    return pred, end_indices, sigma


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


def _pnl_for_threshold(pred, y, direction, thr, cost):
    if direction == "long":
        mask = pred > thr
        pnl = y[mask] - cost
    else:
        mask = pred < -thr
        pnl = -y[mask] - cost
    return float(np.sum(pnl)), int(np.sum(mask))


def main():
    ap = argparse.ArgumentParser(description="Purged CV threshold selection for trade models.")
    ap.add_argument("--features", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--threshold-sigmas", default="0.3,0.4,0.5,0.6,0.8,1.0,1.2")
    ap.add_argument("--cost-bps", type=float, default=15.0)
    ap.add_argument("--purge-gap", type=int, default=0, help="Gap in rows before test fold.")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    args = ap.parse_args()

    df = pd.read_parquet(args.features)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    if args.start:
        df = df[df["timestamp"] >= pd.to_datetime(args.start, utc=True)].reset_index(drop=True)
    if args.end:
        df = df[df["timestamp"] <= pd.to_datetime(args.end, utc=True)].reset_index(drop=True)

    ts = df["timestamp"].to_numpy()
    close = df["close"].to_numpy(dtype=np.float64)
    cost = float(args.cost_bps) / 10000.0

    best_h20 = pd.read_csv("/home/vitamind/my_project/model6/reports/backtest_v7_long_short_sweep_best.csv")
    best_v2 = pd.read_csv("/home/vitamind/my_project/model6/reports/backtest_v7_long_short_v2_sweep_best.csv")

    specs = [
        ModelSpec(
            name="h20_long",
            horizon=20,
            direction="long",
            model_path="/home/vitamind/my_project/model6/new_models/2026-01-18_v7_h20_long/model_15m_itransformer_v7_h20_long.keras",
            stats_path="/home/vitamind/my_project/model6/new_models/2026-01-18_v7_h20_long/norm_stats_v7_h20_long.npz",
        ),
        ModelSpec(
            name="h20_short",
            horizon=20,
            direction="short",
            model_path="/home/vitamind/my_project/model6/new_models/2026-01-18_v7_h20_short/model_15m_itransformer_v7_h20_short.keras",
            stats_path="/home/vitamind/my_project/model6/new_models/2026-01-18_v7_h20_short/norm_stats_v7_h20_short.npz",
        ),
        ModelSpec(
            name="h80_short_v2",
            horizon=80,
            direction="short",
            model_path="/home/vitamind/my_project/model6/new_models/2026-01-18_v7_h80_short_v2/model_15m_itransformer_v7_h80_short_v2.keras",
            stats_path="/home/vitamind/my_project/model6/new_models/2026-01-18_v7_h80_short_v2/norm_stats_v7_h80_short_v2.npz",
        ),
        ModelSpec(
            name="h160_long_v2",
            horizon=160,
            direction="long",
            model_path="/home/vitamind/my_project/model6/new_models/2026-01-18_v7_h160_long_v2/model_15m_itransformer_v7_h160_long_v2.keras",
            stats_path="/home/vitamind/my_project/model6/new_models/2026-01-18_v7_h160_long_v2/norm_stats_v7_h160_long_v2.npz",
        ),
    ]

    thresholds = [float(x.strip()) for x in args.threshold_sigmas.split(",") if x.strip()]

    rows = []
    for spec in specs:
        stats = _load_stats(spec.stats_path)
        model = _load_model(spec.model_path)
        pred, idx, sigma = _pred_for_horizon(model, stats, df, spec.horizon, args.batch_size)

        y = np.log(close[idx + spec.horizon] / close[idx])

        n = len(idx)
        fold_size = n // args.folds
        for f in range(args.folds):
            start = f * fold_size
            end = n if f == args.folds - 1 else (f + 1) * fold_size
            test_mask = np.zeros(n, dtype=bool)
            test_mask[start:end] = True
            train_mask = np.zeros(n, dtype=bool)
            purge_end = max(0, start - int(args.purge_gap))
            train_mask[:purge_end] = True

            if not np.any(train_mask) or not np.any(test_mask):
                continue

            bias_shift = _calc_bias_shift(df, idx[train_mask], spec.horizon, pred[train_mask])
            pred_adj = pred - bias_shift

            best_thr = None
            best_pnl = -1e18
            best_trades = 0
            for thr_sigma in thresholds:
                thr = thr_sigma * sigma
                pnl, trades = _pnl_for_threshold(pred_adj[test_mask], y[test_mask], spec.direction, thr, cost)
                if pnl > best_pnl:
                    best_pnl = pnl
                    best_thr = thr
                    best_trades = trades
            rows.append(
                {
                    "model": spec.name,
                    "horizon": spec.horizon,
                    "direction": spec.direction,
                    "fold": f,
                    "best_threshold": best_thr,
                    "best_pnl": best_pnl,
                    "trades": best_trades,
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
