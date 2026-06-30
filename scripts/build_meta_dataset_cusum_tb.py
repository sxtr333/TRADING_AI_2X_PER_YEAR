#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
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
    return pred, end_indices, sigma, seq_len


def _calc_bias_shift(df, idx, horizon, pred_log):
    close = df["close"].to_numpy(dtype=np.float64)
    y = np.log(close[idx + horizon] / close[idx])
    err = pred_log - y
    return float(np.median(err))


def _cusum_events(ret: np.ndarray, h: np.ndarray) -> np.ndarray:
    events = np.zeros(len(ret), dtype=bool)
    s_pos = 0.0
    s_neg = 0.0
    for i in range(len(ret)):
        if not np.isfinite(h[i]):
            continue
        s_pos = max(0.0, s_pos + ret[i])
        s_neg = min(0.0, s_neg + ret[i])
        if s_pos > h[i]:
            s_pos = 0.0
            events[i] = True
        elif s_neg < -h[i]:
            s_neg = 0.0
            events[i] = True
    return events


def _tb_label(close: np.ndarray, vol: np.ndarray, i: int, horizon: int, pt_mult: float, sl_mult: float) -> int:
    if i + horizon >= len(close):
        return 0
    v = float(vol[i])
    if not np.isfinite(v) or v <= 0:
        return 0
    p0 = float(close[i])
    pt = p0 * math.exp(pt_mult * v)
    sl = p0 * math.exp(-sl_mult * v)
    for j in range(i + 1, i + horizon + 1):
        p = float(close[j])
        if p >= pt:
            return 1
        if p <= sl:
            return -1
    return 0


@dataclass
class ModelSpec:
    name: str
    horizon: int
    direction: str
    model_path: str
    stats_path: str
    threshold: float


def main():
    ap = argparse.ArgumentParser(description="Build meta-label dataset using CUSUM events + Triple Barrier labels.")
    ap.add_argument("--features", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--cusum-k", type=float, default=2.5, help="CUSUM threshold multiplier over vol.")
    ap.add_argument("--pt-mult", type=float, default=1.5, help="PT multiplier over vol (scaled by sqrt(h/20)).")
    ap.add_argument("--sl-mult", type=float, default=1.5, help="SL multiplier over vol (scaled by sqrt(h/20)).")
    ap.add_argument("--vol-col", default="rv_short", help="Volatility column name.")
    args = ap.parse_args()

    df = pd.read_parquet(args.features)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    if args.vol_col in df.columns:
        vol = df[args.vol_col].to_numpy(dtype=np.float64)
    elif "rv" in df.columns:
        vol = df["rv"].to_numpy(dtype=np.float64)
    else:
        raise ValueError(f"Vol column '{args.vol_col}' not found and no fallback 'rv'.")

    close = df["close"].to_numpy(dtype=np.float64)
    ret = np.diff(np.log(close), prepend=np.log(close[0]))
    h = args.cusum_k * vol
    event_mask = _cusum_events(ret, h)
    df["cusum_event"] = event_mask.astype(np.int32)

    best_h20 = pd.read_csv("/home/vitamind/my_project/model6/reports/backtest_v7_long_short_sweep_best.csv")
    best_v2 = pd.read_csv("/home/vitamind/my_project/model6/reports/backtest_v7_long_short_v2_sweep_best.csv")

    def get_best(df_best, horizon, direction):
        row = df_best[(df_best.horizon == horizon) & (df_best.direction == direction)].iloc[0]
        return float(row["threshold"])

    specs = [
        ModelSpec(
            name="h20_long",
            horizon=20,
            direction="long",
            model_path="/home/vitamind/my_project/model6/new_models/2026-01-18_v7_h20_long/model_15m_itransformer_v7_h20_long.keras",
            stats_path="/home/vitamind/my_project/model6/new_models/2026-01-18_v7_h20_long/norm_stats_v7_h20_long.npz",
            threshold=get_best(best_h20, 20, "long"),
        ),
        ModelSpec(
            name="h20_short",
            horizon=20,
            direction="short",
            model_path="/home/vitamind/my_project/model6/new_models/2026-01-18_v7_h20_short/model_15m_itransformer_v7_h20_short.keras",
            stats_path="/home/vitamind/my_project/model6/new_models/2026-01-18_v7_h20_short/norm_stats_v7_h20_short.npz",
            threshold=get_best(best_h20, 20, "short"),
        ),
        ModelSpec(
            name="h80_short_v2",
            horizon=80,
            direction="short",
            model_path="/home/vitamind/my_project/model6/new_models/2026-01-18_v7_h80_short_v2/model_15m_itransformer_v7_h80_short_v2.keras",
            stats_path="/home/vitamind/my_project/model6/new_models/2026-01-18_v7_h80_short_v2/norm_stats_v7_h80_short_v2.npz",
            threshold=get_best(best_v2, 80, "short"),
        ),
        ModelSpec(
            name="h160_long_v2",
            horizon=160,
            direction="long",
            model_path="/home/vitamind/my_project/model6/new_models/2026-01-18_v7_h160_long_v2/model_15m_itransformer_v7_h160_long_v2.keras",
            stats_path="/home/vitamind/my_project/model6/new_models/2026-01-18_v7_h160_long_v2/norm_stats_v7_h160_long_v2.npz",
            threshold=get_best(best_v2, 160, "long"),
        ),
    ]

    for spec in specs:
        stats = _load_stats(spec.stats_path)
        model = _load_model(spec.model_path)
        pred, idx, sigma, seq_len = _pred_for_horizon(model, stats, df, spec.horizon, args.batch_size)

        val_start = pd.to_datetime(_unpack_obj(stats["val_start_ts"]), utc=True)
        test_start = pd.to_datetime(_unpack_obj(stats["test_start_ts"]), utc=True)
        ts = df["timestamp"].to_numpy()
        val_mask = (ts[idx] >= val_start) & (ts[idx] < test_start)
        bias_shift = _calc_bias_shift(df, idx[val_mask], spec.horizon, pred[val_mask])
        pred = pred - bias_shift

        thr = spec.threshold + sigma * 0.0
        pred_full = np.full(len(df), np.nan, dtype=np.float32)
        label_full = np.zeros(len(df), dtype=np.int32)
        tb_full = np.zeros(len(df), dtype=np.int32)

        pred_full[idx] = pred

        # scale TB barriers by sqrt(h/20) to keep comparable difficulty
        scale = math.sqrt(spec.horizon / 20.0)
        pt_mult = args.pt_mult * scale
        sl_mult = args.sl_mult * scale

        for i in idx:
            if not event_mask[i]:
                continue
            p = float(pred_full[i])
            if not np.isfinite(p):
                continue
            if spec.direction == "long" and p <= thr:
                continue
            if spec.direction == "short" and p >= -thr:
                continue

            tb = _tb_label(close, vol, int(i), spec.horizon, pt_mult, sl_mult)
            tb_full[int(i)] = tb
            if spec.direction == "long":
                label_full[int(i)] = 1 if tb == 1 else 0
            else:
                label_full[int(i)] = 1 if tb == -1 else 0

        df[f"meta_pred_{spec.name}"] = pred_full
        df[f"meta_label_{spec.name}"] = label_full
        df[f"tb_label_{spec.name}_cusum"] = tb_full

    # sanitize (keep alignment with features)
    df = df.replace([np.inf, -np.inf], np.nan)
    for c in df.columns:
        if c.startswith(("meta_pred_", "meta_label_", "tb_label_")) or c == "cusum_event":
            df[c] = df[c].fillna(0.0)
    df = df.reset_index(drop=True)

    df.to_parquet(args.output, index=False)
    print(f"Saved: {args.output} rows={len(df)}")


if __name__ == "__main__":
    main()
