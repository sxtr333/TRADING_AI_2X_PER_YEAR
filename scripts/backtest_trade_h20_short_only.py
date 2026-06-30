#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
    return pred, end_indices, sigma


def _calc_bias_shift(df, idx, horizon, pred_log):
    close = df["close"].to_numpy(dtype=np.float64)
    y = np.log(close[idx + horizon] / close[idx])
    err = pred_log - y
    return float(np.median(err))


@dataclass
class Position:
    close_idx: int
    notional: float
    entry_price: float


def main():
    start_ts = pd.Timestamp("2025-01-01T00:00:00+00:00")
    end_ts = pd.Timestamp("2026-01-01T00:00:00+00:00")
    cost_rt = 0.0015
    leverage = 1.0
    trade_frac = 0.2
    batch_size = 64
    threshold_bump_sigma = 0.9
    cooldown_steps = 32  # 8h

    model_path = "/home/vitamind/my_project/model6/new_models/2026-01-18_v7_h20_short/model_15m_itransformer_v7_h20_short.keras"
    stats_path = "/home/vitamind/my_project/model6/new_models/2026-01-18_v7_h20_short/norm_stats_v7_h20_short.npz"

    df = pd.read_parquet(
        "/home/vitamind/my_project/model6/data/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes_with_news_pruned.parquet"
    )
    ts = pd.to_datetime(df["timestamp"], utc=True).to_numpy()
    close = df["close"].to_numpy(dtype=np.float64)

    stats = _load_stats(stats_path)
    model = _load_model(model_path)
    pred, idx, sigma = _pred_for_horizon(model, stats, df, 20, batch_size)

    val_start = pd.to_datetime(_unpack_obj(stats["val_start_ts"]), utc=True)
    test_start = pd.to_datetime(_unpack_obj(stats["test_start_ts"]), utc=True)
    val_mask = (ts[idx] >= val_start) & (ts[idx] < test_start)
    pred = pred - _calc_bias_shift(df, idx[val_mask], 20, pred[val_mask])

    if "rv_short" in df.columns:
        rv = df["rv_short"].to_numpy(dtype=np.float64)
    elif "rv" in df.columns:
        rv = df["rv"].to_numpy(dtype=np.float64)
    else:
        rv = None

    period_mask = (ts[idx] >= start_ts) & (ts[idx] < end_ts)
    pred = pred[period_mask]
    idx = idx[period_mask]

    if rv is not None:
        rv_gate = np.percentile(rv[idx], 60.0)
        gate = rv[idx] >= rv_gate
    else:
        gate = np.ones_like(idx, dtype=bool)

    thr = stats.get("price_head_scale", {})
    thr = threshold_bump_sigma * sigma + float(0.0) + 0.0
    # base threshold from sweep best
    base_thr = 0.002556
    thr = base_thr + threshold_bump_sigma * sigma

    equity = 100.0
    positions = []
    trade_count = 0
    total_fees = 0.0
    last_trade = -10**9

    for p, i, g in zip(pred, idx, gate):
        if ts[i] < start_ts or ts[i] >= end_ts:
            continue

        # close positions due at step i
        still = []
        for pos in positions:
            if pos.close_idx == i:
                ret = -((close[i] / pos.entry_price) - 1.0)  # short
                pnl = pos.notional * ret
                fee = pos.notional * cost_rt
                equity += pnl - fee
                total_fees += fee
            else:
                still.append(pos)
        positions = still

        if not g:
            continue
        if p >= -thr:
            continue
        if i - last_trade < cooldown_steps:
            continue

        notional = equity * trade_frac * leverage
        if notional <= 0:
            continue
        positions.append(Position(close_idx=i + 20, notional=notional, entry_price=close[i]))
        last_trade = i
        trade_count += 1

    # close remaining at end
    end_idx = np.where(ts >= end_ts)[0]
    end_i = int(end_idx[0]) if len(end_idx) > 0 else len(df) - 1
    for pos in positions:
        ret = -((close[end_i] / pos.entry_price) - 1.0)
        pnl = pos.notional * ret
        fee = pos.notional * cost_rt
        equity += pnl - fee
        total_fees += fee

    print("Backtest period:", start_ts, "->", end_ts)
    print("Initial equity: $100.00")
    print(f"Final equity: ${equity:.2f}")
    print(f"Net PnL: ${equity-100.0:.2f}")
    print(f"Trades: {trade_count}")
    print(f"Total fees: ${total_fees:.2f}")
    print(f"Threshold used: {thr:.6f}")


if __name__ == "__main__":
    main()
