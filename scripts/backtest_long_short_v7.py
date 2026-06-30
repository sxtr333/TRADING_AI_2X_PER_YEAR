#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import sys
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
    out = {k: _unpack_obj(s[k]) for k in s.files}
    return out


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
    ds = ds.batch(batch_size)
    return ds


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


def _non_overlap_indices(idx: np.ndarray, horizon: int) -> np.ndarray:
    if len(idx) == 0:
        return idx
    return idx[::horizon]


def _metrics(trade_returns: np.ndarray):
    if trade_returns.size == 0:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "sharpe": 0.0,
            "max_dd": 0.0,
            "total": 0.0,
        }
    mean = float(trade_returns.mean())
    std = float(trade_returns.std(ddof=1)) if trade_returns.size > 1 else 0.0
    sharpe = float((mean / std) * math.sqrt(trade_returns.size)) if std > 0 else 0.0
    equity = np.cumsum(trade_returns)
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    max_dd = float(dd.min()) if dd.size else 0.0
    return {
        "trades": int(trade_returns.size),
        "win_rate": float((trade_returns > 0).mean()),
        "mean": mean,
        "std": std,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "total": float(trade_returns.sum()),
    }


def run_direction_backtest(
    df: pd.DataFrame,
    pred_log: np.ndarray,
    end_idx: np.ndarray,
    horizon: int,
    sigma: float,
    threshold_sigma: float,
    cost_rt: float,
    direction: str,
):
    end_idx = _non_overlap_indices(end_idx, horizon)
    pred_log = pred_log[: len(end_idx)]

    thr = threshold_sigma * sigma
    close = df["close"].to_numpy(dtype=np.float64)

    rets = []
    for p, i in zip(pred_log, end_idx):
        if i + horizon >= len(close):
            break
        if direction == "long":
            if p <= thr:
                continue
            signed = math.log(close[i + horizon] / close[i])
        elif direction == "short":
            if p >= -thr:
                continue
            signed = -math.log(close[i + horizon] / close[i])
        else:
            raise ValueError("direction must be 'long' or 'short'")
        net = signed - cost_rt
        rets.append(net)

    rets = np.asarray(rets, dtype=np.float64)
    return {
        "horizon": horizon,
        "threshold": thr,
        "direction": direction,
        "cost_rt": cost_rt,
        **_metrics(rets),
    }


def main():
    # models/stats
    paths = {
        20: {
            "long": {
                "model": "/home/vitamind/my_project/model6/new_models/2026-01-18_v7_h20_long/model_15m_itransformer_v7_h20_long.keras",
                "stats": "/home/vitamind/my_project/model6/new_models/2026-01-18_v7_h20_long/norm_stats_v7_h20_long.npz",
            },
            "short": {
                "model": "/home/vitamind/my_project/model6/new_models/2026-01-18_v7_h20_short/model_15m_itransformer_v7_h20_short.keras",
                "stats": "/home/vitamind/my_project/model6/new_models/2026-01-18_v7_h20_short/norm_stats_v7_h20_short.npz",
            },
        },
        80: {
            "long": {
                "model": "/home/vitamind/my_project/model6/new_models/2026-01-18_v7_h80_long/model_15m_itransformer_v7_h80_long.keras",
                "stats": "/home/vitamind/my_project/model6/new_models/2026-01-18_v7_h80_long/norm_stats_v7_h80_long.npz",
            },
            "short": {
                "model": "/home/vitamind/my_project/model6/new_models/2026-01-18_v7_h80_short/model_15m_itransformer_v7_h80_short.keras",
                "stats": "/home/vitamind/my_project/model6/new_models/2026-01-18_v7_h80_short/norm_stats_v7_h80_short.npz",
            },
        },
        160: {
            "long": {
                "model": "/home/vitamind/my_project/model6/new_models/2026-01-18_v7_h160_long/model_15m_itransformer_v7_h160_long.keras",
                "stats": "/home/vitamind/my_project/model6/new_models/2026-01-18_v7_h160_long/norm_stats_v7_h160_long.npz",
            },
            "short": {
                "model": "/home/vitamind/my_project/model6/new_models/2026-01-18_v7_h160_short/model_15m_itransformer_v7_h160_short.keras",
                "stats": "/home/vitamind/my_project/model6/new_models/2026-01-18_v7_h160_short/norm_stats_v7_h160_short.npz",
            },
        },
    }

    df = pd.read_parquet(
        "/home/vitamind/my_project/model6/data/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes_with_news_pruned.parquet"
    )

    threshold_sigma = 0.5
    cost_rt = 0.0015
    batch_size = 64

    results = []
    for horizon in (20, 80, 160):
        for direction in ("long", "short"):
            model_path = paths[horizon][direction]["model"]
            stats_path = paths[horizon][direction]["stats"]
            stats = _load_stats(stats_path)
            model = _load_model(model_path)

            pred, idx, sigma = _pred_for_horizon(model, stats, df, horizon, batch_size)

            ts = pd.to_datetime(df["timestamp"], utc=True).to_numpy()
            test_start = pd.to_datetime(_unpack_obj(stats["test_start_ts"]), utc=True)
            mask = ts[idx] >= test_start
            pred, idx = pred[mask], idx[mask]

            results.append(
                run_direction_backtest(
                    df,
                    pred,
                    idx,
                    horizon,
                    sigma,
                    threshold_sigma,
                    cost_rt,
                    direction,
                )
            )

    out_path = Path("/home/vitamind/my_project/model6/reports/backtest_v7_long_short_thr05.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_path, index=False)

    print(pd.DataFrame(results))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
