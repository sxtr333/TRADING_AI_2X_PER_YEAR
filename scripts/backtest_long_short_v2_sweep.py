#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
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
    if isinstance(x, np.ndarray):
        if x.dtype == object and x.size == 1:
            return x[0]
        if x.size == 1:
            return x.item()
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
    # Guard against missing feature columns in the current dataset.
    if isinstance(df.columns, pd.Index):
        feature_names = [c for c in feature_names if c in df.columns]
    seq_len = int(_unpack_obj(stats["seq_len"]))
    X_raw = df[feature_names].to_numpy(dtype=np.float32)
    X = apply_norm(X_raw, stats)

    end_indices = np.arange(seq_len - 1, len(df) - horizon - 1, dtype=np.int64)
    ds = build_dataset(X, end_indices, seq_len=seq_len, batch_size=batch_size)
    if ds is None:
        return np.array([], dtype=np.float32), end_indices, 0.0
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


def _calc_bias_shift(df, idx, horizon, pred_log):
    close = df["close"].to_numpy(dtype=np.float64)
    y = np.log(close[idx + horizon] / close[idx])
    err = pred_log - y
    return float(np.median(err))


def run_direction_backtest(
    df: pd.DataFrame,
    pred_log: np.ndarray,
    end_idx: np.ndarray,
    horizon: int,
    sigma: float,
    threshold_sigma: float,
    cost_rt: float,
    direction: str,
    overlap: bool,
    gate_mask: np.ndarray | None,
):
    if gate_mask is not None:
        gate_mask = gate_mask.astype(bool)
        pred_log = pred_log[gate_mask]
        end_idx = end_idx[gate_mask]

    if not overlap:
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
        "threshold": thr,
        "direction": direction,
        "overlap": bool(overlap),
        "cost_rt": cost_rt,
        **_metrics(rets),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="/home/vitamind/my_project/model6/data/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes_with_news_pruned.parquet")
    ap.add_argument("--model-dir", default="/home/vitamind/my_project/model6/new_models")
    ap.add_argument("--model-tag", default="2026-01-18_v7", help="Prefix for model subdirs, e.g. 2026-01-18_v7")
    ap.add_argument("--out-csv", default="/home/vitamind/my_project/model6/reports/backtest_v7_long_short_v2_sweep.csv")
    ap.add_argument("--bias-csv", default="/home/vitamind/my_project/model6/reports/bias_shift_long_short_v2.csv")
    ap.add_argument("--start", default=None, help="Optional start timestamp (UTC).")
    ap.add_argument("--end", default=None, help="Optional end timestamp (UTC).")
    args = ap.parse_args()

    def mp(*parts):
        return str(Path(args.model_dir, *parts))

    paths = {
        80: {
            "long": {
                "model": mp(f"{args.model_tag}_h80_long_v2", "model_15m_itransformer_v7_h80_long_v2.keras"),
                "stats": mp(f"{args.model_tag}_h80_long_v2", "norm_stats_v7_h80_long_v2.npz"),
            },
            "short": {
                "model": mp(f"{args.model_tag}_h80_short_v2", "model_15m_itransformer_v7_h80_short_v2.keras"),
                "stats": mp(f"{args.model_tag}_h80_short_v2", "norm_stats_v7_h80_short_v2.npz"),
            },
        },
        160: {
            "long": {
                "model": mp(f"{args.model_tag}_h160_long_v2", "model_15m_itransformer_v7_h160_long_v2.keras"),
                "stats": mp(f"{args.model_tag}_h160_long_v2", "norm_stats_v7_h160_long_v2.npz"),
            },
            "short": {
                "model": mp(f"{args.model_tag}_h160_short_v2", "model_15m_itransformer_v7_h160_short_v2.keras"),
                "stats": mp(f"{args.model_tag}_h160_short_v2", "norm_stats_v7_h160_short_v2.npz"),
            },
        },
    }

    df = pd.read_parquet(
        args.features
    )
    if "timestamp" not in df.columns:
        raise ValueError("features parquet must include 'timestamp'")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    if args.start:
        st = pd.to_datetime(args.start, utc=True)
        df = df[df["timestamp"] >= st].reset_index(drop=True)
    if args.end:
        en = pd.to_datetime(args.end, utc=True)
        df = df[df["timestamp"] <= en].reset_index(drop=True)

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2]
    cost_rt = 0.0015
    batch_size = 64

    results = []
    bias_rows = []

    if "rv_short" in df.columns:
        rv = df["rv_short"].to_numpy(dtype=np.float64)
    elif "rv" in df.columns:
        rv = df["rv"].to_numpy(dtype=np.float64)
    else:
        rv = None

    for horizon in (80, 160):
        for direction in ("long", "short"):
            model_path = paths[horizon][direction]["model"]
            stats_path = paths[horizon][direction]["stats"]
            stats = _load_stats(stats_path)
            model = _load_model(model_path)

            pred, idx, sigma = _pred_for_horizon(model, stats, df, horizon, batch_size)
            # Guard against rare off-by-one between pred and idx lengths
            if len(pred) != len(idx):
                n = min(len(pred), len(idx))
                pred = pred[:n]
                idx = idx[:n]

            ts = pd.to_datetime(df["timestamp"], utc=True).to_numpy()
            test_start = pd.to_datetime(_unpack_obj(stats["test_start_ts"]), utc=True)
            val_start = pd.to_datetime(_unpack_obj(stats["val_start_ts"]), utc=True)

            val_mask = (ts[idx] >= val_start) & (ts[idx] < test_start)
            test_mask = ts[idx] >= test_start
            if not np.any(test_mask):
                test_mask = np.ones_like(idx, dtype=bool)

            bias_shift = _calc_bias_shift(df, idx[val_mask], horizon, pred[val_mask])
            pred_bs = pred - bias_shift

            bias_rows.append(
                {
                    "horizon": horizon,
                    "direction": direction,
                    "bias_shift": bias_shift,
                    "sigma": sigma,
                }
            )

            if rv is not None and idx[test_mask].size > 0:
                rv_gate = np.percentile(rv[idx[test_mask]], 60.0)
                gate_mask = rv[idx] >= rv_gate
            else:
                gate_mask = None

            for thr in thresholds:
                base = run_direction_backtest(
                    df,
                    pred_bs[test_mask],
                    idx[test_mask],
                    horizon,
                    sigma,
                    thr,
                    cost_rt,
                    direction,
                    overlap=False,
                    gate_mask=None,
                )
                base.update(
                    {
                        "horizon": horizon,
                        "strategy": "base",
                    }
                )
                results.append(base)

                gated = run_direction_backtest(
                    df,
                    pred_bs[test_mask],
                    idx[test_mask],
                    horizon,
                    sigma,
                    thr,
                    cost_rt,
                    direction,
                    overlap=False,
                    gate_mask=gate_mask[test_mask] if gate_mask is not None else None,
                )
                gated.update(
                    {
                        "horizon": horizon,
                        "strategy": "vol_gate_p60",
                    }
                )
                results.append(gated)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_path, index=False)

    bias_path = Path(args.bias_csv)
    pd.DataFrame(bias_rows).to_csv(bias_path, index=False)

    print(pd.DataFrame(results))
    print(f"Saved: {out_path}")
    print(f"Bias saved: {bias_path}")


if __name__ == "__main__":
    main()
