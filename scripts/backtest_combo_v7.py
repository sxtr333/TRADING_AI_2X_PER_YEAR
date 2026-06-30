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

    # normalize model output to dict
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

    return pred, end_indices, float(scale_map.get(key, np.std(pred)))


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


def run_backtest_from_preds(
    df: pd.DataFrame,
    pred_log: np.ndarray,
    end_idx: np.ndarray,
    horizon: int,
    sigma: float,
    threshold_sigma: float,
    cost_rt: float,
    overlap: bool,
    long_only: bool,
    gate_mask: np.ndarray | None,
    strategy: str,
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
        if long_only and p <= 0:
            continue
        if abs(p) < thr:
            continue
        if i + horizon >= len(close):
            break
        r = math.log(close[i + horizon] / close[i])
        signed = r if long_only else math.copysign(r, p)
        net = signed - cost_rt
        rets.append(net)

    rets = np.asarray(rets, dtype=np.float64)
    return {
        "horizon": horizon,
        "threshold": thr,
        "overlap": bool(overlap),
        "long_only": bool(long_only),
        "strategy": strategy,
        "cost_rt": cost_rt,
        **_metrics(rets),
    }


def main():
    # models
    h20_model = "/home/vitamind/my_project/model6/new_models/2026-01-18_v7_pruned/model_15m_itransformer_v7_pruned.keras"
    h20_stats = "/home/vitamind/my_project/model6/new_models/2026-01-18_v7_pruned/norm_stats_v7_pruned.npz"

    h80_model = "/home/vitamind/my_project/model6/new_models/2026-01-18_v7_long_daily_v3/model_15m_itransformer_v7_long_daily_v3.keras"
    h80_stats = "/home/vitamind/my_project/model6/new_models/2026-01-18_v7_long_daily_v3/norm_stats_v7_long_daily_v3.npz"

    h160_model = "/home/vitamind/my_project/model6/new_models/2026-01-18_v7_long_daily_v2/model_15m_itransformer_v7_long_daily_v2.keras"
    h160_stats = "/home/vitamind/my_project/model6/new_models/2026-01-18_v7_long_daily_v2/norm_stats_v7_long_daily_v2.npz"

    # datasets
    df_pruned = pd.read_parquet(
        "/home/vitamind/my_project/model6/data/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes_with_news_pruned.parquet"
    )
    df_full = pd.read_parquet(
        "/home/vitamind/my_project/model6/data/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes_with_news.parquet"
    )

    thresholds = [0.75, 0.5, 0.35]
    cost_rt = 0.0015  # 0.15% round-trip
    batch_size = 64

    # load stats/models once
    h20_stats_obj = _load_stats(h20_stats)
    h80_stats_obj = _load_stats(h80_stats)
    h160_stats_obj = _load_stats(h160_stats)

    h20_model_obj = _load_model(h20_model)
    h80_model_obj = _load_model(h80_model)
    h160_model_obj = _load_model(h160_model)

    # predictions (once)
    p20, idx20, sigma20 = _pred_for_horizon(h20_model_obj, h20_stats_obj, df_pruned, 20, batch_size)
    p80, idx80, sigma80 = _pred_for_horizon(h80_model_obj, h80_stats_obj, df_full, 80, batch_size)
    p160, idx160, sigma160 = _pred_for_horizon(h160_model_obj, h160_stats_obj, df_full, 160, batch_size)

    # test window mask
    ts20 = pd.to_datetime(df_pruned["timestamp"], utc=True).to_numpy()
    tsfull = pd.to_datetime(df_full["timestamp"], utc=True).to_numpy()
    test_start20 = pd.to_datetime(_unpack_obj(h20_stats_obj["test_start_ts"]), utc=True)
    test_start80 = pd.to_datetime(_unpack_obj(h80_stats_obj["test_start_ts"]), utc=True)
    test_start160 = pd.to_datetime(_unpack_obj(h160_stats_obj["test_start_ts"]), utc=True)

    m20 = ts20[idx20] >= test_start20
    m80 = tsfull[idx80] >= test_start80
    m160 = tsfull[idx160] >= test_start160
    p20, idx20 = p20[m20], idx20[m20]
    p80, idx80 = p80[m80], idx80[m80]
    p160, idx160 = p160[m160], idx160[m160]

    # volatility gate for h80/h160 using rv_short (or rv)
    if "rv_short" in df_full.columns:
        rv = df_full["rv_short"].to_numpy(dtype=np.float64)
    else:
        rv = df_full["rv"].to_numpy(dtype=np.float64)
    rv_gate = np.percentile(rv[idx80], 60.0)
    gate80 = rv[idx80] >= rv_gate
    gate160 = rv[idx160] >= rv_gate

    results = []
    for thr in thresholds:
        # base (long/short)
        results.append(run_backtest_from_preds(df_pruned, p20, idx20, 20, sigma20, thr, cost_rt, overlap=False, long_only=False, gate_mask=None, strategy="base"))
        results.append(run_backtest_from_preds(df_full, p80, idx80, 80, sigma80, thr, cost_rt, overlap=False, long_only=False, gate_mask=None, strategy="base"))
        results.append(run_backtest_from_preds(df_full, p160, idx160, 160, sigma160, thr, cost_rt, overlap=False, long_only=False, gate_mask=None, strategy="base"))

        # long-only
        results.append(run_backtest_from_preds(df_pruned, p20, idx20, 20, sigma20, thr, cost_rt, overlap=False, long_only=True, gate_mask=None, strategy="long_only"))
        results.append(run_backtest_from_preds(df_full, p80, idx80, 80, sigma80, thr, cost_rt, overlap=False, long_only=True, gate_mask=None, strategy="long_only"))
        results.append(run_backtest_from_preds(df_full, p160, idx160, 160, sigma160, thr, cost_rt, overlap=False, long_only=True, gate_mask=None, strategy="long_only"))

        # vol-gated (h80/h160 only)
        results.append(run_backtest_from_preds(df_full, p80, idx80, 80, sigma80, thr, cost_rt, overlap=False, long_only=False, gate_mask=gate80, strategy="vol_gate"))
        results.append(run_backtest_from_preds(df_full, p160, idx160, 160, sigma160, thr, cost_rt, overlap=False, long_only=False, gate_mask=gate160, strategy="vol_gate"))

        # overlap variant for 0.5σ
        if abs(thr - 0.5) < 1e-9:
            results.append(run_backtest_from_preds(df_pruned, p20, idx20, 20, sigma20, thr, cost_rt, overlap=True, long_only=False, gate_mask=None, strategy="overlap"))
            results.append(run_backtest_from_preds(df_full, p80, idx80, 80, sigma80, thr, cost_rt, overlap=True, long_only=False, gate_mask=None, strategy="overlap"))
            results.append(run_backtest_from_preds(df_full, p160, idx160, 160, sigma160, thr, cost_rt, overlap=True, long_only=False, gate_mask=None, strategy="overlap"))

    out_path = Path("/home/vitamind/my_project/model6/reports/backtest_v7_combo_variants.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_path, index=False)

    print(pd.DataFrame(results))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
