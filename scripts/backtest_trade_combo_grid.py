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
class ModelSpec:
    name: str
    horizon: int
    direction: str
    strategy: str
    model_path: str
    stats_path: str
    threshold: float


@dataclass
class Position:
    close_idx: int
    notional: float
    direction: str
    entry_price: float


def _build_open_map(ts, start_ts, end_ts, preds, idx, sigma, spec, gate_mask, threshold_bump_sigma):
    thr = spec.threshold + threshold_bump_sigma * sigma
    period_mask = (ts[idx] >= start_ts) & (ts[idx] < end_ts)
    pred = preds[period_mask]
    idx = idx[period_mask]
    if gate_mask is not None:
        gate_mask = gate_mask[period_mask]

    open_map = {}
    if gate_mask is None:
        gate_iter = [True] * len(idx)
    else:
        gate_iter = gate_mask

    for p, i, g in zip(pred, idx, gate_iter):
        if not g:
            continue
        if spec.direction == "long" and p <= thr:
            continue
        if spec.direction == "short" and p >= -thr:
            continue
        open_map.setdefault(int(i), []).append(spec)
    return open_map


def _simulate(df, ts, close, open_maps, specs, start_ts, end_ts, cost_rt, leverage, trade_frac, cooldown_steps, max_concurrent):
    equity = 100.0
    positions = []
    trade_count = 0
    total_fees = 0.0
    last_trade_step = {}

    for i in range(len(df)):
        if ts[i] < start_ts or ts[i] >= end_ts:
            continue

        if positions:
            still = []
            for pos in positions:
                if pos.close_idx == i:
                    ret = (close[i] / pos.entry_price) - 1.0
                    if pos.direction == "short":
                        ret = -ret
                    pnl = pos.notional * ret
                    fee = pos.notional * cost_rt
                    equity += pnl - fee
                    total_fees += fee
                else:
                    still.append(pos)
            positions = still

        for spec in specs:
            omap = open_maps[spec.name]
            if i not in omap:
                continue
            if len(positions) >= max_concurrent:
                break
            last_i = last_trade_step.get(spec.name, -10**9)
            if i - last_i < cooldown_steps:
                continue
            notional = equity * trade_frac * leverage
            if notional <= 0:
                continue
            positions.append(
                Position(
                    close_idx=i + spec.horizon,
                    notional=notional,
                    direction=spec.direction,
                    entry_price=close[i],
                )
            )
            last_trade_step[spec.name] = i
            trade_count += 1

    end_idx = np.where(ts >= end_ts)[0]
    end_i = int(end_idx[0]) if len(end_idx) > 0 else len(df) - 1

    for pos in positions:
        ret = (close[end_i] / pos.entry_price) - 1.0
        if pos.direction == "short":
            ret = -ret
        pnl = pos.notional * ret
        fee = pos.notional * cost_rt
        equity += pnl - fee
        total_fees += fee

    return {
        "final_equity": equity,
        "pnl": equity - 100.0,
        "trades": trade_count,
        "fees": total_fees,
    }


def main():
    start_ts = pd.Timestamp("2025-01-01T00:00:00+00:00")
    end_ts = pd.Timestamp("2026-01-01T00:00:00+00:00")
    cost_rt = 0.0015
    leverage = 1.0
    batch_size = 64

    df = pd.read_parquet(
        "/home/vitamind/my_project/model6/data/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes_with_news_pruned.parquet"
    )
    ts = pd.to_datetime(df["timestamp"], utc=True).to_numpy()
    close = df["close"].to_numpy(dtype=np.float64)

    best_h20 = pd.read_csv("/home/vitamind/my_project/model6/reports/backtest_v7_long_short_sweep_best.csv")
    best_v2 = pd.read_csv("/home/vitamind/my_project/model6/reports/backtest_v7_long_short_v2_sweep_best.csv")

    def get_best(df_best, horizon, direction):
        row = df_best[(df_best.horizon == horizon) & (df_best.direction == direction)].iloc[0]
        return float(row["threshold"]), str(row["strategy"])

    h20_long_thr, h20_long_strat = get_best(best_h20, 20, "long")
    h20_short_thr, h20_short_strat = get_best(best_h20, 20, "short")
    h80_short_thr, h80_short_strat = get_best(best_v2, 80, "short")
    h160_long_thr, h160_long_strat = get_best(best_v2, 160, "long")

    specs = [
        ModelSpec(
            name="h20_long",
            horizon=20,
            direction="long",
            strategy=h20_long_strat,
            model_path="/home/vitamind/my_project/model6/new_models/2026-01-18_v7_h20_long/model_15m_itransformer_v7_h20_long.keras",
            stats_path="/home/vitamind/my_project/model6/new_models/2026-01-18_v7_h20_long/norm_stats_v7_h20_long.npz",
            threshold=h20_long_thr,
        ),
        ModelSpec(
            name="h20_short",
            horizon=20,
            direction="short",
            strategy=h20_short_strat,
            model_path="/home/vitamind/my_project/model6/new_models/2026-01-18_v7_h20_short/model_15m_itransformer_v7_h20_short.keras",
            stats_path="/home/vitamind/my_project/model6/new_models/2026-01-18_v7_h20_short/norm_stats_v7_h20_short.npz",
            threshold=h20_short_thr,
        ),
        ModelSpec(
            name="h80_short_v2",
            horizon=80,
            direction="short",
            strategy=h80_short_strat,
            model_path="/home/vitamind/my_project/model6/new_models/2026-01-18_v7_h80_short_v2/model_15m_itransformer_v7_h80_short_v2.keras",
            stats_path="/home/vitamind/my_project/model6/new_models/2026-01-18_v7_h80_short_v2/norm_stats_v7_h80_short_v2.npz",
            threshold=h80_short_thr,
        ),
        ModelSpec(
            name="h160_long_v2",
            horizon=160,
            direction="long",
            strategy=h160_long_strat,
            model_path="/home/vitamind/my_project/model6/new_models/2026-01-18_v7_h160_long_v2/model_15m_itransformer_v7_h160_long_v2.keras",
            stats_path="/home/vitamind/my_project/model6/new_models/2026-01-18_v7_h160_long_v2/norm_stats_v7_h160_long_v2.npz",
            threshold=h160_long_thr,
        ),
    ]

    if "rv_short" in df.columns:
        rv = df["rv_short"].to_numpy(dtype=np.float64)
    elif "rv" in df.columns:
        rv = df["rv"].to_numpy(dtype=np.float64)
    else:
        rv = None

    model_outputs = {}
    for spec in specs:
        stats = _load_stats(spec.stats_path)
        model = _load_model(spec.model_path)
        pred, idx, sigma = _pred_for_horizon(model, stats, df, spec.horizon, batch_size)

        val_start = pd.to_datetime(_unpack_obj(stats["val_start_ts"]), utc=True)
        test_start = pd.to_datetime(_unpack_obj(stats["test_start_ts"]), utc=True)
        val_mask = (ts[idx] >= val_start) & (ts[idx] < test_start)
        bias_shift = _calc_bias_shift(df, idx[val_mask], spec.horizon, pred[val_mask])
        pred = pred - bias_shift

        if spec.strategy == "vol_gate_p60" and rv is not None:
            period_mask = (ts[idx] >= start_ts) & (ts[idx] < end_ts)
            rv_gate = np.percentile(rv[idx[period_mask]], 60.0)
            gate_mask = rv[idx] >= rv_gate
        else:
            gate_mask = None

        model_outputs[spec.name] = {
            "spec": spec,
            "pred": pred,
            "idx": idx,
            "gate": gate_mask,
            "sigma": sigma,
        }

    threshold_bumps = [0.4, 0.7, 1.0]
    cooldown_steps_list = [0, 16, 32]
    max_concurrent_list = [1, 2]
    trade_frac_list = [0.1, 0.2]

    rows = []
    for bump in threshold_bumps:
        open_maps = {}
        for name, out in model_outputs.items():
            spec = out["spec"]
            open_maps[name] = _build_open_map(
                ts,
                start_ts,
                end_ts,
                out["pred"],
                out["idx"],
                out["sigma"],
                spec,
                out["gate"],
                bump,
            )
        for cooldown in cooldown_steps_list:
            for max_concurrent in max_concurrent_list:
                for trade_frac in trade_frac_list:
                    res = _simulate(
                        df,
                        ts,
                        close,
                        open_maps,
                        specs,
                        start_ts,
                        end_ts,
                        cost_rt,
                        leverage,
                        trade_frac,
                        cooldown,
                        max_concurrent,
                    )
                    rows.append(
                        {
                            "threshold_bump_sigma": bump,
                            "cooldown_steps": cooldown,
                            "max_concurrent": max_concurrent,
                            "trade_frac": trade_frac,
                            "final_equity": res["final_equity"],
                            "pnl": res["pnl"],
                            "trades": res["trades"],
                            "fees": res["fees"],
                        }
                    )

    out_df = pd.DataFrame(rows).sort_values("final_equity", ascending=False)
    out_path = Path("/home/vitamind/my_project/model6/reports/backtest_trade_combo_grid.csv")
    out_df.to_csv(out_path, index=False)
    print("Saved grid results:", out_path)
    print(out_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
