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

from train_keras import apply_norm as apply_norm_v5
from train_keras_v7 import apply_norm as apply_norm_v7, TimePositionalEncoding
from model_layers import RevIN, TSMixerBlock, ITransformerBlock, LastStep, DropPath


def _unpack_obj(x):
    if isinstance(x, np.ndarray) and x.size == 1:
        return x.reshape(-1)[0]
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
    X = apply_norm_v5(X_raw, stats)

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


def _pred_meta_probs(model, stats, df, batch_size: int):
    feature_names = list(_unpack_obj(stats["feature_names"]))
    seq_len = int(_unpack_obj(stats["seq_len"]))
    X_raw = df[feature_names].to_numpy(dtype=np.float32)
    X = apply_norm_v7(X_raw, stats)

    end_indices = np.arange(seq_len - 1, len(df), dtype=np.int64)
    ds = build_dataset(X, end_indices, seq_len=seq_len, batch_size=batch_size)
    preds = model.predict(ds, verbose=0)

    if isinstance(preds, dict):
        if "cls" in preds:
            prob = preds["cls"]
        else:
            first_key = list(preds.keys())[0]
            prob = preds[first_key]
    elif isinstance(preds, (list, tuple)):
        prob = preds[0]
    else:
        prob = preds

    prob = np.asarray(prob).reshape(-1)
    out = np.full(len(df), np.nan, dtype=np.float32)
    out[end_indices] = prob.astype(np.float32)
    return out


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _fit_platt(prob, y, lr=0.1, steps=400):
    eps = 1e-6
    p = np.clip(prob, eps, 1.0 - eps)
    x = np.log(p / (1.0 - p))
    a = 1.0
    b = 0.0
    for _ in range(steps):
        logits = a * x + b
        pred = _sigmoid(logits)
        grad_a = np.mean((pred - y) * x)
        grad_b = np.mean(pred - y)
        a -= lr * grad_a
        b -= lr * grad_b
    return a, b


def _apply_platt(prob, a, b):
    eps = 1e-6
    p = np.clip(prob, eps, 1.0 - eps)
    x = np.log(p / (1.0 - p))
    return _sigmoid(a * x + b).astype(np.float32)


def _calc_bias_shift(df, idx, horizon, pred_log):
    close = df["close"].to_numpy(dtype=np.float64)
    y = np.log(close[idx + horizon] / close[idx])
    err = pred_log - y
    return float(np.median(err))


@dataclass
class ModelSpec:
    name: str
    horizon: int
    direction: str  # long or short
    strategy: str   # base or vol_gate_p60
    model_path: str
    stats_path: str
    threshold: float


@dataclass
class MetaSpec:
    name: str
    model_path: str
    stats_path: str


def _get_atr_series(df: pd.DataFrame, atr_col: str = "atr", window: int = 14) -> np.ndarray:
    """Return ATR series in price units.

    If `atr_col` exists, it's used (with ffill/bfill). Otherwise ATR is computed from OHLC
    using a rolling mean of True Range.

    This helper is intentionally self-contained so the backtest stays portable even if the
    feature parquet doesn't include a precomputed ATR column.
    """
    if atr_col and atr_col in df.columns:
        arr = df[atr_col].to_numpy(dtype=np.float64)
        if np.isfinite(arr).any():
            return pd.Series(arr).ffill().bfill().to_numpy(dtype=np.float64)

    # Fallback: compute TR/ATR from OHLC
    if {"high", "low", "close"}.issubset(df.columns):
        high = df["high"].to_numpy(dtype=np.float64)
        low = df["low"].to_numpy(dtype=np.float64)
        close = df["close"].to_numpy(dtype=np.float64)
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    else:
        # Last-resort fallback: 1-bar abs close change (still gives a scale for trailing-stop)
        close = df["close"].to_numpy(dtype=np.float64)
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        tr = np.abs(close - prev_close)

    atr = pd.Series(tr).rolling(window=window, min_periods=1).mean().to_numpy(dtype=np.float64)
    return atr


@dataclass
class Position:
    open_idx: int
    min_exit_idx: int   # earliest bar we are allowed to exit (usually == open_idx + horizon)
    max_exit_idx: int   # time stop (usually > min_exit_idx when dynamic exit is enabled)
    notional: float
    direction: str
    entry_price: float
    spec_name: str
    meta_thr: float
    best_price: float   # max close since entry (long) or min close since entry (short)


def _simulate(
    df,
    ts,
    close,
    open_map,
    start_ts,
    end_ts,
    cost_rt,
    trade_frac,
    trade_frac_long,
    trade_frac_short,
    leverage,
    cooldown_steps,
    max_concurrent,
    meta_sizing,
    meta_size_scale,
    meta_size_power,
    equity_curve_out,
    exit_mode: str = "fixed",
    exit_soft_delta: float = 0.05,
    exit_min_hold_mult: float = 1.0,
    exit_max_hold_mult: float = 4.0,
    exit_profit_buffer: float = 0.0,
    exit_atr_mult: float = 3.0,
    exit_atr_col: str = "atr",
    meta_prob_map: dict | None = None,
):
    """Simulate trades.

    exit_mode:
      - fixed: close exactly at horizon (backward-compatible)
      - soft: keep winners after horizon; exit when meta_prob < (meta_thr - exit_soft_delta)
      - atr: keep winners after horizon; exit on ATR trailing stop
      - atr_soft: ATR trailing OR soft meta exit

    Notes:
      - For dynamic exit modes (soft/atr/atr_soft) we never extend losers by default:
        at min_exit_idx we close if return <= cost_rt + exit_profit_buffer.
    """
    exit_mode = str(exit_mode or "fixed").lower().strip()
    if exit_mode not in ("fixed", "soft", "atr", "atr_soft"):
        raise ValueError(f"Unknown exit_mode: {exit_mode}")

    use_soft = exit_mode in ("soft", "atr_soft")
    use_atr = exit_mode in ("atr", "atr_soft")

    # Precompute ATR only if needed
    atr = None
    if use_atr:
        atr = _get_atr_series(df, atr_col=exit_atr_col)

    # sanitize params
    exit_soft_delta = float(exit_soft_delta)
    exit_min_hold_mult = float(exit_min_hold_mult)
    exit_max_hold_mult = float(exit_max_hold_mult)
    exit_profit_buffer = float(exit_profit_buffer)
    exit_atr_mult = float(exit_atr_mult)

    equity = 100.0
    positions: list[Position] = []
    trade_count = 0
    total_fees = 0.0
    last_trade_step = {}
    peak = equity
    max_dd = 0.0

    for i in range(len(df)):
        if ts[i] < start_ts or ts[i] >= end_ts:
            continue

        # ---- Close / manage positions
        if positions:
            still: list[Position] = []
            for pos in positions:
                # update best favorable price (close-based; we execute at close in this sim)
                if pos.direction == "long":
                    if close[i] > pos.best_price:
                        pos.best_price = float(close[i])
                else:
                    if close[i] < pos.best_price:
                        pos.best_price = float(close[i])

                close_now = False

                if exit_mode == "fixed":
                    # exact horizon close (legacy)
                    if i == pos.min_exit_idx:
                        close_now = True
                else:
                    # hard time stop
                    if i >= pos.max_exit_idx:
                        close_now = True
                    elif i >= pos.min_exit_idx:
                        # profit-only extension gate at min_exit_idx:
                        # if we are not net-profitable (after cost), close at min_exit_idx
                        ret = (close[i] / pos.entry_price) - 1.0
                        if pos.direction == "short":
                            ret = -ret
                        if i == pos.min_exit_idx and ret <= (cost_rt + exit_profit_buffer):
                            close_now = True
                        else:
                            # soft meta exit
                            if use_soft and not close_now:
                                probs = meta_prob_map.get(pos.spec_name) if meta_prob_map is not None else None
                                if probs is not None:
                                    p = float(probs[i])
                                    if (not math.isfinite(p)) or (p < (pos.meta_thr - exit_soft_delta)):
                                        close_now = True

                            # ATR trailing stop
                            if use_atr and not close_now:
                                a = float(atr[i]) if atr is not None else float("nan")
                                if (not math.isfinite(a)) or (a <= 0.0):
                                    # fallback: 1-bar abs move as a weak proxy
                                    a = float(abs(close[i] - close[i - 1])) if i > 0 else 0.0
                                if a > 0.0:
                                    if pos.direction == "long":
                                        stop = pos.best_price - exit_atr_mult * a
                                        if close[i] <= stop:
                                            close_now = True
                                    else:
                                        stop = pos.best_price + exit_atr_mult * a
                                        if close[i] >= stop:
                                            close_now = True

                if close_now:
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

        # ---- Drawdown tracking (equity changes only on closes)
        if equity > peak:
            peak = equity
        dd = (equity - peak) / peak
        if dd < max_dd:
            max_dd = dd

        # ---- Open new positions
        if i in open_map:
            for spec, meta_prob, meta_thr in open_map[i]:
                if len(positions) >= max_concurrent:
                    break
                last_i = last_trade_step.get(spec.name, -10**9)
                if i - last_i < cooldown_steps:
                    continue

                size_factor = 1.0
                if meta_sizing:
                    denom = max(1.0 - meta_thr, 1e-9)
                    size_factor = (meta_prob - meta_thr) / denom
                    if size_factor < 0.0:
                        size_factor = 0.0
                    elif size_factor > 1.0:
                        size_factor = 1.0
                    if meta_size_power != 1.0:
                        size_factor = size_factor ** meta_size_power
                    size_factor *= meta_size_scale
                    if size_factor <= 0.0:
                        continue

                if spec.direction == "long":
                    tf = trade_frac_long if trade_frac_long is not None else trade_frac
                else:
                    tf = trade_frac_short if trade_frac_short is not None else trade_frac
                notional = equity * tf * leverage * size_factor
                if notional <= 0:
                    continue

                if exit_mode == "fixed":
                    min_bars = int(spec.horizon)
                    max_bars = int(spec.horizon)
                else:
                    # NOTE: min_bars >= 1 so we don't instantly close in the same bar.
                    min_bars = max(1, int(round(spec.horizon * exit_min_hold_mult)))
                    max_bars = max(min_bars, int(round(spec.horizon * exit_max_hold_mult)))

                positions.append(
                    Position(
                        open_idx=int(i),
                        min_exit_idx=int(i + min_bars),
                        max_exit_idx=int(i + max_bars),
                        notional=float(notional),
                        direction=str(spec.direction),
                        entry_price=float(close[i]),
                        spec_name=str(spec.name),
                        meta_thr=float(meta_thr),
                        best_price=float(close[i]),
                    )
                )
                last_trade_step[spec.name] = i
                trade_count += 1

        if equity_curve_out is not None:
            equity_curve_out.append((ts[i], equity))

    # ---- Force-close remaining positions at end_ts
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
        if equity > peak:
            peak = equity
        dd = (equity - peak) / peak
        if dd < max_dd:
            max_dd = dd

    if equity_curve_out is not None:
        equity_curve_out.append((ts[end_i], equity))
    return equity, trade_count, total_fees, max_dd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="/home/vitamind/my_project/model6/data/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes_with_news_pruned.parquet")
    ap.add_argument("--meta-features", default="/home/vitamind/my_project/model6/data/meta/meta_dataset_pruned.parquet")
    ap.add_argument("--out-csv", default="/home/vitamind/my_project/model6/reports/backtest_trade_combo_meta_sweep.csv")
    ap.add_argument("--start", default="2025-01-01T00:00:00+00:00")
    ap.add_argument("--end", default="2026-01-01T00:00:00+00:00")
    ap.add_argument("--cost-rt", type=float, default=0.0015)
    ap.add_argument("--trade-frac", type=float, default=0.2)
    ap.add_argument("--trade-frac-long", type=float, default=None,
                    help="Override trade fraction for long trades.")
    ap.add_argument("--trade-frac-short", type=float, default=None,
                    help="Override trade fraction for short trades.")
    ap.add_argument("--leverage", type=float, default=1.0)
    ap.add_argument("--threshold-bump-sigma", type=float, default=0.4)
    ap.add_argument("--meta-sizing", action="store_true",
                    help="Scale position size by meta_prob above threshold.")
    ap.add_argument("--meta-size-scale", type=float, default=1.0,
                    help="Scale factor applied to meta sizing (default 1.0).")
    ap.add_argument("--meta-size-power", type=float, default=1.0,
                    help="Power applied to normalized meta sizing (default 1.0).")

    # --- Exit logic (optional; default keeps legacy fixed horizon)
    ap.add_argument(
        "--exit-mode",
        choices=["fixed", "soft", "atr", "atr_soft"],
        default="fixed",
        help="Exit logic: fixed (legacy horizon), soft (meta_prob decay after horizon), "
             "atr (ATR trailing stop after horizon), atr_soft (ATR trailing OR meta_prob decay).",
    )
    ap.add_argument(
        "--exit-soft-delta",
        type=float,
        default=0.05,
        help="Soft-exit delta: exit if meta_prob < meta_thr - delta (default 0.05).",
    )
    ap.add_argument(
        "--exit-min-hold-mult",
        type=float,
        default=1.0,
        help="Dynamic exits: earliest exit is horizon * mult bars (default 1.0).",
    )
    ap.add_argument(
        "--exit-max-hold-mult",
        type=float,
        default=4.0,
        help="Dynamic exits: time stop is horizon * mult bars (default 4.0).",
    )
    ap.add_argument(
        "--exit-profit-buffer",
        type=float,
        default=0.0,
        help="Dynamic exits: at min_exit_idx extend only if ret > cost_rt + buffer (default 0).",
    )
    ap.add_argument(
        "--exit-atr-mult",
        type=float,
        default=3.0,
        help="ATR trailing multiplier (default 3.0).",
    )
    ap.add_argument(
        "--exit-atr-col",
        default="atr",
        help="ATR column name (default 'atr'). If missing, ATR is computed from OHLC.",
    )
    ap.add_argument("--out-equity-csv", type=str, default=None,
                    help="Optional CSV path to write equity curve (timestamp,equity).")
    ap.add_argument("--rv-gate-pct", type=float, default=None,
                    help="If set, require rv >= percentile gate over the backtest period.")
    ap.add_argument("--rv-gate-col", default="rv_short",
                    help="Volatility column for rv gate (default rv_short, fallback rv).")
    ap.add_argument("--require-news-present", action="store_true",
                    help="Skip trades when news_missing==1 (requires news_missing column).")
    ap.add_argument("--cooldown-steps", type=int, default=32)
    ap.add_argument("--max-concurrent", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--meta-prob-min", type=float, default=0.50)
    ap.add_argument("--meta-prob-max", type=float, default=0.90)
    ap.add_argument("--meta-prob-step", type=float, default=0.05)
    ap.add_argument("--meta-prob-thr", type=float, default=None,
                    help="Single threshold; if set, skip sweep range.")
    ap.add_argument("--meta-prob-per-model", type=str, default=None,
                    help="Fixed per-model thresholds, e.g. h20_long=0.6,h20_short=0.65,"
                         "h80_short_v2=0.7,h160_long_v2=0.75")
    ap.add_argument("--agreement-gate", action="store_true",
                    help="Require agreement between (h20_long & h160_long_v2) or (h20_short & h80_short_v2).")
    ap.add_argument("--agreement-mode", choices=["strict", "soft"], default="strict",
                    help="strict: h20_long+h160_long_v2 OR h20_short+h80_short_v2. "
                         "soft: h20_long + (h160_long_v2 OR h80_short_v2) or "
                         "h20_short + (h80_short_v2 OR h160_long_v2).")
    ap.add_argument("--meta-prob-h20-long", type=str, default=None,
                    help="Comma-separated list for per-model sweep.")
    ap.add_argument("--meta-prob-h20-short", type=str, default=None,
                    help="Comma-separated list for per-model sweep.")
    ap.add_argument("--meta-prob-h80-short", type=str, default=None,
                    help="Comma-separated list for per-model sweep.")
    ap.add_argument("--meta-prob-h160-long", type=str, default=None,
                    help="Comma-separated list for per-model sweep.")
    ap.add_argument("--meta-model-dir", type=str,
                    default="/home/vitamind/my_project/model6/new_models/meta_2026-01-19",
                    help="Directory with meta_*.keras and *_stats.npz files.")
    ap.add_argument("--best-h20", type=str,
                    default="/home/vitamind/my_project/model6/reports/backtest_v7_long_short_sweep_best.csv")
    ap.add_argument("--best-v2", type=str,
                    default="/home/vitamind/my_project/model6/reports/backtest_v7_long_short_v2_sweep_best.csv")
    ap.add_argument("--use-purged-cv", action="store_true",
                    help="Use median thresholds from reports/purged_cv_thresholds.csv.")
    ap.add_argument("--calibrate-meta", action="store_true",
                    help="Calibrate meta probabilities on each model's val window (Platt scaling).")
    ap.add_argument("--calib-lr", type=float, default=0.1,
                    help="Learning rate for Platt scaling.")
    ap.add_argument("--calib-steps", type=int, default=400,
                    help="Gradient steps for Platt scaling.")
    args = ap.parse_args()

    start_ts = pd.to_datetime(args.start, utc=True)
    end_ts = pd.to_datetime(args.end, utc=True)

    df = pd.read_parquet(args.features)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    meta_df = pd.read_parquet(args.meta_features)
    meta_df["timestamp"] = pd.to_datetime(meta_df["timestamp"], utc=True)
    meta_df = meta_df.sort_values("timestamp").reset_index(drop=True)

    if len(df) != len(meta_df) or not np.all(df["timestamp"].to_numpy() == meta_df["timestamp"].to_numpy()):
        raise ValueError("features and meta-features are not aligned by timestamp/length.")

    ts = df["timestamp"].to_numpy()
    close = df["close"].to_numpy(dtype=np.float64)
    news_present = None
    if args.require_news_present:
        if "news_missing" not in df.columns:
            raise ValueError("--require-news-present needs 'news_missing' column in features.")
        news_present = (df["news_missing"].to_numpy(dtype=np.float32) == 0.0)

    best_h20 = pd.read_csv(args.best_h20)
    best_v2 = pd.read_csv(args.best_v2)

    def get_best(df_best, horizon, direction):
        row = df_best[(df_best.horizon == horizon) & (df_best.direction == direction)].iloc[0]
        return float(row["threshold"]), str(row["strategy"])

    h20_long_thr, h20_long_strat = get_best(best_h20, 20, "long")
    h20_short_thr, h20_short_strat = get_best(best_h20, 20, "short")
    h80_short_thr, h80_short_strat = get_best(best_v2, 80, "short")
    h160_long_thr, h160_long_strat = get_best(best_v2, 160, "long")

    if args.use_purged_cv:
        purged = pd.read_csv("/home/vitamind/my_project/model6/reports/purged_cv_thresholds.csv")
        med = purged.groupby("model")["best_threshold"].median().to_dict()
        h20_long_thr = float(med.get("h20_long", h20_long_thr))
        h20_short_thr = float(med.get("h20_short", h20_short_thr))
        h80_short_thr = float(med.get("h80_short_v2", h80_short_thr))
        h160_long_thr = float(med.get("h160_long_v2", h160_long_thr))

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

    meta_dir = Path(args.meta_model_dir).expanduser().resolve()
    meta_specs = {
        "h20_long": MetaSpec(
            name="h20_long",
            model_path=str(meta_dir / "meta_h20_long.keras"),
            stats_path=str(meta_dir / "meta_h20_long_stats.npz"),
        ),
        "h20_short": MetaSpec(
            name="h20_short",
            model_path=str(meta_dir / "meta_h20_short.keras"),
            stats_path=str(meta_dir / "meta_h20_short_stats.npz"),
        ),
        "h80_short_v2": MetaSpec(
            name="h80_short_v2",
            model_path=str(meta_dir / "meta_h80_short_v2.keras"),
            stats_path=str(meta_dir / "meta_h80_short_v2_stats.npz"),
        ),
        "h160_long_v2": MetaSpec(
            name="h160_long_v2",
            model_path=str(meta_dir / "meta_h160_long_v2.keras"),
            stats_path=str(meta_dir / "meta_h160_long_v2_stats.npz"),
        ),
    }

    # volatility gate
    rv = None
    if args.rv_gate_pct is not None:
        if args.rv_gate_col in df.columns:
            rv = df[args.rv_gate_col].to_numpy(dtype=np.float64)
        elif "rv" in df.columns:
            rv = df["rv"].to_numpy(dtype=np.float64)

    # meta probabilities per spec
    meta_prob_map = {}
    for name, mspec in meta_specs.items():
        stats = _load_stats(mspec.stats_path)
        model = _load_model(mspec.model_path)
        meta_prob_map[name] = _pred_meta_probs(model, stats, meta_df, args.batch_size)

    if args.calibrate_meta:
        label_map = {
            "h20_long": "meta_label_h20_long",
            "h20_short": "meta_label_h20_short",
            "h80_short_v2": "meta_label_h80_short_v2",
            "h160_long_v2": "meta_label_h160_long_v2",
        }
        for name, mspec in meta_specs.items():
            stats = _load_stats(mspec.stats_path)
            val_start = pd.to_datetime(_unpack_obj(stats["val_start_ts"]), utc=True)
            test_start = pd.to_datetime(_unpack_obj(stats["test_start_ts"]), utc=True)
            label_col = label_map.get(name)
            if label_col not in meta_df.columns:
                continue
            y = meta_df[label_col].to_numpy(dtype=np.float32)
            prob = meta_prob_map.get(name)
            if prob is None:
                continue
            mask = (ts >= val_start) & (ts < test_start) & ~np.isnan(y) & ~np.isnan(prob)
            if mask.sum() < 50 or len(np.unique(y[mask])) < 2:
                continue
            a, b = _fit_platt(prob[mask], y[mask], lr=args.calib_lr, steps=args.calib_steps)
            meta_prob_map[name] = _apply_platt(prob, a, b)

    model_outputs = {}
    for spec in specs:
        stats = _load_stats(spec.stats_path)
        model = _load_model(spec.model_path)
        pred, idx, sigma = _pred_for_horizon(model, stats, df, spec.horizon, args.batch_size)

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

        rv_gate_mask = None
        if args.rv_gate_pct is not None and rv is not None:
            period_mask = (ts[idx] >= start_ts) & (ts[idx] < end_ts)
            rv_gate = np.percentile(rv[idx[period_mask]], float(args.rv_gate_pct))
            rv_gate_mask = rv[idx] >= rv_gate

        model_outputs[spec.name] = {
            "spec": spec,
            "pred": pred,
            "idx": idx,
            "gate": gate_mask,
            "rv_gate": rv_gate_mask,
            "sigma": sigma,
        }

    def _parse_list(s):
        return [float(x) for x in s.split(",") if x.strip()]

    per_model_fixed = None
    if args.meta_prob_per_model:
        per_model_fixed = {}
        for part in args.meta_prob_per_model.split(","):
            if not part.strip():
                continue
            k, v = part.split("=")
            per_model_fixed[k.strip()] = float(v)

    per_model_lists = {
        "h20_long": _parse_list(args.meta_prob_h20_long) if args.meta_prob_h20_long else None,
        "h20_short": _parse_list(args.meta_prob_h20_short) if args.meta_prob_h20_short else None,
        "h80_short_v2": _parse_list(args.meta_prob_h80_short) if args.meta_prob_h80_short else None,
        "h160_long_v2": _parse_list(args.meta_prob_h160_long) if args.meta_prob_h160_long else None,
    }
    if any(v is not None for v in per_model_lists.values()):
        missing = [k for k, v in per_model_lists.items() if v is None]
        if missing:
            raise ValueError(f"Per-model sweep requires all lists; missing: {missing}")
        per_model_grid = [
            {"h20_long": a, "h20_short": b, "h80_short_v2": c, "h160_long_v2": d}
            for a in per_model_lists["h20_long"]
            for b in per_model_lists["h20_short"]
            for c in per_model_lists["h80_short_v2"]
            for d in per_model_lists["h160_long_v2"]
        ]
        thr_list = None
    elif per_model_fixed is not None:
        thr_list = None
        per_model_grid = [per_model_fixed]
    else:
        if args.meta_prob_thr is not None:
            thr_list = [float(args.meta_prob_thr)]
        else:
            thr_list = list(np.arange(args.meta_prob_min, args.meta_prob_max + 1e-9, args.meta_prob_step))
        per_model_grid = None

    rows = []
    if per_model_grid is not None:
        sweep_iter = [(None, m) for m in per_model_grid]
    else:
        sweep_iter = [(t, None) for t in thr_list]

    for meta_thr, meta_thr_map in sweep_iter:
        open_map = {}
        for out in model_outputs.values():
            spec = out["spec"]
            pred = out["pred"]
            idx = out["idx"]
            gate = out["gate"]
            rv_gate = out["rv_gate"]
            sigma = float(out["sigma"])
            thr = spec.threshold + args.threshold_bump_sigma * sigma
            meta_probs = meta_prob_map.get(spec.name)
            if meta_thr_map is not None:
                meta_thr_use = float(meta_thr_map.get(spec.name, 0.0))
            else:
                meta_thr_use = float(meta_thr)

            period_mask = (ts[idx] >= start_ts) & (ts[idx] < end_ts)
            pred = pred[period_mask]
            idx = idx[period_mask]
            if gate is not None:
                gate = gate[period_mask]
            if rv_gate is not None:
                rv_gate = rv_gate[period_mask]

            if gate is None and rv_gate is None:
                gate_iter = [True] * len(idx)
            else:
                g1 = gate if gate is not None else np.ones(len(idx), dtype=bool)
                g2 = rv_gate if rv_gate is not None else np.ones(len(idx), dtype=bool)
                gate_iter = (g1 & g2)

            for p, i, g in zip(pred, idx, gate_iter):
                if not g:
                    continue
                if news_present is not None and not bool(news_present[i]):
                    continue
                if spec.direction == "long" and p <= thr:
                    continue
                if spec.direction == "short" and p >= -thr:
                    continue
                if meta_probs is None or math.isnan(float(meta_probs[i])) or float(meta_probs[i]) < meta_thr_use:
                    continue
                open_map.setdefault(int(i), []).append((spec, float(meta_probs[i]), float(meta_thr_use)))

        if args.agreement_gate:
            keep_names = {"h20_long", "h160_long_v2", "h20_short", "h80_short_v2"}
            for i in list(open_map.keys()):
                names = {spec.name for spec, _, _ in open_map[i]}
                if args.agreement_mode == "soft":
                    long_ok = "h20_long" in names and ("h160_long_v2" in names or "h80_short_v2" in names)
                    short_ok = "h20_short" in names and ("h80_short_v2" in names or "h160_long_v2" in names)
                else:
                    long_ok = "h20_long" in names and "h160_long_v2" in names
                    short_ok = "h20_short" in names and "h80_short_v2" in names
                if not (long_ok or short_ok):
                    del open_map[i]
                else:
                    open_map[i] = [item for item in open_map[i] if item[0].name in keep_names]

        equity_curve = [] if args.out_equity_csv else None
        equity, trades, fees, max_dd = _simulate(
            df,
            ts,
            close,
            open_map,
            start_ts,
            end_ts,
            args.cost_rt,
            args.trade_frac,
            args.trade_frac_long,
            args.trade_frac_short,
            args.leverage,
            args.cooldown_steps,
            args.max_concurrent,
            args.meta_sizing,
            args.meta_size_scale,
            args.meta_size_power,
            equity_curve,
            exit_mode=args.exit_mode,
            exit_soft_delta=args.exit_soft_delta,
            exit_min_hold_mult=args.exit_min_hold_mult,
            exit_max_hold_mult=args.exit_max_hold_mult,
            exit_profit_buffer=args.exit_profit_buffer,
            exit_atr_mult=args.exit_atr_mult,
            exit_atr_col=args.exit_atr_col,
            meta_prob_map=meta_prob_map,
        )

        if args.out_equity_csv and equity_curve:
            eq_df = pd.DataFrame(equity_curve, columns=["timestamp", "equity"])
            eq_df["timestamp"] = pd.to_datetime(eq_df["timestamp"], utc=True)
            eq_df.to_csv(args.out_equity_csv, index=False)
        row = {
            "meta_prob_thr": float(meta_thr) if meta_thr is not None else None,
            "exit_mode": str(args.exit_mode),
            "exit_min_hold_mult": float(args.exit_min_hold_mult),
            "exit_max_hold_mult": float(args.exit_max_hold_mult),
            "exit_profit_buffer": float(args.exit_profit_buffer),
            "exit_soft_delta": float(args.exit_soft_delta),
            "exit_atr_mult": float(args.exit_atr_mult),
            "exit_atr_col": str(args.exit_atr_col),
            "final_equity": float(equity),
            "pnl": float(equity - 100.0),
            "trades": int(trades),
            "fees": float(fees),
            "max_dd": float(max_dd),
        }
        if meta_thr_map is not None:
            row.update({
                "meta_prob_h20_long": meta_thr_map.get("h20_long"),
                "meta_prob_h20_short": meta_thr_map.get("h20_short"),
                "meta_prob_h80_short_v2": meta_thr_map.get("h80_short_v2"),
                "meta_prob_h160_long_v2": meta_thr_map.get("h160_long_v2"),
            })
        rows.append(row)

    out_df = pd.DataFrame(rows)
    if "meta_prob_thr" in out_df.columns:
        out_df = out_df.sort_values("meta_prob_thr")
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print("Saved:", out_path)
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
