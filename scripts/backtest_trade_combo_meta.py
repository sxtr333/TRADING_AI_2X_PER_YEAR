#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import sys
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd
import tensorflow as tf
import importlib
try:
    import joblib
except Exception:  # optional; only needed for TF-IDF gate bundle
    joblib = None
try:
    from scipy.sparse import csr_matrix, hstack
except Exception:  # optional; only needed for TF-IDF gate bundle
    csr_matrix = None
    hstack = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_keras import apply_norm as apply_norm_v5
from train_keras_v7 import apply_norm as apply_norm_v7, TimePositionalEncoding
from model_layers import RevIN, TSMixerBlock, ITransformerBlock, LastStep, DropPath


def _install_keras_module_aliases():
    aliases = [
        ("keras.src.models.functional", "tf_keras.src.models.functional"),
        ("keras.src.ops.numpy", "tf_keras.src.ops.numpy"),
        ("keras.src.ops", "tf_keras.src.ops"),
        ("keras.src.layers", "tf_keras.src.layers"),
    ]
    for src_name, dst_name in aliases:
        if dst_name in sys.modules:
            continue
        try:
            sys.modules[dst_name] = importlib.import_module(src_name)
        except Exception:
            pass


_install_keras_module_aliases()


def _register_custom_objects():
    registry = {
        "model6>RevIN": RevIN,
        "model6>TSMixerBlock": TSMixerBlock,
        "model6>ITransformerBlock": ITransformerBlock,
        "model6>LastStep": LastStep,
        "model6>DropPath": DropPath,
        "RevIN": RevIN,
        "TSMixerBlock": TSMixerBlock,
        "ITransformerBlock": ITransformerBlock,
        "LastStep": LastStep,
        "DropPath": DropPath,
        "TimePositionalEncoding": TimePositionalEncoding,
    }
    try:
        tf.keras.utils.get_custom_objects().update(registry)
    except Exception:
        pass
    try:
        import keras
        keras.utils.get_custom_objects().update(registry)
    except Exception:
        pass
    return registry


CUSTOM_OBJECTS = _register_custom_objects()


def _unpack_obj(x):
    if isinstance(x, np.ndarray) and x.size == 1:
        return x.reshape(-1)[0]
    return x


def _load_stats(path: str) -> dict:
    s = np.load(path, allow_pickle=True)
    return {k: _unpack_obj(s[k]) for k in s.files}


def _load_model(path: str) -> tf.keras.Model:
    try:
        import keras
        return keras.models.load_model(
            path,
            custom_objects=CUSTOM_OBJECTS,
            compile=False,
        )
    except Exception:
        return tf.keras.models.load_model(
            path,
            custom_objects=CUSTOM_OBJECTS,
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


def _build_crowd_gate_mask(
    ts: np.ndarray,
    model_path: str,
    signals_path: str,
    thr_good: float,
    max_age_hours: float | None,
) -> np.ndarray:
    """Backward-compatible hard gate mask."""
    state = _build_crowd_gate_state(
        ts=ts,
        model_path=model_path,
        signals_path=signals_path,
        thr_good=thr_good,
        max_age_hours=max_age_hours,
    )
    return state["hard_pass"]


def _build_crowd_gate_state(
    ts: np.ndarray,
    model_path: str,
    signals_path: str,
    thr_good: float,
    max_age_hours: float | None,
) -> dict:
    """Return per-bar Telegram crowd state aligned to bars.

    Keys:
      - prob_good: float32 P(good), NaN when no signal
      - age_h: float32 hours since latest signal, inf when no signal
      - has_signal: bool latest signal exists
      - is_fresh: bool has_signal and (age <= max_age_hours if configured)
      - hard_pass: bool is_fresh and prob_good >= thr_good
    """
    if joblib is None or csr_matrix is None or hstack is None:
        raise RuntimeError("crowd gate dependencies are unavailable in this runtime.")
    bundle = joblib.load(model_path)
    vec = bundle["vectorizer"]
    clf = bundle["classifier"]
    labels = list(bundle.get("labels", []))
    num_cols = list(bundle.get("num_cols", ["direction_long", "parse_confidence", "text_len", "is_candidate_f"]))
    if "good" not in labels:
        raise ValueError("crowd gate model has no 'good' class in labels.")
    good_idx = labels.index("good")

    s = pd.read_parquet(signals_path).copy()
    s["timestamp_utc"] = pd.to_datetime(s["timestamp_utc"], utc=True, errors="coerce")
    s = s[s["timestamp_utc"].notna()].sort_values("timestamp_utc").reset_index(drop=True)
    if s.empty:
        n = len(ts)
        return {
            "prob_good": np.full(n, np.nan, dtype=np.float32),
            "age_h": np.full(n, np.inf, dtype=np.float32),
            "has_signal": np.zeros(n, dtype=bool),
            "is_fresh": np.zeros(n, dtype=bool),
            "hard_pass": np.zeros(n, dtype=bool),
        }

    symbol = s.get("symbol", "").fillna("").astype(str)
    direction = s.get("direction", "").fillna("").astype(str)
    text = s.get("text", "").fillna("").astype(str)
    s["text_in"] = (symbol + " " + direction + " " + text).str.strip()
    s["direction_long"] = (direction.str.lower() == "long").astype(np.float32)
    s["parse_confidence"] = pd.to_numeric(s.get("parse_confidence", 0.0), errors="coerce").fillna(0.0).astype(np.float32)
    s["text_len"] = s["text_in"].astype(str).str.len().astype(np.float32)
    s["is_candidate_f"] = pd.to_numeric(s.get("is_candidate", False), errors="coerce").fillna(0.0).astype(np.float32)

    x_text = vec.transform(s["text_in"].astype(str))
    x_num = csr_matrix(s[num_cols].to_numpy(dtype=np.float32))
    x = hstack([x_text, x_num], format="csr")
    prob = clf.predict_proba(x)
    s["gate_prob_good"] = prob[:, good_idx].astype(np.float32)

    bars = pd.DataFrame({"timestamp": pd.to_datetime(ts, utc=True)})
    merged = pd.merge_asof(
        bars.sort_values("timestamp"),
        s[["timestamp_utc", "gate_prob_good"]].sort_values("timestamp_utc"),
        left_on="timestamp",
        right_on="timestamp_utc",
        direction="backward",
        allow_exact_matches=True,
    )
    prob_good = merged["gate_prob_good"].to_numpy(dtype=np.float32)
    has_signal = merged["timestamp_utc"].notna().to_numpy(dtype=bool)
    age_h = (merged["timestamp"] - merged["timestamp_utc"]).dt.total_seconds() / 3600.0
    age_h = age_h.fillna(np.inf).to_numpy(dtype=np.float32)

    if max_age_hours is not None and max_age_hours > 0:
        is_fresh = has_signal & (age_h <= float(max_age_hours))
    else:
        is_fresh = has_signal.copy()

    hard_pass = is_fresh & np.isfinite(prob_good) & (prob_good >= float(thr_good))
    return {
        "prob_good": prob_good.astype(np.float32),
        "age_h": age_h.astype(np.float32),
        "has_signal": has_signal.astype(bool),
        "is_fresh": is_fresh.astype(bool),
        "hard_pass": hard_pass.astype(bool),
    }


def _build_crowd_soft_size_mult(
    crowd_state: dict,
    thr_good: float,
    half_life_hours: float,
    alpha_up: float,
    alpha_down: float,
    min_mult: float,
    max_mult: float,
) -> np.ndarray:
    """Per-bar size multiplier from Telegram quality (neutral=1.0)."""
    prob = np.asarray(crowd_state["prob_good"], dtype=np.float32)
    age_h = np.asarray(crowd_state["age_h"], dtype=np.float32)
    is_fresh = np.asarray(crowd_state["is_fresh"], dtype=bool)

    mult = np.ones(len(prob), dtype=np.float32)
    valid = is_fresh & np.isfinite(prob)
    if not np.any(valid):
        return mult

    if half_life_hours is not None and float(half_life_hours) > 0:
        decay = np.exp(-np.maximum(age_h, 0.0) / float(half_life_hours)).astype(np.float32)
    else:
        decay = np.ones(len(prob), dtype=np.float32)

    thr = float(thr_good)
    good_mask = valid & (prob >= thr)
    bad_mask = valid & (prob < thr)

    if np.any(good_mask):
        good_norm = (prob[good_mask] - thr) / max(1.0 - thr, 1e-9)
        mult[good_mask] = 1.0 + float(alpha_up) * good_norm * decay[good_mask]

    if np.any(bad_mask) and float(alpha_down) > 0.0:
        bad_norm = (thr - prob[bad_mask]) / max(thr, 1e-9)
        mult[bad_mask] = 1.0 - float(alpha_down) * bad_norm * decay[bad_mask]

    mult = np.clip(mult, float(min_mult), float(max_mult))
    return mult.astype(np.float32)


def _pick_first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _build_mm_supervisor_state(
    ts: np.ndarray,
    signals_path: str,
    max_age_hours: float | None = 72.0,
) -> dict:
    """Align MM/VLM supervisor outputs to bar timestamps (last known state).

    Expected columns (flexible names):
      - timestamp: one of [timestamp, timestamp_utc, ts]
      - mm_stop_hunt_long_prob (fallback stop_hunt_long_prob)
      - mm_stop_hunt_short_prob (fallback stop_hunt_short_prob)
      - mm_fake_reversal_prob (fallback fake_reversal_prob)
      - mm_confidence (fallback confidence)
      - mm_crowd_imbalance_side (fallback crowd_imbalance_side)
      - mm_plan (fallback plan/mm_plan)
    """
    sig_path = str(signals_path)
    if sig_path.lower().endswith(".parquet"):
        sig = pd.read_parquet(sig_path)
    else:
        sig = pd.read_csv(sig_path)

    ts_col = _pick_first_col(sig, ["timestamp", "timestamp_utc", "ts"])
    if ts_col is None:
        raise ValueError("mm supervisor file must contain one of: timestamp, timestamp_utc, ts")
    sig[ts_col] = pd.to_datetime(sig[ts_col], utc=True, errors="coerce")
    sig = sig.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)
    if len(sig) == 0:
        n = len(ts)
        return {
            "long_risk": np.full(n, np.nan, dtype=np.float32),
            "short_risk": np.full(n, np.nan, dtype=np.float32),
            "fake_prob": np.full(n, np.nan, dtype=np.float32),
            "conf": np.full(n, np.nan, dtype=np.float32),
            "crowd_side": np.full(n, "", dtype=object),
            "plan": np.full(n, "", dtype=object),
            "age_h": np.full(n, np.nan, dtype=np.float32),
            "has_signal": np.zeros(n, dtype=bool),
            "is_fresh": np.zeros(n, dtype=bool),
        }

    col_long = _pick_first_col(sig, ["mm_stop_hunt_long_prob", "stop_hunt_long_prob"])
    col_short = _pick_first_col(sig, ["mm_stop_hunt_short_prob", "stop_hunt_short_prob"])
    col_fake = _pick_first_col(sig, ["mm_fake_reversal_prob", "fake_reversal_prob"])
    col_conf = _pick_first_col(sig, ["mm_confidence", "confidence"])
    col_side = _pick_first_col(sig, ["mm_crowd_imbalance_side", "crowd_imbalance_side"])
    col_plan = _pick_first_col(sig, ["mm_plan", "plan"])

    work = pd.DataFrame({
        "sig_ts": sig[ts_col],
        "long_risk": pd.to_numeric(sig[col_long], errors="coerce") if col_long else np.nan,
        "short_risk": pd.to_numeric(sig[col_short], errors="coerce") if col_short else np.nan,
        "fake_prob": pd.to_numeric(sig[col_fake], errors="coerce") if col_fake else np.nan,
        "conf": pd.to_numeric(sig[col_conf], errors="coerce") if col_conf else np.nan,
        "crowd_side": (sig[col_side].astype(str).str.lower().str.strip() if col_side else ""),
        "plan": (sig[col_plan].astype(str).str.lower().str.strip() if col_plan else ""),
    }).sort_values("sig_ts")

    # Keep last state per timestamp to avoid duplicate clashes.
    work = work.groupby("sig_ts", as_index=False).last()

    bars = pd.DataFrame({"bar_ts": pd.to_datetime(ts, utc=True)})
    merged = pd.merge_asof(
        bars.sort_values("bar_ts"),
        work,
        left_on="bar_ts",
        right_on="sig_ts",
        direction="backward",
    )
    age_h = (merged["bar_ts"] - merged["sig_ts"]).dt.total_seconds() / 3600.0
    has_signal = merged["sig_ts"].notna().to_numpy(dtype=bool)
    if max_age_hours is not None and float(max_age_hours) > 0:
        is_fresh = has_signal & (age_h.to_numpy(dtype=np.float32) <= float(max_age_hours))
    else:
        is_fresh = has_signal.copy()

    def _clip01(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        arr = np.where(np.isfinite(arr), arr, np.nan)
        return np.clip(arr, 0.0, 1.0)

    crowd_side = merged["crowd_side"].fillna("").astype(str).str.lower().str.strip()
    plan = merged["plan"].fillna("").astype(str).str.lower().str.strip()
    return {
        "long_risk": _clip01(merged["long_risk"].to_numpy(dtype=np.float32)),
        "short_risk": _clip01(merged["short_risk"].to_numpy(dtype=np.float32)),
        "fake_prob": _clip01(merged["fake_prob"].to_numpy(dtype=np.float32)),
        "conf": _clip01(merged["conf"].to_numpy(dtype=np.float32)),
        "crowd_side": crowd_side.to_numpy(dtype=object),
        "plan": plan.to_numpy(dtype=object),
        "age_h": age_h.to_numpy(dtype=np.float32),
        "has_signal": has_signal.astype(bool),
        "is_fresh": is_fresh.astype(bool),
    }


def _build_mm_supervisor_mult(
    mm_state: dict,
    alpha_up: float,
    alpha_down: float,
    plan_bias: float,
    fake_reversal_down: float,
    fake_reversal_dir_bias: float,
    min_mult: float,
    max_mult: float,
    half_life_hours: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-bar (long_mult, short_mult) from MM state."""
    n = len(mm_state["is_fresh"])
    long_mult = np.ones(n, dtype=np.float32)
    short_mult = np.ones(n, dtype=np.float32)

    is_fresh = np.asarray(mm_state["is_fresh"], dtype=bool)
    age_h = np.asarray(mm_state["age_h"], dtype=np.float32)
    long_risk = np.asarray(mm_state["long_risk"], dtype=np.float32)
    short_risk = np.asarray(mm_state["short_risk"], dtype=np.float32)
    fake_prob = np.asarray(mm_state["fake_prob"], dtype=np.float32)
    conf = np.asarray(mm_state["conf"], dtype=np.float32)
    crowd_side = np.asarray(mm_state["crowd_side"], dtype=object)
    plan = np.asarray(mm_state["plan"], dtype=object)

    valid = is_fresh & np.isfinite(long_risk) & np.isfinite(short_risk)
    if not np.any(valid):
        return long_mult, short_mult

    if half_life_hours is not None and float(half_life_hours) > 0:
        decay = np.exp(-np.maximum(age_h, 0.0) / float(half_life_hours)).astype(np.float32)
    else:
        decay = np.ones(n, dtype=np.float32)

    # Base risk transfer: penalize side with higher sweep risk, slightly boost opposite side.
    lr = np.where(np.isfinite(long_risk), long_risk, 0.0) * np.where(np.isfinite(conf), conf, 0.0)
    sr = np.where(np.isfinite(short_risk), short_risk, 0.0) * np.where(np.isfinite(conf), conf, 0.0)
    fp = np.where(np.isfinite(fake_prob), fake_prob, 0.0) * np.where(np.isfinite(conf), conf, 0.0)

    long_mult[valid] = (
        1.0
        - float(alpha_down) * lr[valid] * decay[valid]
        + float(alpha_up) * sr[valid] * decay[valid]
        - float(fake_reversal_down) * fp[valid] * decay[valid]
    )
    short_mult[valid] = (
        1.0
        - float(alpha_down) * sr[valid] * decay[valid]
        + float(alpha_up) * lr[valid] * decay[valid]
        - float(fake_reversal_down) * fp[valid] * decay[valid]
    )

    # Directional fake-reversal penalty:
    # if stop-hunt risk is imbalanced to one side, "fake reversal" should penalize
    # mainly that side instead of both sides equally.
    if float(fake_reversal_dir_bias) > 0:
        risk_diff = (lr - sr).astype(np.float32)
        # Smoothly map imbalance to [0..1] side-penalty strength.
        imb_long = np.clip(np.tanh(np.maximum(risk_diff, 0.0) / 0.15), 0.0, 1.0)
        imb_short = np.clip(np.tanh(np.maximum(-risk_diff, 0.0) / 0.15), 0.0, 1.0)
        long_mult[valid] -= float(fake_reversal_dir_bias) * fp[valid] * decay[valid] * imb_long[valid]
        short_mult[valid] -= float(fake_reversal_dir_bias) * fp[valid] * decay[valid] * imb_short[valid]

    # Plan-conditioned bias from "crowd side + mm plan".
    for i in np.where(valid)[0]:
        side = str(crowd_side[i])
        pl = str(plan[i])
        if side not in ("long", "short"):
            continue
        if pl == "sweep_against_crowd":
            if side == "long":
                long_mult[i] -= float(plan_bias) * float(decay[i])
                short_mult[i] += float(plan_bias) * float(decay[i])
            else:
                short_mult[i] -= float(plan_bias) * float(decay[i])
                long_mult[i] += float(plan_bias) * float(decay[i])
        elif pl == "push_with_crowd":
            if side == "long":
                long_mult[i] += float(plan_bias) * float(decay[i])
                short_mult[i] -= float(plan_bias) * float(decay[i])
            else:
                short_mult[i] += float(plan_bias) * float(decay[i])
                long_mult[i] -= float(plan_bias) * float(decay[i])

    long_mult = np.clip(long_mult, float(min_mult), float(max_mult)).astype(np.float32)
    short_mult = np.clip(short_mult, float(min_mult), float(max_mult)).astype(np.float32)
    return long_mult, short_mult


def _build_mm_crowd_first_mult(
    mm_state: dict,
    min_conf: float,
    risk_gap: float,
    alpha_dir: float,
    min_mult: float,
    max_mult: float,
    half_life_hours: float | None = None,
    require_plan: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build per-bar directional multipliers for crowd-first logic (no hard blocking).

    Logic priority:
      1) If fresh + confident and explicit plan/side exists:
         - sweep_against_crowd: trade against crowd side
         - push_with_crowd: trade with crowd side
      2) If plan is absent/unknown and require_plan=False:
         fallback by risk gap between long/short stop-hunt risk.
    """
    n = len(mm_state["is_fresh"])
    long_mult = np.ones(n, dtype=np.float32)
    short_mult = np.ones(n, dtype=np.float32)
    active = np.zeros(n, dtype=bool)

    is_fresh = np.asarray(mm_state["is_fresh"], dtype=bool)
    age_h = np.asarray(mm_state["age_h"], dtype=np.float32)
    conf = np.asarray(mm_state["conf"], dtype=np.float32)
    long_risk = np.asarray(mm_state["long_risk"], dtype=np.float32)
    short_risk = np.asarray(mm_state["short_risk"], dtype=np.float32)
    crowd_side = np.asarray(mm_state["crowd_side"], dtype=object)
    plan = np.asarray(mm_state["plan"], dtype=object)

    if half_life_hours is not None and float(half_life_hours) > 0:
        decay = np.exp(-np.maximum(age_h, 0.0) / float(half_life_hours)).astype(np.float32)
    else:
        decay = np.ones(n, dtype=np.float32)

    alpha_dir = max(0.0, float(alpha_dir))
    valid = is_fresh & np.isfinite(conf) & (conf >= float(min_conf))
    for i in np.where(valid)[0]:
        side = str(crowd_side[i])
        pl = str(plan[i])
        applied = False
        up = 1.0 + alpha_dir * float(decay[i])
        dn = 1.0 - alpha_dir * float(decay[i])

        if side in ("long", "short"):
            if pl == "sweep_against_crowd":
                if side == "long":
                    long_mult[i] *= dn
                    short_mult[i] *= up
                else:
                    short_mult[i] *= dn
                    long_mult[i] *= up
                applied = True
            elif pl == "push_with_crowd":
                if side == "long":
                    short_mult[i] *= dn
                    long_mult[i] *= up
                else:
                    long_mult[i] *= dn
                    short_mult[i] *= up
                applied = True

        if (not applied) and (not require_plan):
            lr = float(long_risk[i]) if np.isfinite(long_risk[i]) else np.nan
            sr = float(short_risk[i]) if np.isfinite(short_risk[i]) else np.nan
            if math.isfinite(lr) and math.isfinite(sr):
                diff = lr - sr
                if diff >= float(risk_gap):
                    long_mult[i] *= dn
                    short_mult[i] *= up
                    applied = True
                elif diff <= -float(risk_gap):
                    short_mult[i] *= dn
                    long_mult[i] *= up
                    applied = True

        if applied:
            active[i] = True

    long_mult = np.clip(long_mult, float(min_mult), float(max_mult)).astype(np.float32)
    short_mult = np.clip(short_mult, float(min_mult), float(max_mult)).astype(np.float32)
    return long_mult, short_mult, active


def _build_regime_risk_threshold(
    df: pd.DataFrame,
    base_thr: float,
    trend_relax: float,
    flat_tighten: float,
    adx_col: str,
    rv_col: str,
    adx_trend_thr: float,
    adx_flat_thr: float,
    rv_trend_pct: float,
    rv_flat_pct: float,
) -> np.ndarray:
    n = len(df)
    thr = np.full(n, float(base_thr), dtype=np.float32)
    if adx_col not in df.columns or rv_col not in df.columns:
        return thr

    adx = pd.to_numeric(df[adx_col], errors="coerce").to_numpy(dtype=np.float32)
    rv = pd.to_numeric(df[rv_col], errors="coerce").to_numpy(dtype=np.float32)
    finite_rv = rv[np.isfinite(rv)]
    if finite_rv.size < 100:
        return thr
    q_hi = float(np.quantile(finite_rv, float(rv_trend_pct)))
    q_lo = float(np.quantile(finite_rv, float(rv_flat_pct)))

    trend_mask = (np.isfinite(adx) & (adx >= float(adx_trend_thr))) | (np.isfinite(rv) & (rv >= q_hi))
    flat_mask = (np.isfinite(adx) & (adx <= float(adx_flat_thr))) & (np.isfinite(rv) & (rv <= q_lo))

    thr[trend_mask] = float(base_thr) + float(trend_relax)
    thr[flat_mask] = float(base_thr) - float(flat_tighten)
    return np.clip(thr, 0.0, 0.999).astype(np.float32)


def _build_mm_independent_open_map(
    mm_state: dict | None,
    spec_long: "ModelSpec",
    spec_short: "ModelSpec",
    min_conf: float,
    risk_gap: float,
    min_prob: float,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    ts: np.ndarray,
) -> dict[int, list[tuple["ModelSpec", float, float]]]:
    """Build independent MM-only entries map.

    Decision order:
      1) if plan+crowd_side exist -> follow plan (against/with crowd),
      2) else fallback by risk imbalance with minimum risk gap.
    """
    out: dict[int, list[tuple["ModelSpec", float, float]]] = {}
    if mm_state is None:
        return out

    is_fresh = np.asarray(mm_state["is_fresh"], dtype=bool)
    conf = np.asarray(mm_state["conf"], dtype=np.float32)
    long_risk = np.asarray(mm_state["long_risk"], dtype=np.float32)
    short_risk = np.asarray(mm_state["short_risk"], dtype=np.float32)
    crowd_side = np.asarray(mm_state["crowd_side"], dtype=object)
    plan = np.asarray(mm_state["plan"], dtype=object)

    for i in range(len(ts)):
        if ts[i] < start_ts or ts[i] >= end_ts:
            continue
        if (not bool(is_fresh[i])) or (not np.isfinite(conf[i])) or float(conf[i]) < float(min_conf):
            continue

        lr = float(long_risk[i]) if np.isfinite(long_risk[i]) else np.nan
        sr = float(short_risk[i]) if np.isfinite(short_risk[i]) else np.nan
        side = str(crowd_side[i]).lower().strip()
        pl = str(plan[i]).lower().strip()

        chosen = None
        score = 0.0
        if side in ("long", "short"):
            if pl == "sweep_against_crowd":
                chosen = "short" if side == "long" else "long"
            elif pl == "push_with_crowd":
                chosen = side

        if chosen is not None:
            score = (sr if chosen == "short" else lr)
        else:
            if (not np.isfinite(lr)) or (not np.isfinite(sr)):
                continue
            diff = lr - sr
            if diff >= float(risk_gap):
                chosen, score = "short", sr
            elif diff <= -float(risk_gap):
                chosen, score = "long", lr
            else:
                continue

        if (not np.isfinite(score)) or float(score) < float(min_prob):
            continue

        spec = spec_long if chosen == "long" else spec_short
        out.setdefault(int(i), []).append((spec, float(score), 0.0))
    return out


def _limit_open_map_activity(
    open_map: dict[int, list[tuple["ModelSpec", float, float]]],
    ts: np.ndarray,
    cooldown_steps: int,
    max_trades_per_day: int,
) -> dict[int, list[tuple["ModelSpec", float, float]]]:
    """Throttle open-map activity to avoid MM overtrading."""
    if not open_map:
        return open_map
    cd = max(0, int(cooldown_steps))
    per_day_cap = max(0, int(max_trades_per_day))
    if cd <= 0 and per_day_cap <= 0:
        return open_map

    out: dict[int, list[tuple["ModelSpec", float, float]]] = {}
    last_i_by_spec: dict[str, int] = {}
    daily_count: dict[tuple[str, str], int] = {}

    for i in sorted(open_map.keys()):
        kept = []
        tsi = pd.Timestamp(ts[i])
        if tsi.tzinfo is None:
            tsi = tsi.tz_localize("UTC")
        else:
            tsi = tsi.tz_convert("UTC")
        day = str(tsi.date())
        for item in open_map[i]:
            spec = item[0]
            sname = str(spec.name)
            last_i = last_i_by_spec.get(sname, -10**9)
            if cd > 0 and (i - last_i) < cd:
                continue
            if per_day_cap > 0:
                key = (sname, day)
                c = daily_count.get(key, 0)
                if c >= per_day_cap:
                    continue
                daily_count[key] = c + 1
            kept.append(item)
            last_i_by_spec[sname] = i
        if kept:
            out[i] = kept
    return out


def _build_chronos_ret_getter(
    ts: np.ndarray,
    close: np.ndarray,
    model_id: str,
    prediction_length: int,
    context_length: int,
    num_samples: int,
    device: str,
) -> Callable[[int], float]:
    """Return a cached function idx -> median forecast log-return from Chronos.

    The getter lazily loads Chronos and computes forecasts only for requested indices.
    Returns NaN when forecast cannot be computed.
    """
    prediction_length = max(1, int(prediction_length))
    context_length = max(8, int(context_length))
    num_samples = max(1, int(num_samples))

    model = None
    torch = None
    cache: dict[int, float] = {}
    predict_mode: dict[str, object] = {
        "input_rank": None,         # 2 or 3
        "use_num_samples": None,    # bool
    }

    def _load():
        nonlocal model, torch
        if model is not None:
            return model, torch

        import importlib
        torch = importlib.import_module("torch")
        chronos_mod = importlib.import_module("chronos")
        pipeline_cls = getattr(chronos_mod, "Chronos2Pipeline", None) or getattr(chronos_mod, "ChronosPipeline", None)
        if pipeline_cls is None:
            raise RuntimeError("Chronos package is installed but no ChronosPipeline/Chronos2Pipeline class found.")

        dev = (device or "auto").lower()
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"

        model = pipeline_cls.from_pretrained(model_id, device_map=dev)
        return model, torch

    def _get(i: int) -> float:
        i = int(i)
        if i in cache:
            return cache[i]

        try:
            if i < 0 or i >= len(close):
                cache[i] = float("nan")
                return cache[i]

            m, th = _load()
            s = max(0, i - context_length + 1)
            context = np.asarray(close[s : i + 1], dtype=np.float32)
            if context.size < 8 or not np.isfinite(context).all():
                cache[i] = float("nan")
                return cache[i]

            ctx1 = th.tensor(context, dtype=th.float32)

            def _ctx_by_rank(rank: int):
                if rank == 2:
                    return ctx1.unsqueeze(0)  # [1, L]
                return ctx1.unsqueeze(0).unsqueeze(0)  # [1, 1, L] for Chronos-2

            def _call_predict(inp, use_num_samples: bool):
                kwargs = {"prediction_length": prediction_length}
                if use_num_samples:
                    kwargs["num_samples"] = num_samples
                return m.predict(inp, **kwargs)

            pred = None
            if predict_mode["input_rank"] is not None and predict_mode["use_num_samples"] is not None:
                pred = _call_predict(
                    _ctx_by_rank(int(predict_mode["input_rank"])),
                    bool(predict_mode["use_num_samples"]),
                )
            else:
                last_err = None
                for rank in (2, 3):
                    for use_ns in (True, False):
                        try:
                            pred = _call_predict(_ctx_by_rank(rank), use_ns)
                            predict_mode["input_rank"] = rank
                            predict_mode["use_num_samples"] = use_ns
                            last_err = None
                            break
                        except Exception as e:
                            last_err = e
                    if pred is not None:
                        break
                if pred is None:
                    raise last_err if last_err is not None else RuntimeError("Chronos predict failed.")

            if isinstance(pred, (list, tuple)):
                if len(pred) == 0:
                    cache[i] = float("nan")
                    return cache[i]
                pred = pred[0]
            if hasattr(pred, "detach"):
                arr = pred.detach().cpu().numpy()
            else:
                arr = np.asarray(pred)
            arr = np.squeeze(arr)
            if arr.ndim == 0:
                cache[i] = float("nan")
                return cache[i]

            step = min(prediction_length - 1, arr.shape[-1] - 1)
            y_hat = float(np.median(arr[..., step]))

            p0 = max(float(close[i]), 1e-12)
            y_hat = max(float(y_hat), 1e-12)
            cache[i] = float(np.log(y_hat / p0))
            return cache[i]
        except Exception:
            cache[i] = float("nan")
            return cache[i]

    return _get


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
    meta_prob: float    # meta prob at entry
    meta_thr: float
    size_factor: float  # final size factor applied to base notional
    crowd_mult: float   # crowd multiplier at entry (1.0 when disabled)
    mm_mult: float      # MM multiplier at entry (1.0 when disabled)
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
    trades_out,
    exit_mode: str = "fixed",
    exit_soft_delta: float = 0.05,
    exit_min_hold_mult: float = 1.0,
    exit_max_hold_mult: float = 4.0,
    exit_profit_buffer: float = 0.0,
    exit_atr_mult: float = 3.0,
    exit_atr_col: str = "atr",
    meta_prob_map: dict | None = None,
    crowd_size_mult: np.ndarray | None = None,
    mm_long_mult: np.ndarray | None = None,
    mm_short_mult: np.ndarray | None = None,
    chronos_ret_getter: Callable[[int], float] | None = None,
    chronos_exit_enable: bool = False,
    chronos_exit_check_every: int = 4,
    chronos_exit_flip_logret: float = 0.0,
    mm_independent_size_mult: float = 1.0,
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

                            # Chronos-based optional exit override.
                            if (
                                chronos_exit_enable
                                and (not close_now)
                                and chronos_ret_getter is not None
                                and chronos_exit_check_every > 0
                                and (i % int(chronos_exit_check_every) == 0)
                            ):
                                cr = float(chronos_ret_getter(i))
                                flip = float(max(chronos_exit_flip_logret, 0.0))
                                if math.isfinite(cr):
                                    if pos.direction == "long" and cr <= -flip:
                                        close_now = True
                                    if pos.direction == "short" and cr >= flip:
                                        close_now = True

                if close_now:
                    ret = (close[i] / pos.entry_price) - 1.0
                    if pos.direction == "short":
                        ret = -ret
                    pnl = pos.notional * ret
                    fee = pos.notional * cost_rt
                    equity += pnl - fee
                    total_fees += fee
                    if trades_out is not None:
                        trades_out.append(
                            {
                                "entry_ts": ts[pos.open_idx],
                                "exit_ts": ts[i],
                                "entry_price": pos.entry_price,
                                "exit_price": float(close[i]),
                                "direction": pos.direction,
                                "model": pos.spec_name,
                                "notional": float(pos.notional),
                                "ret": float(ret),
                                "pnl_gross": float(pnl),
                                "fee": float(fee),
                                "pnl_net": float(pnl - fee),
                                "hold_bars": int(i - pos.open_idx),
                                "meta_prob_entry": float(pos.meta_prob),
                                "meta_thr": float(pos.meta_thr),
                                "size_factor": float(pos.size_factor),
                                "crowd_mult": float(pos.crowd_mult),
                                "mm_mult": float(pos.mm_mult),
                            }
                        )
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
                crowd_m = 1.0
                mm_m = 1.0
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
                if crowd_size_mult is not None:
                    cm = float(crowd_size_mult[i])
                    if not math.isfinite(cm) or cm <= 0.0:
                        continue
                    crowd_m = cm
                    size_factor *= cm
                    if size_factor <= 0.0:
                        continue
                if spec.direction == "long" and mm_long_mult is not None:
                    mm = float(mm_long_mult[i])
                    if not math.isfinite(mm) or mm <= 0.0:
                        continue
                    mm_m = mm
                    size_factor *= mm
                    if size_factor <= 0.0:
                        continue
                if spec.direction == "short" and mm_short_mult is not None:
                    mm = float(mm_short_mult[i])
                    if not math.isfinite(mm) or mm <= 0.0:
                        continue
                    mm_m = mm
                    size_factor *= mm
                    if size_factor <= 0.0:
                        continue

                if spec.direction == "long":
                    tf = trade_frac_long if trade_frac_long is not None else trade_frac
                else:
                    tf = trade_frac_short if trade_frac_short is not None else trade_frac
                if str(spec.name).startswith("mm_only_"):
                    size_factor *= float(mm_independent_size_mult)
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
                        meta_prob=float(meta_prob),
                        meta_thr=float(meta_thr),
                        size_factor=float(size_factor),
                        crowd_mult=float(crowd_m),
                        mm_mult=float(mm_m),
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
        if trades_out is not None:
            trades_out.append(
                {
                    "entry_ts": ts[pos.open_idx],
                    "exit_ts": ts[end_i],
                    "entry_price": pos.entry_price,
                    "exit_price": float(close[end_i]),
                    "direction": pos.direction,
                    "model": pos.spec_name,
                    "notional": float(pos.notional),
                    "ret": float(ret),
                    "pnl_gross": float(pnl),
                    "fee": float(fee),
                    "pnl_net": float(pnl - fee),
                    "hold_bars": int(end_i - pos.open_idx),
                    "meta_prob_entry": float(pos.meta_prob),
                    "meta_thr": float(pos.meta_thr),
                    "size_factor": float(pos.size_factor),
                    "crowd_mult": float(pos.crowd_mult),
                    "mm_mult": float(pos.mm_mult),
                }
            )
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
    ap.add_argument("--out-trades-csv", type=str, default=None,
                    help="Optional CSV path to write trades (entry_ts, exit_ts, prices, direction, model).")
    ap.add_argument("--out-monthly-csv", type=str, default=None,
                    help="Optional CSV path to write monthly breakdown (requires --out-trades-csv).")
    ap.add_argument("--out-summary-json", type=str, default=None,
                    help="Optional JSON path to write a short run summary (best row + settings).")
    ap.add_argument("--preset", choices=["mm_r13_balance", "mm_r13_aggressive"], default=None,
                    help="Convenience preset for reproducible runs. Does not override explicitly provided CLI flags.")
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
    ap.add_argument("--crowd-gate-model", type=str, default=None,
                    help="Optional joblib gate model (TF-IDF+LR).")
    ap.add_argument("--crowd-gate-signals", type=str, default=None,
                    help="Signals parquet used to compute gate probabilities.")
    ap.add_argument("--crowd-gate-thr-good", type=float, default=0.40,
                    help="Allow new trades only when P(good) >= threshold.")
    ap.add_argument("--crowd-gate-max-age-hours", type=float, default=72.0,
                    help="Max age of last signal used by crowd gate.")
    ap.add_argument("--crowd-gate-mode", choices=["hard", "soft", "hybrid"], default="hard",
                    help="hard: block trades by gate; soft: size-only crowd effect; "
                         "hybrid: hard-block only very bad fresh signals + size effect.")
    ap.add_argument("--crowd-hybrid-bad-thr", type=float, default=0.20,
                    help="For hybrid mode: block only when fresh P(good) < threshold.")
    ap.add_argument("--crowd-soft-alpha-up", type=float, default=0.25,
                    help="Soft mode: max positive size boost from fresh high-quality crowd signals.")
    ap.add_argument("--crowd-soft-alpha-down", type=float, default=0.10,
                    help="Soft mode: max negative size reduction from fresh low-quality crowd signals.")
    ap.add_argument("--crowd-soft-half-life-hours", type=float, default=24.0,
                    help="Soft mode: exponential half-life for crowd effect by signal age.")
    ap.add_argument("--crowd-soft-min-mult", type=float, default=0.85,
                    help="Soft mode: lower clip for crowd size multiplier.")
    ap.add_argument("--crowd-soft-max-mult", type=float, default=1.25,
                    help="Soft mode: upper clip for crowd size multiplier.")
    ap.add_argument("--crowd-soft-min-coverage", type=float, default=0.10,
                    help="If fresh crowd coverage is below this, disable down-scaling (alpha_down=0).")
    ap.add_argument("--mm-supervisor-signals", type=str, default=None,
                    help="Optional MM/VLM supervisor file (.parquet/.csv) with per-image/per-time mm_* outputs.")
    ap.add_argument("--mm-supervisor-max-age-hours", type=float, default=72.0,
                    help="Max age of latest MM supervisor signal used for a bar.")
    ap.add_argument("--mm-supervisor-mode", choices=["off", "soft", "hybrid", "crowd_first"], default="off",
                    help="off: disabled; soft: direction-aware sizing only; hybrid: sizing + hard skip on extreme risk; "
                         "crowd_first: prioritize side from crowd/MM plan with fallback risk-gap.")
    ap.add_argument("--mm-supervisor-alpha-up", type=float, default=0.20,
                    help="Soft mode: boost opposite side when sweep risk concentrates.")
    ap.add_argument("--mm-supervisor-alpha-down", type=float, default=0.25,
                    help="Soft mode: reduce side exposed to likely stop hunt.")
    ap.add_argument("--mm-supervisor-plan-bias", type=float, default=0.10,
                    help="Extra directional bias from mm_plan + crowd_imbalance_side.")
    ap.add_argument("--mm-supervisor-fake-reversal-down", type=float, default=0.15,
                    help="Penalize both sides when fake-reversal risk is high.")
    ap.add_argument("--mm-supervisor-fake-reversal-dir-bias", type=float, default=0.10,
                    help="Extra directional penalty for fake-reversal based on side risk imbalance.")
    ap.add_argument("--mm-supervisor-half-life-hours", type=float, default=24.0,
                    help="MM signal age half-life for sizing effect (<=0 disables decay).")
    ap.add_argument("--mm-supervisor-min-mult", type=float, default=0.75,
                    help="Lower clip for MM direction multipliers.")
    ap.add_argument("--mm-supervisor-max-mult", type=float, default=1.25,
                    help="Upper clip for MM direction multipliers.")
    ap.add_argument("--mm-supervisor-hard-risk-thr", type=float, default=0.85,
                    help="Hybrid mode: hard-block entries when side risk*conf >= threshold.")
    ap.add_argument("--mm-supervisor-hard-conf-thr", type=float, default=0.60,
                    help="Hybrid mode: hard-block requires confidence >= threshold.")
    ap.add_argument("--mm-supervisor-crowd-first-min-conf", type=float, default=0.55,
                    help="Crowd-first mode: minimum MM confidence to activate directional filter.")
    ap.add_argument("--mm-supervisor-crowd-first-risk-gap", type=float, default=0.15,
                    help="Crowd-first fallback: min abs(long_risk-short_risk) to force one-sided entries.")
    ap.add_argument("--mm-supervisor-crowd-first-alpha-dir", type=float, default=0.12,
                    help="Crowd-first mode: directional soft prior strength (size boost/reduction).")
    ap.add_argument("--mm-supervisor-crowd-first-require-plan", action="store_true",
                    help="Crowd-first: if set, do not use risk-gap fallback when plan is absent.")
    ap.add_argument("--regime-enable", action="store_true",
                    help="Enable regime-adaptive hard risk threshold via rv/adx.")
    ap.add_argument("--regime-rv-col", type=str, default="rv_short",
                    help="RV column name for regime logic.")
    ap.add_argument("--regime-adx-col", type=str, default="adx14",
                    help="ADX column name for regime logic.")
    ap.add_argument("--regime-adx-trend-thr", type=float, default=20.0,
                    help="ADX threshold for trend regime.")
    ap.add_argument("--regime-adx-flat-thr", type=float, default=14.0,
                    help="ADX threshold for flat regime.")
    ap.add_argument("--regime-rv-trend-pct", type=float, default=0.70,
                    help="RV quantile for trend-like volatility regime.")
    ap.add_argument("--regime-rv-flat-pct", type=float, default=0.30,
                    help="RV quantile for flat/quiet regime.")
    ap.add_argument("--regime-hard-risk-relax", type=float, default=0.05,
                    help="Increase hard risk threshold in trend regime.")
    ap.add_argument("--regime-hard-risk-tighten", type=float, default=0.05,
                    help="Decrease hard risk threshold in flat regime.")
    ap.add_argument("--target-min-trades", type=int, default=0,
                    help="If >0, adaptively relax thresholds until at least this many opens are found.")
    ap.add_argument("--target-relax-steps", type=int, default=4,
                    help="Number of adaptive relaxation steps for min-trades logic.")
    ap.add_argument("--target-relax-meta", type=float, default=0.04,
                    help="Max reduction of meta threshold under full relaxation.")
    ap.add_argument("--target-relax-risk", type=float, default=0.12,
                    help="Max increase of hard risk threshold under full relaxation.")
    ap.add_argument("--mm-independent-head", action="store_true",
                    help="Run MM as an independent trading head (separate from base heads).")
    ap.add_argument("--mm-independent-min-conf", type=float, default=0.55,
                    help="MM-only: minimum confidence to emit a trade.")
    ap.add_argument("--mm-independent-risk-gap", type=float, default=0.15,
                    help="MM-only: minimum |long_risk-short_risk| if plan/side is unavailable.")
    ap.add_argument("--mm-independent-min-prob", type=float, default=0.55,
                    help="MM-only: minimum side risk probability for entry.")
    ap.add_argument("--mm-independent-horizon-bars", type=int, default=20,
                    help="MM-only: fixed holding horizon in bars.")
    ap.add_argument("--mm-independent-trade-frac-scale", type=float, default=0.10,
                    help="MM-only: scale of trade fraction vs base (risk budget).")
    ap.add_argument("--mm-independent-cooldown-steps", type=int, default=96,
                    help="MM-only: cooldown between entries of the same MM side.")
    ap.add_argument("--mm-independent-max-trades-per-day", type=int, default=2,
                    help="MM-only: cap trades per day per MM side.")
    ap.add_argument("--mm-independent-max-concurrent", type=int, default=1,
                    help="MM-only: max concurrent MM positions in mm_only simulation.")
    ap.add_argument("--chronos-model", type=str, default=None,
                    help="Optional Chronos model id, e.g. amazon/chronos-2.")
    ap.add_argument("--chronos-device", type=str, default="auto",
                    help="Chronos device: auto|cpu|cuda.")
    ap.add_argument("--chronos-context-len", type=int, default=512,
                    help="Chronos context length in bars.")
    ap.add_argument("--chronos-pred-len", type=int, default=20,
                    help="Chronos prediction length in bars.")
    ap.add_argument("--chronos-num-samples", type=int, default=32,
                    help="Chronos number of forecast samples.")
    ap.add_argument("--chronos-gate-entries", action="store_true",
                    help="Use Chronos forecast as additional entry gate.")
    ap.add_argument("--chronos-min-logret-long", type=float, default=0.0,
                    help="For long entries: require Chronos median log-return >= threshold.")
    ap.add_argument("--chronos-min-logret-short", type=float, default=0.0,
                    help="For short entries: require Chronos median log-return <= -threshold.")
    ap.add_argument("--chronos-exit-enable", action="store_true",
                    help="Enable Chronos-based exit override during open positions.")
    ap.add_argument("--chronos-exit-check-every", type=int, default=4,
                    help="Evaluate Chronos exit every N bars (default 4).")
    ap.add_argument("--chronos-exit-flip-logret", type=float, default=0.0,
                    help="Exit if Chronos median log-return flips against position by this threshold.")
    args = ap.parse_args()

    # --- Optional reproducible presets (do not override explicitly provided CLI flags)
    def _cli_has(*flags: str) -> bool:
        argv = set(sys.argv[1:])
        return any(f in argv for f in flags)

    def _apply_if_not_set(attr: str, value, *flags: str):
        if not _cli_has(*flags):
            setattr(args, attr, value)

    if args.preset:
        # Common MM supervisor configuration (case: mm_r13 from reports/mm_soft_sweep_2025_refine16_full.csv)
        _apply_if_not_set("mm_supervisor_mode", "hybrid", "--mm-supervisor-mode")
        _apply_if_not_set("mm_supervisor_alpha_up", 0.08, "--mm-supervisor-alpha-up")
        _apply_if_not_set("mm_supervisor_alpha_down", 0.10, "--mm-supervisor-alpha-down")
        _apply_if_not_set("mm_supervisor_plan_bias", 0.06, "--mm-supervisor-plan-bias")
        _apply_if_not_set("mm_supervisor_fake_reversal_down", 0.04, "--mm-supervisor-fake-reversal-down")
        _apply_if_not_set("mm_supervisor_fake_reversal_dir_bias", 0.05, "--mm-supervisor-fake-reversal-dir-bias")
        _apply_if_not_set("mm_supervisor_half_life_hours", 12.0, "--mm-supervisor-half-life-hours")
        _apply_if_not_set("mm_supervisor_hard_risk_thr", 0.90, "--mm-supervisor-hard-risk-thr")
        _apply_if_not_set("mm_supervisor_hard_conf_thr", 0.68, "--mm-supervisor-hard-conf-thr")
        _apply_if_not_set("mm_supervisor_min_mult", 0.90, "--mm-supervisor-min-mult")
        _apply_if_not_set("mm_supervisor_max_mult", 1.10, "--mm-supervisor-max-mult")

        if not _cli_has("--mm-supervisor-signals") and not args.mm_supervisor_signals:
            args.mm_supervisor_signals = str(ROOT / "data/telegram/vision_mm_stop_hunt_2025.parquet")

        if args.preset == "mm_r13_balance":
            _apply_if_not_set("trade_frac", 0.50, "--trade-frac")
        elif args.preset == "mm_r13_aggressive":
            _apply_if_not_set("trade_frac", 1.00, "--trade-frac")


    start_ts = pd.to_datetime(args.start, utc=True)
    end_ts = pd.to_datetime(args.end, utc=True)

    def _tagify(s: str) -> str:
        s = str(s)
        s = s.replace(".", "p").replace("-", "m").replace("+", "")
        s = re.sub(r"[^a-zA-Z0-9_]+", "_", s)
        return s.strip("_")

    def _path_with_tag(path_str: str, tag: str) -> str:
        p = Path(path_str)
        if not tag:
            return str(p)
        return str(p.with_name(p.stem + "__" + tag + p.suffix))

    def _monthly_stats(trades_df: pd.DataFrame) -> pd.DataFrame:
        if trades_df is None or len(trades_df) == 0:
            return pd.DataFrame(columns=["month", "trades", "wins", "winrate", "pnl_net", "fees", "pnl_gross",
                                         "avg_pnl_net", "avg_fee", "avg_hold_bars"])
        dfm = trades_df.copy()
        dfm["exit_ts"] = pd.to_datetime(dfm["exit_ts"], utc=True, errors="coerce")
        dfm = dfm[dfm["exit_ts"].notna()].copy()
        if len(dfm) == 0:
            return pd.DataFrame(columns=["month", "trades", "wins", "winrate", "pnl_net", "fees", "pnl_gross",
                                         "avg_pnl_net", "avg_fee", "avg_hold_bars"])
        dfm["month"] = dfm["exit_ts"].dt.to_period("M").astype(str)
        dfm["is_win"] = pd.to_numeric(dfm.get("pnl_net", 0.0), errors="coerce").fillna(0.0) > 0.0
        grp = dfm.groupby("month", as_index=False).agg(
            trades=("month", "size"),
            wins=("is_win", "sum"),
            pnl_net=("pnl_net", "sum"),
            fees=("fee", "sum"),
            pnl_gross=("pnl_gross", "sum"),
            avg_pnl_net=("pnl_net", "mean"),
            avg_fee=("fee", "mean"),
            avg_hold_bars=("hold_bars", "mean"),
        )
        grp["winrate"] = grp["wins"] / grp["trades"].clip(lower=1)
        # Reorder columns
        cols = ["month", "trades", "wins", "winrate", "pnl_net", "fees", "pnl_gross",
                "avg_pnl_net", "avg_fee", "avg_hold_bars"]
        return grp[cols].sort_values("month").reset_index(drop=True)


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
    chronos_ret_getter = None
    if args.chronos_model:
        chronos_ret_getter = _build_chronos_ret_getter(
            ts=ts,
            close=close,
            model_id=str(args.chronos_model),
            prediction_length=int(args.chronos_pred_len),
            context_length=int(args.chronos_context_len),
            num_samples=int(args.chronos_num_samples),
            device=str(args.chronos_device),
        )
    news_present = None
    if args.require_news_present:
        if "news_missing" not in df.columns:
            raise ValueError("--require-news-present needs 'news_missing' column in features.")
        news_present = (df["news_missing"].to_numpy(dtype=np.float32) == 0.0)

    crowd_state = None
    crowd_gate_mask = None
    crowd_size_mult = None
    crowd_fresh_coverage = np.nan
    crowd_hard_coverage = np.nan
    if args.crowd_gate_model:
        if not args.crowd_gate_signals:
            raise ValueError("--crowd-gate-model requires --crowd-gate-signals")
        crowd_state = _build_crowd_gate_state(
            ts=ts,
            model_path=str(args.crowd_gate_model),
            signals_path=str(args.crowd_gate_signals),
            thr_good=float(args.crowd_gate_thr_good),
            max_age_hours=float(args.crowd_gate_max_age_hours) if args.crowd_gate_max_age_hours is not None else None,
        )
        crowd_gate_mask = crowd_state["hard_pass"]
        crowd_fresh_coverage = float(np.mean(crowd_state["is_fresh"]))
        crowd_hard_coverage = float(np.mean(crowd_gate_mask))
        if args.crowd_gate_mode in ("soft", "hybrid"):
            alpha_down = float(args.crowd_soft_alpha_down)
            if crowd_fresh_coverage < float(args.crowd_soft_min_coverage):
                alpha_down = 0.0
            crowd_size_mult = _build_crowd_soft_size_mult(
                crowd_state=crowd_state,
                thr_good=float(args.crowd_gate_thr_good),
                half_life_hours=float(args.crowd_soft_half_life_hours),
                alpha_up=float(args.crowd_soft_alpha_up),
                alpha_down=alpha_down,
                min_mult=float(args.crowd_soft_min_mult),
                max_mult=float(args.crowd_soft_max_mult),
            )

    mm_state = None
    mm_long_mult = None
    mm_short_mult = None
    mm_fresh_coverage = np.nan
    mm_crowd_first_coverage = np.nan
    mm_mean_long_mult = np.nan
    mm_mean_short_mult = np.nan
    if args.mm_supervisor_mode != "off" or args.mm_independent_head:
        if not args.mm_supervisor_signals:
            raise ValueError("--mm-supervisor-signals is required when MM supervisor or MM-independent head is enabled")
        mm_state = _build_mm_supervisor_state(
            ts=ts,
            signals_path=str(args.mm_supervisor_signals),
            max_age_hours=float(args.mm_supervisor_max_age_hours)
            if args.mm_supervisor_max_age_hours is not None
            else None,
        )
        mm_fresh_coverage = float(np.mean(mm_state["is_fresh"]))
        if args.mm_supervisor_mode != "off":
            mm_long_mult, mm_short_mult = _build_mm_supervisor_mult(
                mm_state=mm_state,
                alpha_up=float(args.mm_supervisor_alpha_up),
                alpha_down=float(args.mm_supervisor_alpha_down),
                plan_bias=float(args.mm_supervisor_plan_bias),
                fake_reversal_down=float(args.mm_supervisor_fake_reversal_down),
                fake_reversal_dir_bias=float(args.mm_supervisor_fake_reversal_dir_bias),
                min_mult=float(args.mm_supervisor_min_mult),
                max_mult=float(args.mm_supervisor_max_mult),
                half_life_hours=float(args.mm_supervisor_half_life_hours),
            )
            mm_mean_long_mult = float(np.mean(mm_long_mult))
            mm_mean_short_mult = float(np.mean(mm_short_mult))
            if args.mm_supervisor_mode == "crowd_first":
                mm_long_cf, mm_short_cf, mm_active = _build_mm_crowd_first_mult(
                    mm_state=mm_state,
                    min_conf=float(args.mm_supervisor_crowd_first_min_conf),
                    risk_gap=float(args.mm_supervisor_crowd_first_risk_gap),
                    alpha_dir=float(args.mm_supervisor_crowd_first_alpha_dir),
                    min_mult=float(args.mm_supervisor_min_mult),
                    max_mult=float(args.mm_supervisor_max_mult),
                    half_life_hours=float(args.mm_supervisor_half_life_hours),
                    require_plan=bool(args.mm_supervisor_crowd_first_require_plan),
                )
                mm_long_mult = mm_long_mult * mm_long_cf
                mm_short_mult = mm_short_mult * mm_short_cf
                mm_long_mult = np.clip(mm_long_mult, float(args.mm_supervisor_min_mult), float(args.mm_supervisor_max_mult))
                mm_short_mult = np.clip(mm_short_mult, float(args.mm_supervisor_min_mult), float(args.mm_supervisor_max_mult))
                mm_mean_long_mult = float(np.mean(mm_long_mult))
                mm_mean_short_mult = float(np.mean(mm_short_mult))
                mm_crowd_first_coverage = float(np.mean(mm_active))

    hard_risk_thr_arr = np.full(len(df), float(args.mm_supervisor_hard_risk_thr), dtype=np.float32)
    if args.regime_enable:
        hard_risk_thr_arr = _build_regime_risk_threshold(
            df=df,
            base_thr=float(args.mm_supervisor_hard_risk_thr),
            trend_relax=float(args.regime_hard_risk_relax),
            flat_tighten=float(args.regime_hard_risk_tighten),
            adx_col=str(args.regime_adx_col),
            rv_col=str(args.regime_rv_col),
            adx_trend_thr=float(args.regime_adx_trend_thr),
            adx_flat_thr=float(args.regime_adx_flat_thr),
            rv_trend_pct=float(args.regime_rv_trend_pct),
            rv_flat_pct=float(args.regime_rv_flat_pct),
        )

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
    mm_indep_spec_long = ModelSpec(
        name="mm_only_long",
        horizon=int(max(1, args.mm_independent_horizon_bars)),
        direction="long",
        strategy="mm_only",
        model_path="",
        stats_path="",
        threshold=0.0,
    )
    mm_indep_spec_short = ModelSpec(
        name="mm_only_short",
        horizon=int(max(1, args.mm_independent_horizon_bars)),
        direction="short",
        strategy="mm_only",
        model_path="",
        stats_path="",
        threshold=0.0,
    )

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

    multi_run = len(sweep_iter) > 1

    for meta_thr, meta_thr_map in sweep_iter:
        relax_steps = max(1, int(args.target_relax_steps) + 1)
        relax_levels = np.linspace(0.0, 1.0, relax_steps)
        selected_relax = 0.0
        open_map = {}

        def _build_open_map(relax_k: float):
            omap = {}
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
                    meta_thr_base = float(meta_thr_map.get(spec.name, 0.0))
                else:
                    meta_thr_base = float(meta_thr)
                meta_thr_use = max(0.0, meta_thr_base - float(args.target_relax_meta) * float(relax_k))

                period_mask = (ts[idx] >= start_ts) & (ts[idx] < end_ts)
                pred_loc = pred[period_mask]
                idx_loc = idx[period_mask]
                gate_loc = gate[period_mask] if gate is not None else None
                rv_gate_loc = rv_gate[period_mask] if rv_gate is not None else None

                if gate_loc is None and rv_gate_loc is None:
                    gate_iter = [True] * len(idx_loc)
                else:
                    g1 = gate_loc if gate_loc is not None else np.ones(len(idx_loc), dtype=bool)
                    g2 = rv_gate_loc if rv_gate_loc is not None else np.ones(len(idx_loc), dtype=bool)
                    gate_iter = (g1 & g2)

                for p, i, g in zip(pred_loc, idx_loc, gate_iter):
                    if not g:
                        continue

                    crowd_bad = False
                    if crowd_state is not None:
                        if args.crowd_gate_mode == "hard":
                            if not bool(crowd_gate_mask[i]):
                                continue
                        elif args.crowd_gate_mode == "hybrid":
                            cp = float(crowd_state["prob_good"][i])
                            bad_thr_eff = max(0.0, float(args.crowd_hybrid_bad_thr) * (1.0 - 0.6 * float(relax_k)))
                            crowd_bad = bool(crowd_state["is_fresh"][i]) and math.isfinite(cp) and cp < bad_thr_eff

                    if news_present is not None and not bool(news_present[i]):
                        continue
                    if spec.direction == "long" and p <= thr:
                        continue
                    if spec.direction == "short" and p >= -thr:
                        continue
                    if meta_probs is None or math.isnan(float(meta_probs[i])) or float(meta_probs[i]) < meta_thr_use:
                        continue
                    if args.chronos_gate_entries and chronos_ret_getter is not None:
                        cr = float(chronos_ret_getter(int(i)))
                        if not math.isfinite(cr):
                            continue
                        if spec.direction == "long" and cr < float(args.chronos_min_logret_long):
                            continue
                        if spec.direction == "short" and cr > -float(args.chronos_min_logret_short):
                            continue

                    # Hard block only in extreme condition: high mm risk*conf AND bad fresh crowd quality.
                    if mm_state is not None and args.mm_supervisor_mode in ("hybrid", "crowd_first"):
                        conf = float(mm_state["conf"][i]) if np.isfinite(mm_state["conf"][i]) else 0.0
                        side_risk = (
                            float(mm_state["long_risk"][i]) if spec.direction == "long"
                            else float(mm_state["short_risk"][i])
                        )
                        risk_thr_eff = min(
                            0.999,
                            float(hard_risk_thr_arr[i]) + float(args.target_relax_risk) * float(relax_k),
                        )
                        mm_extreme = (
                            bool(mm_state["is_fresh"][i])
                            and conf >= float(args.mm_supervisor_hard_conf_thr)
                            and math.isfinite(side_risk)
                            and (side_risk * conf) >= risk_thr_eff
                        )
                        if mm_extreme and crowd_bad:
                            continue

                    omap.setdefault(int(i), []).append((spec, float(meta_probs[i]), float(meta_thr_use)))
            return omap

        for rk in relax_levels:
            cand = _build_open_map(float(rk))
            open_cnt = sum(len(v) for v in cand.values())
            open_map = cand
            selected_relax = float(rk)
            if int(args.target_min_trades) <= 0 or open_cnt >= int(args.target_min_trades):
                break

        base_open_map = open_map
        if args.agreement_gate:
            keep_names = {"h20_long", "h160_long_v2", "h20_short", "h80_short_v2"}
            for i in list(base_open_map.keys()):
                names = {spec.name for spec, _, _ in base_open_map[i]}
                if args.agreement_mode == "soft":
                    long_ok = "h20_long" in names and ("h160_long_v2" in names or "h80_short_v2" in names)
                    short_ok = "h20_short" in names and ("h80_short_v2" in names or "h160_long_v2" in names)
                else:
                    long_ok = "h20_long" in names and "h160_long_v2" in names
                    short_ok = "h20_short" in names and "h80_short_v2" in names
                if not (long_ok or short_ok):
                    del base_open_map[i]
                else:
                    base_open_map[i] = [item for item in base_open_map[i] if item[0].name in keep_names]

        mm_open_map = {}
        if args.mm_independent_head:
            mm_open_map = _build_mm_independent_open_map(
                mm_state=mm_state,
                spec_long=mm_indep_spec_long,
                spec_short=mm_indep_spec_short,
                min_conf=float(args.mm_independent_min_conf),
                risk_gap=float(args.mm_independent_risk_gap),
                min_prob=float(args.mm_independent_min_prob),
                start_ts=start_ts,
                end_ts=end_ts,
                ts=ts,
            )
            mm_open_map = _limit_open_map_activity(
                open_map=mm_open_map,
                ts=ts,
                cooldown_steps=int(args.mm_independent_cooldown_steps),
                max_trades_per_day=int(args.mm_independent_max_trades_per_day),
            )

        combined_open_map = {}
        all_keys = set(base_open_map.keys()) | set(mm_open_map.keys())
        for k in all_keys:
            combined_open_map[k] = []
            if k in base_open_map:
                combined_open_map[k].extend(base_open_map[k])
            if k in mm_open_map:
                combined_open_map[k].extend(mm_open_map[k])

        open_count_base = int(sum(len(v) for v in base_open_map.values()))
        open_count_mm = int(sum(len(v) for v in mm_open_map.values()))
        open_count = int(sum(len(v) for v in combined_open_map.values()))

        equity_curve = [] if args.out_equity_csv else None
        trades_rows = [] if args.out_trades_csv else None
        eq_base, trades_base, fees_base, dd_base = _simulate(
            df,
            ts,
            close,
            base_open_map,
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
            None,
            None,
            exit_mode=args.exit_mode,
            exit_soft_delta=args.exit_soft_delta,
            exit_min_hold_mult=args.exit_min_hold_mult,
            exit_max_hold_mult=args.exit_max_hold_mult,
            exit_profit_buffer=args.exit_profit_buffer,
            exit_atr_mult=args.exit_atr_mult,
            exit_atr_col=args.exit_atr_col,
            meta_prob_map=meta_prob_map,
            crowd_size_mult=crowd_size_mult,
            mm_long_mult=mm_long_mult,
            mm_short_mult=mm_short_mult,
            chronos_ret_getter=chronos_ret_getter,
            chronos_exit_enable=bool(args.chronos_exit_enable and chronos_ret_getter is not None),
            chronos_exit_check_every=int(args.chronos_exit_check_every),
            chronos_exit_flip_logret=float(args.chronos_exit_flip_logret),
        )
        eq_mm, trades_mm, fees_mm, dd_mm = _simulate(
            df,
            ts,
            close,
            mm_open_map,
            start_ts,
            end_ts,
            args.cost_rt,
            args.trade_frac * float(args.mm_independent_trade_frac_scale),
            args.trade_frac_long,
            args.trade_frac_short,
            args.leverage,
            int(args.mm_independent_cooldown_steps),
            int(args.mm_independent_max_concurrent),
            args.meta_sizing,
            args.meta_size_scale,
            args.meta_size_power,
            None,
            None,
            exit_mode=args.exit_mode,
            exit_soft_delta=args.exit_soft_delta,
            exit_min_hold_mult=args.exit_min_hold_mult,
            exit_max_hold_mult=args.exit_max_hold_mult,
            exit_profit_buffer=args.exit_profit_buffer,
            exit_atr_mult=args.exit_atr_mult,
            exit_atr_col=args.exit_atr_col,
            meta_prob_map=meta_prob_map,
            crowd_size_mult=None,
            mm_long_mult=None,
            mm_short_mult=None,
            chronos_ret_getter=chronos_ret_getter,
            chronos_exit_enable=bool(args.chronos_exit_enable and chronos_ret_getter is not None),
            chronos_exit_check_every=int(args.chronos_exit_check_every),
            chronos_exit_flip_logret=float(args.chronos_exit_flip_logret),
            mm_independent_size_mult=1.0,
        )
        equity, trades, fees, max_dd = _simulate(
            df,
            ts,
            close,
            combined_open_map,
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
            trades_rows,
            exit_mode=args.exit_mode,
            exit_soft_delta=args.exit_soft_delta,
            exit_min_hold_mult=args.exit_min_hold_mult,
            exit_max_hold_mult=args.exit_max_hold_mult,
            exit_profit_buffer=args.exit_profit_buffer,
            exit_atr_mult=args.exit_atr_mult,
            exit_atr_col=args.exit_atr_col,
            meta_prob_map=meta_prob_map,
            crowd_size_mult=crowd_size_mult,
            mm_long_mult=mm_long_mult,
            mm_short_mult=mm_short_mult,
            chronos_ret_getter=chronos_ret_getter,
            chronos_exit_enable=bool(args.chronos_exit_enable and chronos_ret_getter is not None),
            chronos_exit_check_every=int(args.chronos_exit_check_every),
            chronos_exit_flip_logret=float(args.chronos_exit_flip_logret),
            mm_independent_size_mult=float(args.mm_independent_trade_frac_scale),
        )
        run_tag = ""
        if multi_run:
            if meta_thr_map is not None:
                parts = []
                for k in ("h20_long", "h20_short", "h80_short_v2", "h160_long_v2"):
                    v = meta_thr_map.get(k)
                    if v is None:
                        continue
                    parts.append(f"{k}{float(v):.3f}")
                run_tag = _tagify("map_" + "_".join(parts) if parts else "map")
            else:
                run_tag = _tagify(f"thr{float(meta_thr):.3f}" if meta_thr is not None else "thr_na")

        if args.out_trades_csv and trades_rows is not None:
            trades_df = pd.DataFrame(trades_rows)
            out_trades = _path_with_tag(args.out_trades_csv, run_tag)
            trades_df.to_csv(out_trades, index=False)
            if args.out_monthly_csv:
                out_monthly = _path_with_tag(args.out_monthly_csv, run_tag)
                _monthly_stats(trades_df).to_csv(out_monthly, index=False)

        if args.out_equity_csv and equity_curve:
            eq_df = pd.DataFrame(equity_curve, columns=["timestamp", "equity"])
            eq_df["timestamp"] = pd.to_datetime(eq_df["timestamp"], utc=True)
            out_eq = _path_with_tag(args.out_equity_csv, run_tag)
            eq_df.to_csv(out_eq, index=False)
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
            "base_only_equity": float(eq_base),
            "base_only_pnl": float(eq_base - 100.0),
            "base_only_trades": int(trades_base),
            "base_only_fees": float(fees_base),
            "base_only_max_dd": float(dd_base),
            "mm_only_equity": float(eq_mm),
            "mm_only_pnl": float(eq_mm - 100.0),
            "mm_only_trades": int(trades_mm),
            "mm_only_fees": float(fees_mm),
            "mm_only_max_dd": float(dd_mm),
            "crowd_gate_enabled": bool(crowd_gate_mask is not None),
            "crowd_gate_mode": str(args.crowd_gate_mode),
            "crowd_gate_thr_good": float(args.crowd_gate_thr_good),
            "crowd_gate_max_age_hours": float(args.crowd_gate_max_age_hours),
            "crowd_gate_coverage": float(crowd_hard_coverage),
            "crowd_fresh_coverage": float(crowd_fresh_coverage),
            "crowd_soft_alpha_up": float(args.crowd_soft_alpha_up),
            "crowd_soft_alpha_down": float(args.crowd_soft_alpha_down),
            "crowd_soft_half_life_hours": float(args.crowd_soft_half_life_hours),
            "crowd_soft_min_mult": float(args.crowd_soft_min_mult),
            "crowd_soft_max_mult": float(args.crowd_soft_max_mult),
            "crowd_soft_mean_mult": float(np.mean(crowd_size_mult)) if crowd_size_mult is not None else np.nan,
            "mm_supervisor_enabled": bool(mm_state is not None),
            "mm_supervisor_mode": str(args.mm_supervisor_mode),
            "mm_supervisor_signals": str(args.mm_supervisor_signals) if args.mm_supervisor_signals else "",
            "mm_supervisor_max_age_hours": float(args.mm_supervisor_max_age_hours),
            "mm_supervisor_fresh_coverage": float(mm_fresh_coverage),
            "mm_supervisor_alpha_up": float(args.mm_supervisor_alpha_up),
            "mm_supervisor_alpha_down": float(args.mm_supervisor_alpha_down),
            "mm_supervisor_plan_bias": float(args.mm_supervisor_plan_bias),
            "mm_supervisor_fake_reversal_down": float(args.mm_supervisor_fake_reversal_down),
            "mm_supervisor_fake_reversal_dir_bias": float(args.mm_supervisor_fake_reversal_dir_bias),
            "mm_supervisor_half_life_hours": float(args.mm_supervisor_half_life_hours),
            "mm_supervisor_min_mult": float(args.mm_supervisor_min_mult),
            "mm_supervisor_max_mult": float(args.mm_supervisor_max_mult),
            "mm_supervisor_hard_risk_thr": float(args.mm_supervisor_hard_risk_thr),
            "mm_supervisor_hard_conf_thr": float(args.mm_supervisor_hard_conf_thr),
            "mm_supervisor_crowd_first_min_conf": float(args.mm_supervisor_crowd_first_min_conf),
            "mm_supervisor_crowd_first_risk_gap": float(args.mm_supervisor_crowd_first_risk_gap),
            "mm_supervisor_crowd_first_alpha_dir": float(args.mm_supervisor_crowd_first_alpha_dir),
            "mm_supervisor_crowd_first_require_plan": bool(args.mm_supervisor_crowd_first_require_plan),
            "mm_supervisor_crowd_first_coverage": float(mm_crowd_first_coverage),
            "mm_supervisor_mean_long_mult": float(mm_mean_long_mult),
            "mm_supervisor_mean_short_mult": float(mm_mean_short_mult),
            "regime_enable": bool(args.regime_enable),
            "regime_rv_col": str(args.regime_rv_col),
            "regime_adx_col": str(args.regime_adx_col),
            "regime_adx_trend_thr": float(args.regime_adx_trend_thr),
            "regime_adx_flat_thr": float(args.regime_adx_flat_thr),
            "regime_rv_trend_pct": float(args.regime_rv_trend_pct),
            "regime_rv_flat_pct": float(args.regime_rv_flat_pct),
            "regime_hard_risk_relax": float(args.regime_hard_risk_relax),
            "regime_hard_risk_tighten": float(args.regime_hard_risk_tighten),
            "target_min_trades": int(args.target_min_trades),
            "target_relax_steps": int(args.target_relax_steps),
            "target_relax_meta": float(args.target_relax_meta),
            "target_relax_risk": float(args.target_relax_risk),
            "selected_relax": float(selected_relax),
            "open_candidates": int(open_count),
            "open_candidates_base": int(open_count_base),
            "open_candidates_mm": int(open_count_mm),
            "mm_independent_head": bool(args.mm_independent_head),
            "mm_independent_min_conf": float(args.mm_independent_min_conf),
            "mm_independent_risk_gap": float(args.mm_independent_risk_gap),
            "mm_independent_min_prob": float(args.mm_independent_min_prob),
            "mm_independent_horizon_bars": int(args.mm_independent_horizon_bars),
            "mm_independent_trade_frac_scale": float(args.mm_independent_trade_frac_scale),
            "mm_independent_cooldown_steps": int(args.mm_independent_cooldown_steps),
            "mm_independent_max_trades_per_day": int(args.mm_independent_max_trades_per_day),
            "mm_independent_max_concurrent": int(args.mm_independent_max_concurrent),
            "chronos_enabled": bool(chronos_ret_getter is not None),
            "chronos_model": str(args.chronos_model) if args.chronos_model else "",
            "chronos_gate_entries": bool(args.chronos_gate_entries),
            "chronos_pred_len": int(args.chronos_pred_len),
            "chronos_context_len": int(args.chronos_context_len),
            "chronos_num_samples": int(args.chronos_num_samples),
            "chronos_min_logret_long": float(args.chronos_min_logret_long),
            "chronos_min_logret_short": float(args.chronos_min_logret_short),
            "chronos_exit_enable": bool(args.chronos_exit_enable and chronos_ret_getter is not None),
            "chronos_exit_check_every": int(args.chronos_exit_check_every),
            "chronos_exit_flip_logret": float(args.chronos_exit_flip_logret),
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

    if args.out_summary_json:
        def _clean(v):
            if v is None:
                return None
            if isinstance(v, (np.floating,)):
                return float(v)
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.bool_,)):
                return bool(v)
            if isinstance(v, (pd.Timestamp,)):
                return v.isoformat()
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            try:
                if pd.isna(v):
                    return None
            except Exception:
                pass
            return v

        best = out_df.sort_values("pnl", ascending=False).iloc[0].to_dict()
        best = {k: _clean(v) for k, v in best.items()}

        summary = {
            "window": {"start": args.start, "end": args.end},
            "inputs": {
                "features": str(args.features),
                "meta_features": str(args.meta_features),
            },
            "preset": args.preset,
            "rows": int(len(out_df)),
            "best_by_pnl": best,
            "notes": [
                "pnl is net (after fees) relative to 100 initial equity",
                "fees are accumulated as notional * cost_rt per closed trade in this sim",
            ],
        }
        out_sum = Path(args.out_summary_json)
        out_sum.parent.mkdir(parents=True, exist_ok=True)
        out_sum.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Saved:", out_path)
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
