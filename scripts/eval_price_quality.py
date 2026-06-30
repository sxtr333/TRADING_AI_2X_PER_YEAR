#!/usr/bin/env python3
import argparse
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import tensorflow as tf

# Ensure project root is on path when running from scripts/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from train_keras import apply_norm
from model_layers import RevIN, ITransformerBlock, TSMixerBlock, LastStep, DropPath


def _load_model_keras3(model_path: str):
    import keras as _keras

    class RevIN(_keras.layers.Layer):
        def __init__(self, affine: bool = True, eps: float = 1e-5, **kwargs):
            super().__init__(**kwargs)
            self.affine = bool(affine)
            self.eps = float(eps)
            self._mu = None
            self._std = None

        def build(self, input_shape):
            feat_dim = int(input_shape[-1])
            if self.affine:
                self.gamma = self.add_weight(
                    name="gamma", shape=(1, 1, feat_dim), initializer="ones", trainable=True
                )
                self.beta = self.add_weight(
                    name="beta", shape=(1, 1, feat_dim), initializer="zeros", trainable=True
                )
            super().build(input_shape)

        def call(self, x, mode: str = "norm"):
            if mode == "norm":
                mu = tf.reduce_mean(x, axis=1, keepdims=True)
                var = tf.reduce_mean(tf.square(x - mu), axis=1, keepdims=True)
                std = tf.sqrt(var + self.eps)
                self._mu = mu
                self._std = std
                y = (x - mu) / std
                if self.affine:
                    y = y * self.gamma + self.beta
                return y
            if mode == "denorm":
                if self._mu is None or self._std is None:
                    raise ValueError("RevIN denorm called before norm.")
                y = x
                if self.affine:
                    y = (y - self.beta) / (self.gamma + self.eps)
                return y * self._std + self._mu
            raise ValueError("mode must be 'norm' or 'denorm'")

        def get_config(self):
            return {"affine": self.affine, "eps": self.eps}

    class DropPath(_keras.layers.Layer):
        def __init__(self, drop_prob: float = 0.0, **kwargs):
            super().__init__(**kwargs)
            self.drop_prob = float(drop_prob)

        def call(self, x, training=False):
            if not training or self.drop_prob == 0.0:
                return x
            keep = 1.0 - self.drop_prob
            shape = (tf.shape(x)[0],) + (1,) * (len(x.shape) - 1)
            rnd = keep + tf.random.uniform(shape, dtype=x.dtype)
            mask = tf.floor(rnd)
            return (x / keep) * mask

        def get_config(self):
            return {"drop_prob": self.drop_prob}

    class LastStep(_keras.layers.Layer):
        def call(self, x):
            return x[:, -1, :]

    class TSMixerBlock(_keras.layers.Layer):
        def __init__(self, mlp_dim: int, dropout: float = 0.0, **kwargs):
            super().__init__(**kwargs)
            self.mlp_dim = int(mlp_dim)
            self.dropout = float(dropout)
            self.ln_time = _keras.layers.LayerNormalization(epsilon=1e-5)
            self.ln_feat = _keras.layers.LayerNormalization(epsilon=1e-5)
            self.do = _keras.layers.Dropout(self.dropout)

        def build(self, input_shape):
            seq_len = int(input_shape[1])
            feat_dim = int(input_shape[2])
            self.time_dense = _keras.layers.Dense(seq_len)
            self.ff1 = _keras.layers.Dense(self.mlp_dim, activation="gelu")
            self.ff2 = _keras.layers.Dense(feat_dim)
            super().build(input_shape)

        def call(self, x, training=False):
            y = self.ln_time(x)
            y = tf.transpose(y, [0, 2, 1])
            y = self.time_dense(y)
            y = tf.transpose(y, [0, 2, 1])
            y = self.do(y, training=training)
            x = x + y

            y = self.ln_feat(x)
            y = self.ff1(y)
            y = self.do(y, training=training)
            y = self.ff2(y)
            y = self.do(y, training=training)
            return x + y

        def get_config(self):
            return {"mlp_dim": self.mlp_dim, "dropout": self.dropout}

    class ITransformerBlock(_keras.layers.Layer):
        def __init__(self, seq_len: int, d_model: int, heads: int, dropout: float = 0.0, **kwargs):
            super().__init__(**kwargs)
            self.seq_len = int(seq_len)
            self.d_model = int(d_model)
            self.heads = int(heads)
            self.dropout = float(dropout)
            self.ln1 = _keras.layers.LayerNormalization(epsilon=1e-5)
            self.ln2 = _keras.layers.LayerNormalization(epsilon=1e-5)

        def build(self, input_shape):
            key_dim = max(self.d_model // max(self.heads, 1), 8)
            self.proj = _keras.layers.Dense(self.d_model)
            self.mha = _keras.layers.MultiHeadAttention(
                num_heads=self.heads, key_dim=key_dim, dropout=self.dropout
            )
            self.ff1 = _keras.layers.Dense(self.d_model * 4, activation="gelu")
            self.ff2 = _keras.layers.Dense(self.d_model)
            self.to_time = _keras.layers.Dense(self.seq_len)
            self.do = _keras.layers.Dropout(self.dropout)
            super().build(input_shape)

        def call(self, x, training=False):
            xt = tf.transpose(x, [0, 2, 1])
            h = self.proj(xt)
            h1 = self.ln1(h)
            h1 = self.mha(h1, h1)
            h1 = self.do(h1, training=training)
            h = h + h1

            h2 = self.ln2(h)
            h2 = self.ff1(h2)
            h2 = self.do(h2, training=training)
            h2 = self.ff2(h2)
            h2 = self.do(h2, training=training)
            h = h + h2

            yt = self.to_time(h)
            y = tf.transpose(yt, [0, 2, 1])
            return x + y

        def get_config(self):
            return {
                "seq_len": self.seq_len,
                "d_model": self.d_model,
                "heads": self.heads,
                "dropout": self.dropout,
            }

    @_keras.saving.register_keras_serializable(package="keras.layers")
    class SpatialDropout1D(_keras.layers.SpatialDropout1D):
        def __init__(self, *args, **kwargs):
            kwargs.pop("trainable", None)
            super().__init__(*args, **kwargs)

        @classmethod
        def from_config(cls, config):
            config.pop("trainable", None)
            return cls(**config)

    custom_objects = {
        "RevIN": RevIN,
        "model6>RevIN": RevIN,
        "TSMixerBlock": TSMixerBlock,
        "model6>TSMixerBlock": TSMixerBlock,
        "ITransformerBlock": ITransformerBlock,
        "model6>ITransformerBlock": ITransformerBlock,
        "LastStep": LastStep,
        "model6>LastStep": LastStep,
        "DropPath": DropPath,
        "model6>DropPath": DropPath,
        "SpatialDropout1D": SpatialDropout1D,
        "keras.layers.SpatialDropout1D": SpatialDropout1D,
        "keras.src.layers.regularization.spatial_dropout.SpatialDropout1D": SpatialDropout1D,
        "Functional": _keras.Model,
    }
    _keras.layers.SpatialDropout1D = SpatialDropout1D
    try:
        import keras.src.layers.regularization.spatial_dropout as _sd
        _sd.SpatialDropout1D = SpatialDropout1D
    except Exception:
        pass
    return _keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)


def _unpack_obj(val):
    if isinstance(val, np.ndarray) and val.dtype == object:
        if val.size == 1:
            return val.item()
    return val


def load_stats(path: str) -> dict:
    s = np.load(path, allow_pickle=True)
    stats = {
        "feature_names": s["feature_names"].tolist(),
        "mean": s["mean"].astype(np.float32),
        "std": s["std"].astype(np.float32),
        "q_low": s["q_low"].astype(np.float32),
        "q_high": s["q_high"].astype(np.float32),
        "seq_len": int(s["seq_len"][0]) if "seq_len" in s else 256,
        "test_start_ts": str(s["test_start_ts"][0]) if "test_start_ts" in s else None,
        "price_head_scale": _unpack_obj(s["price_head_scale"]) if "price_head_scale" in s else None,
        "price_target_mode": _unpack_obj(s["price_target_mode"]) if "price_target_mode" in s else None,
        "price_multi_horizons": _unpack_obj(s["price_multi_horizons"]) if "price_multi_horizons" in s else None,
    }
    return stats


def build_dataset(X: np.ndarray, end_indices: np.ndarray, seq_len: int, batch_size: int):
    X_tf = tf.convert_to_tensor(X, dtype=tf.float32)
    end_indices = end_indices.astype(np.int64)
    ds = tf.data.Dataset.from_tensor_slices(end_indices)

    def map_fn(i):
        i = tf.cast(i, tf.int32)
        start = i - (seq_len - 1)
        x_seq = X_tf[start:i + 1]
        x_seq = tf.ensure_shape(x_seq, [seq_len, X.shape[1]])
        return x_seq

    ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds


def safe_smape(pred: np.ndarray, actual: np.ndarray) -> float:
    denom = np.abs(pred) + np.abs(actual)
    denom = np.where(denom < 1e-9, 1e-9, denom)
    return float(np.mean(2.0 * np.abs(pred - actual) / denom) * 100.0)

def _tail_stats(abs_err: np.ndarray) -> dict:
    if abs_err.size == 0:
        return {"p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {
        "p50": float(np.quantile(abs_err, 0.50)),
        "p90": float(np.quantile(abs_err, 0.90)),
        "p95": float(np.quantile(abs_err, 0.95)),
        "p99": float(np.quantile(abs_err, 0.99)),
        "max": float(np.max(abs_err)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--horizons", default="20,80,160")
    ap.add_argument("--single-horizon", type=int, default=None,
                    help="If set, evaluate a single-horizon model that outputs 'price' head.")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--start", default=None, help="ISO start timestamp (UTC). If omitted, uses stats test_start_ts.")
    ap.add_argument("--out", default=None, help="Optional CSV output path.")
    ap.add_argument("--line-metrics", action="store_true", help="Compute LineMAE/LineRMSE across steps 1..h")
    ap.add_argument("--bias-shift", choices=["none", "val", "window"], default="none",
                    help="Apply bias shift by subtracting median error on a calibration window.")
    ap.add_argument("--bias-shift-start", default=None,
                    help="Start timestamp (UTC) for bias-shift window (use with --bias-shift window).")
    ap.add_argument("--bias-shift-end", default=None,
                    help="End timestamp (UTC) for bias-shift window (use with --bias-shift window).")
    args = ap.parse_args()

    stats = load_stats(args.stats)
    feature_names = stats["feature_names"]
    seq_len = stats["seq_len"]

    df = pd.read_parquet(args.features)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    else:
        raise ValueError("features parquet must include 'timestamp'")

    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X_raw = df[feature_names].to_numpy(np.float32)
    X = apply_norm(X_raw, stats)

    close = df["close"].to_numpy(np.float64)
    ts = df["timestamp"].to_numpy()

    if args.single_horizon is not None:
        horizons = [int(args.single_horizon)]
    else:
        horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    max_h = max(horizons)

    # Determine start timestamp
    if args.start:
        start_ts = pd.to_datetime(args.start, utc=True)
    elif stats.get("test_start_ts"):
        start_ts = pd.to_datetime(stats["test_start_ts"], utc=True)
    else:
        start_ts = pd.to_datetime(df["timestamp"].iloc[0], utc=True)

    # Choose prediction window to avoid unnecessary compute
    start_idx = int(np.searchsorted(df["timestamp"].to_numpy(), start_ts))
    start_idx = max(start_idx, seq_len - 1)
    end_idx = len(df) - max_h - 1
    if end_idx <= start_idx:
        raise ValueError("Not enough rows for evaluation window.")

    end_indices = np.arange(start_idx, end_idx + 1, dtype=np.int64)
    ds = build_dataset(X, end_indices, seq_len=seq_len, batch_size=args.batch_size)

    custom_objects = {
        "RevIN": RevIN,
        "ITransformerBlock": ITransformerBlock,
        "TSMixerBlock": TSMixerBlock,
        "LastStep": LastStep,
        "DropPath": DropPath,
    }
    use_keras3 = os.environ.get("USE_KERAS3", "0") == "1"
    if use_keras3:
        model = _load_model_keras3(args.model)
    else:
        model = tf.keras.models.load_model(args.model, custom_objects=custom_objects, compile=False, safe_mode=False)
    preds = model.predict(ds, verbose=0)

    # Some models return multiple outputs as list/tuple (e.g., [price, cls])
    if isinstance(preds, (list, tuple)):
        if args.single_horizon is None:
            raise ValueError("Model returned multiple outputs; use --single-horizon to select price head.")
        if len(preds) < 1:
            raise ValueError("Model returned empty outputs list.")
        preds = {"price": np.asarray(preds[0])}

    if not isinstance(preds, dict):
        if args.single_horizon is None:
            raise ValueError("Model must return dict with price_h{h} heads for horizons.")
        preds = {"price": np.asarray(preds)}

    def _pred_dict_from_outputs(pred_out, idx_len: int) -> dict:
        if isinstance(pred_out, (list, tuple)):
            if args.single_horizon is None:
                raise ValueError("Model returned multiple outputs; use --single-horizon to select price head.")
            if len(pred_out) < 1:
                raise ValueError("Model returned empty outputs list.")
            pred_out = {"price": np.asarray(pred_out[0])}

        if not isinstance(pred_out, dict):
            if args.single_horizon is None:
                raise ValueError("Model must return dict with price_h{h} heads for horizons.")
            pred_out = {"price": np.asarray(pred_out)}

        pred_by = {}
        for h in horizons:
            key = f"price_h{h}"
            if key in pred_out:
                pred_by[h] = np.asarray(pred_out[key]).reshape(-1)
            elif "price" in pred_out and args.single_horizon is not None:
                pred_by[h] = np.asarray(pred_out["price"]).reshape(-1)
            else:
                raise ValueError(f"Missing prediction head: {key}")
            if len(pred_by[h]) != idx_len:
                raise ValueError(f"Pred length mismatch for {key}: {len(pred_by[h])} vs {idx_len}")
        return pred_by

    pred_by_h = _pred_dict_from_outputs(preds, len(end_indices))

    # optional bias-shift (median error on calibration window)
    bias_map = {}
    if args.bias_shift != "none":
        if args.bias_shift == "val":
            if not stats.get("val_start_ts") or not stats.get("test_start_ts"):
                raise ValueError("--bias-shift val requires stats val_start_ts and test_start_ts")
            bias_start = pd.to_datetime(stats["val_start_ts"], utc=True)
            bias_end = pd.to_datetime(stats["test_start_ts"], utc=True)
        else:
            if not args.bias_shift_start or not args.bias_shift_end:
                raise ValueError("--bias-shift window requires --bias-shift-start and --bias-shift-end")
            bias_start = pd.to_datetime(args.bias_shift_start, utc=True)
            bias_end = pd.to_datetime(args.bias_shift_end, utc=True)

        bias_idx = np.arange(max(seq_len - 1, 0), len(df) - max_h - 1, dtype=np.int64)
        bias_mask = (pd.to_datetime(ts[bias_idx], utc=True) >= bias_start) & (pd.to_datetime(ts[bias_idx], utc=True) < bias_end)
        bias_idx = bias_idx[bias_mask]

        if len(bias_idx) < 100:
            raise ValueError("Bias-shift window too small.")

        ds_bias = build_dataset(X, bias_idx, seq_len=seq_len, batch_size=args.batch_size)
        preds_bias = model.predict(ds_bias, verbose=0)
        pred_bias_by_h = _pred_dict_from_outputs(preds_bias, len(bias_idx))

        for h in horizons:
            valid = bias_idx + h < len(df)
            valid &= np.isfinite(close[bias_idx]) & np.isfinite(close[bias_idx + h])
            if not np.any(valid):
                bias_map[h] = 0.0
                continue
            i = bias_idx[valid]
            base = close[i]
            actual = close[i + h]
            pred_log = pred_bias_by_h[h][valid]
            pred_price = base * np.exp(pred_log)
            err = pred_price - actual
            bias_map[h] = float(np.median(err))

        print(f"[bias_shift] {bias_map}")

    # apply per-head scaling (if any)
    scale_map = stats.get("price_head_scale") or {}
    if isinstance(scale_map, dict) and scale_map:
        for h in pred_by_h.keys():
            key = f"price_h{h}" if f"price_h{h}" in scale_map else "price"
            if key in scale_map:
                pred_by_h[h] = pred_by_h[h] * float(scale_map[key])

    # convert segment deltas to cumulative if needed
    target_mode = stats.get("price_target_mode") or "cumulative"
    if target_mode == "segment_deltas":
        cum = 0.0
        for h in sorted(pred_by_h.keys()):
            cum = cum + pred_by_h[h]
            pred_by_h[h] = cum

    results = []
    for h in horizons:
        idx = end_indices
        # valid horizon range
        valid = idx + h < len(df)
        # time filter
        valid &= (pd.to_datetime(ts[idx], utc=True) >= start_ts)
        # close availability
        valid &= np.isfinite(close[idx]) & np.isfinite(close[idx + h])

        if not np.any(valid):
            results.append({"horizon": h, "n": 0})
            continue

        i = idx[valid]
        base = close[i]
        actual = close[i + h]
        pred_log = pred_by_h[h][valid]
        pred_price = base * np.exp(pred_log)
        if bias_map:
            pred_price = pred_price - bias_map.get(h, 0.0)

        err = pred_price - actual
        abs_err = np.abs(err)
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mape = float(np.mean(np.abs(err) / np.maximum(actual, 1e-9)) * 100.0)
        smape = safe_smape(pred_price, actual)
        bias_usd = float(np.mean(err))
        bias_pct = float(np.mean(err / np.maximum(actual, 1e-9)) * 100.0)
        tail = _tail_stats(abs_err)

        # Direction accuracy: sign of change vs base close
        actual_dir = np.sign(actual - base)
        pred_dir = np.sign(pred_price - base)
        dir_acc = float(np.mean(actual_dir == pred_dir) * 100.0)

        line_mae = None
        line_rmse = None
        line_bias = None
        if args.line_metrics:
            abs_sum = 0.0
            sq_sum = 0.0
            bias_sum = 0.0
            count = 0
            for step in range(1, h + 1):
                pred_step = base * np.exp(pred_log * (float(step) / float(h)))
                actual_step = close[i + step]
                e = pred_step - actual_step
                abs_sum += float(np.sum(np.abs(e)))
                sq_sum += float(np.sum(e ** 2))
                bias_sum += float(np.sum(e))
                count += len(i)
            if count > 0:
                line_mae = abs_sum / float(count)
                line_rmse = float(np.sqrt(sq_sum / float(count)))
                line_bias = bias_sum / float(count)

        results.append({
            "horizon": h,
            "n": int(len(i)),
            "mae_usd": mae,
            "rmse_usd": rmse,
            "mape_pct": mape,
            "smape_pct": smape,
            "bias_usd": bias_usd,
            "bias_pct": bias_pct,
            "tail_abs_p50": tail["p50"],
            "tail_abs_p90": tail["p90"],
            "tail_abs_p95": tail["p95"],
            "tail_abs_p99": tail["p99"],
            "tail_abs_max": tail["max"],
            "direction_acc_pct": dir_acc,
            "line_mae_usd": line_mae,
            "line_rmse_usd": line_rmse,
            "line_bias_usd": line_bias,
            "start_ts": str(start_ts),
        })

    out_df = pd.DataFrame(results)
    print(out_df.to_string(index=False))

    if args.out:
        out_df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
