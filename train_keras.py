#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import argparse
from typing import Optional, List
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from model_layers import RevIN, TSMixerBlock, ITransformerBlock, LastStep


# -----------------------------
# Utils
# -----------------------------
def parse_ts(s: str) -> pd.Timestamp:
    # принимает "2025-08-01 15:15:00+00:00", "2025-08-01T15:15:00Z", etc.
    return pd.to_datetime(s, utc=True)

def pick_feature_cols(df: pd.DataFrame, explicit: list[str] | None = None) -> list[str]:
    if explicit:
        for c in explicit:
            if c not in df.columns:
                raise ValueError(f"Feature column '{c}' not found in parquet.")
        return list(explicit)

    exclude_prefixes = ("target_", "label_", "tb_", "y_true_", "y_pred_", "future_")
    exclude_cols = {
        "timestamp",
        "label_3cls",
        "label_3cls_vf",
        "label_3cls_vf2",
        "tb_label",
        "tb_tth",
        "y_true_cls",
        "y_pred_cls",
    }
    cols = []
    for c in df.columns:
        if c in exclude_cols:
            continue
        if any(c.startswith(p) for p in exclude_prefixes):
            continue
        if df[c].dtype.kind in "ifb":  # numeric
            cols.append(c)

    if not cols:
        raise ValueError("No numeric feature columns found. Provide --feature-cols.")
    return cols

def make_binary_label(df: pd.DataFrame, target_col: str) -> np.ndarray:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available: {list(df.columns)[:50]} ...")

    s = df[target_col]

    # 3-class -> binary UP vs NOT-UP (neutral and down are 0)
    if target_col in ("label_3cls", "label_3cls_vf", "label_3cls_vf2", "target_dir", "target_dir_vf", "target_dir_vf2"):
        return (s.astype(int) == 1).astype(np.int32).to_numpy()

    # tb_label обычно 0/1/2 (где 2 == up)
    if target_col == "tb_label":
        return (s.astype(int) == 2).astype(np.int32).to_numpy()

    # если это continuous типа target_next_close (возврат), то UP = >0
    if pd.api.types.is_numeric_dtype(s):
        return (s.to_numpy(dtype=np.float32) > 0).astype(np.int32)

    raise ValueError(f"Don't know how to build binary label from '{target_col}' dtype={s.dtype}")

def make_price_target(df: pd.DataFrame, price_col: str | None) -> np.ndarray | None:
    if not price_col:
        return None
    if price_col not in df.columns:
        raise ValueError(f"Price target column '{price_col}' not found.")
    return df[price_col].to_numpy(dtype=np.float32)

def compute_norm_stats(x: np.ndarray, q_low: float, q_high: float) -> dict:
    # x: (N, F) float32
    lo = np.quantile(x, q_low, axis=0).astype(np.float32)
    hi = np.quantile(x, q_high, axis=0).astype(np.float32)
    x_clip = np.clip(x, lo, hi)
    mean = x_clip.mean(axis=0).astype(np.float32)
    std = x_clip.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    return {"mean": mean, "std": std, "q_low": lo, "q_high": hi}

def apply_norm(x: np.ndarray, stats: dict) -> np.ndarray:
    x = np.clip(x, stats["q_low"], stats["q_high"])
    return ((x - stats["mean"]) / stats["std"]).astype(np.float32)

def cosine_warmup_schedule(base_lr: float, warmup_steps: int, total_steps: int):
    base_lr = float(base_lr)
    warmup_steps = int(warmup_steps)
    total_steps = int(total_steps)

    if total_steps <= 0:
        raise ValueError("total_steps must be > 0")

    def lr_fn(step):
        step = tf.cast(step, tf.float32)
        if warmup_steps > 0:
            warm = tf.minimum(1.0, step / float(warmup_steps))
        else:
            warm = 1.0
        # cosine from 1 -> 0 after warmup
        t = tf.clip_by_value((step - float(warmup_steps)) / float(max(total_steps - warmup_steps, 1)), 0.0, 1.0)
        cosine = 0.5 * (1.0 + tf.cos(np.pi * t))
        return base_lr * warm * cosine

    return lr_fn


class CosineWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr: float, warmup_steps: int, total_steps: int, name: str = "cosine_warmup"):
        super().__init__()
        self.base_lr = float(base_lr)
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.name = name

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warm = tf.minimum(1.0, step / float(self.warmup_steps)) if self.warmup_steps > 0 else 1.0
        t = tf.clip_by_value((step - float(self.warmup_steps)) / float(max(self.total_steps - self.warmup_steps, 1)), 0.0, 1.0)
        cosine = 0.5 * (1.0 + tf.cos(np.pi * t))
        return self.base_lr * warm * cosine

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "name": self.name,
        }

class DropPath(tf.keras.layers.Layer):
    def __init__(self, drop_prob=0.0, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = float(drop_prob)

    def call(self, x, training=None):
        if (not training) or self.drop_prob <= 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (tf.shape(x)[0],) + (1,) * (len(x.shape) - 1)
        rnd = keep + tf.random.uniform(shape, dtype=x.dtype)
        mask = tf.floor(rnd)
        return (x / keep) * mask

def transformer_block(x, heads: int, dropout: float, drop_path: float):
    dim = x.shape[-1]
    if dim is None:
        dim = tf.keras.backend.int_shape(x)[-1]
    key_dim = max(int(dim // heads), 8)

    # MHSA
    h = tf.keras.layers.LayerNormalization(epsilon=1e-5)(x)
    h = tf.keras.layers.MultiHeadAttention(
        num_heads=heads, key_dim=key_dim, dropout=dropout
    )(h, h)
    h = tf.keras.layers.Dropout(dropout)(h)
    h = DropPath(drop_path)(h)
    x = tf.keras.layers.Add()([x, h])

    # FFN
    h = tf.keras.layers.LayerNormalization(epsilon=1e-5)(x)
    h = tf.keras.layers.Dense(int(dim * 4), activation="gelu")(h)
    h = tf.keras.layers.Dropout(dropout)(h)
    h = tf.keras.layers.Dense(int(dim))(h)
    h = tf.keras.layers.Dropout(dropout)(h)
    h = DropPath(drop_path)(h)
    x = tf.keras.layers.Add()([x, h])
    return x

def build_model(
    seq_len: int,
    n_features: int,
    num_classes: int,
    d_model: int,
    layers: int,
    heads: int,
    feature_dropout: float,
    dropout: float,
    drop_path: float,
    pooling: str,
    arch: str = "transformer",
    patch_size: int = 16,
    patch_stride: int = 16,
    tsmixer_mlp: int = 256,
    var_layers: int = 2,
    time_layers: int = 2,
    use_revin: bool = False,
    revin_affine: bool = True,
    quantiles: Optional[List[float]] = None,
    cls_heads: Optional[List[str]] = None,
    price_heads: Optional[List[str]] = None,
):
    inp = tf.keras.Input(shape=(seq_len, n_features), name="x")

    # "feature dropout": SpatialDropout1D дропает каналы (фичи) целиком по времени
    x = inp
    if use_revin:
        x = RevIN(affine=revin_affine)(x)
    if feature_dropout and feature_dropout > 0:
        x = tf.keras.layers.SpatialDropout1D(rate=float(feature_dropout))(x)

    arch = arch.lower()
    if arch == "patchtst":
        x = tf.keras.layers.Conv1D(
            filters=d_model,
            kernel_size=int(patch_size),
            strides=int(patch_stride),
            padding="same",
            name="patch_embed",
        )(x)
        num_patches = int(np.ceil(seq_len / float(patch_stride)))
        pos = tf.keras.layers.Embedding(input_dim=num_patches, output_dim=d_model, name="patch_pos")
        x = x + pos(tf.range(num_patches))
        for i in range(layers):
            dp = drop_path * (i / max(layers - 1, 1))
            x = transformer_block(x, heads=heads, dropout=dropout, drop_path=dp)
    elif arch == "tsmixer":
        x = tf.keras.layers.Dense(d_model)(x)
        for _ in range(layers):
            x = TSMixerBlock(mlp_dim=int(tsmixer_mlp), dropout=dropout)(x)
    elif arch == "itransformer":
        for _ in range(max(1, var_layers)):
            x = ITransformerBlock(seq_len=seq_len, d_model=d_model, heads=heads, dropout=dropout)(x)
        x = tf.keras.layers.Dense(d_model)(x)
        for i in range(max(1, time_layers)):
            dp = drop_path * (i / max(time_layers - 1, 1))
            x = transformer_block(x, heads=heads, dropout=dropout, drop_path=dp)
    elif arch == "transformer":
        x = tf.keras.layers.Dense(d_model)(x)
        for i in range(layers):
            dp = drop_path * (i / max(layers - 1, 1))
            x = transformer_block(x, heads=heads, dropout=dropout, drop_path=dp)
    else:
        raise ValueError(f"Unknown arch='{arch}' (use transformer/patchtst/tsmixer/itransformer)")

    # pooling
    pooling = pooling.lower()
    if pooling == "mean":
        h = tf.keras.layers.GlobalAveragePooling1D()(x)
    elif pooling == "max":
        h = tf.keras.layers.GlobalMaxPooling1D()(x)
    elif pooling == "last":
        h = LastStep()(x)
    elif pooling == "multi":
        h1 = tf.keras.layers.GlobalAveragePooling1D()(x)
        h2 = tf.keras.layers.GlobalMaxPooling1D()(x)
        h3 = LastStep()(x)
        h = tf.keras.layers.Concatenate()([h1, h2, h3])
    else:
        raise ValueError(f"Unknown pooling='{pooling}' (use mean/max/last/multi)")

    h = tf.keras.layers.Dense(d_model, activation="gelu")(h)
    h = tf.keras.layers.Dropout(dropout)(h)

    # outputs
    outputs = {}
    if price_heads:
        if quantiles:
            raise ValueError("quantiles are not supported with price_heads")
        for head in price_heads:
            outputs[head] = tf.keras.layers.Dense(1, name=head)(h)
    else:
        # price head (regression or quantiles)
        if quantiles:
            price_out = tf.keras.layers.Dense(len(quantiles), name="price_q")(h)
        else:
            price_out = tf.keras.layers.Dense(1, name="price")(h)
        outputs["price_q" if quantiles else "price"] = price_out

    # cls head(s)
    if cls_heads:
        for head in cls_heads:
            if num_classes <= 1:
                outputs[head] = tf.keras.layers.Dense(1, activation="sigmoid", name=head)(h)
            else:
                outputs[head] = tf.keras.layers.Dense(num_classes, activation="softmax", name=head)(h)
    else:
        if num_classes <= 1:
            outputs["cls"] = tf.keras.layers.Dense(1, activation="sigmoid", name="cls")(h)
        else:
            outputs["cls"] = tf.keras.layers.Dense(num_classes, activation="softmax", name="cls")(h)

    model = tf.keras.Model(inputs=inp, outputs=outputs, name="model")
    return model

def binary_focal_loss(alpha: float = 0.25, gamma: float = 2.0):
    alpha = float(alpha)
    gamma = float(gamma)

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        pt = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        w = alpha * y_true + (1.0 - alpha) * (1.0 - y_true)
        return -w * tf.pow(1.0 - pt, gamma) * tf.math.log(pt)
    return loss_fn

def pinball_loss(quantiles):
    qs = tf.constant(quantiles, dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        # y_true: (B,) or (B,1), y_pred: (B, Q)
        y = tf.cast(y_true, tf.float32)
        if len(y.shape) == 2:
            y = tf.squeeze(y, axis=-1)
        y = y[:, None]
        e = y - y_pred
        q = qs[None, :]
        return tf.reduce_mean(tf.maximum(q * e, (q - 1.0) * e))

    return loss_fn

class QuantileMAE(tf.keras.metrics.Metric):
    def __init__(self, q_index: int, name: str = "q_mae", **kwargs):
        super().__init__(name=name, **kwargs)
        self.q_index = int(q_index)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y = tf.cast(y_true, tf.float32)
        if len(y.shape) == 2:
            y = tf.squeeze(y, axis=-1)
        yq = tf.cast(y_pred[:, self.q_index], tf.float32)
        err = tf.abs(y - yq)
        if sample_weight is not None:
            sw = tf.cast(sample_weight, tf.float32)
            err = err * sw
            n = tf.reduce_sum(sw)
        else:
            n = tf.cast(tf.size(err), tf.float32)
        self.total.assign_add(tf.reduce_sum(err))
        self.count.assign_add(n)

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)

def make_window_dataset(X: np.ndarray, y_price: np.ndarray, y_cls: np.ndarray,
                        end_indices: np.ndarray, seq_len: int, batch_size: int,
                        shuffle: bool, sample_weight: Optional[np.ndarray] = None,
                        sample_weight_cls: Optional[np.ndarray] = None,
                        price_key: str = "price"):
    # X: (N,F) float32
    X_tf = tf.convert_to_tensor(X, dtype=tf.float32)
    y_price_tf = tf.convert_to_tensor(y_price, dtype=tf.float32)
    y_cls_tf = tf.convert_to_tensor(y_cls, dtype=tf.int32)
    if sample_weight is None:
        sample_weight = np.ones_like(y_price, dtype=np.float32)
    if sample_weight_cls is None:
        sample_weight_cls = np.ones_like(y_price, dtype=np.float32)
    w_tf = tf.convert_to_tensor(sample_weight, dtype=tf.float32)
    w_cls_tf = tf.convert_to_tensor(sample_weight_cls, dtype=tf.float32)

    end_indices = end_indices.astype(np.int64)
    ds = tf.data.Dataset.from_tensor_slices(end_indices)

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(end_indices), 20000), reshuffle_each_iteration=True)

    def map_fn(i):
        i = tf.cast(i, tf.int32)
        start = i - (seq_len - 1)
        x_seq = X_tf[start:i + 1]  # (seq_len, F)
        x_seq = tf.ensure_shape(x_seq, [seq_len, X.shape[1]])
        yp = y_price_tf[i]
        yc = y_cls_tf[i]
        w = w_tf[i]
        wc = w_cls_tf[i]
        return x_seq, {price_key: yp, "cls": yc}, {price_key: w, "cls": wc}

    ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds

def make_window_dataset_multi(X: np.ndarray, y_price: np.ndarray, y_cls_dict: dict,
                              end_indices: np.ndarray, seq_len: int, batch_size: int,
                              shuffle: bool, sample_weight: Optional[np.ndarray] = None,
                              sample_weight_cls: Optional[np.ndarray] = None,
                              price_key: str = "price", y_price_dict: Optional[dict] = None,
                              y_price_mask_dict: Optional[dict] = None):
    X_tf = tf.convert_to_tensor(X, dtype=tf.float32)
    y_price_tf = tf.convert_to_tensor(y_price, dtype=tf.float32)
    y_cls_tf = {k: tf.convert_to_tensor(v, dtype=tf.int32) for k, v in y_cls_dict.items()}
    y_price_tf_dict = {k: tf.convert_to_tensor(v, dtype=tf.float32) for k, v in y_price_dict.items()} if y_price_dict else {}
    y_price_mask_tf_dict = {k: tf.convert_to_tensor(v, dtype=tf.float32) for k, v in y_price_mask_dict.items()} if y_price_mask_dict else {}
    if sample_weight is None:
        sample_weight = np.ones_like(y_price, dtype=np.float32)
    if sample_weight_cls is None:
        sample_weight_cls = np.ones_like(y_price, dtype=np.float32)
    w_tf = tf.convert_to_tensor(sample_weight, dtype=tf.float32)
    w_cls_tf = tf.convert_to_tensor(sample_weight_cls, dtype=tf.float32)

    end_indices = end_indices.astype(np.int64)
    ds = tf.data.Dataset.from_tensor_slices(end_indices)
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(end_indices), 20000), reshuffle_each_iteration=True)

    def map_fn(i):
        i = tf.cast(i, tf.int32)
        start = i - (seq_len - 1)
        x_seq = X_tf[start:i + 1]
        x_seq = tf.ensure_shape(x_seq, [seq_len, X.shape[1]])
        w = w_tf[i]
        wc = w_cls_tf[i]
        y = {}
        wdict = {}
        if y_price_dict:
            for k, t in y_price_tf_dict.items():
                y[k] = t[i]
                if y_price_mask_dict:
                    wdict[k] = w * y_price_mask_tf_dict[k][i]
                else:
                    wdict[k] = w
        else:
            yp = y_price_tf[i]
            y[price_key] = yp
            wdict[price_key] = w
        for k, t in y_cls_tf.items():
            y[k] = tf.reshape(t[i], [])
            wdict[k] = wc
        return x_seq, y, wdict

    ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="Parquet with features+targets")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--model-out", required=True)
    ap.add_argument("--stats-out", required=True)

    ap.add_argument("--target-col", default=None, help="Classification target column (default auto)")
    ap.add_argument("--price-col", default=None, help="Regression target column (default auto)")
    ap.add_argument("--purge-gap", type=int, default=0, help="Gap (rows) between splits to avoid leakage")
    ap.add_argument("--train-end", default=None, help="Optional train end timestamp (ISO, e.g. 2024-01-01T00:00:00Z)")
    ap.add_argument("--val-end", default=None, help="Optional val end timestamp (ISO, e.g. 2024-07-01T00:00:00Z)")
    ap.add_argument("--target-horizon", type=int, default=0, help="Forecast horizon in rows (used to prevent split leakage)")
    ap.add_argument("--multi-horizons", default=None,
                    help="Comma-separated horizons for multi-horizon cls heads (e.g., 20,80,160). "
                         "Uses label_3cls_h{h} / tb_label_h{h} / target_dir_h{h} columns.")
    ap.add_argument("--price-multi-horizons", default=None,
                    help="Comma-separated horizons for multi-horizon price heads (e.g., 20,80,160). "
                         "Uses target_ret_h{h} columns and creates price_h{h} heads.")

    ap.add_argument("--num-classes", type=int, default=1)
    ap.add_argument("--cls-weight", type=float, default=1.0)
    ap.add_argument("--price-weight", type=float, default=0.0)
    ap.add_argument("--pos-weight", type=float, default=None, help="Optional positive class weight for binary cls")
    ap.add_argument("--auto-pos-weight", action="store_true",
                    help="Auto compute pos_weight from train labels if pos_weight not set")
    ap.add_argument("--cls-loss", choices=["bce", "focal"], default="bce")
    ap.add_argument("--focal-alpha", type=float, default=0.25)
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--label-smoothing", type=float, default=0.0,
                    help="Label smoothing for binary BCE (0 disables)")

    ap.add_argument("--price-loss", choices=["huber", "mse", "logcosh", "mae"], default="huber")
    ap.add_argument("--huber-delta", type=float, default=1.0)
    ap.add_argument("--price-clip-q-low", type=float, default=None,
                    help="Optional lower quantile for clipping price targets (e.g. 0.001)")
    ap.add_argument("--price-clip-q-high", type=float, default=None,
                    help="Optional upper quantile for clipping price targets (e.g. 0.999)")
    ap.add_argument("--quantiles", default=None,
                    help="Comma-separated quantiles for price head, e.g. '0.1,0.5,0.9'. "
                         "Enables pinball loss on price head.")

    ap.add_argument("--train-ratio", type=float, default=0.70)
    ap.add_argument("--val-ratio", type=float, default=0.15)

    ap.add_argument("--feature-cols", default=None,
                    help="JSON list of feature columns to force order. Example: '[\"f1\",\"f2\"]'")

    ap.add_argument("--q-low", type=float, default=0.001)
    ap.add_argument("--q-high", type=float, default=0.999)

    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--arch", choices=["transformer", "patchtst", "tsmixer", "itransformer"], default="transformer")
    ap.add_argument("--pooling", default="multi")
    ap.add_argument("--patch-size", type=int, default=16)
    ap.add_argument("--patch-stride", type=int, default=16)
    ap.add_argument("--tsmixer-mlp", type=int, default=256)
    ap.add_argument("--var-layers", type=int, default=2)
    ap.add_argument("--time-layers", type=int, default=2)
    ap.add_argument("--revin", action="store_true")
    ap.add_argument("--revin-affine", action="store_true")
    ap.add_argument("--feature-dropout", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--drop-path", type=float, default=0.05)

    ap.add_argument("--cosine", action="store_true")
    ap.add_argument("--warmup-steps", type=int, default=0)

    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--min-delta", type=float, default=1e-4)

    ap.add_argument("--start", default=None, help="Filter df timestamp >= start (UTC parseable)")
    ap.add_argument("--end", default=None, help="Filter df timestamp <= end (UTC parseable)")
    ap.add_argument("--sample-weight-col", default=None,
                    help="Optional column to weight samples (e.g., rv, rv_long).")
    ap.add_argument("--sample-weight-k", type=float, default=0.0,
                    help="Weight scale: w = 1 + k * zscore(col). 0 disables.")
    ap.add_argument("--sample-weight-clip", type=float, default=3.0,
                    help="Clip weights to [1/clip, clip].")
    ap.add_argument("--price-weight-mode", default="none", choices=["none", "close"],
                    help="Price-weighting scheme for price heads. 'close' weights by close/mean_close.")
    ap.add_argument("--price-weight-power", type=float, default=1.0,
                    help="Exponent for close-based weights.")
    ap.add_argument("--price-weight-clip", type=float, default=3.0,
                    help="Clip price weights to [1/clip, clip].")
    ap.add_argument("--price-segment-deltas", action="store_true",
                    help="For price_multi_horizons, train on segment deltas instead of cumulative returns.")
    ap.add_argument("--price-head-scale", choices=["none", "std"], default="none",
                    help="Per-head scaling for price targets (e.g., std on train).")

    args = ap.parse_args()

    multi_horizons = None
    if args.multi_horizons:
        multi_horizons = [int(x.strip()) for x in args.multi_horizons.split(",") if x.strip()]
        if not multi_horizons:
            multi_horizons = None

    # parse quantiles early so price-multi-horizon can validate against it
    quantiles = None
    if args.quantiles:
        quantiles = [float(q.strip()) for q in args.quantiles.split(",") if q.strip()]
        if len(quantiles) < 2:
            raise ValueError("--quantiles must provide at least two values, e.g. 0.1,0.5,0.9")
        if not (0 < min(quantiles) < max(quantiles) < 1):
            raise ValueError("--quantiles must be in (0,1) and strictly increasing")

    price_multi_horizons = None
    if args.price_multi_horizons:
        price_multi_horizons = [int(x.strip()) for x in args.price_multi_horizons.split(",") if x.strip()]
        if not price_multi_horizons:
            price_multi_horizons = None

    df = pd.read_parquet(args.features)

    if "timestamp" not in df.columns:
        raise ValueError("Parquet must contain 'timestamp' column.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
    df = df.sort_values("timestamp").reset_index(drop=True)

    if args.start:
        st = parse_ts(args.start)
        df = df[df["timestamp"] >= st].reset_index(drop=True)
    if args.end:
        en = parse_ts(args.end)
        df = df[df["timestamp"] <= en].reset_index(drop=True)

    if len(df) < args.seq_len + 10:
        raise ValueError(f"Too few rows after filtering: {len(df)} for seq_len={args.seq_len}")

    # choose target columns
    if multi_horizons is None:
        if args.target_col is None:
            if "target_dir" in df.columns:
                args.target_col = "target_dir"
            elif "label_3cls" in df.columns:
                args.target_col = "label_3cls"
            elif "tb_label" in df.columns:
                args.target_col = "tb_label"
            elif "target_next_close" in df.columns:
                args.target_col = "target_next_close"
            else:
                raise ValueError("Can't infer --target-col. Provide it explicitly.")

    if args.price_col is None:
        # If quantiles are requested, prefer signed returns for a directional distribution.
        if args.quantiles and "target_ret" in df.columns:
            args.price_col = "target_ret"
        elif "target_amp_abs" in df.columns:
            args.price_col = "target_amp_abs"
        elif "target_range" in df.columns:
            args.price_col = "target_range"
        else:
            args.price_col = None  # можно тренить cls-only

    feature_cols = None
    if args.feature_cols:
        feature_cols = json.loads(args.feature_cols)
    feature_cols = pick_feature_cols(df, feature_cols)

    # build targets
    cls_heads = None
    y_cls = None
    y_cls_dict = None
    def _map_dir_to_3cls(arr: np.ndarray) -> np.ndarray:
        # Accept either {-1,0,1} or {0,1,2}
        uniq = set(np.unique(arr).tolist())
        if uniq.issubset({0, 1, 2}):
            return arr.astype(np.int32)
        return np.where(arr < 0, 0, np.where(arr > 0, 2, 1)).astype(np.int32)

    if multi_horizons is None:
        y_cls = make_binary_label(df, args.target_col)
        y_cls_dict = None
    else:
        cls_heads = [f"cls_h{h}" for h in multi_horizons]
        y_cls_dict = {}
        for h in multi_horizons:
            col_3cls = f"label_3cls_h{h}"
            col_tb = f"tb_label_h{h}"
            col_dir = f"target_dir_h{h}"
            if col_3cls in df.columns:
                if args.num_classes > 1:
                    y = _map_dir_to_3cls(df[col_3cls].to_numpy()).reshape(-1)
                else:
                    y = (df[col_3cls].astype(int) == 1).astype(np.int32).to_numpy()
            elif col_tb in df.columns:
                if args.num_classes > 1:
                    y = df[col_tb].astype(int).to_numpy().reshape(-1)
                else:
                    y = (df[col_tb].astype(int) == 2).astype(np.int32).to_numpy()
            elif col_dir in df.columns:
                if args.num_classes > 1:
                    y = _map_dir_to_3cls(df[col_dir].to_numpy()).reshape(-1)
                else:
                    y = (df[col_dir].astype(int) == 1).astype(np.int32).to_numpy()
            else:
                raise ValueError(f"Missing multi-h label column for h={h} (expected {col_3cls}/{col_tb}/{col_dir})")
            y_cls_dict[f"cls_h{h}"] = y

    price_heads = None
    y_price_dict = None
    y_price_mask_dict = None
    price_head_scale = {}
    price_target_mode = "cumulative"
    price_clip_meta = {}
    if price_multi_horizons:
        if multi_horizons is None:
            y_cls_dict = {"cls": y_cls}
        if quantiles:
            raise ValueError("--price-multi-horizons is not compatible with --quantiles")
        price_heads = [f"price_h{h}" for h in price_multi_horizons]
        y_price_dict = {}
        y_price_mask_dict = {}
        for h in price_multi_horizons:
            col = f"target_ret_h{h}"
            if col not in df.columns:
                raise ValueError(f"Missing price target column for h={h}: {col}")
            arr = df[col].to_numpy(dtype=np.float32)
            mask = np.isfinite(arr).astype(np.float32)
            if not np.all(mask):
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            if args.price_clip_q_low is not None or args.price_clip_q_high is not None:
                valid = arr[mask > 0]
                lo = np.quantile(valid, args.price_clip_q_low) if args.price_clip_q_low is not None else None
                hi = np.quantile(valid, args.price_clip_q_high) if args.price_clip_q_high is not None else None
                arr = np.clip(arr, lo if lo is not None else -np.inf, hi if hi is not None else np.inf)
                price_clip_meta[f"price_h{h}"] = (lo, hi)
            y_price_dict[f"price_h{h}"] = arr
            y_price_mask_dict[f"price_h{h}"] = mask
        if args.price_segment_deltas:
            price_target_mode = "segment_deltas"
            sorted_h = sorted(price_multi_horizons)
            delta_dict = {}
            delta_mask_dict = {}
            prev_h = None
            for h in sorted_h:
                key = f"price_h{h}"
                cur = y_price_dict[key]
                cur_mask = y_price_mask_dict[key]
                if prev_h is None:
                    delta = cur
                    delta_mask = cur_mask
                else:
                    prev_key = f"price_h{prev_h}"
                    prev = y_price_dict[prev_key]
                    prev_mask = y_price_mask_dict[prev_key]
                    delta = cur - prev
                    delta_mask = cur_mask * prev_mask
                delta_dict[key] = delta
                delta_mask_dict[key] = delta_mask
                prev_h = h
            y_price_dict = delta_dict
            y_price_mask_dict = delta_mask_dict
        y_price = np.zeros((len(df),), dtype=np.float32)
        args.price_col = "target_ret_multi"
    else:
        y_price = make_price_target(df, args.price_col)
        if y_price is None:
            y_price = np.zeros((len(df),), dtype=np.float32)
        elif args.price_clip_q_low is not None or args.price_clip_q_high is not None:
            valid = y_price[np.isfinite(y_price)]
            lo = np.quantile(valid, args.price_clip_q_low) if args.price_clip_q_low is not None else None
            hi = np.quantile(valid, args.price_clip_q_high) if args.price_clip_q_high is not None else None
            y_price = np.clip(y_price, lo if lo is not None else -np.inf, hi if hi is not None else np.inf)
            price_clip_meta["price"] = (lo, hi)

    # train/val/test split by time index (end index of window)
    n = len(df)
    if args.train_end:
        train_end_ts = parse_ts(args.train_end)
        n_train = int(df.index[df["timestamp"] <= train_end_ts].max()) + 1
    else:
        n_train = int(n * args.train_ratio)
    if args.val_end:
        val_end_ts = parse_ts(args.val_end)
        val_end = int(df.index[df["timestamp"] <= val_end_ts].max()) + 1
    else:
        n_val = int(n * args.val_ratio)
        val_end = min(n_train + n_val, n - 1)

    # window ends must be >= seq_len-1
    min_end = args.seq_len - 1
    purge = max(0, int(args.purge_gap))
    hgap = max(0, int(args.target_horizon))
    if multi_horizons:
        hgap = max(hgap, max(multi_horizons))
    if price_multi_horizons:
        hgap = max(hgap, max(price_multi_horizons))
    train_max = max(min_end, n_train - hgap)
    val_max = max(min_end, val_end - hgap)
    test_max = max(min_end, n - hgap)
    train_ends = np.arange(min_end, train_max, dtype=np.int64)
    val_start = max(n_train + purge, min_end)
    test_start = max(val_end + purge, min_end)
    val_ends = np.arange(val_start, val_max, dtype=np.int64)
    test_ends = np.arange(test_start, test_max, dtype=np.int64)

    if len(train_ends) < 100 or len(val_ends) < 50 or len(test_ends) < 50:
        raise ValueError(f"Split too small. train={len(train_ends)} val={len(val_ends)} test={len(test_ends)}")

    # optional per-head scaling (train-only)
    if args.price_head_scale != "none":
        if price_multi_horizons:
            if args.price_head_scale == "std":
                for h in price_multi_horizons:
                    key = f"price_h{h}"
                    arr = y_price_dict[key]
                    mask = y_price_mask_dict[key]
                    valid = arr[:n_train][mask[:n_train] > 0]
                    scale = float(np.std(valid)) if valid.size > 0 else 1.0
                    if not np.isfinite(scale) or scale < 1e-8:
                        scale = 1.0
                    y_price_dict[key] = (arr / scale).astype(np.float32)
                    price_head_scale[key] = scale
        else:
            if args.price_head_scale == "std":
                valid = y_price[:n_train]
                scale = float(np.std(valid)) if valid.size > 0 else 1.0
                if not np.isfinite(scale) or scale < 1e-8:
                    scale = 1.0
                y_price = (y_price / scale).astype(np.float32)
                price_head_scale["price"] = scale

    # compute norm stats ONLY on train rows (0..n_train-1)
    X_raw = df[feature_cols].to_numpy(dtype=np.float32)
    stats = compute_norm_stats(X_raw[:n_train], q_low=args.q_low, q_high=args.q_high)
    X = apply_norm(X_raw, stats)

    # save stats
    np.savez(
        args.stats_out,
        feature_names=np.array(feature_cols, dtype=object),
        mean=stats["mean"],
        std=stats["std"],
        q_low=stats["q_low"],
        q_high=stats["q_high"],
        quantiles=np.array(quantiles if quantiles else [], dtype=np.float32),
        target_col=np.array([args.target_col], dtype=object),
        price_col=np.array([args.price_col if args.price_col else ""], dtype=object),
        price_clip_meta=np.array([price_clip_meta], dtype=object),
        price_head_scale=np.array([price_head_scale], dtype=object),
        price_target_mode=np.array([price_target_mode], dtype=object),
        seq_len=np.array([args.seq_len], dtype=np.int32),
        multi_horizons=np.array(multi_horizons if multi_horizons else [], dtype=np.int32),
        price_multi_horizons=np.array(price_multi_horizons if price_multi_horizons else [], dtype=np.int32),
        train_end_ts=np.array([str(df.iloc[n_train - 1]["timestamp"])], dtype=object),
        val_start_ts=np.array([str(df.iloc[max(n_train, 0)]["timestamp"])], dtype=object),
        test_start_ts=np.array([str(df.iloc[max(val_end, 0)]["timestamp"])], dtype=object),
    )

    print(f"[data] rows={n} features={len(feature_cols)} seq_len={args.seq_len}")
    print(f"[split] train_ends={len(train_ends)} val_ends={len(val_ends)} test_ends={len(test_ends)}")
    print(f"[split] train_end_ts={df.iloc[n_train-1]['timestamp']}")
    print(f"[split] test_start_ts={df.iloc[max(val_end,0)]['timestamp']}")
    if multi_horizons is None:
        print(f"[label] target_col={args.target_col} pos_rate={float(y_cls.mean()):.4f}")
    else:
        pr = {k: float(v[train_ends].mean()) for k, v in y_cls_dict.items()}
        print(f"[label] multi_horizons={multi_horizons} pos_rate(train)={pr}")
    if args.price_col:
        print(f"[price] price_col={args.price_col} mean={float(y_price.mean()):.6g} std={float(y_price.std()):.6g}")
        if price_clip_meta:
            print(f"[price] clip={price_clip_meta}")
        if price_head_scale:
            print(f"[price_scale] {price_head_scale}")
    else:
        print("[price] disabled (no price_col)")

    price_key = "price_q" if quantiles else "price"

    # optional sample weights (volatility-aware or custom column)
    sample_weight = np.ones((n,), dtype=np.float32)
    if args.sample_weight_col and float(args.sample_weight_k) != 0.0:
        if args.sample_weight_col not in df.columns:
            raise ValueError(f"--sample-weight-col '{args.sample_weight_col}' not found in df")
        col = df[args.sample_weight_col].to_numpy(dtype=np.float32)
        # z-score on train only
        mu = float(np.mean(col[:n_train]))
        sd = float(np.std(col[:n_train]) + 1e-9)
        z = (col - mu) / sd
        w = 1.0 + float(args.sample_weight_k) * z
        clip = float(args.sample_weight_clip)
        if clip > 0:
            w = np.clip(w, 1.0 / clip, clip)
        sample_weight = w.astype(np.float32)
        print(f"[weights] col={args.sample_weight_col} k={args.sample_weight_k} clip={args.sample_weight_clip}")

    # price-weighting (align training with $-error); apply only to price heads
    sample_weight_price = sample_weight.copy()
    sample_weight_cls = np.ones((n,), dtype=np.float32)
    if args.price_weight_mode == "close":
        if "close" not in df.columns:
            raise ValueError("--price-weight-mode close requires 'close' column")
        close = df["close"].to_numpy(dtype=np.float32)
        mean_close = float(np.mean(close[:n_train]) + 1e-9)
        w = (close / mean_close) ** float(args.price_weight_power)
        clip = float(args.price_weight_clip)
        if clip > 0:
            w = np.clip(w, 1.0 / clip, clip)
        sample_weight_price = (sample_weight_price * w.astype(np.float32)).astype(np.float32)
        print(f"[price_weight] mode=close power={args.price_weight_power} clip={args.price_weight_clip}")

    # datasets
    if multi_horizons is None and not price_multi_horizons:
        ds_train = make_window_dataset(
            X, y_price, y_cls, train_ends, args.seq_len, args.batch_size,
            shuffle=True, sample_weight=sample_weight_price, sample_weight_cls=sample_weight_cls, price_key=price_key
        )
        ds_val = make_window_dataset(
            X, y_price, y_cls, val_ends, args.seq_len, args.batch_size,
            shuffle=False, sample_weight=sample_weight_price, sample_weight_cls=sample_weight_cls, price_key=price_key
        )
        ds_test = make_window_dataset(
            X, y_price, y_cls, test_ends, args.seq_len, args.batch_size,
            shuffle=False, sample_weight=sample_weight_price, sample_weight_cls=sample_weight_cls, price_key=price_key
        )
    else:
        ds_train = make_window_dataset_multi(
            X, y_price, y_cls_dict, train_ends, args.seq_len, args.batch_size,
            shuffle=True, sample_weight=sample_weight_price, sample_weight_cls=sample_weight_cls, price_key=price_key,
            y_price_dict=y_price_dict, y_price_mask_dict=y_price_mask_dict
        )
        ds_val = make_window_dataset_multi(
            X, y_price, y_cls_dict, val_ends, args.seq_len, args.batch_size,
            shuffle=False, sample_weight=sample_weight_price, sample_weight_cls=sample_weight_cls, price_key=price_key,
            y_price_dict=y_price_dict, y_price_mask_dict=y_price_mask_dict
        )
        ds_test = make_window_dataset_multi(
            X, y_price, y_cls_dict, test_ends, args.seq_len, args.batch_size,
            shuffle=False, sample_weight=sample_weight_price, sample_weight_cls=sample_weight_cls, price_key=price_key,
            y_price_dict=y_price_dict, y_price_mask_dict=y_price_mask_dict
        )

    # model
    model = build_model(
        seq_len=args.seq_len,
        n_features=len(feature_cols),
        num_classes=args.num_classes,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        feature_dropout=args.feature_dropout,
        dropout=args.dropout,
        drop_path=args.drop_path,
        pooling=args.pooling,
        arch=args.arch,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        tsmixer_mlp=args.tsmixer_mlp,
        var_layers=args.var_layers,
        time_layers=args.time_layers,
        use_revin=args.revin,
        revin_affine=args.revin_affine,
        quantiles=quantiles,
        cls_heads=cls_heads,
        price_heads=price_heads,
    )

    # losses
    if args.num_classes <= 1:
        if args.cls_loss == "focal":
            cls_loss = binary_focal_loss(alpha=args.focal_alpha, gamma=args.focal_gamma)
        else:
            cls_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=float(args.label_smoothing))
        cls_metrics = [
            tf.keras.metrics.BinaryAccuracy(name="acc"),
            tf.keras.metrics.Precision(name="precision_up"),
            tf.keras.metrics.Recall(name="recall_up"),
            tf.keras.metrics.AUC(curve="PR", name="auc_pr"),
            tf.keras.metrics.AUC(curve="ROC", name="auc_roc"),
        ]
    else:
        cls_loss = tf.keras.losses.SparseCategoricalCrossentropy()
        cls_metrics = [
            tf.keras.metrics.SparseCategoricalAccuracy(name="acc"),
        ]

    if quantiles:
        price_loss = pinball_loss(quantiles)
        q_med = int(np.argmin([abs(q - 0.5) for q in quantiles]))
        def make_price_metrics():
            return [QuantileMAE(q_med, name=f"mae_q{quantiles[q_med]:.2f}")]
    elif args.price_loss == "mse":
        price_loss = tf.keras.losses.MeanSquaredError()
        def make_price_metrics():
            return [tf.keras.metrics.MeanAbsoluteError(name="mae")]
    elif args.price_loss == "mae":
        price_loss = tf.keras.losses.MeanAbsoluteError()
        def make_price_metrics():
            return [tf.keras.metrics.MeanAbsoluteError(name="mae")]
    elif args.price_loss == "logcosh":
        price_loss = tf.keras.losses.LogCosh()
        def make_price_metrics():
            return [tf.keras.metrics.MeanAbsoluteError(name="mae")]
    else:
        price_loss = tf.keras.losses.Huber(delta=float(args.huber_delta))
        def make_price_metrics():
            return [tf.keras.metrics.MeanAbsoluteError(name="mae")]

    if price_heads:
        price_keys = list(price_heads)
    else:
        price_keys = [price_key]

    if cls_heads:
        loss_weights = {k: float(args.price_weight) for k in price_keys}
        for h in cls_heads:
            loss_weights[h] = float(args.cls_weight)
    else:
        loss_weights = {k: float(args.price_weight) for k in price_keys}
        loss_weights["cls"] = float(args.cls_weight)

    # optimizer / lr schedule
    steps_per_epoch = int(np.ceil(len(train_ends) / args.batch_size))
    total_steps = steps_per_epoch * args.epochs

    if args.cosine:
        lr_sched = CosineWarmup(args.lr, args.warmup_steps, total_steps)
        opt = tf.keras.optimizers.Adam(learning_rate=lr_sched)
    else:
        opt = tf.keras.optimizers.Adam(learning_rate=args.lr)

    if cls_heads:
        loss_dict = {k: price_loss for k in price_keys}
        metrics_dict = {k: make_price_metrics() for k in price_keys}
        for h in cls_heads:
            loss_dict[h] = cls_loss
            if args.num_classes <= 1:
                metrics_dict[h] = [
                    tf.keras.metrics.BinaryAccuracy(name=f"{h}_acc"),
                    tf.keras.metrics.Precision(name=f"{h}_precision_up"),
                    tf.keras.metrics.Recall(name=f"{h}_recall_up"),
                    tf.keras.metrics.AUC(curve="PR", name=f"{h}_auc_pr"),
                    tf.keras.metrics.AUC(curve="ROC", name=f"{h}_auc_roc"),
                ]
            else:
                metrics_dict[h] = [
                    tf.keras.metrics.SparseCategoricalAccuracy(name=f"{h}_acc"),
                ]
        model.compile(optimizer=opt, loss=loss_dict, loss_weights=loss_weights, metrics=metrics_dict)
    else:
        model.compile(
            optimizer=opt,
            loss={**{k: price_loss for k in price_keys}, "cls": cls_loss},
            loss_weights=loss_weights,
            metrics={**{k: make_price_metrics() for k in price_keys}, "cls": cls_metrics},
        )
    # class weighting for binary classification (optional)
    pos_w = None
    pos_w_dict = None
    if args.num_classes <= 1:
        if args.pos_weight is not None:
            pos_w = float(args.pos_weight)
        elif args.auto_pos_weight and multi_horizons is None:
            # compute on train ends only
            pos_rate = float(np.mean(y_cls[train_ends]))
            if pos_rate > 1e-6:
                pos_w = float((1.0 - pos_rate) / pos_rate)
                print(f"[pos_weight] auto pos_rate={pos_rate:.4f} -> pos_weight={pos_w:.4f}")
            else:
                print("[pos_weight] auto disabled (pos_rate ~ 0)")
        elif args.auto_pos_weight and multi_horizons is not None:
            pos_w_dict = {}
            for k, v in y_cls_dict.items():
                pos_rate = float(np.mean(v[train_ends]))
                if pos_rate > 1e-6:
                    pos_w_dict[k] = float((1.0 - pos_rate) / pos_rate)
                else:
                    pos_w_dict[k] = 1.0
            print(f"[pos_weight] auto multi {pos_w_dict}")

    cb = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            min_delta=args.min_delta,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    # train
    if pos_w is not None:
        # multiply existing sample weights on cls head only
        def add_pos_weight(x, y, w):
            cls = y["cls"]
            w_cls = tf.where(tf.equal(cls, 1), pos_w, 1.0)
            return x, y, {price_key: w[price_key], "cls": w["cls"] * w_cls}
        ds_train = ds_train.map(add_pos_weight, num_parallel_calls=tf.data.AUTOTUNE)
        ds_val = ds_val.map(add_pos_weight, num_parallel_calls=tf.data.AUTOTUNE)
    elif pos_w_dict is not None:
        def add_pos_weight_multi(x, y, w):
            w_out = {price_key: w[price_key]}
            for k, pw in pos_w_dict.items():
                cls = y[k]
                w_cls = tf.where(tf.equal(cls, 1), pw, 1.0)
                w_out[k] = w[k] * w_cls
            return x, y, w_out
        ds_train = ds_train.map(add_pos_weight_multi, num_parallel_calls=tf.data.AUTOTUNE)
        ds_val = ds_val.map(add_pos_weight_multi, num_parallel_calls=tf.data.AUTOTUNE)

    model.fit(ds_train, validation_data=ds_val, epochs=args.epochs, callbacks=cb, verbose=1)

    # test eval
    metrics = model.evaluate(ds_test, verbose=1)
    print("Test metrics:", metrics)

    # save final model
    model.save(args.model_out)
    print(f"Saved model to {args.model_out}")
    print(f"Saved normalization stats to {args.stats_out}")


if __name__ == "__main__":
    main()
