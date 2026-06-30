#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf

from model_layers import RevIN, LastStep


def parse_ts(s: str) -> pd.Timestamp:
    return pd.to_datetime(s, utc=True)


def pick_feature_cols(df: pd.DataFrame, explicit: list[str] | None = None) -> list[str]:
    if explicit:
        for c in explicit:
            if c not in df.columns:
                raise ValueError(f"Feature column '{c}' not found in parquet.")
        return list(explicit)

    exclude_prefixes = ("target_",)
    exclude_cols = {"timestamp", "label_3cls", "tb_label", "tb_tth", "y_true_cls", "y_pred_cls"}
    cols = []
    for c in df.columns:
        if c in exclude_cols:
            continue
        if any(c.startswith(p) for p in exclude_prefixes):
            continue
        if df[c].dtype.kind in "ifb":
            cols.append(c)
    if not cols:
        raise ValueError("No numeric feature columns found. Provide --feature-cols.")
    return cols


def compute_norm_stats(x: np.ndarray, q_low: float, q_high: float) -> dict:
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


def build_patchtst_backbone(seq_len: int, n_features: int, d_model: int, patch_size: int, patch_stride: int, layers: int, heads: int, dropout: float, use_revin: bool, revin_affine: bool):
    inp = tf.keras.Input(shape=(seq_len, n_features), name="x")
    x = inp
    if use_revin:
        x = RevIN(affine=revin_affine)(x)
    x = tf.keras.layers.Conv1D(filters=d_model, kernel_size=patch_size, strides=patch_stride, padding="same", name="patch_embed")(x)
    num_patches = int(np.ceil(seq_len / float(patch_stride)))
    pos = tf.keras.layers.Embedding(input_dim=num_patches, output_dim=d_model, name="patch_pos")
    x = x + pos(tf.range(num_patches))
    for _ in range(layers):
        h = tf.keras.layers.LayerNormalization(epsilon=1e-5)(x)
        h = tf.keras.layers.MultiHeadAttention(num_heads=heads, key_dim=max(d_model // max(heads, 1), 8), dropout=dropout)(h, h)
        h = tf.keras.layers.Dropout(dropout)(h)
        x = tf.keras.layers.Add()([x, h])
        h = tf.keras.layers.LayerNormalization(epsilon=1e-5)(x)
        h = tf.keras.layers.Dense(d_model * 4, activation="gelu")(h)
        h = tf.keras.layers.Dropout(dropout)(h)
        h = tf.keras.layers.Dense(d_model)(h)
        h = tf.keras.layers.Dropout(dropout)(h)
        x = tf.keras.layers.Add()([x, h])
    pooled = tf.keras.layers.GlobalAveragePooling1D()(x)
    pooled = tf.keras.layers.Dense(d_model, activation="gelu")(pooled)
    pooled = tf.keras.layers.Dropout(dropout)(pooled)
    return inp, pooled


def build_multihorizon_model(seq_len: int, n_features: int, horizons: List[int], d_model: int, patch_size: int, patch_stride: int, layers: int, heads: int, dropout: float, use_revin: bool, revin_affine: bool):
    inp, pooled = build_patchtst_backbone(seq_len, n_features, d_model, patch_size, patch_stride, layers, heads, dropout, use_revin, revin_affine)
    outputs = {}
    for h in horizons:
        outputs[f"price_h{h}"] = tf.keras.layers.Dense(1, name=f"price_h{h}")(pooled)
        outputs[f"cls_h{h}"] = tf.keras.layers.Dense(1, activation="sigmoid", name=f"cls_h{h}")(pooled)
    model = tf.keras.Model(inputs=inp, outputs=outputs, name="patchtst_multihorizon")
    return model


def make_window_dataset(X: np.ndarray, y_price: np.ndarray, y_cls: np.ndarray, end_indices: np.ndarray, seq_len: int, batch_size: int, shuffle: bool):
    X_tf = tf.convert_to_tensor(X, dtype=tf.float32)
    y_price_tf = tf.convert_to_tensor(y_price, dtype=tf.float32)
    y_cls_tf = tf.convert_to_tensor(y_cls, dtype=tf.float32)

    end_indices = end_indices.astype(np.int64)
    ds = tf.data.Dataset.from_tensor_slices(end_indices)
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(end_indices), 20000), reshuffle_each_iteration=True)

    def map_fn(i):
        i = tf.cast(i, tf.int32)
        start = i - (seq_len - 1)
        x_seq = X_tf[start:i + 1]
        x_seq = tf.ensure_shape(x_seq, [seq_len, X.shape[1]])
        y = {}
        for k in range(y_price_tf.shape[1]):
            y[f"price_h{horizons[k]}"] = y_price_tf[i, k]
            y[f"cls_h{horizons[k]}"] = y_cls_tf[i, k]
        return x_seq, y

    ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--horizons", default="4,8,20,48")
    ap.add_argument("--model-out", required=True)
    ap.add_argument("--stats-out", required=True)
    ap.add_argument("--feature-cols", default=None)
    ap.add_argument("--q-low", type=float, default=0.001)
    ap.add_argument("--q-high", type=float, default=0.999)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--patch-size", type=int, default=16)
    ap.add_argument("--patch-stride", type=int, default=16)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--revin", action="store_true")
    ap.add_argument("--revin-affine", action="store_true")
    ap.add_argument("--train-ratio", type=float, default=0.70)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    args = ap.parse_args()

    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
    if not horizons:
        raise ValueError("Empty horizons list.")

    df = pd.read_parquet(args.features)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
    df = df.sort_values("timestamp").reset_index(drop=True)
    if args.start:
        df = df[df["timestamp"] >= parse_ts(args.start)].reset_index(drop=True)
    if args.end:
        df = df[df["timestamp"] <= parse_ts(args.end)].reset_index(drop=True)

    if len(df) < args.seq_len + max(horizons) + 10:
        raise ValueError("Too few rows for given horizons/seq_len.")

    feature_cols = None
    if args.feature_cols:
        feature_cols = json.loads(args.feature_cols)
    feature_cols = pick_feature_cols(df, feature_cols)

    close = df["close"].to_numpy(dtype=np.float32)
    y_price = []
    y_cls = []
    for h in horizons:
        fut = np.log(close[h:]) - np.log(close[:-h])
        pad = np.full((h,), np.nan, dtype=np.float32)
        targ = np.concatenate([fut, pad], axis=0)
        y_price.append(np.abs(targ))
        y_cls.append((targ > 0).astype(np.float32))
    y_price = np.stack(y_price, axis=1)
    y_cls = np.stack(y_cls, axis=1)

    n = len(df)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    val_end = min(n_train + n_val, n - max(horizons) - 1)
    max_h = max(horizons)
    min_end = args.seq_len - 1
    max_end = n - max_h - 1

    train_ends = np.arange(min_end, min(n_train, max_end), dtype=np.int64)
    val_ends = np.arange(max(n_train, min_end), min(val_end, max_end), dtype=np.int64)
    test_ends = np.arange(max(val_end, min_end), max_end, dtype=np.int64)

    X_raw = df[feature_cols].to_numpy(dtype=np.float32)
    stats = compute_norm_stats(X_raw[:n_train], q_low=args.q_low, q_high=args.q_high)
    X = apply_norm(X_raw, stats)

    np.savez(
        args.stats_out,
        feature_names=np.array(feature_cols, dtype=object),
        mean=stats["mean"],
        std=stats["std"],
        q_low=stats["q_low"],
        q_high=stats["q_high"],
        horizons=np.array(horizons, dtype=np.int32),
        seq_len=np.array([args.seq_len], dtype=np.int32),
    )

    ds_train = make_window_dataset(X, y_price, y_cls, train_ends, args.seq_len, args.batch_size, shuffle=True)
    ds_val = make_window_dataset(X, y_price, y_cls, val_ends, args.seq_len, args.batch_size, shuffle=False)
    ds_test = make_window_dataset(X, y_price, y_cls, test_ends, args.seq_len, args.batch_size, shuffle=False)

    model = build_multihorizon_model(
        seq_len=args.seq_len,
        n_features=len(feature_cols),
        horizons=horizons,
        d_model=args.d_model,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
        use_revin=args.revin,
        revin_affine=args.revin_affine,
    )

    losses = {}
    metrics = {}
    for h in horizons:
        losses[f"price_h{h}"] = tf.keras.losses.Huber()
        losses[f"cls_h{h}"] = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        metrics[f"price_h{h}"] = [tf.keras.metrics.MeanAbsoluteError(name="mae")]
        metrics[f"cls_h{h}"] = [
            tf.keras.metrics.BinaryAccuracy(name="acc"),
            tf.keras.metrics.AUC(name="auc_pr", curve="PR"),
            tf.keras.metrics.AUC(name="auc_roc", curve="ROC"),
        ]

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4), loss=losses, metrics=metrics)
    model.fit(ds_train, validation_data=ds_val, epochs=args.epochs, verbose=1)
    test_metrics = model.evaluate(ds_test, verbose=1)
    print("Test metrics:", test_metrics)
    model.save(args.model_out)
    print(f"Saved model to {args.model_out}")
