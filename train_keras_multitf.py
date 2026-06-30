"""
Training script for explicit multi-timeframe fusion (base + context inputs).
- Base: fine TF (e.g., 1m) with seq_len_base
- Context: coarser TF (e.g., 5m/15m/1h) with seq_len_ctx (ffilled to base timestamps)
- Uses build_multitf_model from trading_keras_multitf.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from trading_keras_core import default_feature_list
from trading_keras_multitf import build_multitf_model


def align_ctx(base: pd.DataFrame, ctx: pd.DataFrame) -> pd.DataFrame:
    """Align context dataframe to base timestamps with forward fill."""
    ctx = ctx.sort_values("timestamp").set_index("timestamp")
    base = base.sort_values("timestamp").set_index("timestamp")
    aligned = base.join(ctx, how="left").ffill().reset_index()
    return aligned


def make_multitf_dataset(
    base_df: pd.DataFrame,
    ctx_df: pd.DataFrame,
    feature_cols: Tuple[str, ...],
    target_col: str,
    seq_len_base: int,
    seq_len_ctx: int,
    batch_size: int,
    shuffle: bool = True,
):
    feats_base = base_df[list(feature_cols)].to_numpy(np.float32)
    feats_ctx = ctx_df[list(feature_cols)].to_numpy(np.float32)
    targets = base_df[target_col].to_numpy(np.float32)
    total = len(base_df)
    max_len = max(seq_len_base, seq_len_ctx)
    if total < max_len + 1:
        raise ValueError("Not enough rows for requested sequence lengths.")

    base_windows = []
    ctx_windows = []
    ys = []
    for i in range(total - max_len):
        base_windows.append(feats_base[i : i + seq_len_base])
        ctx_windows.append(feats_ctx[i : i + seq_len_ctx])
        ys.append(targets[i + seq_len_base])  # predict next after base window
    base_windows = np.stack(base_windows, axis=0)
    ctx_windows = np.stack(ctx_windows, axis=0)
    ys = np.array(ys, dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices(((base_windows, ctx_windows), ys))
    if shuffle:
        ds = ds.shuffle(2048)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def time_split(df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15):
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = df.iloc[:n_train]
    val = df.iloc[n_train : n_train + n_val]
    test = df.iloc[n_train + n_val :]
    return train, val, test


def main():
    parser = argparse.ArgumentParser(description="Train multitf cross-attention model.")
    parser.add_argument("--base-features", required=True, help="Base TF features parquet (e.g., 1m)")
    parser.add_argument("--ctx-features", required=True, help="Context TF features parquet (e.g., 5m/15m/1h)")
    parser.add_argument("--seq-len-base", type=int, default=256)
    parser.add_argument("--seq-len-ctx", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--model-out", default="model_multitf.keras")
    args = parser.parse_args()

    base_df = pd.read_parquet(args.base_features).dropna()
    ctx_df = pd.read_parquet(args.ctx_features).dropna()
    # Align context to base timestamps
    aligned = align_ctx(base_df, ctx_df)

    feature_cols = tuple(default_feature_list())
    target_col = "target_next_close"
    aligned = aligned.dropna(subset=list(feature_cols) + [target_col])

    train_df, val_df, test_df = time_split(aligned)

    train_ds = make_multitf_dataset(train_df, train_df, feature_cols, target_col, args.seq_len_base, args.seq_len_ctx, args.batch_size, shuffle=True)
    val_ds = make_multitf_dataset(val_df, val_df, feature_cols, target_col, args.seq_len_base, args.seq_len_ctx, args.batch_size, shuffle=False)
    test_ds = make_multitf_dataset(test_df, test_df, feature_cols, target_col, args.seq_len_base, args.seq_len_ctx, args.batch_size, shuffle=False)

    model = build_multitf_model(
        seq_len_base=args.seq_len_base,
        n_features_base=len(feature_cols),
        seq_len_ctx=args.seq_len_ctx,
        n_features_ctx=len(feature_cols),
        d_model=256,
        dropout=0.1,
        n_heads=4,
        layers_base=2,
        layers_ctx=2,
        gmlp_dim=256,
        use_glu=True,
        pooling="attn",
    )
    model.optimizer.learning_rate = args.lr

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(args.model_out, save_best_only=True, monitor="val_loss"),
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss"),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    test_loss = model.evaluate(test_ds)
    print(f"Test loss: {test_loss:.6f}")
    model.save(args.model_out)
    print(f"Saved multitf model to {args.model_out}")


if __name__ == "__main__":
    main()
