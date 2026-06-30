"""
model6/train_keras.py

Training script for the Keras sequence model on your prepared features parquet.

Что исправлено (по проблемам из чата):
1) Нормализация применяется правильно:
   - mean/std считаются ТОЛЬКО на TRAIN-сплите
   - те же stats применяются к train/val/test внутри tf.data
   - stats сохраняются в --stats-out и должны использоваться в evaluate_keras.py

2) Binary direction head обучается на TB-событиях, но сохраняет полный контекст:
   - используется label_3cls (-1, 0, 1)
   - neutral (0) НЕ выкидываем из последовательностей, но игнорируем в cls loss через sample_weight=0

3) Балансировка классов (по умолчанию включена для binary):
   - up/down события балансируются 50/50 через sample_from_datasets
   - отключить: --no-balance

4) Печатает split timestamps:
   - копируешь test_start_ts и вставляешь в evaluate_keras.py --start

Пример:
  python model6/train_keras.py \
    --features model6/data/BTCUSDT_15m_features_tb.parquet \
    --seq-len 256 --batch-size 64 --epochs 12 --lr 3e-4 \
    --model-out model6/model_15m_bin.keras \
    --stats-out model6/norm_stats_15m_bin.npz \
    --num-classes 1 --cls-weight 1.0 --price-weight 0.0 \
    --pooling multi --layers 2 --heads 4 \
    --feature-dropout 0.05 --drop-path 0.05 \
    --cosine --warmup-steps 500 --patience 3 --min-delta 1e-4
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from trading_keras_core import build_sequence_model, default_feature_list, make_tf_dataset


def _as_utc_timestamp_series(s: pd.Series) -> pd.Series:
    ts = pd.to_datetime(s, utc=True, errors="coerce")
    if ts.isna().any():
        bad = s[ts.isna()].head(5).tolist()
        raise ValueError(f"Failed to parse some timestamps (showing up to 5): {bad}")
    return ts


@dataclass
class SplitInfo:
    n: int
    n_train: int
    n_val: int
    gap: int
    val_start: int
    test_start: int
    test_start_with_gap: int


def compute_time_split_idx(n: int, train_frac: float, val_frac: float, purge_gap: int) -> SplitInfo:
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    gap = int(purge_gap)
    val_start = min(n, n_train + gap)
    test_start = min(n, val_start + n_val)
    test_start_with_gap = min(n, test_start + gap)
    return SplitInfo(
        n=n,
        n_train=n_train,
        n_val=n_val,
        gap=gap,
        val_start=val_start,
        test_start=test_start,
        test_start_with_gap=test_start_with_gap,
    )


def time_split_df(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    purge_gap: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitInfo]:
    info = compute_time_split_idx(len(df), train_frac=train_frac, val_frac=val_frac, purge_gap=purge_gap)
    train = df.iloc[: info.n_train].copy()
    val = df.iloc[info.val_start : info.test_start].copy()
    test = df.iloc[info.test_start_with_gap :].copy()
    return train, val, test, info


class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup -> cosine decay to 0."""

    def __init__(self, base_lr: float, warmup_steps: int, total_steps: int):
        super().__init__()
        self.base_lr = float(base_lr)
        self.warmup_steps = int(max(0, warmup_steps))
        self.total_steps = int(max(1, total_steps))

    def __call__(self, step: tf.Tensor) -> tf.Tensor:
        step_f = tf.cast(step, tf.float32)
        base = tf.cast(self.base_lr, tf.float32)

        if self.warmup_steps > 0:
            warmup = base * tf.minimum(1.0, (step_f + 1.0) / float(self.warmup_steps))
        else:
            warmup = base

        decay_steps = max(1, self.total_steps - self.warmup_steps)
        t = tf.clip_by_value((step_f - float(self.warmup_steps)) / float(decay_steps), 0.0, 1.0)
        cosine = 0.5 * (1.0 + tf.cos(np.pi * t))
        decayed = base * cosine

        return tf.where(step_f < float(self.warmup_steps), warmup, decayed)

    def get_config(self):
        return {"base_lr": self.base_lr, "warmup_steps": self.warmup_steps, "total_steps": self.total_steps}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Keras sequence model on prepared features parquet.")
    parser.add_argument("--features", required=True, help="Path to features parquet (from build_features.py)")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length for model input")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=12, help="Training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Base learning rate")
    parser.add_argument("--cosine", action="store_true", help="Use warmup+cosine LR schedule")
    parser.add_argument("--warmup-steps", type=int, default=500, help="Warmup steps (only if --cosine)")

    parser.add_argument("--model-out", default="model.keras", help="Path to save the trained model (.keras)")
    parser.add_argument("--stats-out", default="norm_stats.npz", help="Where to save mean/std stats (npz)")

    parser.add_argument("--target-col", default="target_amp_abs", help="Target column for price head")
    parser.add_argument("--label-col", default="label_3cls", help="Direction label column (-1/0/1)")

    parser.add_argument("--train-frac", type=float, default=0.70, help="Train fraction (time split)")
    parser.add_argument("--val-frac", type=float, default=0.15, help="Val fraction (time split)")
    parser.add_argument(
        "--purge-gap",
        type=int,
        default=None,
        help="Gap (rows) between splits to avoid leakage; default seq_len+48",
    )

    # Heads / loss weights
    parser.add_argument(
        "--num-classes",
        type=int,
        default=1,
        help="0 => no cls head. 1 => binary sigmoid head. 3 => softmax 3-class head",
    )
    parser.add_argument("--cls-weight", type=float, default=1.0, help="Loss weight for classification head")
    parser.add_argument("--price-weight", type=float, default=0.0, help="Loss weight for price head")
    parser.add_argument("--down-weight", type=float, default=1.0, help="Extra sample-weight multiplier for DOWN events (binary only)")

    # Balancing (binary only) — default ON
    parser.add_argument(
        "--balance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Balance up/down event windows 50/50 (binary only). Default: enabled. Use --no-balance to disable.",
    )

    # Model hyperparams
    parser.add_argument("--positional-encoding", action="store_true", help="Add sinusoidal positional encoding")
    parser.add_argument("--pooling", choices=["last", "mean", "cls", "multi", "attn"], default="last")
    parser.add_argument("--heads", type=int, default=4, help="Transformer attention heads")
    parser.add_argument("--layers", type=int, default=2, help="Transformer encoder layers")
    parser.add_argument("--d-model", type=int, default=256, help="Model hidden size")
    parser.add_argument("--mlp-hidden", type=int, default=128, help="Final MLP hidden size")
    parser.add_argument("--feature-dropout", type=float, default=0.05, help="Dropout on input features before projection")
    parser.add_argument("--drop-path", type=float, default=0.0, help="Stochastic depth drop path rate")
    parser.add_argument("--layerscale", action="store_true", help="Enable LayerScale in transformer blocks")
    parser.add_argument("--depthwise", action="store_true", help="Use depthwise separable convs in TCN")
    parser.add_argument("--se", action="store_true", help="Enable squeeze-excite after TCN")
    parser.add_argument("--se-reduction", type=int, default=4, help="Reduction ratio for squeeze-excite")

    # Training controls
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="Early stopping min delta")
    parser.add_argument("--clipnorm", type=float, default=1.0, help="Gradient clipping norm")
    args = parser.parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_parquet(args.features)
    if df.empty:
        raise ValueError("Features parquet is empty.")

    # Sort by timestamp
    if "timestamp" not in df.columns:
        raise ValueError("No 'timestamp' column in parquet.")
    df["timestamp"] = _as_utc_timestamp_series(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Feature columns
    feature_cols = list(default_feature_list())
    missing_feats = [c for c in feature_cols if c not in df.columns]
    if missing_feats:
        raise ValueError(f"Missing feature columns in parquet: {missing_feats[:10]} (total {len(missing_feats)})")

    # Leakage guard
    bad = [c for c in feature_cols if c.startswith("target_") or c.startswith("label_") or c == "timestamp"]
    if bad:
        raise ValueError(f"Leakage columns in feature list: {bad}")

    if args.target_col not in df.columns:
        raise ValueError(f"Target column '{args.target_col}' not found in parquet.")
    if args.label_col not in df.columns:
        raise ValueError(f"Label column '{args.label_col}' not found in parquet.")

    use_cls = args.num_classes is not None and int(args.num_classes) > 0
    if use_cls:
        lbl = df[args.label_col].to_numpy()

        if args.num_classes == 1:
            # Binary: up=1, down=-1, neutral=0 (neutral ignored via weight 0)
            df["cls_y"] = (lbl == 1).astype(np.int32)   # up=1 else 0
            df["cls_w"] = (lbl != 0).astype(np.float32) # events weight=1, neutral weight=0
            if args.down_weight != 1.0:
                df.loc[df[args.label_col] == -1, "cls_w"] *= float(args.down_weight)
        elif args.num_classes == 3:
            mapping = {-1: 0, 0: 1, 1: 2}
            df["cls_y"] = pd.Series(lbl).map(mapping).astype(np.int32).to_numpy()
            df["cls_w"] = np.ones(len(df), dtype=np.float32)
        else:
            raise ValueError("--num-classes supported values: 0, 1, 3")
    else:
        df["cls_y"] = 0
        df["cls_w"] = 0.0

    # Drop NaNs in needed columns
    needed = feature_cols + [args.target_col, args.label_col, "cls_y", "cls_w"]
    before = len(df)
    df = df.dropna(subset=needed).reset_index(drop=True)
    if len(df) != before:
        print(f"Dropped rows with NaNs: {before} -> {len(df)}")

    # Split
    purge_gap = int(args.purge_gap) if args.purge_gap is not None else (args.seq_len + 48)
    train_df, val_df, test_df, info = time_split_df(df, train_frac=args.train_frac, val_frac=args.val_frac, purge_gap=purge_gap)

    if len(train_df) <= args.seq_len + 1 or len(val_df) <= args.seq_len + 1 or len(test_df) <= args.seq_len + 1:
        raise ValueError(
            f"Not enough rows for seq_len={args.seq_len}. "
            f"train={len(train_df)} val={len(val_df)} test={len(test_df)}"
        )

    # Print split timestamps
    train_end_ts = df.iloc[info.n_train - 1]["timestamp"] if info.n_train > 0 else None
    val_start_ts = df.iloc[info.val_start]["timestamp"] if info.val_start < len(df) else None
    test_start_ts = df.iloc[info.test_start_with_gap]["timestamp"] if info.test_start_with_gap < len(df) else None
    print(f"rows: {len(df)}")
    print(f"train_end_ts: {train_end_ts}")
    print(f"val_start_ts: {val_start_ts}")
    print(f"test_start_ts: {test_start_ts}   <-- use this for evaluate --start")

    # Normalization stats from TRAIN only
    feats_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    mean = feats_train.mean(axis=0).astype(np.float32)
    std = feats_train.std(axis=0).astype(np.float32)
    std = np.where(std == 0, 1.0, std).astype(np.float32)
    np.savez(args.stats_out, mean=mean, std=std, features=np.array(feature_cols))
    print(f"Saved normalization stats to {args.stats_out}")

    mean_tf = tf.constant(mean.reshape(1, 1, -1), dtype=tf.float32)
    std_tf = tf.constant(std.reshape(1, 1, -1), dtype=tf.float32)

    def norm_x(x: tf.Tensor) -> tf.Tensor:
        return (tf.cast(x, tf.float32) - mean_tf) / std_tf

    # Build datasets (make_tf_dataset windows as seq_len+1 -> x[:-1], y[-1])
    def build_ds(split_df: pd.DataFrame, shuffle: bool):
        extra = {"cls": split_df["cls_y"].to_numpy(np.int32), "cls_w": split_df["cls_w"].to_numpy(np.float32)} if use_cls else None
        ds = make_tf_dataset(
            split_df,
            feature_cols=feature_cols,
            target_col=args.target_col,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            stride=1,
            shuffle=shuffle,
            extra_targets=extra,
        )
        ds = ds.map(lambda x, y: (norm_x(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        return ds

    train_ds = build_ds(train_df, shuffle=True)
    val_ds = build_ds(val_df, shuffle=False)
    test_ds = build_ds(test_df, shuffle=False)

    # Build model
    model = build_sequence_model(
        seq_len=args.seq_len,
        n_features=len(feature_cols),
        d_model=args.d_model,
        mlp_hidden=args.mlp_hidden,
        use_positional_encoding=args.positional_encoding,
        feature_dropout=args.feature_dropout,
        pooling=args.pooling,
        num_transformer_layers=args.layers,
        n_heads=args.heads,
        use_se=args.se,
        se_reduction=args.se_reduction,
        drop_path_rate=args.drop_path,
        use_layerscale=args.layerscale,
        use_depthwise=args.depthwise,
        num_classes=args.num_classes if use_cls else None,
    )

    ckpt_path = args.model_out + ".weights.h5"
    if use_cls:
        monitor = "val_cls_auc_pr" if args.num_classes == 1 else "val_cls_loss"
        mode = "max" if monitor.endswith("auc_pr") else "min"
    else:
        monitor = "val_loss"
        mode = "min"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path,
            save_best_only=True,
            save_weights_only=True,
            monitor=monitor,
            mode=mode,
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=args.patience,
            min_delta=args.min_delta,
            restore_best_weights=True,
            monitor=monitor,
            mode=mode,
        ),
    ]

    if use_cls:
        # Convert to (x, y_dict, sample_weight_dict)
        def to_fit(x, y):
            price = tf.cast(y["price"], tf.float32)

            cls_y = y["cls"]
            cls_w = tf.cast(y["cls_w"], tf.float32)

            if args.num_classes == 1:
                cls_y = tf.cast(cls_y, tf.float32)
            else:
                cls_y = tf.cast(cls_y, tf.int32)

            y_out = {"price": price, "cls": cls_y}
            sw_out = {"price": tf.ones_like(price, tf.float32), "cls": cls_w}
            return x, y_out, sw_out

        # Unbatch + filter to event windows (cls_w>0)
        train_u = train_ds.unbatch().map(to_fit, num_parallel_calls=tf.data.AUTOTUNE).filter(lambda x, y, sw: sw["cls"] > 0)
        val_u = val_ds.unbatch().map(to_fit, num_parallel_calls=tf.data.AUTOTUNE).filter(lambda x, y, sw: sw["cls"] > 0)
        test_u = test_ds.unbatch().map(to_fit, num_parallel_calls=tf.data.AUTOTUNE).filter(lambda x, y, sw: sw["cls"] > 0)

        # Event counts at label positions (index >= seq_len)
        events_train = int((train_df["cls_w"].to_numpy()[args.seq_len:] > 0).sum())
        events_val = int((val_df["cls_w"].to_numpy()[args.seq_len:] > 0).sum())
        events_test = int((test_df["cls_w"].to_numpy()[args.seq_len:] > 0).sum())
        print(f"Event windows (label_3cls!=0) after seq shift: train={events_train} val={events_val} test={events_test}")

        steps_per_epoch = max(1, math.ceil(events_train / args.batch_size))
        val_steps = max(1, math.ceil(events_val / args.batch_size))
        test_steps = max(1, math.ceil(events_test / args.batch_size))

        # LR schedule (needs FINAL steps_per_epoch)
        total_steps = steps_per_epoch * args.epochs
        lr = WarmupCosine(args.lr, args.warmup_steps, total_steps) if args.cosine else args.lr
        opt = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=args.clipnorm)

        # Balance only for binary classification
        if args.num_classes == 1 and args.balance:
            ds_up = train_u.filter(lambda x, y, sw: tf.equal(tf.cast(y["cls"], tf.int32), 1)).repeat()
            ds_dn = train_u.filter(lambda x, y, sw: tf.equal(tf.cast(y["cls"], tf.int32), 0)).repeat()
            train_fit = tf.data.Dataset.sample_from_datasets([ds_up, ds_dn], weights=[0.5, 0.5])
            print("Balancing: ON (50/50)")
        else:
            train_fit = train_u.shuffle(50_000, seed=args.seed, reshuffle_each_iteration=True).repeat()
            print("Balancing: OFF")

        train_fit = train_fit.batch(args.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        val_fit = val_u.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
        test_fit = test_u.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

        # Losses & metrics
        if args.num_classes == 1:
            cls_loss = tf.keras.losses.BinaryCrossentropy()
            cls_metrics = [
                tf.keras.metrics.BinaryAccuracy(name="acc"),
                tf.keras.metrics.Precision(name="precision_up"),
                tf.keras.metrics.Recall(name="recall_up"),
                tf.keras.metrics.AUC(name="auc_pr", curve="PR"),
                tf.keras.metrics.AUC(name="auc_roc", curve="ROC"),
            ]
        else:
            cls_loss = tf.keras.losses.SparseCategoricalCrossentropy()
            cls_metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="acc")]

        model.compile(
            optimizer=opt,
            loss={"price": tf.keras.losses.Huber(), "cls": cls_loss},
            loss_weights={"price": float(args.price_weight), "cls": float(args.cls_weight)},
            metrics={"price": [tf.keras.metrics.MeanAbsoluteError(name="mae")], "cls": cls_metrics},
        )

        print("=== TRAIN ===")
        model.fit(
            train_fit,
            validation_data=val_fit,
            steps_per_epoch=steps_per_epoch,
            validation_steps=val_steps,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=1,
        )

        print("=== TEST ===")
        test_res = model.evaluate(test_fit, steps=test_steps, return_dict=True, verbose=1)
        print("Test metrics:", test_res)

    else:
        # Regression-only
        steps_per_epoch = max(1, math.ceil((len(train_df) - args.seq_len) / args.batch_size))
        total_steps = steps_per_epoch * args.epochs
        lr = WarmupCosine(args.lr, args.warmup_steps, total_steps) if args.cosine else args.lr
        opt = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=args.clipnorm)

        model.compile(
            optimizer=opt,
            loss=tf.keras.losses.Huber(),
            metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
        )

        print("=== TRAIN (regression only) ===")
        model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks, verbose=1)

        print("=== TEST ===")
        test_res = model.evaluate(test_ds, return_dict=True, verbose=1)
        print("Test metrics:", test_res)

    # Load best weights (if checkpoint exists)
    if os.path.exists(ckpt_path):
        model.load_weights(ckpt_path)

    model.save(args.model_out)
    print(f"Saved model to {args.model_out}")


if __name__ == "__main__":
    main()
