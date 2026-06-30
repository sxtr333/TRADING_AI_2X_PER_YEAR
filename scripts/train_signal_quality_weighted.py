#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


def build_text(row: pd.Series) -> str:
    symbol = str(row.get("symbol", "") or "")
    direction = str(row.get("direction", "") or "")
    text = str(row.get("text", "") or "")
    return f"{symbol} {direction} {text}".strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Train quality classifier on final_trade_dataset with sample weights")
    ap.add_argument("--data", required=True)
    ap.add_argument("--model-out", required=True)
    ap.add_argument("--summary-json", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=30000)
    ap.add_argument("--seq-len", type=int, default=220)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--ignore-sample-weight", action="store_true")
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--lstm-units", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.35)
    ap.add_argument("--dense-units", type=int, default=64)
    ap.add_argument("--num-units", type=int, default=16)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--lr", type=float, default=7e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    args = ap.parse_args()

    df = pd.read_parquet(args.data).copy()
    if "quality_label" not in df.columns:
        raise ValueError("quality_label column is required")
    if "timestamp_utc" not in df.columns:
        raise ValueError("timestamp_utc column is required")

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df[df["timestamp_utc"].notna()].copy()
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    labels = ["bad", "neutral", "good"]
    label_to_id = {k: i for i, k in enumerate(labels)}
    df = df[df["quality_label"].isin(labels)].copy()
    df["y"] = df["quality_label"].map(label_to_id).astype(np.int32)

    df["text_in"] = df.apply(build_text, axis=1)
    df["direction_long"] = (df["direction"].astype(str).str.lower() == "long").astype(np.float32)
    df["parse_confidence"] = pd.to_numeric(df.get("parse_confidence", 0.0), errors="coerce").fillna(0.0).astype(np.float32)
    df["text_len"] = df["text"].astype(str).str.len().astype(np.float32)
    df["is_candidate_f"] = df.get("is_candidate", False).astype(np.float32)
    if (not args.ignore_sample_weight) and "sample_weight" in df.columns:
        sw = pd.to_numeric(df["sample_weight"], errors="coerce").fillna(1.0).astype(np.float32)
    else:
        sw = pd.Series(np.ones(len(df), dtype=np.float32))
    df["sample_weight"] = sw

    n = len(df)
    n_train = int(n * args.train_frac)
    n_val = int(n * args.val_frac)
    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train : n_train + n_val].copy()
    test = df.iloc[n_train + n_val :].copy()
    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise ValueError("split produced an empty train/val/test part")

    xnum_cols = ["direction_long", "parse_confidence", "text_len", "is_candidate_f"]

    x_text_train = train["text_in"].astype(str).to_numpy()
    x_text_val = val["text_in"].astype(str).to_numpy()
    x_text_test = test["text_in"].astype(str).to_numpy()
    x_num_train = train[xnum_cols].to_numpy(dtype=np.float32)
    x_num_val = val[xnum_cols].to_numpy(dtype=np.float32)
    x_num_test = test[xnum_cols].to_numpy(dtype=np.float32)
    y_train = train["y"].to_numpy(dtype=np.int32)
    y_val = val["y"].to_numpy(dtype=np.int32)
    y_test = test["y"].to_numpy(dtype=np.int32)
    w_train = train["sample_weight"].to_numpy(dtype=np.float32)
    w_val = val["sample_weight"].to_numpy(dtype=np.float32)

    text_vec = tf.keras.layers.TextVectorization(
        max_tokens=args.max_tokens,
        output_mode="int",
        output_sequence_length=args.seq_len,
        standardize="lower_and_strip_punctuation",
    )
    text_vec.adapt(tf.data.Dataset.from_tensor_slices(x_text_train).batch(1024))

    text_in = tf.keras.Input(shape=(), dtype=tf.string, name="text")
    num_in = tf.keras.Input(shape=(len(xnum_cols),), dtype=tf.float32, name="num")

    x = text_vec(text_in)
    reg = tf.keras.regularizers.l2(float(args.l2))
    x = tf.keras.layers.Embedding(args.max_tokens, int(args.embed_dim))(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(int(args.lstm_units), kernel_regularizer=reg, recurrent_regularizer=reg)
    )(x)
    x = tf.keras.layers.Dropout(float(args.dropout))(x)
    y = tf.keras.layers.Dense(int(args.num_units), activation="relu", kernel_regularizer=reg)(num_in)
    y = tf.keras.layers.BatchNormalization()(y)
    z = tf.keras.layers.Concatenate()([x, y])
    z = tf.keras.layers.Dense(int(args.dense_units), activation="relu", kernel_regularizer=reg)(z)
    z = tf.keras.layers.Dropout(float(args.dropout))(z)
    out = tf.keras.layers.Dense(3, activation="softmax")(z)

    model = tf.keras.Model(inputs={"text": text_in, "num": num_in}, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(args.lr)),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )

    cb = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.5, patience=2, min_lr=1e-5),
    ]

    hist = model.fit(
        {"text": x_text_train, "num": x_num_train},
        y_train,
        validation_data=({"text": x_text_val, "num": x_num_val}, y_val, w_val),
        sample_weight=w_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
        callbacks=cb,
    )

    test_metrics = model.evaluate({"text": x_text_test, "num": x_num_test}, y_test, verbose=0)
    pred = model.predict({"text": x_text_test, "num": x_num_test}, verbose=0)
    yhat = np.argmax(pred, axis=1)
    conf = np.zeros((3, 3), dtype=np.int64)
    for a, b in zip(y_test, yhat):
        conf[int(a), int(b)] += 1

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_out)

    summary = {
        "data": args.data,
        "rows_total": int(len(df)),
        "split_rows": {"train": int(len(train)), "val": int(len(val)), "test": int(len(test))},
        "class_counts_total": df["quality_label"].value_counts().to_dict(),
        "test_loss": float(test_metrics[0]),
        "test_acc": float(test_metrics[1]) if len(test_metrics) > 1 else None,
        "confusion_matrix_rows_true_cols_pred_bad_neutral_good": conf.tolist(),
        "labels": labels,
        "history_last": {k: float(v[-1]) for k, v in hist.history.items()},
        "model_out": str(model_out),
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
