"""
Simple inference utility for a trained Keras model on prepared features.
- Loads features parquet and a saved model
- Generates predictions on the latest N sequences and writes CSV
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from trading_keras_core import default_feature_list


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with a trained Keras model.")
    parser.add_argument("--features", required=True, help="Path to features parquet (built via build_features.py)")
    parser.add_argument("--model", required=True, help="Path to saved Keras model (.keras or SavedModel)")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length used during training")
    parser.add_argument("--limit", type=int, default=1000, help="Number of most recent sequences to predict")
    parser.add_argument("--out", default="predictions.csv", help="Output CSV with timestamp and prediction")
    parser.add_argument("--stats", help="Path to npz with mean/std for normalization")
    parser.add_argument("--mc-samples", type=int, default=1, help="Monte Carlo dropout samples for uncertainty; >1 enables stochastic forward with training=True")
    args = parser.parse_args()

    df = pd.read_parquet(args.features).dropna()
    feature_cols = default_feature_list()
    target_col = "target_next_close"

    # Take last `limit + seq_len` rows to form windows
    df_tail = df.tail(args.limit + args.seq_len)
    feats = df_tail[feature_cols].to_numpy()
    timestamps = df_tail["timestamp"].to_numpy()

    windows = []
    ts_out = []
    for i in range(len(feats) - args.seq_len):
        windows.append(feats[i : i + args.seq_len])
        ts_out.append(timestamps[i + args.seq_len])

    x = np.stack(windows, axis=0)
    if args.stats:
        stats = np.load(args.stats)
        mean = stats["mean"]
        std = stats["std"]
        std = np.where(std == 0, 1.0, std)
        x = (x - mean) / std

    x = tf.convert_to_tensor(x, dtype=tf.float32)
    model = tf.keras.models.load_model(args.model)
    if args.mc_samples > 1:
        preds_samples = []
        for _ in range(args.mc_samples):
            preds_samples.append(model(x, training=True).numpy().squeeze())
        preds_samples = np.stack(preds_samples, axis=0)
        preds = preds_samples.mean(axis=0)
        preds_std = preds_samples.std(axis=0)
    else:
        preds = model.predict(x, verbose=0).squeeze()
        preds_std = None

    out_df = pd.DataFrame({"timestamp": ts_out, "prediction": preds})
    if preds_std is not None:
        out_df["pred_std"] = preds_std
    out_path = Path(args.out)
    out_df.to_csv(out_path, index=False)
    print(f"Saved {len(out_df)} predictions to {out_path}")


if __name__ == "__main__":
    main()
