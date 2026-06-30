"""
Ensemble prediction utility: averages predictions from multiple Keras models.
- Supports MC dropout via --mc-samples
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf

from trading_keras_core import default_feature_list


def main():
    parser = argparse.ArgumentParser(description="Ensemble predictions from multiple models.")
    parser.add_argument("--features", required=True, help="Features parquet")
    parser.add_argument("--models", nargs="+", required=True, help="List of Keras model paths")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--out", default="ensemble_preds.csv")
    parser.add_argument("--stats", help="npz with mean/std")
    parser.add_argument("--mc-samples", type=int, default=1)
    args = parser.parse_args()

    df = pd.read_parquet(args.features).dropna()
    df = df.tail(args.limit + args.seq_len)
    feats = df[default_feature_list()].to_numpy(dtype=np.float32)
    ts = df["timestamp"].to_numpy()

    windows = []
    ts_out = []
    for i in range(len(feats) - args.seq_len):
        windows.append(feats[i : i + args.seq_len])
        ts_out.append(ts[i + args.seq_len])
    x = np.stack(windows, axis=0)
    if args.stats:
        stats = np.load(args.stats)
        mean = stats["mean"]
        std = np.where(stats["std"] == 0, 1.0, stats["std"])
        x = (x - mean) / std
    x_tf = tf.convert_to_tensor(x, dtype=tf.float32)

    preds_list: List[np.ndarray] = []
    for model_path in args.models:
        model = tf.keras.models.load_model(model_path)
        if args.mc_samples > 1:
            mc_preds = []
            for _ in range(args.mc_samples):
                mc_preds.append(model(x_tf, training=True).numpy().squeeze())
            mc_preds = np.stack(mc_preds, axis=0).mean(axis=0)
            preds_list.append(mc_preds)
        else:
            preds_list.append(model(x_tf, training=False).numpy().squeeze())

    ensemble = np.mean(preds_list, axis=0)
    out_df = pd.DataFrame({"timestamp": ts_out, "prediction": ensemble})
    out_df.to_csv(Path(args.out), index=False)
    print(f"Saved ensemble predictions to {args.out}")


if __name__ == "__main__":
    main()
