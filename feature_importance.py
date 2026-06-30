"""
Permutation feature importance on a holdout set for the Keras model.
Computes loss increase when shuffling each feature.
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from trading_keras_core import default_feature_list


def main():
    parser = argparse.ArgumentParser(description="Permutation importance for Keras model.")
    parser.add_argument("--features", required=True, help="Features parquet")
    parser.add_argument("--model", required=True, help="Trained Keras model path")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--samples", type=int, default=20000, help="Use last N rows for evaluation")
    args = parser.parse_args()

    df = pd.read_parquet(args.features).dropna()
    if args.samples:
        df = df.tail(args.samples)
    feature_cols = default_feature_list()
    target_col = "target_next_close"

    feats = df[feature_cols].to_numpy(dtype=np.float32)
    targets = df[target_col].to_numpy(dtype=np.float32)

    windows = []
    y_true = []
    for i in range(len(feats) - args.seq_len):
        windows.append(feats[i : i + args.seq_len])
        y_true.append(targets[i + args.seq_len])
    x = np.stack(windows, axis=0)
    y_true = np.array(y_true, dtype=np.float32)

    model = tf.keras.models.load_model(args.model)
    preds = model.predict(x, verbose=0).squeeze()
    base_loss = np.mean((preds - y_true) ** 2)

    importances = []
    rng = np.random.default_rng(seed=42)
    for idx, col in enumerate(feature_cols):
        x_perm = x.copy()
        perm = rng.permutation(x_perm.shape[0])
        x_perm[:, :, idx] = x_perm[perm, :, idx]
        preds_perm = model.predict(x_perm, verbose=0).squeeze()
        loss_perm = np.mean((preds_perm - y_true) ** 2)
        importances.append((col, loss_perm - base_loss))
        print(f"{col}: delta_loss={loss_perm - base_loss:.6f}")

    out = pd.DataFrame(importances, columns=["feature", "delta_mse"]).sort_values("delta_mse", ascending=False)
    out.to_csv("feature_importance.csv", index=False)
    print("Saved importance to feature_importance.csv")


if __name__ == "__main__":
    main()
