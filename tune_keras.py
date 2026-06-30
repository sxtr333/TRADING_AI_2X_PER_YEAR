"""
Hyperparameter search using Keras Tuner (RandomSearch).
- Tunes model dims (d_model, heads, transformer layers, dropout), seq_len, batch size
- Uses a subset of data for speed
"""

from __future__ import annotations

import argparse
from pathlib import Path

import keras_tuner as kt
import pandas as pd
import tensorflow as tf

from trading_keras_core import default_feature_list, make_tf_dataset, build_sequence_model


def build_model(hp, seq_len: int, n_features: int):
    d_model = hp.Choice("d_model", [128, 192, 256, 320])
    heads = hp.Choice("heads", [2, 4, 8])
    layers = hp.Choice("transformer_layers", [1, 2, 3])
    dropout = hp.Float("dropout", 0.05, 0.3, step=0.05)
    mlp_hidden = hp.Choice("mlp_hidden", [64, 128, 192])
    model = build_sequence_model(
        seq_len=seq_len,
        n_features=n_features,
        d_model=d_model,
        n_heads=heads,
        num_transformer_layers=layers,
        dropout=dropout,
        mlp_hidden=mlp_hidden,
    )
    lr = hp.Float("lr", 1e-4, 5e-4, sampling="log")
    model.optimizer.learning_rate = lr
    return model


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for Keras model.")
    parser.add_argument("--features", required=True, help="Features parquet")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length")
    parser.add_argument("--max-trials", type=int, default=10, help="Tuner trials")
    parser.add_argument("--executions", type=int, default=1, help="Executions per trial")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per trial")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--limit", type=int, default=50000, help="Use only last N rows for tuning")
    parser.add_argument("--project", default="kt_tuning", help="Tuner project name")
    args = parser.parse_args()

    df = pd.read_parquet(args.features).dropna()
    if args.limit:
        df = df.tail(args.limit)
    feature_cols = default_feature_list()
    target_col = "target_next_close"

    ds = make_tf_dataset(df, feature_cols, target_col, seq_len=args.seq_len, batch_size=args.batch_size, stride=1, shuffle=True)

    tuner = kt.RandomSearch(
        lambda hp: build_model(hp, seq_len=args.seq_len, n_features=len(feature_cols)),
        objective="loss",
        max_trials=args.max_trials,
        executions_per_trial=args.executions,
        project_name=args.project,
        overwrite=True,
    )

    tuner.search(ds, epochs=args.epochs)
    best = tuner.get_best_hyperparameters(1)[0]
    print("Best hyperparameters:")
    for k, v in best.values.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
