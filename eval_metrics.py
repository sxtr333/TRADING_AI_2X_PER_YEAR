"""
Compute offline metrics for trained models on the test split.

Outputs per model:
- MAE, RMSE in log-return units
- MAE in pct space (exp(logret)-1)
- Sign accuracy (using price head)

Supported models: pass multiple --model/--weights pairs.
If weights are Keras checkpoints (.ckpt), we rebuild the architecture and restore via tf.train.Checkpoint.
If full .keras file is given without weights, we attempt load_model(safe_mode=False).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from trading_keras_core import default_feature_list, make_tf_dataset, build_sequence_model


def time_split(df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15):
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = df.iloc[:n_train]
    val = df.iloc[n_train : n_train + n_val]
    test = df.iloc[n_train + n_val :]
    return train, val, test


def infer_periods_per_year(df: pd.DataFrame) -> float:
    if "timestamp" not in df.columns or len(df) < 3:
        return 365 * 24
    ts = pd.to_datetime(df["timestamp"], utc=True).sort_values()
    deltas = ts.diff().dropna().dt.total_seconds().values
    if deltas.size == 0:
        return 365 * 24
    med = float(np.median(deltas))
    if med <= 0:
        return 365 * 24
    minutes = med / 60.0
    return (365 * 24 * 60) / minutes


def load_model_from_ckpt(seq_len: int, n_features: int, ckpt_prefix: Path):
    model = build_sequence_model(
        seq_len=seq_len,
        n_features=n_features,
        pooling="multi",
        num_transformer_layers=2,
        n_heads=4,
        feature_dropout=0.1,
    )
    ckpt = tf.train.Checkpoint(model=model)
    status = ckpt.restore(str(ckpt_prefix))
    status.expect_partial()
    return model


def evaluate(model, ds, periods_per_year: float):
    preds = model.predict(ds, verbose=0).reshape(-1)
    true = np.array(list(ds.map(lambda _x, y: y).unbatch().as_numpy_iterator())).reshape(-1)
    mae = np.mean(np.abs(true - preds))
    rmse = np.sqrt(np.mean((true - preds) ** 2))
    true_pct = np.expm1(true)
    pred_pct = np.expm1(preds)
    mae_pct = np.mean(np.abs(true_pct - pred_pct))
    sign_acc = ((true > 0) == (preds > 0)).mean()
    # SMAPE on pct returns
    smape = np.mean(2 * np.abs(true_pct - pred_pct) / (np.abs(true_pct) + np.abs(pred_pct) + 1e-9))
    # correlation
    corr = np.corrcoef(true, preds)[0, 1]
    # simple long/flat PnL (no fees)
    position = (preds > 0).astype(float)
    pnl = position * true_pct
    cagr = (1 + pnl).prod() ** (periods_per_year / len(pnl)) - 1
    sharpe = np.mean(pnl) / (np.std(pnl) + 1e-9) * np.sqrt(periods_per_year)
    mdd = 0.0
    equity = np.cumprod(1 + pnl)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    mdd = drawdown.min()
    return {
        "mae_log": mae,
        "rmse_log": rmse,
        "mae_pct": mae_pct,
        "smape_pct": smape,
        "sign_acc": sign_acc,
        "corr": corr,
        "pnl_cagr": cagr,
        "pnl_sharpe": sharpe,
        "pnl_max_dd": mdd,
        "n": len(true),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/BTCUSDT_1h_features.parquet")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument(
        "--model",
        action="append",
        default=[],
        help="Path to .keras file (used only for name/report)",
    )
    ap.add_argument(
        "--weights",
        action="append",
        default=[],
        help="Path prefix to checkpoint (.ckpt) or .keras weights file aligned with --model",
    )
    args = ap.parse_args()

    df = pd.read_parquet(args.features)
    feature_cols = default_feature_list()
    train, val, test = time_split(df)
    test_ds = make_tf_dataset(
        test,
        feature_cols,
        "target_next_close",
        seq_len=args.seq_len,
        batch_size=args.batch,
        stride=1,
        shuffle=False,
    )
    periods_per_year = infer_periods_per_year(test)

    if not args.model and not args.weights:
        print("Nothing to evaluate. Provide --model/--weights.")
        return

    pairs = []
    if args.weights:
        for i, w in enumerate(args.weights):
            name = args.model[i] if i < len(args.model) else Path(w).name
            pairs.append((name, w))
    else:
        for m in args.model:
            pairs.append((m, m))

    results = {}
    for name, wpath in pairs:
        wp = Path(wpath)
        model = None
        if wp.suffix == ".ckpt" or wp.name.endswith(".ckpt"):
            prefix = wp
            if wp.suffix == ".index":
                prefix = wp.with_suffix("")
            elif (wp.with_suffix(".index")).exists():
                prefix = wp
            elif (Path(str(wp) + ".index")).exists():
                prefix = Path(str(wp))
            else:
                print(f"[warn] No .index for {wp}, skipping")
                continue
            try:
                model = load_model_from_ckpt(args.seq_len, len(feature_cols), prefix)
            except Exception as e:
                print(f"[warn] Failed to load ckpt {prefix}: {e}")
                continue
        else:
            try:
                # compile=False to ignore saved losses/metrics strings that may not be callable at load
                model = tf.keras.models.load_model(wp, safe_mode=False, compile=False)
            except Exception as e:
                print(f"[warn] Failed to load model {wp}: {e}")
                continue
        metrics = evaluate(model, test_ds, periods_per_year)
        results[name] = metrics
        print(name, metrics)

    print("Done.")


if __name__ == "__main__":
    main()
