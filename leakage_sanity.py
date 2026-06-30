#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from train_keras import pick_feature_cols


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--sample", type=int, default=30000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--corr-thr", type=float, default=0.2)
    p.add_argument("--future-corr-thr", type=float, default=0.95)
    return p.parse_args()


def make_simple_model(n_features: int) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(n_features,), name="x")
    x = tf.keras.layers.LayerNormalization()(inp)
    x = tf.keras.layers.Dense(64, activation="gelu")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(2e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(curve="ROC", name="auc")],
    )
    return model


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    df = pd.read_parquet(args.features)
    df = df.dropna().reset_index(drop=True)

    if args.label not in df.columns:
        raise SystemExit(f"Label '{args.label}' not in columns.")

    feature_cols = pick_feature_cols(df)

    # 1) Direct leakage scan
    leak_like = [c for c in feature_cols if c.startswith(("target_", "label_", "tb_", "y_true_", "y_pred_", "future_"))]
    print("[check] features:", len(feature_cols))
    if leak_like:
        print("[leak] suspicious features in X:", leak_like)
    else:
        print("[ok] no obvious target/label columns in X")

    # 2) Correlation scan
    y_full = df[args.label]
    if y_full.nunique() > 2:
        y_full = (y_full.astype(int) == 1).astype(np.int32)
    else:
        y_full = y_full.astype(np.int32)

    X_full = df[feature_cols].to_numpy(np.float32)
    # sample for speed
    if len(X_full) > args.sample:
        idx = rng.choice(len(X_full), size=args.sample, replace=False)
        X = X_full[idx]
        y = y_full.iloc[idx].to_numpy()
    else:
        X = X_full
        y = y_full.to_numpy()

    # correlation with label
    y_center = y - y.mean()
    corr = []
    for i in range(X.shape[1]):
        xi = X[:, i]
        denom = (xi.std() * y_center.std())
        c = 0.0 if denom == 0 else np.corrcoef(xi, y_center)[0, 1]
        corr.append(c)
    corr = np.array(corr)
    top_idx = np.argsort(np.abs(corr))[::-1][:10]
    print("[corr] top 10 |corr| features vs label:")
    for i in top_idx:
        print(f"  {feature_cols[i]}: corr={corr[i]:.4f}")

    # 2b) Future-close correlation scan (leakage proxy)
    suspicious = set()
    corr_thr = float(args.corr_thr)
    future_corr_thr = float(args.future_corr_thr)
    for i in top_idx:
        if abs(corr[i]) >= corr_thr:
            suspicious.add(feature_cols[i])
    if "close" in df.columns:
        close = df["close"].to_numpy(np.float32)
        future = pd.Series(close).shift(-args.horizon).to_numpy()
        past = pd.Series(close).shift(args.horizon).to_numpy()
        # align to X_full
        valid = ~np.isnan(future)
        if valid.sum() > 0:
            for i, col in enumerate(feature_cols):
                xi = X_full[:, i]
                xi = xi[valid]
                f = future[valid]
                p = past[valid]
                # corr with future close
                denom_f = xi.std() * np.nanstd(f)
                corr_f = 0.0 if denom_f == 0 else np.corrcoef(xi, f)[0, 1]
                # corr with past close (sanity)
                denom_p = xi.std() * np.nanstd(p)
                corr_p = 0.0 if denom_p == 0 else np.corrcoef(xi, p)[0, 1]
                if abs(corr_f) >= future_corr_thr and abs(corr_f) > abs(corr_p) + 0.05:
                    suspicious.add(col)
            print(f"[future-corr] suspicious features (|corr|>{future_corr_thr}): {sorted(suspicious)[:10]}")
    if suspicious:
        print(f"[suspect] total suspicious features: {len(suspicious)}")

    # 3) Sanity train/val with shuffled labels
    n = len(X)
    split = int(n * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    y_shuf = y_train.copy()
    rng.shuffle(y_shuf)

    model = make_simple_model(X.shape[1])
    model.fit(X_train, y_shuf, epochs=3, batch_size=512, verbose=0)
    auc_shuf = model.evaluate(X_val, y_val, verbose=0)[1]

    # 4) Same model with true labels (baseline)
    model = make_simple_model(X.shape[1])
    model.fit(X_train, y_train, epochs=3, batch_size=512, verbose=0)
    auc_true = model.evaluate(X_val, y_val, verbose=0)[1]

    print(f"[sanity] AUC shuffled-labels: {auc_shuf:.4f}")
    print(f"[sanity] AUC true-labels:     {auc_true:.4f}")

    if auc_shuf > 0.60:
        print("[warn] AUC on shuffled labels is too high -> likely leakage.")

    # 5) Re-run with suspicious features removed (if any)
    if suspicious:
        keep = [c for c in feature_cols if c not in suspicious]
        X2 = df[keep].to_numpy(np.float32)
        if len(X2) > args.sample:
            idx = rng.choice(len(X2), size=args.sample, replace=False)
            X2 = X2[idx]
            y2 = y_full.iloc[idx].to_numpy()
        else:
            y2 = y_full.to_numpy()
        n2 = len(X2)
        split2 = int(n2 * 0.8)
        model = make_simple_model(X2.shape[1])
        model.fit(X2[:split2], y2[:split2], epochs=3, batch_size=512, verbose=0)
        auc_true2 = model.evaluate(X2[split2:], y2[split2:], verbose=0)[1]
        print(f"[sanity] AUC true-labels w/out suspicious: {auc_true2:.4f} (features={len(keep)})")

    # 6) Past-only proxy: shift features by 1 bar and re-evaluate
    X_lag = df[feature_cols].shift(1).to_numpy(np.float32)
    valid = ~np.isnan(X_lag).any(axis=1)
    X_lag = X_lag[valid]
    y_lag = y_full.to_numpy()[valid]
    if len(X_lag) > args.sample:
        idx = rng.choice(len(X_lag), size=args.sample, replace=False)
        X_lag = X_lag[idx]
        y_lag = y_lag[idx]
    n_l = len(X_lag)
    if n_l > 1000:
        split_l = int(n_l * 0.8)
        model = make_simple_model(X_lag.shape[1])
        model.fit(X_lag[:split_l], y_lag[:split_l], epochs=3, batch_size=512, verbose=0)
        auc_lag = model.evaluate(X_lag[split_l:], y_lag[split_l:], verbose=0)[1]
        print(f"[sanity] AUC with lagged features (past-only proxy): {auc_lag:.4f}")


if __name__ == "__main__":
    main()
