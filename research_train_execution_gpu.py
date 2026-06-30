#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score


FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "atr",
    "rv_ratio",
    "rsi14",
    "macd_hist",
    "adx14",
    "stoch_rsi14",
    "wick_up",
    "wick_down",
    "range_norm",
    "close_delta",
    "volume_delta",
    "buy_sell_ratio",
    "oi_delta",
    "bollinger_bandwidth",
    "coinbase_premium_pct",
    "qqq_ret_1d",
    "vix",
    "hy_oas",
    "etf_flow_fbtc_usdm",
    "etf_flow_bitb_usdm",
    "risk_on_flag",
    "news_sentiment",
    "news_shock",
    "liq_long",
    "liq_short",
    "dow",
    "hour",
]


def build_events(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Event families: sweep/reclaim, impulsive continuation, compression pop.
    out["evt_long_sweep"] = (
        (out["wick_down"] >= 0.58)
        & (out["close_delta"] > 0)
        & (out["rsi14"] <= 46)
    ).astype("int8")

    out["evt_short_sweep"] = (
        (out["wick_up"] >= 0.58)
        & (out["close_delta"] < 0)
        & (out["rsi14"] >= 54)
    ).astype("int8")

    out["evt_long_impulse"] = (
        (out["close_delta"] > 0)
        & (out["volume_delta"] > 0)
        & (out["macd_hist"] > 0)
        & (out["adx14"] >= 18)
        & (out["buy_sell_ratio"] >= 1.02)
    ).astype("int8")

    out["evt_short_impulse"] = (
        (out["close_delta"] < 0)
        & (out["volume_delta"] < 0)
        & (out["macd_hist"] < 0)
        & (out["adx14"] >= 18)
        & (out["buy_sell_ratio"] <= 0.98)
    ).astype("int8")

    out["evt_long_compression"] = (
        (out["bollinger_bandwidth"] <= out["bollinger_bandwidth"].rolling(96, min_periods=24).quantile(0.35))
        & (out["close_delta"] > 0)
        & (out["volume_delta"] > 0)
        & (out["coinbase_premium_pct"] > -0.001)
    ).fillna(False).astype("int8")

    out["evt_short_compression"] = (
        (out["bollinger_bandwidth"] <= out["bollinger_bandwidth"].rolling(96, min_periods=24).quantile(0.35))
        & (out["close_delta"] < 0)
        & (out["volume_delta"] < 0)
        & (out["coinbase_premium_pct"] < 0.001)
    ).fillna(False).astype("int8")

    out["long_event"] = (
        (out["evt_long_sweep"] == 1)
        | (out["evt_long_impulse"] == 1)
        | (out["evt_long_compression"] == 1)
    ).astype("int8")

    out["short_event"] = (
        (out["evt_short_sweep"] == 1)
        | (out["evt_short_impulse"] == 1)
        | (out["evt_short_compression"] == 1)
    ).astype("int8")
    return out


def compute_labels(df: pd.DataFrame, horizon: int = 20, stop_atr: float = 1.0, take_atr: float = 1.5):
    n = len(df)
    long_y = np.full(n, np.nan, dtype=np.float32)
    short_y = np.full(n, np.nan, dtype=np.float32)

    high = df["high"].to_numpy(np.float32)
    low = df["low"].to_numpy(np.float32)
    close = df["close"].to_numpy(np.float32)
    atr = np.maximum(df["atr"].to_numpy(np.float32), 1e-8)

    long_ret = np.full(n, np.nan, dtype=np.float32)
    short_ret = np.full(n, np.nan, dtype=np.float32)

    for i in range(n - horizon):
        entry = close[i]
        a = atr[i]
        long_stop = entry - stop_atr * a
        long_take = entry + take_atr * a
        short_stop = entry + stop_atr * a
        short_take = entry - take_atr * a

        long_done = False
        short_done = False

        for j in range(i + 1, i + horizon + 1):
            if not long_done:
                hit_stop = low[j] <= long_stop
                hit_take = high[j] >= long_take
                if hit_stop or hit_take:
                    long_y[i] = 1.0 if hit_take and not hit_stop else 0.0
                    long_done = True
            if not short_done:
                hit_stop = high[j] >= short_stop
                hit_take = low[j] <= short_take
                if hit_stop or hit_take:
                    short_y[i] = 1.0 if hit_take and not hit_stop else 0.0
                    short_done = True
            if long_done and short_done:
                break

        if np.isnan(long_y[i]):
            fut_close = close[i + horizon]
            long_y[i] = 1.0 if fut_close > entry else 0.0
        if np.isnan(short_y[i]):
            fut_close = close[i + horizon]
            short_y[i] = 1.0 if fut_close < entry else 0.0

        fut_slice = close[i + 1 : i + horizon + 1]
        long_ret[i] = (np.nanmax(fut_slice) - entry) / entry
        short_ret[i] = (entry - np.nanmin(fut_slice)) / entry

    return long_y, short_y, long_ret, short_ret


def temporal_split(df: pd.DataFrame, seq_guard: int = 256, horizon: int = 20):
    train_end = pd.Timestamp("2025-03-31T23:59:59Z")
    val_end = pd.Timestamp("2025-09-30T23:59:59Z")
    n = len(df)
    n_train = int(df.index[df["timestamp"] <= train_end].max()) + 1
    val_end_idx = int(df.index[df["timestamp"] <= val_end].max()) + 1
    min_end = seq_guard - 1
    train_max = max(min_end, n_train - horizon)
    val_max = max(min_end, val_end_idx - horizon)
    test_max = max(min_end, n - horizon)
    train_idx = np.arange(min_end, train_max, dtype=np.int64)
    val_idx = np.arange(max(n_train, min_end), val_max, dtype=np.int64)
    test_idx = np.arange(max(val_end_idx, min_end), test_max, dtype=np.int64)
    return train_idx, val_idx, test_idx


def train_xgb_binary(X_train, y_train, X_val, y_val):
    from xgboost import XGBClassifier

    model = XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        device="cuda",
        max_depth=7,
        n_estimators=1800,
        learning_rate=0.03,
        subsample=0.82,
        colsample_bytree=0.82,
        reg_lambda=8.0,
        min_child_weight=4,
        random_state=42,
        eval_metric="logloss",
        early_stopping_rounds=120,
        verbosity=1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
    cfg = model.get_booster().save_config()
    if '"device":"cuda"' not in cfg and '"device":"cuda:0"' not in cfg:
        raise RuntimeError("XGBoost did not stay on CUDA device.")
    return model


def evaluate_binary(y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(np.int32)
    return {
        "acc": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
        "auc": float(roc_auc_score(y_true, proba)),
        "logloss": float(log_loss(y_true, np.vstack([1 - proba, proba]).T, labels=[0, 1])),
        "pos_rate": float(np.mean(y_true)),
        "pred_rate": float(np.mean(pred)),
    }


def run_side(df, side: str, out_dir: Path):
    if side not in {"long", "short"}:
        raise ValueError(side)

    event_col = f"{side}_event"
    label_col = f"{side}_label"
    ret_col = f"{side}_ret"

    train_idx, val_idx, test_idx = temporal_split(df)
    use_cols = FEATURES + [
        "evt_long_sweep",
        "evt_short_sweep",
        "evt_long_impulse",
        "evt_short_impulse",
        "evt_long_compression",
        "evt_short_compression",
        "long_event",
        "short_event",
    ]

    train = df.iloc[train_idx].copy()
    val = df.iloc[val_idx].copy()
    test = df.iloc[test_idx].copy()

    train = train[train[event_col] == 1].copy()
    val = val[val[event_col] == 1].copy()
    test = test[test[event_col] == 1].copy()

    X_train = train[use_cols].to_numpy(np.float32)
    y_train = train[label_col].astype(np.int32).to_numpy()
    X_val = val[use_cols].to_numpy(np.float32)
    y_val = val[label_col].astype(np.int32).to_numpy()
    X_test = test[use_cols].to_numpy(np.float32)
    y_test = test[label_col].astype(np.int32).to_numpy()

    model = train_xgb_binary(X_train, y_train, X_val, y_val)
    val_proba = model.predict_proba(X_val)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "side": side,
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "val": evaluate_binary(y_val, val_proba),
        "test": evaluate_binary(y_test, test_proba),
    }

    imp = pd.DataFrame(
        {"feature": use_cols, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    # Simple trade-style expectancy at confidence thresholds.
    rows = []
    for thr in [0.55, 0.60, 0.65, 0.70]:
        m = test_proba >= thr
        n = int(m.sum())
        if n == 0:
            continue
        hit = float(np.mean(y_test[m]))
        avg_move = float(test.loc[m, ret_col].mean())
        rows.append(
            {
                "threshold": thr,
                "n_signals": n,
                "hit_rate": hit,
                "avg_favorable_move": avg_move,
            }
        )
    thr_df = pd.DataFrame(rows)

    model_path = out_dir / f"xgboost_execution_{side}.json"
    metrics_path = out_dir / f"xgboost_execution_{side}_metrics.json"
    imp_path = out_dir / f"xgboost_execution_{side}_importance.csv"
    thr_path = out_dir / f"xgboost_execution_{side}_thresholds.csv"

    model.save_model(str(model_path))
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    imp.to_csv(imp_path, index=False)
    thr_df.to_csv(thr_path, index=False)

    return metrics, imp.head(15), thr_df


def main():
    base = Path("/home/vitamind/my_project/model6")
    out_dir = base / "research_runs" / "2026-03-29-execution-gpu"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(base / "data" / "meta" / "meta_dataset_pruned.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = build_events(df)
    long_y, short_y, long_ret, short_ret = compute_labels(df, horizon=20, stop_atr=1.0, take_atr=1.5)
    df["long_label"] = long_y
    df["short_label"] = short_y
    df["long_ret"] = long_ret
    df["short_ret"] = short_ret

    print(
        json.dumps(
            {
                "rows": len(df),
                "long_events": int(df["long_event"].sum()),
                "short_events": int(df["short_event"].sum()),
            },
            ensure_ascii=False,
        )
    )

    for side in ["long", "short"]:
        metrics, imp, thr = run_side(df, side, out_dir)
        print(f"\n## {side.upper()} METRICS")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        print(f"\n## {side.upper()} TOP FEATURES")
        print(imp.to_string(index=False))
        print(f"\n## {side.upper()} THRESHOLDS")
        print(thr.to_string(index=False) if not thr.empty else "no thresholds")


if __name__ == "__main__":
    main()
