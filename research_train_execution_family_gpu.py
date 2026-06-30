#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

from research_train_execution_gpu import build_events, temporal_split


BASE_FEATURES = [
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

FAMILY_CONFIG = {
    "long_sweep": {"event_col": "evt_long_sweep", "side": "long", "take": 1.2, "stop": 0.8, "horizon": 16},
    "short_sweep": {"event_col": "evt_short_sweep", "side": "short", "take": 1.2, "stop": 0.8, "horizon": 16},
    "short_impulse": {"event_col": "evt_short_impulse", "side": "short", "take": 1.5, "stop": 1.0, "horizon": 20},
    "long_compression": {"event_col": "evt_long_compression", "side": "long", "take": 1.6, "stop": 0.9, "horizon": 24},
    "short_compression": {"event_col": "evt_short_compression", "side": "short", "take": 1.6, "stop": 0.9, "horizon": 24},
}


def add_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rng = (out["high"] - out["low"]).clip(lower=1e-8)
    out["body_frac"] = (out["close"] - out["open"]) / rng
    out["close_pos"] = (out["close"] - out["low"]) / rng
    out["atr_pct"] = out["atr"] / out["close"].clip(lower=1e-8)

    for w in [4, 16, 48]:
        out[f"ret_{w}"] = out["close"].pct_change(w)
        out[f"vol_z_{w}"] = (
            (out["volume"] - out["volume"].rolling(w, min_periods=max(4, w // 2)).mean())
            / (out["volume"].rolling(w, min_periods=max(4, w // 2)).std().replace(0, np.nan))
        )

    for w in [24, 96]:
        roll_hi = out["high"].rolling(w, min_periods=max(12, w // 2)).max()
        roll_lo = out["low"].rolling(w, min_periods=max(12, w // 2)).min()
        out[f"dist_hi_{w}_atr"] = (roll_hi - out["close"]) / out["atr"].clip(lower=1e-8)
        out[f"dist_lo_{w}_atr"] = (out["close"] - roll_lo) / out["atr"].clip(lower=1e-8)

    bw_med = out["bollinger_bandwidth"].rolling(96, min_periods=24).median()
    out["compression_ratio"] = out["bollinger_bandwidth"] / bw_med.replace(0, np.nan)
    out["premium_x_news"] = out["coinbase_premium_pct"] * out["news_sentiment"]
    out["macro_stress"] = out["vix"] * out["hy_oas"]
    out["qqq_vix_flip"] = out["qqq_ret_1d"] / out["vix"].replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def compute_family_label(df: pd.DataFrame, side: str, horizon: int, stop_atr: float, take_atr: float) -> np.ndarray:
    n = len(df)
    y = np.full(n, np.nan, dtype=np.float32)
    high = df["high"].to_numpy(np.float32)
    low = df["low"].to_numpy(np.float32)
    close = df["close"].to_numpy(np.float32)
    atr = np.maximum(df["atr"].to_numpy(np.float32), 1e-8)

    is_long = side == "long"
    for i in range(n - horizon):
        entry = close[i]
        a = atr[i]
        stop = entry - stop_atr * a if is_long else entry + stop_atr * a
        take = entry + take_atr * a if is_long else entry - take_atr * a
        done = False
        for j in range(i + 1, i + horizon + 1):
            if is_long:
                hit_stop = low[j] <= stop
                hit_take = high[j] >= take
            else:
                hit_stop = high[j] >= stop
                hit_take = low[j] <= take
            if hit_stop or hit_take:
                y[i] = 1.0 if hit_take and not hit_stop else 0.0
                done = True
                break
        if not done:
            fut_close = close[i + horizon]
            y[i] = 1.0 if ((fut_close > entry) if is_long else (fut_close < entry)) else 0.0
    return y


def train_xgb(X_train, y_train, X_val, y_val):
    from xgboost import XGBClassifier

    model = XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        device="cuda",
        max_depth=6,
        n_estimators=1200,
        learning_rate=0.025,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=10.0,
        min_child_weight=5,
        random_state=42,
        eval_metric="logloss",
        early_stopping_rounds=100,
        verbosity=1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
    cfg = model.get_booster().save_config()
    if '"device":"cuda"' not in cfg and '"device":"cuda:0"' not in cfg:
        raise RuntimeError("XGBoost did not stay on CUDA device.")
    return model


def best_threshold(y_true: np.ndarray, p: np.ndarray) -> float:
    best = None
    for thr in np.arange(0.30, 0.76, 0.02):
        pred = (p >= thr).astype(np.int32)
        if pred.mean() < 0.03:
            continue
        f1 = f1_score(y_true, pred, zero_division=0)
        acc = accuracy_score(y_true, pred)
        score = (f1, acc, thr)
        if best is None or score > best:
            best = score
    return float(best[2]) if best else 0.5


def eval_at(y_true: np.ndarray, p: np.ndarray, thr: float) -> dict:
    pred = (p >= thr).astype(np.int32)
    return {
        "thr": float(thr),
        "acc": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, p)),
        "logloss": float(log_loss(y_true, np.vstack([1 - p, p]).T, labels=[0, 1])),
        "pos_rate": float(np.mean(y_true)),
        "pred_rate": float(np.mean(pred)),
    }


def main():
    base = Path("/home/vitamind/my_project/model6")
    out_dir = base / "research_runs" / "2026-03-29-execution-family-gpu"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(base / "data" / "meta" / "meta_dataset_pruned.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = build_events(df)
    df = add_structure_features(df)

    train_idx, val_idx, test_idx = temporal_split(df, horizon=24)
    feat_cols = [c for c in BASE_FEATURES if c in df.columns] + [
        "body_frac",
        "close_pos",
        "atr_pct",
        "ret_4",
        "ret_16",
        "ret_48",
        "vol_z_4",
        "vol_z_16",
        "vol_z_48",
        "dist_hi_24_atr",
        "dist_lo_24_atr",
        "dist_hi_96_atr",
        "dist_lo_96_atr",
        "compression_ratio",
        "premium_x_news",
        "macro_stress",
        "qqq_vix_flip",
    ]

    summary_rows = []

    for family, cfg in FAMILY_CONFIG.items():
        label = compute_family_label(df, cfg["side"], cfg["horizon"], cfg["stop"], cfg["take"])
        work = df.copy()
        work["label"] = label

        tr = work.iloc[train_idx].copy()
        va = work.iloc[val_idx].copy()
        te = work.iloc[test_idx].copy()
        tr = tr[(tr[cfg["event_col"]] == 1) & tr["label"].notna()].copy()
        va = va[(va[cfg["event_col"]] == 1) & va["label"].notna()].copy()
        te = te[(te[cfg["event_col"]] == 1) & te["label"].notna()].copy()

        if min(len(tr), len(va), len(te)) < 200:
            continue

        med = tr[feat_cols].median()
        X_train = tr[feat_cols].fillna(med).to_numpy(np.float32)
        y_train = tr["label"].astype(np.int32).to_numpy()
        X_val = va[feat_cols].fillna(med).to_numpy(np.float32)
        y_val = va["label"].astype(np.int32).to_numpy()
        X_test = te[feat_cols].fillna(med).to_numpy(np.float32)
        y_test = te["label"].astype(np.int32).to_numpy()

        model = train_xgb(X_train, y_train, X_val, y_val)
        pv = model.predict_proba(X_val)[:, 1]
        pt = model.predict_proba(X_test)[:, 1]
        thr = best_threshold(y_val, pv)
        val_metrics = eval_at(y_val, pv, thr)
        test_metrics = eval_at(y_test, pt, thr)

        imp = pd.DataFrame({"feature": feat_cols, "importance": model.feature_importances_}).sort_values(
            "importance", ascending=False
        )

        rows = []
        for t in [thr, max(thr, 0.55), max(thr, 0.60), max(thr, 0.65)]:
            pred = pt >= t
            n = int(pred.sum())
            if n == 0:
                continue
            rows.append(
                {
                    "threshold": float(t),
                    "n_signals": n,
                    "hit_rate": float(np.mean(y_test[pred])),
                    "test_pred_rate": float(np.mean(pred)),
                }
            )
        thr_df = pd.DataFrame(rows).drop_duplicates(subset=["threshold"])

        metrics = {
            "family": family,
            "config": cfg,
            "n_train": int(len(tr)),
            "n_val": int(len(va)),
            "n_test": int(len(te)),
            "val": val_metrics,
            "test": test_metrics,
            "top_features": imp.head(12).to_dict(orient="records"),
        }

        (out_dir / f"{family}_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
        imp.to_csv(out_dir / f"{family}_importance.csv", index=False)
        thr_df.to_csv(out_dir / f"{family}_thresholds.csv", index=False)
        model.save_model(str(out_dir / f"{family}.json"))

        summary_rows.append(
            {
                "family": family,
                "side": cfg["side"],
                "n_test": int(len(te)),
                "test_auc": round(test_metrics["auc"], 4),
                "test_f1": round(test_metrics["f1"], 4),
                "test_acc": round(test_metrics["acc"], 4),
                "thr": round(thr, 3),
                "test_pred_rate": round(test_metrics["pred_rate"], 4),
                "test_pos_rate": round(test_metrics["pos_rate"], 4),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(["test_auc", "test_f1"], ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
