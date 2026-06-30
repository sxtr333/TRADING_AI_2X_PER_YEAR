#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score


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


FAMILIES = {
    "fade_short": {
        "event_col": "evt_false_break_high",
        "side": "short",
        "horizon": 16,
        "stop_atr": 0.9,
        "take_atr": 1.4,
    },
    "fade_long": {
        "event_col": "evt_false_break_low",
        "side": "long",
        "horizon": 16,
        "stop_atr": 0.9,
        "take_atr": 1.4,
    },
    "cont_long": {
        "event_col": "evt_accept_break_high",
        "side": "long",
        "horizon": 20,
        "stop_atr": 1.0,
        "take_atr": 1.8,
    },
    "cont_short": {
        "event_col": "evt_accept_break_low",
        "side": "short",
        "horizon": 20,
        "stop_atr": 1.0,
        "take_atr": 1.8,
    },
}


def add_trap_structure(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)

    for w in [24, 48, 96]:
        prev_hi = out["high"].rolling(w, min_periods=w).max().shift(1)
        prev_lo = out["low"].rolling(w, min_periods=w).min().shift(1)
        out[f"prev_hi_{w}"] = prev_hi
        out[f"prev_lo_{w}"] = prev_lo
        out[f"break_hi_{w}_atr"] = (out["high"] - prev_hi) / out["atr"].clip(lower=1e-8)
        out[f"break_lo_{w}_atr"] = (prev_lo - out["low"]) / out["atr"].clip(lower=1e-8)
        out[f"close_vs_prev_hi_{w}_atr"] = (out["close"] - prev_hi) / out["atr"].clip(lower=1e-8)
        out[f"close_vs_prev_lo_{w}_atr"] = (out["close"] - prev_lo) / out["atr"].clip(lower=1e-8)

    rng = (out["high"] - out["low"]).clip(lower=1e-8)
    out["body_frac"] = (out["close"] - out["open"]) / rng
    out["close_pos"] = (out["close"] - out["low"]) / rng
    out["atr_pct"] = out["atr"] / out["close"].clip(lower=1e-8)
    out["premium_x_news"] = out["coinbase_premium_pct"] * out["news_sentiment"]
    out["macro_stress"] = out["vix"] * out["hy_oas"]
    out["qqq_vix_flip"] = out["qqq_ret_1d"] / out["vix"].replace(0, np.nan)
    out["hour_sin"] = np.sin((out["hour"] * 2 * np.pi) / 1.0)
    out["hour_cos"] = np.cos((out["hour"] * 2 * np.pi) / 1.0)

    out["evt_false_break_high"] = (
        (out["break_hi_96_atr"] >= 0.08)
        & (out["close_vs_prev_hi_96_atr"] <= -0.04)
        & (out["wick_up"] >= 0.38)
        & (out["close_delta"] < 0)
    ).fillna(False).astype("int8")

    out["evt_false_break_low"] = (
        (out["break_lo_96_atr"] >= 0.08)
        & (out["close_vs_prev_lo_96_atr"] >= 0.04)
        & (out["wick_down"] >= 0.38)
        & (out["close_delta"] > 0)
    ).fillna(False).astype("int8")

    out["evt_accept_break_high"] = (
        (out["break_hi_96_atr"] >= 0.07)
        & (out["close_vs_prev_hi_96_atr"] >= 0.05)
        & (out["body_frac"] >= 0.12)
        & (out["volume_delta"] > 0)
        & (out["macd_hist"] > 0)
    ).fillna(False).astype("int8")

    out["evt_accept_break_low"] = (
        (out["break_lo_96_atr"] >= 0.07)
        & (out["close_vs_prev_lo_96_atr"] <= -0.05)
        & (out["body_frac"] <= -0.12)
        & (out["volume_delta"] < 0)
        & (out["macd_hist"] < 0)
    ).fillna(False).astype("int8")

    return out.replace([np.inf, -np.inf], np.nan)


def compute_label(df: pd.DataFrame, side: str, horizon: int, stop_atr: float, take_atr: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(df)
    y = np.full(n, np.nan, dtype=np.float32)
    edge_ret = np.full(n, np.nan, dtype=np.float32)

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

        outcome = None
        for j in range(i + 1, i + horizon + 1):
            if is_long:
                hit_stop = low[j] <= stop
                hit_take = high[j] >= take
            else:
                hit_stop = high[j] >= stop
                hit_take = low[j] <= take
            if hit_stop or hit_take:
                outcome = 1.0 if hit_take and not hit_stop else 0.0
                break

        if outcome is None:
            fut_close = close[i + horizon]
            outcome = 1.0 if ((fut_close > entry) if is_long else (fut_close < entry)) else 0.0

        y[i] = outcome
        edge_ret[i] = ((close[i + horizon] - entry) / entry) if is_long else ((entry - close[i + horizon]) / entry)
    return y, edge_ret


def temporal_split(df: pd.DataFrame, seq_guard: int = 256, horizon: int = 24):
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


def train_xgb(X_train, y_train, X_val, y_val):
    from xgboost import XGBClassifier

    model = XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        device="cuda",
        max_depth=6,
        n_estimators=1400,
        learning_rate=0.025,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=10.0,
        min_child_weight=5,
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


def best_threshold(y_true: np.ndarray, p: np.ndarray) -> float:
    best = None
    for thr in np.arange(0.35, 0.81, 0.02):
        pred = (p >= thr).astype(np.int32)
        rate = pred.mean()
        if rate < 0.02 or rate > 0.60:
            continue
        f1 = f1_score(y_true, pred, zero_division=0)
        score = (f1, -abs(rate - y_true.mean()), thr)
        if best is None or score > best:
            best = score
    return float(best[2]) if best else 0.5


def evaluate(y_true: np.ndarray, p: np.ndarray, thr: float) -> dict:
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
    out_dir = base / "research_runs" / "2026-03-29-trap-gpu"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(base / "data" / "meta" / "meta_dataset_pruned.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = add_trap_structure(df)

    feat_cols = [c for c in BASE_FEATURES if c in df.columns] + [
        "body_frac",
        "close_pos",
        "atr_pct",
        "premium_x_news",
        "macro_stress",
        "qqq_vix_flip",
        "hour_sin",
        "hour_cos",
        "prev_hi_24",
        "prev_lo_24",
        "prev_hi_48",
        "prev_lo_48",
        "prev_hi_96",
        "prev_lo_96",
        "break_hi_24_atr",
        "break_lo_24_atr",
        "break_hi_48_atr",
        "break_lo_48_atr",
        "break_hi_96_atr",
        "break_lo_96_atr",
        "close_vs_prev_hi_24_atr",
        "close_vs_prev_lo_24_atr",
        "close_vs_prev_hi_96_atr",
        "close_vs_prev_lo_96_atr",
        "evt_false_break_high",
        "evt_false_break_low",
        "evt_accept_break_high",
        "evt_accept_break_low",
    ]

    summary_rows = []

    for name, cfg in FAMILIES.items():
        y, edge_ret = compute_label(df, cfg["side"], cfg["horizon"], cfg["stop_atr"], cfg["take_atr"])
        work = df.copy()
        work["label"] = y
        work["edge_ret"] = edge_ret

        train_idx, val_idx, test_idx = temporal_split(work, horizon=cfg["horizon"])
        tr = work.iloc[train_idx].copy()
        va = work.iloc[val_idx].copy()
        te = work.iloc[test_idx].copy()

        tr = tr[(tr[cfg["event_col"]] == 1) & tr["label"].notna()].copy()
        va = va[(va[cfg["event_col"]] == 1) & va["label"].notna()].copy()
        te = te[(te[cfg["event_col"]] == 1) & te["label"].notna()].copy()

        print(
            json.dumps(
                {
                    "family": name,
                    "n_train": int(len(tr)),
                    "n_val": int(len(va)),
                    "n_test": int(len(te)),
                },
                ensure_ascii=False,
            )
        )

        if min(len(tr), len(va), len(te)) < 120:
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
        val_metrics = evaluate(y_val, pv, thr)
        test_metrics = evaluate(y_test, pt, thr)

        thresholds = []
        for t in sorted({thr, 0.55, 0.60, 0.65, 0.70}):
            pred = pt >= t
            n = int(pred.sum())
            if n == 0:
                continue
            thresholds.append(
                {
                    "threshold": float(t),
                    "n_signals": n,
                    "hit_rate": float(np.mean(y_test[pred])),
                    "avg_edge_ret": float(te.loc[pred, "edge_ret"].mean()),
                    "pred_rate": float(np.mean(pred)),
                }
            )

        imp = pd.DataFrame({"feature": feat_cols, "importance": model.feature_importances_}).sort_values(
            "importance", ascending=False
        )

        metrics = {
            "family": name,
            "config": cfg,
            "n_train": int(len(tr)),
            "n_val": int(len(va)),
            "n_test": int(len(te)),
            "val": val_metrics,
            "test": test_metrics,
            "top_features": imp.head(12).to_dict(orient="records"),
        }
        (out_dir / f"{name}_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
        imp.to_csv(out_dir / f"{name}_importance.csv", index=False)
        pd.DataFrame(thresholds).to_csv(out_dir / f"{name}_thresholds.csv", index=False)
        model.save_model(str(out_dir / f"{name}.json"))

        summary_rows.append(
            {
                "family": name,
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

    if not summary_rows:
        print("No trap families passed minimum sample thresholds.")
        return

    summary = pd.DataFrame(summary_rows).sort_values(["test_auc", "test_f1"], ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
