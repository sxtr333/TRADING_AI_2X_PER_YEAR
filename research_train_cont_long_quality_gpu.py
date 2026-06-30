#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from research_train_trap_gpu import add_trap_structure, compute_label


DATA_PATH = "/home/vitamind/my_project/model6/data/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes_hybrid_news.parquet"
OUT_DIR = Path("/home/vitamind/my_project/model6/research_runs/2026-03-31-cont-long-quality-gpu")

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
    "risk_on_flag",
    "liq_long",
    "liq_short",
    "dow",
    "hour",
    "body_frac",
    "close_pos",
    "atr_pct",
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
]


@dataclass(frozen=True)
class QualityConfig:
    break_thr: float
    close_thr: float
    body_thr: float
    horizon: int
    stop_atr: float
    take_atr: float
    top_q: float

    @property
    def name(self) -> str:
        return (
            f"b{int(self.break_thr*1000):03d}_"
            f"c{int(self.close_thr*1000):03d}_"
            f"bf{int(self.body_thr*1000):03d}_"
            f"h{self.horizon}_"
            f"s{int(self.stop_atr*100):03d}_"
            f"t{int(self.take_atr*100):03d}_"
            f"q{int(self.top_q*100):02d}"
        )


FOLDS = [
    {
        "name": "2024-06-30->2025-06-30",
        "train_end": "2024-06-30T23:59:59Z",
        "val_end": "2025-02-28T23:59:59Z",
        "test_end": "2025-06-30T23:59:59Z",
    },
    {
        "name": "2024-10-31->2025-10-31",
        "train_end": "2024-10-31T23:59:59Z",
        "val_end": "2025-06-30T23:59:59Z",
        "test_end": "2025-10-31T23:59:59Z",
    },
    {
        "name": "2025-03-31->2026-03-01",
        "train_end": "2025-03-31T23:59:59Z",
        "val_end": "2025-09-30T23:59:59Z",
        "test_end": "2026-03-01T23:59:59Z",
    },
]


def make_event(df: pd.DataFrame, cfg: QualityConfig) -> pd.Series:
    return (
        (df["break_hi_96_atr"] >= cfg.break_thr)
        & (df["close_vs_prev_hi_96_atr"] >= cfg.close_thr)
        & (df["body_frac"] >= cfg.body_thr)
        & (df["volume_delta"] > 0)
        & (df["macd_hist"] > 0)
        & (df["adx14"] >= 14)
    ).fillna(False)


def split_fold(df: pd.DataFrame, fold: dict):
    train_end = pd.Timestamp(fold["train_end"])
    val_end = pd.Timestamp(fold["val_end"])
    test_end = pd.Timestamp(fold["test_end"])
    train = df[df["timestamp"] <= train_end].copy()
    val = df[(df["timestamp"] > train_end) & (df["timestamp"] <= val_end)].copy()
    test = df[(df["timestamp"] > val_end) & (df["timestamp"] <= test_end)].copy()
    return train, val, test


def train_classifier(X_train, y_train, X_val, y_val):
    model = XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        device="cuda",
        max_depth=5,
        n_estimators=1100,
        learning_rate=0.022,
        subsample=0.85,
        colsample_bytree=0.82,
        reg_lambda=12.0,
        reg_alpha=1.0,
        min_child_weight=6,
        scale_pos_weight=max(1.0, float((len(y_train) - y_train.sum()) / max(y_train.sum(), 1))),
        random_state=42,
        eval_metric="logloss",
        early_stopping_rounds=120,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    booster_cfg = model.get_booster().save_config()
    if '"device":"cuda"' not in booster_cfg and '"device":"cuda:0"' not in booster_cfg:
        raise RuntimeError("XGBoost did not stay on CUDA.")
    return model


def apply_filter(frame: pd.DataFrame, allowed_hours: set[int] | None, allowed_dows: set[int] | None):
    out = frame
    if allowed_hours:
        out = out[out["hour"].isin(sorted(allowed_hours))]
    if allowed_dows:
        out = out[out["dow"].isin(sorted(allowed_dows))]
    return out.copy()


def backtest_daily(frame: pd.DataFrame, score_col: str, edge_col: str, thr: float, cost_bps: int = 20):
    sel = frame[frame[score_col] >= thr].copy()
    if sel.empty:
        return {}
    sel["day"] = sel["timestamp"].dt.floor("D")
    sel = sel.sort_values([score_col, "timestamp"], ascending=[False, True]).groupby("day", as_index=False).first()
    sel["pnl"] = 1.5 * sel[edge_col] - (cost_bps / 10000.0) * 1.5
    eq = (1.0 + sel["pnl"]).cumprod()
    peak = eq.cummax()
    dd = float((eq / peak - 1.0).min())
    return {
        "n_days": int(len(sel)),
        "win_rate": float((sel["pnl"] > 0).mean()),
        "avg_pnl_pct_equity": float(100.0 * sel["pnl"].mean()),
        "compounded_return_pct": float(100.0 * (eq.iloc[-1] - 1.0)),
        "max_drawdown_pct": float(100.0 * dd),
    }


def optimize_threshold(scores: np.ndarray, edge: np.ndarray, cost_bps: int = 20):
    best = None
    for q in [0.70, 0.75, 0.80, 0.85, 0.90, 0.93, 0.95]:
        thr = float(np.quantile(scores, q))
        mask = scores >= thr
        if mask.sum() < 8:
            continue
        avg_edge = float(np.mean(edge[mask]))
        net = 1.5 * avg_edge - (cost_bps / 10000.0) * 1.5
        score = (net, avg_edge, -abs(mask.sum() - 18), q)
        if best is None or score > best[0]:
            best = (score, q, thr, int(mask.sum()), avg_edge, net)
    return best


def top_positive_buckets(frame: pd.DataFrame, score_col: str, edge_col: str, thr: float, bucket: str, top_k: int):
    sel = frame[frame[score_col] >= thr].copy()
    if sel.empty:
        return set()
    agg = sel.groupby(bucket).agg(n=(edge_col, "size"), avg_edge=(edge_col, "mean")).reset_index()
    agg = agg[(agg["n"] >= 2) & (agg["avg_edge"] > 0)].sort_values(["avg_edge", "n"], ascending=False)
    return set(agg.head(top_k)[bucket].tolist())


def select_session_filter(val_frame: pd.DataFrame, thr: float):
    candidates = [{"hours": None, "dows": None, "name": "none"}]
    for hk, dk in itertools.product([2, 3, 4], [2, 3]):
        hours = top_positive_buckets(val_frame, "score", "edge_ret", thr, "hour", hk)
        dows = top_positive_buckets(val_frame, "score", "edge_ret", thr, "dow", dk)
        if hours or dows:
            candidates.append({"hours": hours or None, "dows": dows or None, "name": f"h{hk}_d{dk}"})
    best = None
    for cand in candidates:
        vf = apply_filter(val_frame, cand["hours"], cand["dows"])
        bt = backtest_daily(vf, "score", "edge_ret", thr, cost_bps=20)
        if not bt or bt["n_days"] < 3:
            continue
        score = (bt["compounded_return_pct"], bt["avg_pnl_pct_equity"], -abs(bt["n_days"] - 8))
        if best is None or score > best[0]:
            best = (score, cand, bt)
    return best


def run_cfg(df: pd.DataFrame, cfg: QualityConfig):
    work = df.copy()
    work["event"] = make_event(work, cfg).astype("int8")
    _, edge = compute_label(work, "long", cfg.horizon, cfg.stop_atr, cfg.take_atr)
    work["edge_ret"] = edge

    fold_results = []
    for fold in FOLDS:
        tr, va, te = split_fold(work, fold)
        tr = tr[(tr["event"] == 1) & tr["edge_ret"].notna()].copy()
        va = va[(va["event"] == 1) & va["edge_ret"].notna()].copy()
        te = te[(te["event"] == 1) & te["edge_ret"].notna()].copy()
        if min(len(tr), len(va), len(te)) < 60:
            fold_results.append({
                "fold": fold["name"],
                "skipped": True,
                "n_train": int(len(tr)),
                "n_val": int(len(va)),
                "n_test": int(len(te)),
            })
            continue

        target_thr = float(tr["edge_ret"].quantile(cfg.top_q))
        tr["label"] = (tr["edge_ret"] >= target_thr).astype("int8")
        va["label"] = (va["edge_ret"] >= target_thr).astype("int8")
        te["label"] = (te["edge_ret"] >= target_thr).astype("int8")
        if tr["label"].sum() < 20 or va["label"].sum() < 6:
            fold_results.append({
                "fold": fold["name"],
                "skipped": True,
                "n_train": int(len(tr)),
                "n_val": int(len(va)),
                "n_test": int(len(te)),
                "target_thr": target_thr,
            })
            continue

        med = tr[FEATURES].median(numeric_only=True)
        X_train = tr[FEATURES].fillna(med).to_numpy(np.float32)
        X_val = va[FEATURES].fillna(med).to_numpy(np.float32)
        X_test = te[FEATURES].fillna(med).to_numpy(np.float32)
        y_train = tr["label"].to_numpy(np.int32)
        y_val = va["label"].to_numpy(np.int32)
        y_test = te["label"].to_numpy(np.int32)

        model = train_classifier(X_train, y_train, X_val, y_val)
        val_scores = model.predict_proba(X_val)[:, 1]
        test_scores = model.predict_proba(X_test)[:, 1]

        best_thr = optimize_threshold(val_scores, va["edge_ret"].to_numpy(), cost_bps=20)
        if best_thr is None:
            fold_results.append({
                "fold": fold["name"],
                "skipped": True,
                "n_train": int(len(tr)),
                "n_val": int(len(va)),
                "n_test": int(len(te)),
                "target_thr": target_thr,
            })
            continue

        _, q, thr, _, avg_edge, net = best_thr
        va["score"] = val_scores
        te["score"] = test_scores

        sf = select_session_filter(va, thr)
        if sf is None:
            filt = {"hours": None, "dows": None, "name": "none"}
            val_bt = backtest_daily(va, "score", "edge_ret", thr, cost_bps=20)
        else:
            _, filt, val_bt = sf
        te_f = apply_filter(te, filt["hours"], filt["dows"])
        test_bt = backtest_daily(te_f, "score", "edge_ret", thr, cost_bps=20)

        auc = float(roc_auc_score(y_val, val_scores)) if len(np.unique(y_val)) > 1 else float("nan")
        fold_results.append({
            "fold": fold["name"],
            "skipped": False,
            "n_train": int(len(tr)),
            "n_val": int(len(va)),
            "n_test": int(len(te)),
            "target_thr": target_thr,
            "target_pos_rate_train": float(tr["label"].mean()),
            "target_pos_rate_val": float(va["label"].mean()),
            "auc_val": auc,
            "score_q": q,
            "score_thr": thr,
            "val_avg_edge_pct": 100.0 * avg_edge,
            "val_net_trade_pct_eq_20bps": 100.0 * net,
            "filter_name": filt["name"],
            "allowed_hours": sorted(filt["hours"]) if filt["hours"] else [],
            "allowed_dows": sorted(filt["dows"]) if filt["dows"] else [],
            "val_backtest": val_bt,
            "test_backtest": test_bt,
            "top_features": (
                pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_})
                .sort_values("importance", ascending=False)
                .head(12)
                .to_dict(orient="records")
            ),
        })

    valid = [r for r in fold_results if not r.get("skipped")]
    if not valid:
        return None

    test_comp = [r["test_backtest"].get("compounded_return_pct", np.nan) for r in valid]
    test_avg = [r["test_backtest"].get("avg_pnl_pct_equity", np.nan) for r in valid]
    test_dd = [r["test_backtest"].get("max_drawdown_pct", np.nan) for r in valid]
    val_net = [r["val_net_trade_pct_eq_20bps"] for r in valid]
    latest = valid[-1]
    robust_score = float(np.nanmedian(test_comp)) + 0.45 * float(np.nanmean(test_avg)) + 0.20 * float(np.nanmean(val_net))
    recent_score = float(latest["test_backtest"].get("compounded_return_pct", -999)) + 0.6 * float(latest["test_backtest"].get("avg_pnl_pct_equity", 0))
    score = robust_score + 0.85 * recent_score - 0.18 * abs(float(np.nanmin(test_dd)))
    return {
        "config": asdict(cfg),
        "folds": fold_results,
        "robust_score": score,
        "median_test_compounded_pct": float(np.nanmedian(test_comp)),
        "mean_test_avg_trade_pct": float(np.nanmean(test_avg)),
        "worst_test_dd_pct": float(np.nanmin(test_dd)),
        "latest_test_compounded_pct": float(latest["test_backtest"].get("compounded_return_pct", np.nan)),
        "latest_test_avg_trade_pct": float(latest["test_backtest"].get("avg_pnl_pct_equity", np.nan)),
        "latest_filter": latest.get("filter_name"),
        "latest_hours": latest.get("allowed_hours"),
        "latest_dows": latest.get("allowed_dows"),
        "latest_target_thr": latest.get("target_thr"),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = add_trap_structure(df)

    grid = [
        QualityConfig(*vals)
        for vals in itertools.product(
            [0.05, 0.07, 0.09],
            [0.03, 0.05],
            [0.08, 0.12],
            [16, 20],
            [0.9, 1.0],
            [1.6, 1.8],
            [0.75, 0.80, 0.85, 0.90],
        )
    ]

    rows = []
    for i, cfg in enumerate(grid, start=1):
        print(f"[{i}/{len(grid)}] {cfg.name}", flush=True)
        res = run_cfg(df, cfg)
        if not res:
            continue
        (OUT_DIR / f"{cfg.name}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
        rows.append({
            "cfg": cfg.name,
            "robust_score": round(res["robust_score"], 4),
            "median_test_compounded_pct": round(res["median_test_compounded_pct"], 4),
            "mean_test_avg_trade_pct": round(res["mean_test_avg_trade_pct"], 4),
            "worst_test_dd_pct": round(res["worst_test_dd_pct"], 4),
            "latest_test_compounded_pct": round(res["latest_test_compounded_pct"], 4),
            "latest_test_avg_trade_pct": round(res["latest_test_avg_trade_pct"], 4),
            "latest_filter": res["latest_filter"],
            "latest_target_thr": round(float(res["latest_target_thr"]), 6) if pd.notna(res["latest_target_thr"]) else np.nan,
            "latest_hours": ",".join(map(str, res["latest_hours"] or [])),
            "latest_dows": ",".join(map(str, res["latest_dows"] or [])),
        })

    if rows:
        summary = pd.DataFrame(rows).sort_values(
            ["robust_score", "latest_test_compounded_pct", "median_test_compounded_pct"],
            ascending=False,
        )
        summary.to_csv(OUT_DIR / "summary.csv", index=False)
        print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
