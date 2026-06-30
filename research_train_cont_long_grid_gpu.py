#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from research_train_trap_gpu import add_trap_structure, compute_label, temporal_split, BASE_FEATURES


FEATURES = [*BASE_FEATURES] + [
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
]


@dataclass(frozen=True)
class ContLongConfig:
    break_thr: float
    close_thr: float
    body_thr: float
    horizon: int
    stop_atr: float
    take_atr: float

    @property
    def name(self) -> str:
        return (
            f"b{int(self.break_thr*1000):03d}_"
            f"c{int(self.close_thr*1000):03d}_"
            f"bf{int(self.body_thr*1000):03d}_"
            f"h{self.horizon}_"
            f"s{int(self.stop_atr*100):03d}_"
            f"t{int(self.take_atr*100):03d}"
        )


def make_event(df: pd.DataFrame, cfg: ContLongConfig) -> pd.Series:
    return (
        (df["break_hi_96_atr"] >= cfg.break_thr)
        & (df["close_vs_prev_hi_96_atr"] >= cfg.close_thr)
        & (df["body_frac"] >= cfg.body_thr)
        & (df["volume_delta"] > 0)
        & (df["macd_hist"] > 0)
    ).fillna(False)


def train_regressor(X_train, y_train, X_val, y_val):
    model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        device="cuda",
        max_depth=6,
        n_estimators=1000,
        learning_rate=0.025,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=10.0,
        min_child_weight=5,
        random_state=42,
        eval_metric="rmse",
        early_stopping_rounds=120,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    cfg = model.get_booster().save_config()
    if '"device":"cuda"' not in cfg and '"device":"cuda:0"' not in cfg:
        raise RuntimeError("XGBoost did not stay on CUDA device.")
    return model


def backtest_daily(frame: pd.DataFrame, score_col: str, edge_col: str, thr: float, cost_bps=20):
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
        "median_pnl_pct_equity": float(100.0 * sel["pnl"].median()),
        "compounded_return_pct": float(100.0 * (eq.iloc[-1] - 1.0)),
        "max_drawdown_pct": float(100.0 * dd),
    }


def optimize_quantile(val_scores, val_edge, cost_bps=20):
    best = None
    for q in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.93]:
        thr = float(np.quantile(val_scores, q))
        mask = val_scores >= thr
        if mask.sum() < 20:
            continue
        avg_edge = float(np.mean(val_edge[mask]))
        net = 1.5 * avg_edge - (cost_bps / 10000.0) * 1.5
        score = (net, avg_edge, -q)
        if best is None or score > best[0]:
            best = (score, q, thr, int(mask.sum()), avg_edge, net)
    return best


def top_positive_buckets(frame: pd.DataFrame, score_col: str, edge_col: str, thr: float, bucket: str, top_k: int):
    sel = frame[frame[score_col] >= thr].copy()
    if sel.empty:
        return set()
    agg = (
        sel.groupby(bucket)
        .agg(n=(edge_col, "size"), avg_edge=(edge_col, "mean"))
        .reset_index()
    )
    agg = agg[(agg["n"] >= 2) & (agg["avg_edge"] > 0)].sort_values(["avg_edge", "n"], ascending=False)
    return set(agg.head(top_k)[bucket].tolist())


def apply_filter(frame: pd.DataFrame, allowed_hours: set[int] | None, allowed_dows: set[int] | None):
    out = frame
    if allowed_hours:
        out = out[out["hour"].isin(sorted(allowed_hours))]
    if allowed_dows:
        out = out[out["dow"].isin(sorted(allowed_dows))]
    return out.copy()


def select_session_filter(val_frame: pd.DataFrame, thr: float):
    candidates = [{"hours": None, "dows": None, "name": "none"}]
    for hk, dk in itertools.product([2, 3, 4, 5], [2, 3, 4]):
        hours = top_positive_buckets(val_frame, "score", "edge_ret", thr, "hour", hk)
        dows = top_positive_buckets(val_frame, "score", "edge_ret", thr, "dow", dk)
        if hours or dows:
            candidates.append(
                {
                    "hours": hours or None,
                    "dows": dows or None,
                    "name": f"h{hk}_d{dk}",
                }
            )

    best = None
    for cand in candidates:
        vf = apply_filter(val_frame, cand["hours"], cand["dows"])
        bt = backtest_daily(vf, "score", "edge_ret", thr, cost_bps=20)
        if not bt or bt["n_days"] < 3:
            continue
        score = (
            bt["compounded_return_pct"],
            bt["avg_pnl_pct_equity"],
            -abs(bt["n_days"] - 8),
        )
        if best is None or score > best[0]:
            best = (score, cand, bt)
    return best


def run_one(df: pd.DataFrame, cfg: ContLongConfig):
    work = df.copy()
    work["event"] = make_event(work, cfg).astype("int8")
    _, edge = compute_label(work, "long", cfg.horizon, cfg.stop_atr, cfg.take_atr)
    work["edge_ret"] = edge

    train_idx, val_idx, test_idx = temporal_split(work, horizon=cfg.horizon)
    tr = work.iloc[train_idx].copy()
    va = work.iloc[val_idx].copy()
    te = work.iloc[test_idx].copy()
    tr = tr[(tr["event"] == 1) & tr["edge_ret"].notna()].copy()
    va = va[(va["event"] == 1) & va["edge_ret"].notna()].copy()
    te = te[(te["event"] == 1) & te["edge_ret"].notna()].copy()
    if min(len(tr), len(va), len(te)) < 100:
        return None

    med = tr[FEATURES].median()
    X_train = tr[FEATURES].fillna(med).to_numpy(np.float32)
    X_val = va[FEATURES].fillna(med).to_numpy(np.float32)
    X_test = te[FEATURES].fillna(med).to_numpy(np.float32)
    y_train = tr["edge_ret"].to_numpy(np.float32)
    y_val = va["edge_ret"].to_numpy(np.float32)
    y_test = te["edge_ret"].to_numpy(np.float32)

    model = train_regressor(X_train, y_train, X_val, y_val)
    val_scores = model.predict(X_val)
    test_scores = model.predict(X_test)

    best_q = optimize_quantile(val_scores, y_val, cost_bps=20)
    if best_q is None:
        return None
    _, q, thr, _, avg_edge, net = best_q
    va["score"] = val_scores
    te["score"] = test_scores
    sf = select_session_filter(va, thr)
    if sf is None:
        sf = ((0,), {"hours": None, "dows": None, "name": "none"}, backtest_daily(va, "score", "edge_ret", thr, cost_bps=20))
    _, filt, val_bt = sf
    te_f = apply_filter(te, filt["hours"], filt["dows"])
    test_bt = backtest_daily(te_f, "score", "edge_ret", thr, cost_bps=20)

    imp = (
        pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(12)
        .to_dict(orient="records")
    )
    return {
        "config": cfg,
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "n_test": int(len(te)),
        "q": q,
        "thr": thr,
        "val_avg_edge_pct": 100.0 * avg_edge,
        "val_net_trade_pct_eq_20bps": 100.0 * net,
        "filter_name": filt["name"],
        "allowed_hours": sorted(filt["hours"]) if filt["hours"] else [],
        "allowed_dows": sorted(filt["dows"]) if filt["dows"] else [],
        "val_backtest": val_bt,
        "test_backtest": test_bt,
        "top_features": imp,
    }


def main():
    out_dir = Path("/home/vitamind/my_project/model6/research_runs/2026-03-29-cont-long-grid-gpu")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet("/home/vitamind/my_project/model6/data/meta/meta_dataset_pruned.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = add_trap_structure(df)

    grid = [
        ContLongConfig(*vals)
        for vals in itertools.product(
            [0.05, 0.07, 0.09],
            [0.03, 0.05, 0.07],
            [0.08, 0.12],
            [16, 20],
            [0.9, 1.0],
            [1.6, 1.8, 2.0],
        )
    ]

    rows = []
    for i, cfg in enumerate(grid, start=1):
        print(f"[{i}/{len(grid)}] {cfg.name}", flush=True)
        res = run_one(df, cfg)
        if not res:
            continue
        cfg_name = cfg.name
        (out_dir / f"{cfg_name}.json").write_text(json.dumps({
            "config": res["config"].__dict__,
            "n_train": res["n_train"],
            "n_val": res["n_val"],
            "n_test": res["n_test"],
            "q": res["q"],
            "thr": res["thr"],
            "val_avg_edge_pct": res["val_avg_edge_pct"],
            "val_net_trade_pct_eq_20bps": res["val_net_trade_pct_eq_20bps"],
            "filter_name": res["filter_name"],
            "allowed_hours": res["allowed_hours"],
            "allowed_dows": res["allowed_dows"],
            "val_backtest": res["val_backtest"],
            "test_backtest": res["test_backtest"],
            "top_features": res["top_features"],
        }, ensure_ascii=False, indent=2))
        rows.append({
            "cfg": cfg_name,
            "n_test": res["n_test"],
            "q": round(res["q"], 3),
            "val_net_trade_pct_eq_20bps": round(res["val_net_trade_pct_eq_20bps"], 4),
            "val_compounded_pct": round(res["val_backtest"].get("compounded_return_pct", np.nan), 4),
            "val_days": res["val_backtest"].get("n_days", 0),
            "test_compounded_pct": round(res["test_backtest"].get("compounded_return_pct", np.nan), 4),
            "test_avg_trade_pct": round(res["test_backtest"].get("avg_pnl_pct_equity", np.nan), 4),
            "test_win_rate": round(100.0 * res["test_backtest"].get("win_rate", np.nan), 2),
            "test_dd_pct": round(res["test_backtest"].get("max_drawdown_pct", np.nan), 4),
            "filter": res["filter_name"],
            "hours": ",".join(map(str, res["allowed_hours"])),
            "dows": ",".join(map(str, res["allowed_dows"])),
        })

    if not rows:
        print("No valid configs.")
        return
    summary = pd.DataFrame(rows).sort_values(
        ["val_compounded_pct", "test_compounded_pct", "test_avg_trade_pct"],
        ascending=False,
    )
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
