#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
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


def cont_long_event(df: pd.DataFrame) -> pd.Series:
    return (
        (df["break_hi_96_atr"] >= 0.07)
        & (df["close_vs_prev_hi_96_atr"] >= 0.05)
        & (df["body_frac"] >= 0.12)
        & (df["volume_delta"] > 0)
        & (df["macd_hist"] > 0)
    ).fillna(False)


def cont_short_event(df: pd.DataFrame) -> pd.Series:
    return (
        (df["break_lo_48_atr"] >= 0.04)
        & (df["close_vs_prev_lo_48_atr"] <= -0.02)
        & (df["body_frac"] <= -0.05)
    ).fillna(False)


def train_regressor(X_train, y_train, X_val, y_val):
    model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        device="cuda",
        max_depth=6,
        n_estimators=900,
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


def optimize_quantile(val_scores, val_edge, cost_bps=20):
    best = None
    for q in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]:
        thr = float(np.quantile(val_scores, q))
        m = val_scores >= thr
        if m.sum() < 20:
            continue
        avg_edge = float(np.mean(val_edge[m]))
        net = 1.5 * avg_edge - (cost_bps / 10000.0) * 1.5
        score = (net, avg_edge, -q)
        if best is None or score > best[0]:
            best = (score, q, thr, int(m.sum()), avg_edge, net)
    return best


def backtest_daily(frame: pd.DataFrame, score_col: str, edge_col: str, thr: float, cost_bps=20):
    sel = frame[frame[score_col] >= thr].copy()
    if sel.empty:
        return {}
    sel["day"] = sel["timestamp"].dt.floor("D")
    # keep strongest setup per day
    sel = sel.sort_values([score_col, "timestamp"], ascending=[False, True]).groupby("day", as_index=False).first()
    sel["pnl"] = 1.5 * sel[edge_col] - (cost_bps / 10000.0) * 1.5
    eq = (1 + sel["pnl"]).cumprod()
    peak = eq.cummax()
    dd = (eq / peak - 1).min()
    return {
        "n_days": int(len(sel)),
        "win_rate": float((sel["pnl"] > 0).mean()),
        "avg_pnl_pct_equity": float(100 * sel["pnl"].mean()),
        "median_pnl_pct_equity": float(100 * sel["pnl"].median()),
        "compounded_return_pct": float(100 * (eq.iloc[-1] - 1)),
        "max_drawdown_pct": float(100 * dd),
    }


def run_family(name: str, side: str, event_mask: pd.Series, horizon: int, stop_atr: float, take_atr: float, out_dir: Path):
    base = Path("/home/vitamind/my_project/model6")
    df = pd.read_parquet(base / "data" / "meta" / "meta_dataset_pruned.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = add_trap_structure(df)
    event_mask = event_mask(df)
    _, edge = compute_label(df, side, horizon, stop_atr, take_atr)
    df["event"] = event_mask.astype("int8")
    df["edge_ret"] = edge

    train_idx, val_idx, test_idx = temporal_split(df, horizon=horizon)
    tr = df.iloc[train_idx].copy()
    va = df.iloc[val_idx].copy()
    te = df.iloc[test_idx].copy()
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
    best = optimize_quantile(val_scores, y_val, cost_bps=20)
    if best is None:
        return None
    _, q, thr, n_val, avg_edge, net = best
    va["score"] = val_scores
    te["score"] = test_scores
    val_bt = backtest_daily(va, "score", "edge_ret", thr, cost_bps=20)
    test_bt = backtest_daily(te, "score", "edge_ret", thr, cost_bps=20)

    imp = pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_}).sort_values(
        "importance", ascending=False
    )
    result = {
        "family": name,
        "side": side,
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "n_test": int(len(te)),
        "chosen_quantile": q,
        "chosen_threshold": float(thr),
        "val_signal_count": n_val,
        "val_avg_edge_ret_pct": float(100 * avg_edge),
        "val_net_per_trade_pct_equity_20bps": float(100 * net),
        "val_backtest": val_bt,
        "test_backtest": test_bt,
        "top_features": imp.head(12).to_dict(orient="records"),
    }
    (out_dir / f"{name}_rank_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    imp.to_csv(out_dir / f"{name}_rank_importance.csv", index=False)
    model.save_model(str(out_dir / f"{name}_rank.json"))
    return result


def main():
    out_dir = Path("/home/vitamind/my_project/model6/research_runs/2026-03-29-cont-rank-gpu")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, side, fn, horizon, stop, take in [
        ("cont_long_rank", "long", cont_long_event, 20, 1.0, 1.8),
        ("cont_short_rank", "short", cont_short_event, 20, 1.0, 1.8),
    ]:
        res = run_family(name, side, fn, horizon, stop, take, out_dir)
        if res:
            rows.append(
                {
                    "family": res["family"],
                    "n_test": res["n_test"],
                    "q": res["chosen_quantile"],
                    "val_net_per_trade_pct_equity_20bps": round(res["val_net_per_trade_pct_equity_20bps"], 4),
                    "test_compounded_return_pct": round(res["test_backtest"].get("compounded_return_pct", np.nan), 4),
                    "test_avg_pnl_pct_equity": round(res["test_backtest"].get("avg_pnl_pct_equity", np.nan), 4),
                    "test_win_rate": round(100 * res["test_backtest"].get("win_rate", np.nan), 2),
                    "test_max_drawdown_pct": round(res["test_backtest"].get("max_drawdown_pct", np.nan), 4),
                }
            )
    if rows:
        summary = pd.DataFrame(rows).sort_values("test_compounded_return_pct", ascending=False)
        summary.to_csv(out_dir / "summary.csv", index=False)
        print(summary.to_string(index=False))
    else:
        print("No continuation ranking families produced valid results.")


if __name__ == "__main__":
    main()
