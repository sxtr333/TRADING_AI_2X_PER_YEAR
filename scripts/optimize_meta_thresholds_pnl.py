#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import backtest_trade_combo_meta as bcm  # noqa: E402


def _parse_list(s: str):
    return [float(x) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="/home/vitamind/my_project/model6/data/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes_with_news_pruned.parquet")
    ap.add_argument("--meta-features", default="/home/vitamind/my_project/model6/data/meta/meta_dataset_pruned.parquet")
    ap.add_argument("--start", default="2025-01-01T00:00:00+00:00")
    ap.add_argument("--end", default="2026-01-01T00:00:00+00:00")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--threshold-bump-sigma", type=float, default=0.4)
    ap.add_argument("--trade-frac", type=float, default=0.4)
    ap.add_argument("--cooldown-steps", type=int, default=32)
    ap.add_argument("--max-concurrent", type=int, default=2)
    ap.add_argument("--cost-rt", type=float, default=0.0015)
    ap.add_argument("--trade-penalty", type=float, default=0.0)
    ap.add_argument("--out-csv", default="/home/vitamind/my_project/model6/reports/meta_thresholds_pnl_val.csv")
    ap.add_argument("--calibrate-meta", action="store_true")
    ap.add_argument("--calib-lr", type=float, default=0.1)
    ap.add_argument("--calib-steps", type=int, default=400)
    ap.add_argument("--meta-prob-h20-long", required=True)
    ap.add_argument("--meta-prob-h20-short", required=True)
    ap.add_argument("--meta-prob-h80-short", required=True)
    ap.add_argument("--meta-prob-h160-long", required=True)
    args = ap.parse_args()

    start_ts = pd.to_datetime(args.start, utc=True)
    end_ts = pd.to_datetime(args.end, utc=True)

    df = pd.read_parquet(args.features)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    meta_df = pd.read_parquet(args.meta_features)
    meta_df["timestamp"] = pd.to_datetime(meta_df["timestamp"], utc=True)
    meta_df = meta_df.sort_values("timestamp").reset_index(drop=True)

    if len(df) != len(meta_df) or not np.all(df["timestamp"].to_numpy() == meta_df["timestamp"].to_numpy()):
        raise ValueError("features and meta-features are not aligned by timestamp/length.")

    ts = df["timestamp"].to_numpy()
    close = df["close"].to_numpy(dtype=np.float64)

    best_h20 = pd.read_csv("/home/vitamind/my_project/model6/reports/backtest_v7_long_short_sweep_best.csv")
    best_v2 = pd.read_csv("/home/vitamind/my_project/model6/reports/backtest_v7_long_short_v2_sweep_best.csv")

    def get_best(df_best, horizon, direction):
        row = df_best[(df_best.horizon == horizon) & (df_best.direction == direction)].iloc[0]
        return float(row["threshold"]), str(row["strategy"])

    h20_long_thr, h20_long_strat = get_best(best_h20, 20, "long")
    h20_short_thr, h20_short_strat = get_best(best_h20, 20, "short")
    h80_short_thr, h80_short_strat = get_best(best_v2, 80, "short")
    h160_long_thr, h160_long_strat = get_best(best_v2, 160, "long")

    specs = [
        bcm.ModelSpec(
            name="h20_long",
            horizon=20,
            direction="long",
            strategy=h20_long_strat,
            model_path="/home/vitamind/my_project/model6/new_models/2026-01-18_v7_h20_long/model_15m_itransformer_v7_h20_long.keras",
            stats_path="/home/vitamind/my_project/model6/new_models/2026-01-18_v7_h20_long/norm_stats_v7_h20_long.npz",
            threshold=h20_long_thr,
        ),
        bcm.ModelSpec(
            name="h20_short",
            horizon=20,
            direction="short",
            strategy=h20_short_strat,
            model_path="/home/vitamind/my_project/model6/new_models/2026-01-18_v7_h20_short/model_15m_itransformer_v7_h20_short.keras",
            stats_path="/home/vitamind/my_project/model6/new_models/2026-01-18_v7_h20_short/norm_stats_v7_h20_short.npz",
            threshold=h20_short_thr,
        ),
        bcm.ModelSpec(
            name="h80_short_v2",
            horizon=80,
            direction="short",
            strategy=h80_short_strat,
            model_path="/home/vitamind/my_project/model6/new_models/2026-01-18_v7_h80_short_v2/model_15m_itransformer_v7_h80_short_v2.keras",
            stats_path="/home/vitamind/my_project/model6/new_models/2026-01-18_v7_h80_short_v2/norm_stats_v7_h80_short_v2.npz",
            threshold=h80_short_thr,
        ),
        bcm.ModelSpec(
            name="h160_long_v2",
            horizon=160,
            direction="long",
            strategy=h160_long_strat,
            model_path="/home/vitamind/my_project/model6/new_models/2026-01-18_v7_h160_long_v2/model_15m_itransformer_v7_h160_long_v2.keras",
            stats_path="/home/vitamind/my_project/model6/new_models/2026-01-18_v7_h160_long_v2/norm_stats_v7_h160_long_v2.npz",
            threshold=h160_long_thr,
        ),
    ]

    meta_specs = {
        "h20_long": bcm.MetaSpec(
            name="h20_long",
            model_path="/home/vitamind/my_project/model6/new_models/meta_2026-01-19/meta_h20_long.keras",
            stats_path="/home/vitamind/my_project/model6/new_models/meta_2026-01-19/meta_h20_long_stats.npz",
        ),
        "h20_short": bcm.MetaSpec(
            name="h20_short",
            model_path="/home/vitamind/my_project/model6/new_models/meta_2026-01-19/meta_h20_short.keras",
            stats_path="/home/vitamind/my_project/model6/new_models/meta_2026-01-19/meta_h20_short_stats.npz",
        ),
        "h80_short_v2": bcm.MetaSpec(
            name="h80_short_v2",
            model_path="/home/vitamind/my_project/model6/new_models/meta_2026-01-19/meta_h80_short_v2.keras",
            stats_path="/home/vitamind/my_project/model6/new_models/meta_2026-01-19/meta_h80_short_v2_stats.npz",
        ),
        "h160_long_v2": bcm.MetaSpec(
            name="h160_long_v2",
            model_path="/home/vitamind/my_project/model6/new_models/meta_2026-01-19/meta_h160_long_v2.keras",
            stats_path="/home/vitamind/my_project/model6/new_models/meta_2026-01-19/meta_h160_long_v2_stats.npz",
        ),
    }

    # volatility gate
    if "rv_short" in df.columns:
        rv = df["rv_short"].to_numpy(dtype=np.float64)
    elif "rv" in df.columns:
        rv = df["rv"].to_numpy(dtype=np.float64)
    else:
        rv = None

    # meta probabilities per spec
    meta_prob_map = {}
    meta_val_starts = []
    meta_test_starts = []
    for name, mspec in meta_specs.items():
        stats = bcm._load_stats(mspec.stats_path)
        meta_val_starts.append(pd.to_datetime(bcm._unpack_obj(stats["val_start_ts"]), utc=True))
        meta_test_starts.append(pd.to_datetime(bcm._unpack_obj(stats["test_start_ts"]), utc=True))
        model = bcm._load_model(mspec.model_path)
        meta_prob_map[name] = bcm._pred_meta_probs(model, stats, meta_df, args.batch_size)

    if args.calibrate_meta:
        label_map = {
            "h20_long": "meta_label_h20_long",
            "h20_short": "meta_label_h20_short",
            "h80_short_v2": "meta_label_h80_short_v2",
            "h160_long_v2": "meta_label_h160_long_v2",
        }
        for name, mspec in meta_specs.items():
            stats = bcm._load_stats(mspec.stats_path)
            val_start = pd.to_datetime(bcm._unpack_obj(stats["val_start_ts"]), utc=True)
            test_start = pd.to_datetime(bcm._unpack_obj(stats["test_start_ts"]), utc=True)
            label_col = label_map.get(name)
            if label_col not in meta_df.columns:
                continue
            y = meta_df[label_col].to_numpy(dtype=np.float32)
            prob = meta_prob_map.get(name)
            if prob is None:
                continue
            mask = (ts >= val_start) & (ts < test_start) & ~np.isnan(y) & ~np.isnan(prob)
            if mask.sum() < 50 or len(np.unique(y[mask])) < 2:
                continue
            a, b = bcm._fit_platt(prob[mask], y[mask], lr=args.calib_lr, steps=args.calib_steps)
            meta_prob_map[name] = bcm._apply_platt(prob, a, b)

    model_outputs = {}
    base_val_starts = []
    base_test_starts = []
    for spec in specs:
        stats = bcm._load_stats(spec.stats_path)
        base_val_starts.append(pd.to_datetime(bcm._unpack_obj(stats["val_start_ts"]), utc=True))
        base_test_starts.append(pd.to_datetime(bcm._unpack_obj(stats["test_start_ts"]), utc=True))
        model = bcm._load_model(spec.model_path)
        pred, idx, sigma = bcm._pred_for_horizon(model, stats, df, spec.horizon, args.batch_size)

        val_start = pd.to_datetime(bcm._unpack_obj(stats["val_start_ts"]), utc=True)
        test_start = pd.to_datetime(bcm._unpack_obj(stats["test_start_ts"]), utc=True)
        val_mask = (ts[idx] >= val_start) & (ts[idx] < test_start)
        bias_shift = bcm._calc_bias_shift(df, idx[val_mask], spec.horizon, pred[val_mask])
        pred = pred - bias_shift

        if spec.strategy == "vol_gate_p60" and rv is not None:
            period_mask = (ts[idx] >= start_ts) & (ts[idx] < end_ts)
            rv_gate = np.percentile(rv[idx[period_mask]], 60.0)
            gate_mask = rv[idx] >= rv_gate
        else:
            gate_mask = None

        model_outputs[spec.name] = {
            "spec": spec,
            "pred": pred,
            "idx": idx,
            "gate": gate_mask,
            "sigma": sigma,
        }

    # validation window: intersection of all model val windows, within requested start/end
    val_start = max([start_ts] + base_val_starts + meta_val_starts)
    val_end = min([end_ts] + base_test_starts + meta_test_starts)
    if val_end <= val_start:
        raise ValueError("Validation window is empty; check start/end or stats timestamps.")

    h20_long_list = _parse_list(args.meta_prob_h20_long)
    h20_short_list = _parse_list(args.meta_prob_h20_short)
    h80_short_list = _parse_list(args.meta_prob_h80_short)
    h160_long_list = _parse_list(args.meta_prob_h160_long)

    rows = []
    for a, b, c, d in itertools.product(h20_long_list, h20_short_list, h80_short_list, h160_long_list):
        thr_map = {
            "h20_long": a,
            "h20_short": b,
            "h80_short_v2": c,
            "h160_long_v2": d,
        }
        open_map = {}
        for out in model_outputs.values():
            spec = out["spec"]
            pred = out["pred"]
            idx = out["idx"]
            gate = out["gate"]
            sigma = float(out["sigma"])
            thr = spec.threshold + args.threshold_bump_sigma * sigma
            meta_probs = meta_prob_map.get(spec.name)
            meta_thr_use = float(thr_map.get(spec.name, 0.0))

            period_mask = (ts[idx] >= val_start) & (ts[idx] < val_end)
            pred = pred[period_mask]
            idx = idx[period_mask]
            if gate is not None:
                gate = gate[period_mask]

            for p, i, g in zip(pred, idx, gate if gate is not None else [True] * len(idx)):
                if not g:
                    continue
                if spec.direction == "long" and p <= thr:
                    continue
                if spec.direction == "short" and p >= -thr:
                    continue
                if meta_probs is None or np.isnan(float(meta_probs[i])) or float(meta_probs[i]) < meta_thr_use:
                    continue
                open_map.setdefault(int(i), []).append(spec)

        equity, trades, fees, max_dd = bcm._simulate(
            df,
            ts,
            close,
            open_map,
            val_start,
            val_end,
            args.cost_rt,
            args.trade_frac,
            1.0,
            args.cooldown_steps,
            args.max_concurrent,
        )
        pnl = float(equity - 100.0)
        score = pnl - args.trade_penalty * float(trades)
        rows.append(
            {
                "meta_prob_h20_long": a,
                "meta_prob_h20_short": b,
                "meta_prob_h80_short_v2": c,
                "meta_prob_h160_long_v2": d,
                "val_start": str(val_start),
                "val_end": str(val_end),
                "pnl": pnl,
                "score": score,
                "trades": int(trades),
                "fees": float(fees),
                "max_dd": float(max_dd),
            }
        )

    out_df = pd.DataFrame(rows).sort_values("score", ascending=False)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print("Saved:", out_path)
    print(out_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
