#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import backtest_trade_combo_meta as bcm  # noqa: E402


def _parse_map(s: str):
    out = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        k, v = part.split("=")
        out[k.strip()] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="/home/vitamind/my_project/model6/data/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes_with_news_pruned.parquet")
    ap.add_argument("--meta-features", default="/home/vitamind/my_project/model6/data/meta/meta_dataset_pruned.parquet")
    ap.add_argument("--start", default="2025-01-01T00:00:00+00:00")
    ap.add_argument("--end", default="2026-01-01T00:00:00+00:00")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--threshold-bump-sigma", type=float, default=0.4)
    ap.add_argument("--cooldown-steps", type=int, default=32)
    ap.add_argument("--max-concurrent", type=int, default=2)
    ap.add_argument("--meta-prob-per-model", required=True)
    ap.add_argument("--out-png", default="/home/vitamind/my_project/model6/reports/trades_plot_meta_best.png")
    ap.add_argument("--out-trades", default="/home/vitamind/my_project/model6/reports/trades_meta_best.csv")
    args = ap.parse_args()

    meta_thr_map = _parse_map(args.meta_prob_per_model)

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

    if "rv_short" in df.columns:
        rv = df["rv_short"].to_numpy(dtype=np.float64)
    elif "rv" in df.columns:
        rv = df["rv"].to_numpy(dtype=np.float64)
    else:
        rv = None

    meta_prob_map = {}
    for name, mspec in meta_specs.items():
        stats = bcm._load_stats(mspec.stats_path)
        model = bcm._load_model(mspec.model_path)
        meta_prob_map[name] = bcm._pred_meta_probs(model, stats, meta_df, args.batch_size)

    model_outputs = {}
    for spec in specs:
        stats = bcm._load_stats(spec.stats_path)
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

    open_map = {}
    for out in model_outputs.values():
        spec = out["spec"]
        pred = out["pred"]
        idx = out["idx"]
        gate = out["gate"]
        sigma = float(out["sigma"])
        thr = spec.threshold + args.threshold_bump_sigma * sigma
        meta_probs = meta_prob_map.get(spec.name)
        meta_thr_use = float(meta_thr_map.get(spec.name, 0.0))

        period_mask = (ts[idx] >= start_ts) & (ts[idx] < end_ts)
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

    positions = []
    trades = []
    last_trade_step = {}

    for i in range(len(df)):
        if ts[i] < start_ts or ts[i] >= end_ts:
            continue

        if positions:
            still = []
            for pos in positions:
                if pos["close_idx"] == i:
                    trades.append(
                        {
                            "entry_idx": pos["entry_idx"],
                            "exit_idx": i,
                            "entry_ts": ts[pos["entry_idx"]],
                            "exit_ts": ts[i],
                            "direction": pos["direction"],
                            "entry_price": pos["entry_price"],
                            "exit_price": close[i],
                            "model": pos["model"],
                        }
                    )
                else:
                    still.append(pos)
            positions = still

        if i in open_map:
            for spec in open_map[i]:
                if len(positions) >= args.max_concurrent:
                    break
                last_i = last_trade_step.get(spec.name, -10**9)
                if i - last_i < args.cooldown_steps:
                    continue
                positions.append(
                    {
                        "entry_idx": i,
                        "close_idx": i + spec.horizon,
                        "direction": spec.direction,
                        "entry_price": close[i],
                        "model": spec.name,
                    }
                )
                last_trade_step[spec.name] = i

    end_idx = np.where(ts >= end_ts)[0]
    end_i = int(end_idx[0]) if len(end_idx) > 0 else len(df) - 1
    for pos in positions:
        trades.append(
            {
                "entry_idx": pos["entry_idx"],
                "exit_idx": end_i,
                "entry_ts": ts[pos["entry_idx"]],
                "exit_ts": ts[end_i],
                "direction": pos["direction"],
                "entry_price": pos["entry_price"],
                "exit_price": close[end_i],
                "model": pos["model"],
            }
        )

    trades_df = pd.DataFrame(trades)
    trades_df.to_csv(args.out_trades, index=False)

    # Plot (only the requested window)
    plot_mask = (df["timestamp"] >= start_ts) & (df["timestamp"] < end_ts)
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(df.loc[plot_mask, "timestamp"], df.loc[plot_mask, "close"], color="#1f2937", linewidth=0.8, label="Close")

    if not trades_df.empty:
        longs = trades_df[trades_df["direction"] == "long"]
        shorts = trades_df[trades_df["direction"] == "short"]
        ax.scatter(longs["entry_ts"], longs["entry_price"], marker="^", s=25, color="#16a34a", label="Long entry")
        ax.scatter(longs["exit_ts"], longs["exit_price"], marker="x", s=25, color="#16a34a", label="Long exit")
        ax.scatter(shorts["entry_ts"], shorts["entry_price"], marker="v", s=25, color="#dc2626", label="Short entry")
        ax.scatter(shorts["exit_ts"], shorts["exit_price"], marker="x", s=25, color="#dc2626", label="Short exit")

    ax.set_title("Meta-trade entries/exits (buy/sell)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend(loc="best", ncol=2, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    plt.savefig(args.out_png, dpi=160)
    print(f"Saved plot: {args.out_png}")
    print(f"Saved trades: {args.out_trades}")


if __name__ == "__main__":
    main()
