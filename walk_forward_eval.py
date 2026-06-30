#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Walk-forward evaluation with PnL by month.
Uses saved normalization stats and a trained Keras model (price + cls heads).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from model_layers import RevIN, TSMixerBlock, ITransformerBlock, LastStep
from train_keras import DropPath


def parse_ts(s: str) -> pd.Timestamp:
    return pd.to_datetime(s, utc=True)


def load_stats(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    stats = {k: data[k] for k in data.files}
    stats["mean"] = stats["mean"].astype(np.float32)
    stats["std"] = stats["std"].astype(np.float32)
    return stats


def build_end_indices(n: int, seq_len: int, train_end: int, val_end: int, purge_gap: int, horizon: int) -> np.ndarray:
    min_end = seq_len - 1
    purge = max(0, int(purge_gap))
    hgap = max(0, int(horizon))
    test_start = max(val_end + purge, min_end)
    test_max = max(min_end, n - hgap)
    return np.arange(test_start, test_max, dtype=np.int64)


def predict_cls(model, x_batch):
    out = model(x_batch, training=False)
    if isinstance(out, dict):
        # prefer short-horizon head if available
        if "cls" in out:
            cls = out["cls"]
        elif "cls_h20" in out:
            cls = out["cls_h20"]
        else:
            # fallback: first key in dict
            first_key = next(iter(out.keys()))
            cls = out[first_key]
    elif isinstance(out, (list, tuple)) and len(out) >= 2:
        cls = out[1]
    else:
        cls = out
    cls = tf.convert_to_tensor(cls)
    if cls.shape.rank == 2 and cls.shape[-1] > 1:
        # take prob of "up" class if softmax
        cls = cls[:, -1]
    return tf.reshape(cls, [-1]).numpy()


def predict_price_q(model, x_batch):
    out = model(x_batch, training=False)
    if isinstance(out, (list, tuple)) and len(out) >= 1:
        price = out[0]
    else:
        price = out
    price = tf.convert_to_tensor(price)
    return price.numpy()


def pnl_metrics(pnl: np.ndarray, periods_per_year: float) -> dict:
    pnl = np.asarray(pnl, dtype=np.float64)
    equity = np.cumprod(1.0 + pnl)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    cagr = (equity[-1] ** (periods_per_year / len(pnl)) - 1.0) if len(pnl) else 0.0
    sharpe = (pnl.mean() / (pnl.std() + 1e-12)) * np.sqrt(periods_per_year) if len(pnl) else 0.0
    max_dd = float(drawdown.min() if len(pnl) else 0.0)
    calmar = float(cagr / (abs(max_dd) + 1e-12)) if len(pnl) else 0.0
    return {
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "max_dd": max_dd,
        "calmar": calmar,
        "total_return": float(equity[-1] - 1.0) if len(pnl) else 0.0,
    }


def max_dd_with_reset(equity: np.ndarray, ts_idx: pd.Series, dd_reset: str) -> float:
    if len(equity) == 0:
        return 0.0
    peak = equity[0]
    max_dd = 0.0
    last_period = None
    for i in range(len(equity)):
        ts = ts_idx.iloc[i]
        if dd_reset == "daily":
            period = ts.date()
        elif dd_reset == "monthly":
            period = (ts.year, ts.month)
        elif dd_reset == "quarterly":
            q = (ts.month - 1) // 3 + 1
            period = (ts.year, q)
        else:
            period = None
        if period is not None and period != last_period:
            peak = equity[i]
            last_period = period
        peak = max(peak, equity[i])
        dd = (equity[i] - peak) / peak
        if dd < max_dd:
            max_dd = dd
    return float(max_dd)


def infer_periods_per_year(ts: pd.Series) -> float:
    ts = pd.to_datetime(ts, utc=True).sort_values()
    deltas = ts.diff().dropna().dt.total_seconds().values
    if deltas.size == 0:
        return 365 * 24
    med = float(np.median(deltas))
    minutes = med / 60.0
    return (365 * 24 * 60) / minutes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--train-end", required=True)
    ap.add_argument("--val-end", required=True)
    ap.add_argument("--purge-gap", type=int, default=0)
    ap.add_argument("--target-horizon", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--short-threshold", type=float, default=None)
    ap.add_argument("--use-quantile-signal", action="store_true",
                    help="Use price_q head quantiles for signals instead of cls prob.")
    ap.add_argument("--quantiles", default=None,
                    help="Comma-separated quantiles used in training, e.g. '0.1,0.5,0.9'")
    ap.add_argument("--q-min", type=float, default=0.0,
                    help="Min absolute return threshold for quantile signal.")
    ap.add_argument("--mode", choices=["long_only", "long_short"], default="long_short")
    ap.add_argument("--opt-metric", choices=["none", "sharpe", "calmar"], default="none")
    ap.add_argument("--thr-min", type=float, default=0.50)
    ap.add_argument("--thr-max", type=float, default=0.70)
    ap.add_argument("--thr-step", type=float, default=0.01)
    ap.add_argument("--fee-bps", type=float, default=0.0)
    ap.add_argument("--slip-bps", type=float, default=0.0)
    ap.add_argument("--min-hold", type=int, default=1)
    ap.add_argument("--cooldown", type=int, default=0)
    ap.add_argument("--max-trades-per-day", type=int, default=0)
    ap.add_argument("--vol-col", type=str, default=None)
    ap.add_argument("--vol-k", type=float, default=0.0)
    ap.add_argument("--vol-clip", type=float, default=0.25)
    ap.add_argument("--vol-block", type=float, default=999.0)
    ap.add_argument("--vol-size-k", type=float, default=0.0)
    ap.add_argument("--max-dd", type=float, default=0.0)
    ap.add_argument("--daily-limit", type=float, default=0.0)
    ap.add_argument("--dd-reset", choices=["none", "daily", "monthly", "quarterly"], default="none")
    ap.add_argument("--prob-ema", type=float, default=0.0,
                    help="EMA smoothing factor for probs (0 disables).")
    ap.add_argument("--trade-band", type=float, default=0.0,
                    help="Dead-band around 0.5; no trade if |prob-0.5| < band.")
    ap.add_argument("--report-csv", type=str, default=None)
    ap.add_argument("--equity-csv", type=str, default=None)
    ap.add_argument("--equity-png", type=str, default=None)
    ap.add_argument("--model-name", type=str, default=None)
    args = ap.parse_args()

    df = pd.read_parquet(args.features).sort_values("timestamp").reset_index(drop=True)
    stats = load_stats(Path(args.stats))
    feature_names = [str(x) for x in stats.get("feature_names", [])]
    if not feature_names:
        # fallback: numeric columns excluding targets/timestamps
        feature_names = [
            c for c in df.columns
            if df[c].dtype.kind in "ifb"
            and not c.startswith("target_")
            and c not in ("timestamp", "label_3cls", "tb_label", "tb_tth")
        ]

    seq_len = int(args.seq_len or (stats.get("seq_len", np.array([256]))[0]))

    # target_ret + direction are required
    if "target_ret" not in df.columns:
        raise ValueError("target_ret not found in features parquet.")

    X_raw = df[feature_names].to_numpy(dtype=np.float32)
    X = (X_raw - stats["mean"]) / stats["std"]
    y_ret = df["target_ret"].to_numpy(dtype=np.float32)
    ts = pd.to_datetime(df["timestamp"], utc=True)

    # split boundaries by timestamp
    train_end_ts = parse_ts(args.train_end)
    val_end_ts = parse_ts(args.val_end)
    train_end_idx = int(df.index[ts <= train_end_ts].max()) + 1
    val_end_idx = int(df.index[ts <= val_end_ts].max()) + 1

    ends = build_end_indices(
        n=len(df),
        seq_len=seq_len,
        train_end=train_end_idx,
        val_end=val_end_idx,
        purge_gap=args.purge_gap,
        horizon=args.target_horizon,
    )

    # build dataset
    X_tf = tf.convert_to_tensor(X, dtype=tf.float32)
    ends_tf = tf.data.Dataset.from_tensor_slices(ends.astype(np.int64))

    def map_fn(i):
        i = tf.cast(i, tf.int32)
        start = i - (seq_len - 1)
        x_seq = X_tf[start:i + 1]
        x_seq = tf.ensure_shape(x_seq, [seq_len, X.shape[1]])
        return x_seq, i

    ds = ends_tf.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE).batch(256).prefetch(tf.data.AUTOTUNE)

    custom = {
        "RevIN": RevIN,
        "TSMixerBlock": TSMixerBlock,
        "ITransformerBlock": ITransformerBlock,
        "LastStep": LastStep,
        "DropPath": DropPath,
    }
    model = tf.keras.models.load_model(Path(args.model), safe_mode=False, compile=False, custom_objects=custom)

    use_q = bool(args.use_quantile_signal)
    if use_q:
        if not args.quantiles:
            raise ValueError("--use-quantile-signal requires --quantiles (e.g. 0.1,0.5,0.9)")
        q_list = [float(q.strip()) for q in args.quantiles.split(",") if q.strip()]
        if len(q_list) < 2:
            raise ValueError("--quantiles must provide at least two values.")
        q_arr = np.array(q_list, dtype=np.float32)
        q_low_idx = int(np.argmin(q_arr))
        q_high_idx = int(np.argmax(q_arr))

        all_q = []
        all_idx = []
        for xb, ib in ds:
            qpred = predict_price_q(model, xb)
            all_q.append(qpred)
            all_idx.append(ib.numpy())
        qpred = np.concatenate(all_q)
        idx = np.concatenate(all_idx)

        q_low = qpred[:, q_low_idx]
        q_high = qpred[:, q_high_idx]
        q_min = float(args.q_min)
        probs = np.full_like(q_low, 0.5, dtype=np.float32)
        probs[q_low > q_min] = 1.0
        probs[q_high < -q_min] = 0.0
    else:
        all_probs = []
        all_idx = []
        for xb, ib in ds:
            probs = predict_cls(model, xb)
            all_probs.append(probs)
            all_idx.append(ib.numpy())
        probs = np.concatenate(all_probs)
        idx = np.concatenate(all_idx)
        if float(args.prob_ema) > 0.0:
            alpha = float(args.prob_ema)
            p = probs.astype(np.float32).copy()
            for i in range(1, len(p)):
                p[i] = alpha * p[i] + (1.0 - alpha) * p[i - 1]
            probs = p

    y_ret_idx = y_ret[idx]
    ts_idx = ts.iloc[idx].reset_index(drop=True)
    periods_per_year = infer_periods_per_year(ts_idx)

    vol_adj = None
    if args.vol_col:
        if args.vol_col not in df.columns:
            raise ValueError(f"vol_col '{args.vol_col}' not found in features.")
        vol_raw = df[args.vol_col].to_numpy(dtype=np.float32)[idx]
        vmean = float(np.nanmean(vol_raw))
        vstd = float(np.nanstd(vol_raw)) + 1e-6
        vol_adj = (vol_raw - vmean) / vstd
        vol_adj = np.clip(vol_adj, -float(args.vol_clip), float(args.vol_clip))

    def compute_pos(thr_long: float, thr_short: float) -> np.ndarray:
        pos = np.zeros_like(probs, dtype=np.float32)
        for i in range(len(probs)):
            tl = thr_long
            ts_ = thr_short
            if vol_adj is not None and args.vol_k != 0.0:
                tl = float(np.clip(thr_long + args.vol_k * vol_adj[i], 0.01, 0.99))
                ts_ = float(np.clip(thr_short - args.vol_k * vol_adj[i], 0.01, 0.99))
            if vol_adj is not None and abs(float(vol_adj[i])) > float(args.vol_block):
                pos[i] = 0.0
                continue
            if float(args.trade_band) > 0.0 and abs(float(probs[i]) - 0.5) < float(args.trade_band):
                pos[i] = 0.0
                continue
            if args.mode == "long_only":
                pos[i] = 1.0 if probs[i] >= tl else 0.0
            else:
                if probs[i] >= tl:
                    pos[i] = 1.0
                elif probs[i] <= ts_:
                    pos[i] = -1.0
                else:
                    pos[i] = 0.0
        return pos

    def compute_pnl(pos: np.ndarray) -> np.ndarray:
        pnl = pos * np.expm1(y_ret_idx)
        cost_bps = float(args.fee_bps + args.slip_bps)
        if cost_bps > 0:
            cost = cost_bps / 10000.0
            turnover = np.abs(np.diff(pos, prepend=pos[:1]))
            pnl = pnl - turnover * cost
        return pnl

    def apply_vol_sizing(pos: np.ndarray) -> np.ndarray:
        if vol_adj is None or args.vol_size_k == 0.0:
            return pos
        scale = 1.0 / (1.0 + float(args.vol_size_k) * np.abs(vol_adj))
        return pos * scale.astype(np.float32)

    def simulate_with_limits(pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        max_dd = float(args.max_dd)
        daily_limit = float(args.daily_limit)
        cost_bps = float(args.fee_bps + args.slip_bps)
        cost = cost_bps / 10000.0 if cost_bps > 0 else 0.0

        pos_out = np.zeros_like(pos, dtype=np.float32)
        pnl_out = np.zeros_like(pos, dtype=np.float32)
        equity = 1.0
        peak = 1.0
        day = None
        day_pnl = 0.0
        dd_stop = False
        day_stop = False
        prev_pos = 0.0
        dd_period = None

        for i in range(len(pos)):
            d = ts_idx.iloc[i].date()
            if day != d:
                day = d
                day_pnl = 0.0
                day_stop = False
            if args.dd_reset != "none":
                if args.dd_reset == "daily":
                    cur_period = d
                elif args.dd_reset == "monthly":
                    cur_period = (ts_idx.iloc[i].year, ts_idx.iloc[i].month)
                else:
                    q = (ts_idx.iloc[i].month - 1) // 3 + 1
                    cur_period = (ts_idx.iloc[i].year, q)
                if dd_period != cur_period:
                    dd_period = cur_period
                    dd_stop = False
                    peak = equity

            cur_pos = 0.0 if (dd_stop or day_stop) else pos[i]
            turnover = abs(cur_pos - prev_pos)
            pnl_i = cur_pos * np.expm1(y_ret_idx[i]) - turnover * cost

            # Provisional equity update
            equity_next = equity * (1.0 + pnl_i)
            peak_next = max(peak, equity_next)
            dd_next = (equity_next - peak_next) / peak_next

            # Enforce daily loss cap (hard cap)
            if daily_limit > 0 and (day_pnl + pnl_i) < -daily_limit:
                pnl_i = -daily_limit - day_pnl
                equity_next = equity * (1.0 + pnl_i)
                day_stop = True

            # Enforce max drawdown cap (hard cap)
            if max_dd < 0:
                # drawdown relative to current peak (do not move peak on losses)
                dd_next = (equity_next - peak) / peak
                if dd_next <= max_dd:
                    equity_next = peak * (1.0 + max_dd)
                    pnl_i = equity_next / equity - 1.0
                    dd_stop = True

            equity = equity_next
            peak = max(peak, equity)
            day_pnl += pnl_i

            pos_out[i] = cur_pos
            pnl_out[i] = pnl_i
            prev_pos = cur_pos

        return pos_out, pnl_out

    def apply_risk_controls(pos_raw: np.ndarray) -> np.ndarray:
        if args.min_hold <= 1 and args.cooldown <= 0 and args.max_trades_per_day <= 0:
            return pos_raw
        pos_adj = np.zeros_like(pos_raw)
        cur = 0.0
        hold = 0
        cd = 0
        trades_today = 0
        day = None
        for i in range(len(pos_raw)):
            d = ts_idx.iloc[i].date()
            if day != d:
                day = d
                trades_today = 0
            desired = pos_raw[i]
            if cd > 0:
                desired = 0.0
            if hold > 0 and desired != cur:
                desired = cur
            if desired != cur:
                if args.max_trades_per_day > 0 and trades_today >= args.max_trades_per_day:
                    desired = cur
                else:
                    trades_today += 1
            if desired != cur:
                cur = desired
                hold = max(0, int(args.min_hold) - 1) if cur != 0 else 0
                if cur == 0:
                    cd = int(args.cooldown)
            pos_adj[i] = cur
            if hold > 0:
                hold -= 1
            if cd > 0:
                cd -= 1
        return pos_adj

    def eval_threshold(thr_long: float, thr_short: float) -> dict:
        pos = compute_pos(thr_long, thr_short)
        pos = apply_risk_controls(pos)
        pos = apply_vol_sizing(pos)
        if args.max_dd < 0 or args.daily_limit > 0:
            pos, pnl = simulate_with_limits(pos)
        else:
            pnl = compute_pnl(pos)
        metrics = pnl_metrics(pnl, periods_per_year)
        if args.dd_reset != "none":
            equity_local = np.cumprod(1.0 + pnl)
            md = max_dd_with_reset(equity_local, ts_idx, args.dd_reset)
            metrics["max_dd"] = md
            metrics["calmar"] = float(metrics["cagr"] / (abs(md) + 1e-12)) if len(pnl) else 0.0
        metrics["threshold"] = float(thr_long)
        metrics["short_threshold"] = float(thr_short)
        return metrics

    if args.opt_metric == "none":
        thr_long = float(args.threshold)
        thr_short = float(args.short_threshold) if args.short_threshold is not None else 1.0 - thr_long
        base = eval_threshold(thr_long, thr_short)
    else:
        best = None
        thr = float(args.thr_min)
        thr_max = float(args.thr_max)
        step = float(args.thr_step)
        while thr <= thr_max + 1e-9:
            thr_long = thr
            thr_short = float(args.short_threshold) if args.short_threshold is not None else 1.0 - thr_long
            metrics = eval_threshold(thr_long, thr_short)
            score = metrics[args.opt_metric]
            if best is None or score > best[0]:
                best = (score, metrics)
            thr += step
        base = best[1] if best else eval_threshold(float(args.threshold), 1.0 - float(args.threshold))

    pos = compute_pos(base["threshold"], base["short_threshold"])
    pos = apply_risk_controls(pos)
    pos = apply_vol_sizing(pos)
    if args.max_dd < 0 or args.daily_limit > 0:
        pos, pnl = simulate_with_limits(pos)
    else:
        pnl = compute_pnl(pos)
    equity = np.cumprod(1.0 + pnl)

    # monthly walk-forward stats
    month = ts_idx.dt.to_period("M").astype(str)
    df_pnl = pd.DataFrame({"month": month, "pnl": pnl})
    def month_stats(x):
        stats = pnl_metrics(x.values, periods_per_year)
        stats["n"] = int(len(x))
        stats["mean_pnl"] = float(np.mean(x.values)) if len(x) else 0.0
        return pd.Series(stats)
    monthly = df_pnl.groupby("month")["pnl"].apply(month_stats).unstack()

    print("[overall]", base)
    print("\n[monthly]")
    print(monthly.tail(12).to_string())

    if args.report_csv:
        row = {
            "model": args.model_name or Path(args.model).stem,
            "mode": args.mode,
            "opt_metric": args.opt_metric,
            "threshold": base["threshold"],
            "short_threshold": base["short_threshold"],
            "cagr": base["cagr"],
            "sharpe": base["sharpe"],
            "calmar": base["calmar"],
            "max_dd": base["max_dd"],
            "total_return": base["total_return"],
            "fee_bps": float(args.fee_bps),
            "slip_bps": float(args.slip_bps),
            "min_hold": int(args.min_hold),
            "cooldown": int(args.cooldown),
            "max_trades_per_day": int(args.max_trades_per_day),
            "vol_col": args.vol_col or "",
            "vol_k": float(args.vol_k),
            "vol_block": float(args.vol_block),
            "vol_size_k": float(args.vol_size_k),
            "max_dd_cap": float(args.max_dd),
            "daily_limit": float(args.daily_limit),
            "prob_ema": float(args.prob_ema),
            "trade_band": float(args.trade_band),
            "n_trades": int(np.sum(np.abs(np.diff(pos, prepend=pos[:1])) > 0)),
            "n_samples": int(len(pnl)),
        }
        out_path = Path(args.report_csv)
        df_row = pd.DataFrame([row])
        if out_path.exists():
            df_row.to_csv(out_path, mode="a", index=False, header=False)
        else:
            df_row.to_csv(out_path, index=False)

    if args.equity_csv or args.equity_png:
        df_eq = pd.DataFrame(
            {
                "timestamp": ts_idx.values,
                "pnl": pnl,
                "equity": equity,
                "pos": pos,
                "prob": probs,
            }
        )
        if args.equity_csv:
            Path(args.equity_csv).parent.mkdir(parents=True, exist_ok=True)
            df_eq.to_csv(args.equity_csv, index=False)
        if args.equity_png:
            try:
                import matplotlib.pyplot as plt
            except Exception as e:
                raise RuntimeError("matplotlib is required for --equity-png") from e
            plt.figure(figsize=(10, 4))
            plt.plot(df_eq["timestamp"], df_eq["equity"], lw=1.0)
            plt.title(f"Equity Curve: {args.model_name or Path(args.model).stem}")
            plt.xlabel("Time (UTC)")
            plt.ylabel("Equity")
            plt.tight_layout()
            Path(args.equity_png).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(args.equity_png, dpi=150)
            plt.close()


if __name__ == "__main__":
    main()
