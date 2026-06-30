#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Forecast next 24h from a trained iTransformer (15m) model.
Notes:
- Uses the last seq_len window from features parquet.
- Produces a naive 24h curve by repeating the next-step log return for each step.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from model_layers import RevIN, TSMixerBlock, ITransformerBlock, LastStep  # noqa: F401
from trading_keras_core import DropPath, default_feature_list


def load_window(features_path: Path, seq_len: int, stats_path: Path | None) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_parquet(features_path).dropna().sort_values("timestamp")
    feature_cols = None
    if stats_path:
        try:
            stats = np.load(stats_path, allow_pickle=True)
            if "feature_names" in stats:
                feature_cols = [str(x) for x in stats["feature_names"]]
        except Exception:
            feature_cols = None
    if not feature_cols:
        feature_cols = default_feature_list()
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    window = df.tail(seq_len)
    if len(window) < seq_len:
        raise ValueError(f"Not enough rows: have {len(window)}, need {seq_len}")
    return window, feature_cols


def normalize(x: np.ndarray, stats_path: Path | None) -> np.ndarray:
    if not stats_path:
        return x
    stats = np.load(stats_path)
    mean = stats["mean"]
    std = stats["std"]
    std = np.where(std == 0, 1.0, std)
    if mean.shape[0] != x.shape[-1]:
        raise ValueError(f"Stats feature length mismatch: {mean.shape[0]} vs {x.shape[-1]}")
    return (x - mean) / std


def load_model(model_path: Path) -> tf.keras.Model:
    return tf.keras.models.load_model(
        model_path,
        custom_objects={
            "RevIN": RevIN,
            "TSMixerBlock": TSMixerBlock,
            "ITransformerBlock": ITransformerBlock,
            "LastStep": LastStep,
            "DropPath": DropPath,
        },
        compile=False,
    )


def predict_next_log_return(model: tf.keras.Model, window: pd.DataFrame, feature_cols: list[str], stats: Path | None) -> float:
    x = window[feature_cols].to_numpy(dtype=np.float32)[None, ...]
    x = normalize(x, stats)
    pred = model.predict(x, verbose=0)
    if isinstance(pred, (list, tuple)):
        pred = pred[0]
    pred = np.asarray(pred)
    if pred.ndim == 2 and pred.shape[1] > 1:
        # quantile head -> take median (closest to 0.5)
        q = None
        if stats is not None:
            try:
                q = np.load(stats, allow_pickle=True).get("quantiles", None)
            except Exception:
                q = None
        if q is not None and len(q) > 0:
            q = np.asarray(q, dtype=np.float32)
            idx = int(np.argmin(np.abs(q - 0.5)))
        else:
            idx = pred.shape[1] // 2
        return float(pred[0, idx])
    return float(np.squeeze(pred))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="Features parquet (15m)")
    ap.add_argument("--model", required=True, help="Model .keras")
    ap.add_argument("--stats", help="Normalization stats .npz")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--step-min", type=int, default=15)
    ap.add_argument("--out-csv", default="forecast_24h.csv")
    ap.add_argument("--out-html", default="forecast_24h.html")
    ap.add_argument("--max-abs-logret", type=float, default=0.05, help="Clamp per-step log-return to avoid blow-ups; set 0 to disable")
    ap.add_argument("--ema-alpha", type=float, default=0.7, help="EMA smoothing factor for predicted log-return")
    ap.add_argument("--out-png", default="forecast_24h.png")
    ap.add_argument("--range", action="store_true", help="Output uncertainty bands using recent realized volatility")
    ap.add_argument("--rv-window", type=int, default=96, help="Steps for realized volatility (e.g., 96=24h for 15m)")
    ap.add_argument("--max-drift", type=float, default=0.0, help="Clamp total 24h drift in percent (e.g., 0.05 for ±5%). 0 disables.")
    ap.add_argument("--revert", type=float, default=0.0, help="Mean-reversion factor [0..1] applied to median path")
    ap.add_argument("--scenarios", type=int, default=0, help="Number of stochastic scenarios to generate (0 disables)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--noise-mode", choices=["gaussian", "bootstrap"], default="bootstrap")
    ap.add_argument("--noise-scale", type=float, default=1.0, help="Scale factor for noise")
    args = ap.parse_args()

    stats_path = Path(args.stats) if args.stats else None
    window, feature_cols = load_window(Path(args.features), args.seq_len, stats_path)
    last_ts = pd.to_datetime(window["timestamp"].iloc[-1], utc=True)
    last_close = float(window["close"].iloc[-1])

    model = load_model(Path(args.model))
    steps = int((args.hours * 60) // args.step_min)
    times = [last_ts + pd.Timedelta(minutes=args.step_min * (i + 1)) for i in range(steps)]
    closes = []
    price = last_close
    prev_close = last_close
    ema_log = 0.0
    mu_logs = []
    for i in range(steps):
        pred_log = predict_next_log_return(model, window, feature_cols, stats_path)
        ema_log = args.ema_alpha * ema_log + (1.0 - args.ema_alpha) * pred_log
        pred_log = float(ema_log)
        if args.max_abs_logret > 0:
            pred_log = float(np.clip(pred_log, -args.max_abs_logret, args.max_abs_logret))
        mu_logs.append(pred_log)
        price = float(price * np.exp(pred_log))
        if args.revert > 0:
            price = float(price * (1.0 - args.revert) + last_close * args.revert)
        if args.max_drift > 0:
            min_p = last_close * (1.0 - args.max_drift)
            max_p = last_close * (1.0 + args.max_drift)
            price = float(np.clip(price, min_p, max_p))
        closes.append(price)

        # Build a new row by copying last row and updating price-only fields
        last_row = window.iloc[-1].copy()
        last_row["open"] = prev_close
        last_row["high"] = max(prev_close, price)
        last_row["low"] = min(prev_close, price)
        last_row["close"] = price
        last_row["vwap"] = price
        last_row["close_delta"] = price - prev_close
        last_row["log_return_1m"] = float(np.log(price) - np.log(prev_close))
        last_row["range_norm"] = (last_row["high"] - last_row["low"]) / max(price, 1e-8)
        last_row["wick_up"] = (last_row["high"] - max(last_row["open"], price)) / max(price, 1e-8)
        last_row["wick_down"] = (min(last_row["open"], price) - last_row["low"]) / max(price, 1e-8)
        last_row["volume_delta"] = 0.0

        prev_close = price
        window = pd.concat([window.iloc[1:], last_row.to_frame().T], ignore_index=True)

        # Recompute rolling indicators on the updated window for realism
        close_series = window["close"].astype(float)
        log_ret = np.log(close_series).diff()
        # Bollinger (20)
        ma = close_series.rolling(20).mean()
        sd = close_series.rolling(20).std()
        window.loc[window.index[-1], "bollinger_upper"] = (ma + 2 * sd).iloc[-1]
        window.loc[window.index[-1], "bollinger_lower"] = (ma - 2 * sd).iloc[-1]
        window.loc[window.index[-1], "bollinger_bandwidth"] = ((2 * sd) / ma).iloc[-1]
        # ATR (14)
        high = window["high"].astype(float)
        low = window["low"].astype(float)
        prev_c = close_series.shift(1)
        tr = pd.concat([(high - low).abs(), (high - prev_c).abs(), (low - prev_c).abs()], axis=1).max(axis=1)
        window.loc[window.index[-1], "atr"] = tr.rolling(14).mean().iloc[-1]
        # RV (short/long)
        window.loc[window.index[-1], "rv"] = log_ret.rolling(30).std().iloc[-1]
        window.loc[window.index[-1], "rv_short"] = log_ret.rolling(10).std().iloc[-1]
        window.loc[window.index[-1], "rv_long"] = log_ret.rolling(60).std().iloc[-1]
        rv_s = window.loc[window.index[-1], "rv_short"]
        rv_l = window.loc[window.index[-1], "rv_long"]
        window.loc[window.index[-1], "rv_ratio"] = float(rv_s / rv_l) if rv_l not in (0, np.nan) else 0.0

    out_df = pd.DataFrame({"timestamp": times, "pred_close": closes})
    if args.range:
        close_series = window["close"].astype(float)
        log_ret_hist = np.log(close_series).diff().dropna()
        if len(log_ret_hist) >= 2:
            sigma = float(log_ret_hist.tail(args.rv_window).std())
        else:
            sigma = 0.0
        if sigma <= 0:
            sigma = 1e-6

        cum_mu = np.cumsum(mu_logs)
        k = np.arange(1, len(cum_mu) + 1, dtype=float)
        vol = sigma * np.sqrt(k)

        out_df["upper_1s"] = last_close * np.exp(cum_mu + 1.0 * vol)
        out_df["lower_1s"] = last_close * np.exp(cum_mu - 1.0 * vol)
        out_df["upper_2s"] = last_close * np.exp(cum_mu + 2.0 * vol)
        out_df["lower_2s"] = last_close * np.exp(cum_mu - 2.0 * vol)

    if args.scenarios and args.range:
        rng = np.random.default_rng(args.seed)
        hist = log_ret_hist.tail(args.rv_window).to_numpy()
        for s in range(args.scenarios):
            if args.noise_mode == "bootstrap" and len(hist) > 0:
                noise = rng.choice(hist, size=len(mu_logs), replace=True)
            else:
                noise = rng.standard_normal(size=len(mu_logs)) * sigma
            noise = noise * args.noise_scale
            scen_log = np.cumsum(mu_logs + noise)
            scen_price = last_close * np.exp(scen_log)
            if args.max_drift > 0:
                min_p = last_close * (1.0 - args.max_drift)
                max_p = last_close * (1.0 + args.max_drift)
                scen_price = np.clip(scen_price, min_p, max_p)
            out_df[f"scenario_{s+1}"] = scen_price
    out_df.to_csv(args.out_csv, index=False)

    # Simple HTML line chart (self-contained SVG)
    y = np.array(closes, dtype=float)
    y_min, y_max = float(y.min()), float(y.max())
    if args.range:
        y_min = float(min(y_min, out_df["lower_2s"].min()))
        y_max = float(max(y_max, out_df["upper_2s"].max()))
    if y_max == y_min:
        y_max = y_min + 1.0
    w, h, pad = 960, 360, 40
    pts = []
    for i, v in enumerate(y):
        px = pad + (w - 2 * pad) * (i / max(len(y) - 1, 1))
        py = pad + (h - 2 * pad) * (1 - (v - y_min) / (y_max - y_min))
        pts.append(f"{px:.1f},{py:.1f}")
    svg = f"""
<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">
  <rect x="0" y="0" width="{w}" height="{h}" fill="#fff"/>
  <polyline fill="none" stroke="#2563eb" stroke-width="2" points="{' '.join(pts)}"/>
  <text x="{pad}" y="{pad-10}" font-size="14" font-family="Arial">iTransformer 24h forecast (15m step)</text>
  <text x="{pad}" y="{h-10}" font-size="12" font-family="Arial">min={y_min:.2f} max={y_max:.2f}</text>
</svg>
"""

    Path(args.out_html).write_text(svg, encoding="utf-8")

    # PNG plot (if matplotlib available)
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 4))
        if args.range:
            plt.fill_between(times, out_df["lower_2s"], out_df["upper_2s"], color="#93c5fd", alpha=0.3, label="±2σ")
            plt.fill_between(times, out_df["lower_1s"], out_df["upper_1s"], color="#60a5fa", alpha=0.4, label="±1σ")
            if args.scenarios:
                for s in range(args.scenarios):
                    plt.plot(times, out_df[f"scenario_{s+1}"], color="#94a3b8", linewidth=0.7, alpha=0.5)
        plt.plot(times, closes, color="#2563eb", linewidth=2, label="Median")
        plt.title("iTransformer 24h forecast (15m step)")
        plt.xlabel("Time (UTC)")
        plt.ylabel("Predicted close")
        if args.range:
            plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(args.out_png, dpi=150)
        plt.close()
    except Exception as exc:  # pragma: no cover
        print({"warning": "png_failed", "error": str(exc)})

    print(
        {
            "pred_next_log_return": pred_log,
            "last_close": last_close,
            "pred_close_15m": closes[0] if closes else None,
            "pred_close_24h": closes[-1] if closes else None,
            "out_csv": args.out_csv,
            "out_html": args.out_html,
            "out_png": args.out_png,
        }
    )


if __name__ == "__main__":
    main()
