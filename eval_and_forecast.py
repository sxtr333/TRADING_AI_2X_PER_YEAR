"""
Evaluate the trained price model and produce a naive 24h projection with an HTML plot.

- Uses the quantile price-only model by default (model_1h_q.keras).
- Forecast beyond +1h is naive: repeats the next-hour predicted log-return for 24 steps.
- Outputs: CSV with forecast points and an HTML (self-contained, no external deps) showing a simple SVG line chart.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from model_layers import RevIN, TSMixerBlock, ITransformerBlock, LastStep  # noqa: F401
from trading_keras_core import DropPath, default_feature_list


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>24h Forecast</title>
  <style>
    body { font-family: sans-serif; margin: 24px; }
    h1 { margin-top: 0; }
    #chart { width: 100%; max-width: 900px; height: 400px; border: 1px solid #ddd; }
    table { border-collapse: collapse; margin-top: 16px; }
    th, td { border: 1px solid #ccc; padding: 6px 10px; }
  </style>
</head>
<body>
  <h1>Naive 24h Projection</h1>
  <p>Model: {{model}}</p>
  <p>Last close: {{last_close}}</p>
  <p>Predicted next-hour log-return: {{pred_log}}</p>
  <div id="chart"></div>
  <table>
    <thead><tr><th>Hour ahead</th><th>Pred close</th></tr></thead>
    <tbody id="rows"></tbody>
  </table>
<script>
const data = {{data_json}};
const chart = document.getElementById('chart');
// compute min/max
let minY = Math.min(...data.map(d=>d.pred_close));
let maxY = Math.max(...data.map(d=>d.pred_close));
const pad = (maxY - minY) * 0.05 || 1;
minY -= pad; maxY += pad;
const w = chart.clientWidth || 900;
const h = chart.clientHeight || 400;
function yScale(v){ return h - (v - minY) / (maxY - minY) * h; }
function xScale(i){ return i / (data.length - 1) * w; }
let path = "";
data.forEach((d,i)=>{
  const x = xScale(i); const y = yScale(d.pred_close);
  path += (i===0 ? "M" : "L") + x.toFixed(2) + " " + y.toFixed(2) + " ";
});
chart.innerHTML = `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">
  <rect x="0" y="0" width="${w}" height="${h}" fill="white" stroke="#e0e0e0"/>
  <path d="${path}" fill="none" stroke="#0074D9" stroke-width="2"/>
</svg>`;
const rows = document.getElementById('rows');
rows.innerHTML = data.map(d=>`<tr><td>${d.hour_ahead}</td><td>${d.pred_close.toFixed(2)}</td></tr>`).join('');
</script>
</body>
</html>
"""


def load_stats(stats_path: Path) -> dict | None:
    if not stats_path:
        return None
    data = np.load(stats_path, allow_pickle=True)
    stats = {k: data[k] for k in data.files}
    if "mean" in stats:
        stats["mean"] = stats["mean"].astype(np.float32)
    if "std" in stats:
        stats["std"] = stats["std"].astype(np.float32)
    return stats


def load_window(features_path: Path, seq_len: int, stats: dict | None):
    df = pd.read_parquet(features_path)
    if len(df) < seq_len:
        raise ValueError("Not enough rows for the requested sequence length.")
    dropped = []
    if stats and "mean" in stats:
        stats_len = int(stats["mean"].shape[0])
        # prefer feature_names from stats if available
        if "feature_names" in stats:
            candidate = [str(x) for x in stats["feature_names"]]
        else:
            candidate = [
                c for c in df.columns
                if df[c].dtype.kind in "ifb"
                and not c.startswith("target_")
                and c not in ("timestamp", "label_3cls", "tb_label", "tb_tth")
            ]
        if len(candidate) < stats_len:
            raise ValueError(f"Not enough numeric feature columns ({len(candidate)}) for stats_len {stats_len}")
        # drop constant columns only if we still have enough features
        const_cols = [c for c in candidate if df[c].nunique(dropna=True) <= 1]
        if len(candidate) - len(const_cols) >= stats_len:
            feature_cols = [c for c in candidate if c not in const_cols]
            dropped.extend(const_cols)
        else:
            feature_cols = candidate
        if len(feature_cols) > stats_len:
            dropped.extend(feature_cols[stats_len:])
            feature_cols = feature_cols[:stats_len]
        if len(feature_cols) != stats_len:
            raise ValueError(f"Feature count mismatch after trimming: {len(feature_cols)} vs stats_len {stats_len}")
    else:
        feature_cols = default_feature_list()

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        for c in missing:
            df[c] = 0.0
    window = df.iloc[-seq_len:]
    last_close = float(df.iloc[-1]["close"])
    return window, feature_cols, last_close, missing + dropped


def predict_next_log(model_path: Path, ckpt_path: Path | None, window: pd.DataFrame, feature_cols, seq_len: int, stats: dict | None):
    x = window[feature_cols].to_numpy(dtype=np.float32)
    if stats and "mean" in stats and "std" in stats:
        mean = stats["mean"]
        std = stats["std"]
        std = np.where(std == 0, 1.0, std)
        x = (x - mean) / std
    x = x[None, ...]
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "RevIN": RevIN,
            "TSMixerBlock": TSMixerBlock,
            "ITransformerBlock": ITransformerBlock,
            "LastStep": LastStep,
            "DropPath": DropPath,
        },
        compile=False,
        safe_mode=False,
    )
    if ckpt_path:
        try:
            model.load_weights(str(ckpt_path))
        except Exception:
            pass
    pred = model.predict(x, verbose=0)
    if isinstance(pred, (list, tuple)):
        pred = pred[0]
    pred = np.asarray(pred)
    if pred.ndim == 2 and pred.shape[1] > 1:
        q = stats.get("quantiles", None) if stats else None
        if q is not None and len(q) > 0:
            q = np.asarray(q, dtype=np.float32)
            idx = int(np.argmin(np.abs(q - 0.5)))
        else:
            idx = pred.shape[1] // 2
        return float(pred[0, idx])
    return float(np.squeeze(pred))


def build_projection(last_close: float, pred_log: float, hours: int = 24):
    hrs = np.arange(1, hours + 1)
    prices = last_close * np.exp(pred_log * hrs)
    return hrs.tolist(), prices.tolist()


def write_outputs(hours, prices, out_csv: Path, out_html: Path, model_name: str, last_close: float, pred_log: float):
    df = pd.DataFrame({"hour_ahead": hours, "pred_close": prices})
    df.to_csv(out_csv, index=False)
    data_json = json.dumps(df.to_dict(orient="records"))
    html = (
        HTML_TEMPLATE.replace("{{data_json}}", data_json)
        .replace("{{model}}", model_name)
        .replace("{{last_close}}", f"{last_close:.2f}")
        .replace("{{pred_log}}", f"{pred_log:.6f}")
    )
    out_html.write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/BTCUSDT_1h_features.parquet")
    parser.add_argument("--model", default="model_1h_q.keras")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--stats", default=None, help="Optional npz with mean/std and feature_names")
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--out-csv", default="forecast_24h.csv")
    parser.add_argument("--out-html", default="forecast_24h.html")
    args = parser.parse_args()

    stats = load_stats(Path(args.stats)) if args.stats else None
    window, feature_cols, last_close, missing = load_window(Path(args.features), args.seq_len, stats)
    if stats and "mean" in stats and "std" in stats:
        if stats["mean"].shape[0] != len(feature_cols):
            print(
                {
                    "warning": "stats feature length mismatch; skipping normalization",
                    "stats_len": int(stats["mean"].shape[0]),
                    "feature_len": int(len(feature_cols)),
                }
            )
            stats = None
    pred_log = predict_next_log(Path(args.model), Path(args.weights) if args.weights else None, window, feature_cols, args.seq_len, stats)
    hours, prices = build_projection(last_close, pred_log, args.hours)
    write_outputs(hours, prices, Path(args.out_csv), Path(args.out_html), args.model, last_close, pred_log)

    print(
        {
            "pred_next_log_return": pred_log,
            "last_close": last_close,
            "pred_close_1h": prices[0],
            "pred_close_24h": prices[-1],
            "out_csv": args.out_csv,
            "out_html": args.out_html,
            "missing_features_filled": missing,
        }
    )


if __name__ == "__main__":
    main()
