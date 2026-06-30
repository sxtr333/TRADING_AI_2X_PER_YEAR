#!/usr/bin/env python3
"""Approximate signal outcome by mapping direction to BTC candles.

This script intentionally ignores explicit entry/stop/tp levels from posts.
It maps each signal timestamp to the nearest future BTCUSDT 15m candle and
estimates directional quality from raw market movement over fixed horizons.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write_table(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
        return path
    try:
        df.to_parquet(path, index=False)
        return path
    except Exception:
        fallback = path.with_suffix(".csv")
        df.to_csv(fallback, index=False)
        return fallback


def _parse_ts(series: pd.Series, col_name: str) -> pd.Series:
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    if ts.isna().all():
        raise ValueError(f"cannot parse timestamps from '{col_name}'")
    return ts


def _safe_pct(a: float, b: float) -> float:
    if not np.isfinite(a) or abs(a) < 1e-12 or not np.isfinite(b):
        return np.nan
    return (b / a - 1.0) * 100.0


def _reason_for_bad(max_fav: float, max_adv: float, signed_ret: float, min_move: float) -> str:
    if not np.isfinite(signed_ret):
        return "insufficient_data"
    if max_fav < min_move:
        return "no_follow_through"
    if max_adv >= max_fav:
        return "wrong_direction_or_deep_drawdown"
    if signed_ret <= 0:
        return "gave_back_profit"
    return "good"


def main() -> None:
    ap = argparse.ArgumentParser(description="Map long/short signals to BTC move percentages")
    ap.add_argument("--signals", required=True, help="signals parquet/csv with timestamp_utc,direction")
    ap.add_argument("--btc-ohlcv", required=True, help="BTC OHLCV parquet/csv with timestamp,open,high,low,close")
    ap.add_argument("--horizons-hours", default="24,72,168", help="comma-separated horizons in hours")
    ap.add_argument("--bar-minutes", type=int, default=15, help="bar size in minutes for horizon conversion")
    ap.add_argument("--min-move-pct", type=float, default=0.35, help="minimum favorable move to avoid 'no_follow_through'")
    ap.add_argument("--output", default="data/telegram/signals_directional_btc_move.parquet")
    ap.add_argument("--summary-json", default="reports/signals_directional_btc_move_summary.json")
    args = ap.parse_args()

    sig = _read_table(Path(args.signals)).copy()
    btc = _read_table(Path(args.btc_ohlcv)).copy()

    if "timestamp_utc" not in sig.columns:
        raise ValueError("signals must have 'timestamp_utc'")
    if "direction" not in sig.columns:
        raise ValueError("signals must have 'direction'")
    need = {"timestamp", "high", "low", "close"}
    miss = need - set(btc.columns)
    if miss:
        raise ValueError(f"btc ohlcv missing columns: {sorted(miss)}")

    sig["timestamp_utc"] = _parse_ts(sig["timestamp_utc"], "timestamp_utc")
    btc["timestamp"] = _parse_ts(btc["timestamp"], "timestamp")

    sig = sig.sort_values("timestamp_utc").reset_index(drop=True)
    btc = btc.sort_values("timestamp").reset_index(drop=True)

    horizons: List[int] = []
    for x in str(args.horizons_hours).split(","):
        x = x.strip()
        if not x:
            continue
        horizons.append(int(x))
    horizons = sorted(set(h for h in horizons if h > 0))
    if not horizons:
        raise ValueError("at least one positive horizon is required")

    bars_per_hour = max(1, int(round(60 / int(args.bar_minutes))))
    horizon_bars: Dict[int, int] = {h: h * bars_per_hour for h in horizons}

    ts_arr = btc["timestamp"].to_numpy()
    ts_ns = btc["timestamp"].astype("int64").to_numpy()
    hi_arr = btc["high"].to_numpy(dtype=np.float64)
    lo_arr = btc["low"].to_numpy(dtype=np.float64)
    cl_arr = btc["close"].to_numpy(dtype=np.float64)

    res = sig.copy()
    res["direction_norm"] = res["direction"].astype(str).str.lower().str.strip()
    res["anchor_idx"] = -1
    res["anchor_timestamp_utc"] = pd.Series([pd.NaT] * len(res), dtype="datetime64[ns, UTC]")
    res["anchor_price"] = np.nan
    res["bars_to_anchor"] = np.nan
    res["direction_move_status"] = "unknown"

    for h in horizons:
        res[f"signed_ret_{h}h_pct"] = np.nan
        res[f"raw_ret_{h}h_pct"] = np.nan
        res[f"max_favorable_{h}h_pct"] = np.nan
        res[f"max_adverse_{h}h_pct"] = np.nan
        res[f"is_profitable_{h}h"] = 0
        res[f"error_reason_{h}h"] = "insufficient_data"

    for i, row in res.iterrows():
        d = row["direction_norm"]
        if d not in {"long", "short"}:
            res.at[i, "direction_move_status"] = "invalid_direction"
            continue
        t = row["timestamp_utc"]
        if pd.isna(t):
            res.at[i, "direction_move_status"] = "invalid_time"
            continue

        t_ns = int(pd.Timestamp(t).value)
        if t_ns < int(ts_ns[0]) or t_ns > int(ts_ns[-1]):
            res.at[i, "direction_move_status"] = "out_of_range"
            continue

        start_idx = int(np.searchsorted(ts_ns, t_ns, side="left"))
        if start_idx >= len(ts_arr):
            res.at[i, "direction_move_status"] = "out_of_range"
            continue

        entry = float(cl_arr[start_idx])
        if not np.isfinite(entry) or entry <= 0:
            res.at[i, "direction_move_status"] = "bad_anchor_price"
            continue

        res.at[i, "anchor_idx"] = start_idx
        res.at[i, "anchor_timestamp_utc"] = btc.iloc[start_idx]["timestamp"]
        res.at[i, "anchor_price"] = entry
        delta_min = (int(ts_ns[start_idx]) - t_ns) / 60_000_000_000.0
        res.at[i, "bars_to_anchor"] = max(0.0, delta_min / float(args.bar_minutes))

        row_ok = True
        for h in horizons:
            nbar = horizon_bars[h]
            end_idx = min(len(cl_arr) - 1, start_idx + nbar)
            if end_idx <= start_idx:
                row_ok = False
                continue
            wnd_hi = float(np.nanmax(hi_arr[start_idx : end_idx + 1]))
            wnd_lo = float(np.nanmin(lo_arr[start_idx : end_idx + 1]))
            end_close = float(cl_arr[end_idx])

            raw_ret = _safe_pct(entry, end_close)
            if d == "long":
                signed_ret = raw_ret
                max_fav = _safe_pct(entry, wnd_hi)
                max_adv = _safe_pct(wnd_lo, entry)  # drawdown magnitude
            else:
                signed_ret = _safe_pct(end_close, entry)
                max_fav = _safe_pct(wnd_lo, entry)
                max_adv = _safe_pct(entry, wnd_hi)

            if np.isfinite(max_fav):
                max_fav = max(0.0, max_fav)
            if np.isfinite(max_adv):
                max_adv = max(0.0, max_adv)

            res.at[i, f"signed_ret_{h}h_pct"] = signed_ret
            res.at[i, f"raw_ret_{h}h_pct"] = raw_ret
            res.at[i, f"max_favorable_{h}h_pct"] = max_fav
            res.at[i, f"max_adverse_{h}h_pct"] = max_adv
            res.at[i, f"is_profitable_{h}h"] = int(np.isfinite(signed_ret) and signed_ret > 0.0)
            res.at[i, f"error_reason_{h}h"] = _reason_for_bad(
                max_fav=max_fav if np.isfinite(max_fav) else np.nan,
                max_adv=max_adv if np.isfinite(max_adv) else np.nan,
                signed_ret=signed_ret if np.isfinite(signed_ret) else np.nan,
                min_move=float(args.min_move_pct),
            )

        if row_ok:
            res.at[i, "direction_move_status"] = "ok"
        else:
            res.at[i, "direction_move_status"] = "partial"

    out_path = _write_table(res, Path(args.output))

    status_counts = res["direction_move_status"].value_counts(dropna=False).to_dict()
    summary: Dict[str, object] = {
        "signals_rows": int(len(res)),
        "status_counts": status_counts,
        "horizons_hours": horizons,
        "bar_minutes": int(args.bar_minutes),
        "min_move_pct": float(args.min_move_pct),
        "output_file": str(out_path),
    }
    for h in horizons:
        k_prof = f"is_profitable_{h}h"
        k_signed = f"signed_ret_{h}h_pct"
        k_reason = f"error_reason_{h}h"
        m = res[k_signed].notna()
        summary[f"h{h}_rows_with_metric"] = int(m.sum())
        summary[f"h{h}_profitable_rate"] = float(res.loc[m, k_prof].mean()) if m.any() else np.nan
        summary[f"h{h}_avg_signed_ret_pct"] = float(res.loc[m, k_signed].mean()) if m.any() else np.nan
        summary[f"h{h}_error_reason_counts"] = res.loc[m, k_reason].value_counts(dropna=False).to_dict()

    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] saved: {out_path}")
    print(f"[ok] summary: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
