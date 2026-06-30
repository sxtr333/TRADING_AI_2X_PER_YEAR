#!/usr/bin/env python3
"""Label extracted signals against OHLCV candles.

Expected signal columns: timestamp_utc, direction, entry_min, entry_max, stop, tp1
Expected OHLCV columns: timestamp, open, high, low, close
"""

import argparse
import json
from pathlib import Path
from typing import Optional

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
        raise ValueError(f"Cannot parse timestamps from column '{col_name}'")
    return ts


def _entry_hit(direction: str, low: float, high: float, entry_low: float, entry_high: float) -> bool:
    # Entry zone is always [entry_low, entry_high], irrespective of side.
    return low <= entry_high and high >= entry_low


def _label_one(
    direction: str,
    entry_mid: float,
    stop: float,
    tp_levels: list,
    low_arr: np.ndarray,
    high_arr: np.ndarray,
    start_idx: int,
    max_hold: int,
    same_bar_policy: str,
) -> tuple:
    end_idx = min(len(low_arr), start_idx + max_hold)

    for i in range(start_idx, end_idx):
        low = float(low_arr[i])
        high = float(high_arr[i])

        if direction == "long":
            stop_hit = low <= stop
            tp_hit_idx = None
            for k, tp in enumerate(tp_levels):
                if high >= tp:
                    tp_hit_idx = k
                    break
        else:
            stop_hit = high >= stop
            tp_hit_idx = None
            for k, tp in enumerate(tp_levels):
                if low <= tp:
                    tp_hit_idx = k
                    break

        if stop_hit and tp_hit_idx is not None:
            if same_bar_policy == "tp_first":
                stop_hit = False
            else:
                tp_hit_idx = None

        if tp_hit_idx is not None:
            tp_value = float(tp_levels[tp_hit_idx])
            risk = abs(entry_mid - stop)
            if risk <= 0:
                rr = np.nan
            elif direction == "long":
                rr = (tp_value - entry_mid) / risk
            else:
                rr = (entry_mid - tp_value) / risk
            return "win", i, tp_hit_idx + 1, rr

        if stop_hit:
            return "loss", i, 0, -1.0

    return "open", end_idx - 1, 0, np.nan


def main() -> None:
    ap = argparse.ArgumentParser(description="Label signal outcomes with OHLCV history")
    ap.add_argument("--signals", required=True, help="signals_raw.parquet/csv from extract script")
    ap.add_argument("--ohlcv", required=True, help="OHLCV table with timestamp/open/high/low/close")
    ap.add_argument("--output", default="data/telegram/signals_labeled.parquet")
    ap.add_argument("--summary-json", default="reports/signal_label_quality.json")
    ap.add_argument("--max-wait-entry-bars", type=int, default=48)
    ap.add_argument("--max-hold-bars", type=int, default=192)
    ap.add_argument("--same-bar-policy", choices=["stop_first", "tp_first"], default="stop_first")
    args = ap.parse_args()

    sig_path = Path(args.signals)
    ohlcv_path = Path(args.ohlcv)

    sig = _read_table(sig_path)
    ohlcv = _read_table(ohlcv_path)

    if "timestamp_utc" not in sig.columns:
        raise ValueError("Signals must contain timestamp_utc")

    needed_ohlcv = {"timestamp", "open", "high", "low", "close"}
    missing = needed_ohlcv - set(ohlcv.columns)
    if missing:
        raise ValueError(f"OHLCV is missing columns: {sorted(missing)}")

    sig["timestamp_utc"] = _parse_ts(sig["timestamp_utc"], "timestamp_utc")
    ohlcv["timestamp"] = _parse_ts(ohlcv["timestamp"], "timestamp")

    sig = sig.sort_values("timestamp_utc").reset_index(drop=True)
    ohlcv = ohlcv.sort_values("timestamp").reset_index(drop=True)

    ts_arr = ohlcv["timestamp"].to_numpy()
    low_arr = ohlcv["low"].to_numpy(dtype=np.float64)
    high_arr = ohlcv["high"].to_numpy(dtype=np.float64)

    outcomes = []

    def _outcome_stub(name: str):
        return (name, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, pd.NaT)

    for row in sig.itertuples(index=False):
        direction = (getattr(row, "direction", None) or "").lower().strip()
        entry_min = getattr(row, "entry_min", np.nan)
        entry_max = getattr(row, "entry_max", np.nan)
        stop = getattr(row, "stop", np.nan)
        tp1 = getattr(row, "tp1", np.nan)
        tp2 = getattr(row, "tp2", np.nan)
        tp3 = getattr(row, "tp3", np.nan)

        if direction not in {"long", "short"}:
            outcomes.append(_outcome_stub("invalid"))
            continue

        if pd.isna(entry_min) and pd.notna(entry_max):
            entry_min = entry_max
        if pd.isna(entry_max) and pd.notna(entry_min):
            entry_max = entry_min

        if pd.isna(entry_min) or pd.isna(entry_max) or pd.isna(stop) or pd.isna(tp1):
            outcomes.append(_outcome_stub("insufficient"))
            continue

        entry_low = float(min(entry_min, entry_max))
        entry_high = float(max(entry_min, entry_max))
        entry_mid = (entry_low + entry_high) / 2.0
        stop = float(stop)

        tp_levels = [x for x in [tp1, tp2, tp3] if pd.notna(x)]
        tp_levels = [float(x) for x in tp_levels]

        if direction == "long":
            tp_levels = sorted(tp_levels)
        else:
            tp_levels = sorted(tp_levels, reverse=True)

        start_idx = int(np.searchsorted(ts_arr, np.datetime64(row.timestamp_utc.to_datetime64()), side="left"))

        if start_idx >= len(ts_arr):
            outcomes.append(_outcome_stub("out_of_range"))
            continue

        # find first entry touch
        entry_idx: Optional[int] = None
        entry_search_end = min(len(ts_arr), start_idx + int(args.max_wait_entry_bars))
        for i in range(start_idx, entry_search_end):
            if _entry_hit(direction, low_arr[i], high_arr[i], entry_low, entry_high):
                entry_idx = i
                break

        if entry_idx is None:
            outcomes.append(_outcome_stub("no_entry"))
            continue

        outcome, exit_idx, hit_tp, rr = _label_one(
            direction=direction,
            entry_mid=entry_mid,
            stop=stop,
            tp_levels=tp_levels,
            low_arr=low_arr,
            high_arr=high_arr,
            start_idx=entry_idx,
            max_hold=int(args.max_hold_bars),
            same_bar_policy=args.same_bar_policy,
        )

        bars_to_entry = int(entry_idx - start_idx)
        bars_in_trade = int(exit_idx - entry_idx) if pd.notna(exit_idx) else np.nan
        exit_ts = ohlcv.iloc[int(exit_idx)]["timestamp"] if pd.notna(exit_idx) else pd.NaT

        outcomes.append((outcome, entry_idx, exit_idx, bars_to_entry, bars_in_trade, hit_tp, rr, exit_ts))

    out_cols = [
        "outcome",
        "entry_bar_idx",
        "exit_bar_idx",
        "bars_to_entry",
        "bars_in_trade",
        "hit_tp",
        "realized_rr",
        "exit_timestamp_utc",
    ]
    out_df = pd.DataFrame(outcomes, columns=out_cols)

    merged = pd.concat([sig.reset_index(drop=True), out_df], axis=1)

    saved = _write_table(merged, Path(args.output))

    summary = {
        "signals_rows": int(len(sig)),
        "labeled_rows": int(len(merged)),
        "outcome_counts": merged["outcome"].value_counts(dropna=False).to_dict(),
        "win_rate_on_closed": float(
            (merged["outcome"] == "win").sum()
            / max(1, ((merged["outcome"] == "win") | (merged["outcome"] == "loss")).sum())
        ),
        "avg_rr_closed": float(
            merged.loc[merged["outcome"].isin(["win", "loss"]), "realized_rr"].mean()
            if not merged.empty
            else np.nan
        ),
        "output_file": str(saved),
    }

    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[label] rows={len(merged)} output={saved}")
    print(f"[label] summary={summary_path}")


if __name__ == "__main__":
    main()
