#!/usr/bin/env python3
"""Build leak-safe crowd priors from directional BTC move labels.

Leak-safe rule:
- for horizon H, row at time T can use only rows with timestamp <= T - H.
- only historical aggregates are exposed (counts/rates/means), never future of current row.
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


def _as_float01(x: pd.Series) -> np.ndarray:
    return pd.to_numeric(x, errors="coerce").fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=np.float64)


def _as_float(x: pd.Series) -> np.ndarray:
    return pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build leak-safe crowd priors from directional labels")
    ap.add_argument("--directional", required=True, help="directional labels parquet/csv")
    ap.add_argument("--horizons-hours", default="24,72,168")
    ap.add_argument("--min-samples", type=int, default=5, help="min matured samples before trusting directional prior")
    ap.add_argument("--output", default="data/telegram/crowd_priors_strict_s3.parquet")
    ap.add_argument("--summary-json", default="reports/crowd_priors_strict_s3_summary.json")
    args = ap.parse_args()

    df = _read_table(Path(args.directional)).copy()
    need = {"message_id", "timestamp_utc", "direction"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"directional table missing required columns: {sorted(miss)}")

    horizons: List[int] = []
    for x in str(args.horizons_hours).split(","):
        x = x.strip()
        if x:
            horizons.append(int(x))
    horizons = sorted(set(h for h in horizons if h > 0))
    if not horizons:
        raise ValueError("at least one positive horizon required")

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df["direction_norm"] = df["direction"].astype(str).str.lower().str.strip()
    df = df[df["timestamp_utc"].notna()].copy()
    df = df[df["direction_norm"].isin(["long", "short"])].copy()
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    if df.empty:
        raise ValueError("no valid rows after timestamp/direction filtering")

    ts = df["timestamp_utc"].astype("int64").to_numpy()
    dir_arr = df["direction_norm"].to_numpy()

    out = pd.DataFrame(
        {
            "message_id": df["message_id"].astype(str),
            "timestamp_utc": df["timestamp_utc"].astype(str),
            "crowd_samples_prev_all": np.zeros(len(df), dtype=np.float32),
            "crowd_samples_prev_long": np.zeros(len(df), dtype=np.float32),
            "crowd_samples_prev_short": np.zeros(len(df), dtype=np.float32),
        }
    )

    for h in horizons:
        is_prof_col = f"is_profitable_{h}h"
        ret_col = f"signed_ret_{h}h_pct"
        err_col = f"error_reason_{h}h"
        miss_h = [c for c in [is_prof_col, ret_col, err_col] if c not in df.columns]
        if miss_h:
            raise ValueError(f"directional table missing horizon {h} columns: {miss_h}")

        dt_ns = int(pd.Timedelta(hours=h).value)
        mature_cut = ts - dt_ns

        ip = _as_float01(df[is_prof_col])  # 0/1
        sr = _as_float(df[ret_col])
        err = df[err_col].fillna("unknown").astype(str).str.lower().str.strip()
        e_wrong = (err == "wrong_direction_or_deep_drawdown").astype(np.float64).to_numpy()
        e_nofol = (err == "no_follow_through").astype(np.float64).to_numpy()
        e_giveb = (err == "gave_back_profit").astype(np.float64).to_numpy()

        ok_metric = np.isfinite(sr).astype(np.float64)
        sr0 = np.where(np.isfinite(sr), sr, 0.0)

        # Global cumulative stats.
        c_ok = np.cumsum(ok_metric)
        c_ip = np.cumsum(np.where(ok_metric > 0, ip, 0.0))
        c_sr = np.cumsum(np.where(ok_metric > 0, sr0, 0.0))
        c_wrong = np.cumsum(np.where(ok_metric > 0, e_wrong, 0.0))
        c_nofol = np.cumsum(np.where(ok_metric > 0, e_nofol, 0.0))
        c_giveb = np.cumsum(np.where(ok_metric > 0, e_giveb, 0.0))

        # Direction-specific cumulative stats.
        is_long = (dir_arr == "long").astype(np.float64)
        is_short = (dir_arr == "short").astype(np.float64)

        c_ok_long = np.cumsum(ok_metric * is_long)
        c_ok_short = np.cumsum(ok_metric * is_short)
        c_ip_long = np.cumsum(np.where(ok_metric > 0, ip * is_long, 0.0))
        c_ip_short = np.cumsum(np.where(ok_metric > 0, ip * is_short, 0.0))
        c_sr_long = np.cumsum(np.where(ok_metric > 0, sr0 * is_long, 0.0))
        c_sr_short = np.cumsum(np.where(ok_metric > 0, sr0 * is_short, 0.0))
        c_wrong_long = np.cumsum(np.where(ok_metric > 0, e_wrong * is_long, 0.0))
        c_wrong_short = np.cumsum(np.where(ok_metric > 0, e_wrong * is_short, 0.0))
        c_nofol_long = np.cumsum(np.where(ok_metric > 0, e_nofol * is_long, 0.0))
        c_nofol_short = np.cumsum(np.where(ok_metric > 0, e_nofol * is_short, 0.0))
        c_giveb_long = np.cumsum(np.where(ok_metric > 0, e_giveb * is_long, 0.0))
        c_giveb_short = np.cumsum(np.where(ok_metric > 0, e_giveb * is_short, 0.0))

        # For each row i, matured rows are indices <= j where ts[j] <= ts[i]-h.
        j = np.searchsorted(ts, mature_cut, side="right") - 1
        valid = j >= 0

        n_all = np.where(valid, c_ok[j], 0.0)
        n_long = np.where(valid, c_ok_long[j], 0.0)
        n_short = np.where(valid, c_ok_short[j], 0.0)

        out["crowd_samples_prev_all"] = np.maximum(out["crowd_samples_prev_all"], n_all.astype(np.float32))
        out["crowd_samples_prev_long"] = np.maximum(out["crowd_samples_prev_long"], n_long.astype(np.float32))
        out["crowd_samples_prev_short"] = np.maximum(out["crowd_samples_prev_short"], n_short.astype(np.float32))

        eps = 1e-9
        out[f"crowd_h{h}_winrate_all"] = (np.where(valid, c_ip[j], 0.0) / np.maximum(n_all, eps)).astype(np.float32)
        out[f"crowd_h{h}_avg_signed_ret_all"] = (
            np.where(valid, c_sr[j], 0.0) / np.maximum(n_all, eps)
        ).astype(np.float32)
        out[f"crowd_h{h}_err_wrong_all"] = (np.where(valid, c_wrong[j], 0.0) / np.maximum(n_all, eps)).astype(np.float32)
        out[f"crowd_h{h}_err_nofollow_all"] = (
            np.where(valid, c_nofol[j], 0.0) / np.maximum(n_all, eps)
        ).astype(np.float32)
        out[f"crowd_h{h}_err_giveback_all"] = (
            np.where(valid, c_giveb[j], 0.0) / np.maximum(n_all, eps)
        ).astype(np.float32)

        # Directional priors chosen by current direction.
        use_long = dir_arr == "long"
        n_dir = np.where(use_long, n_long, n_short)

        ip_dir_num = np.where(
            use_long,
            np.where(valid, c_ip_long[j], 0.0),
            np.where(valid, c_ip_short[j], 0.0),
        )
        sr_dir_num = np.where(
            use_long,
            np.where(valid, c_sr_long[j], 0.0),
            np.where(valid, c_sr_short[j], 0.0),
        )
        wrong_dir_num = np.where(
            use_long,
            np.where(valid, c_wrong_long[j], 0.0),
            np.where(valid, c_wrong_short[j], 0.0),
        )
        nofol_dir_num = np.where(
            use_long,
            np.where(valid, c_nofol_long[j], 0.0),
            np.where(valid, c_nofol_short[j], 0.0),
        )
        giveb_dir_num = np.where(
            use_long,
            np.where(valid, c_giveb_long[j], 0.0),
            np.where(valid, c_giveb_short[j], 0.0),
        )

        out[f"crowd_h{h}_samples_dir"] = n_dir.astype(np.float32)
        out[f"crowd_h{h}_winrate_dir"] = (ip_dir_num / np.maximum(n_dir, eps)).astype(np.float32)
        out[f"crowd_h{h}_avg_signed_ret_dir"] = (sr_dir_num / np.maximum(n_dir, eps)).astype(np.float32)
        out[f"crowd_h{h}_err_wrong_dir"] = (wrong_dir_num / np.maximum(n_dir, eps)).astype(np.float32)
        out[f"crowd_h{h}_err_nofollow_dir"] = (nofol_dir_num / np.maximum(n_dir, eps)).astype(np.float32)
        out[f"crowd_h{h}_err_giveback_dir"] = (giveb_dir_num / np.maximum(n_dir, eps)).astype(np.float32)
        out[f"crowd_h{h}_dir_prior_ready"] = (n_dir >= float(args.min_samples)).astype(np.float32)

    out_path = _write_table(out, Path(args.output))
    summary = {
        "rows": int(len(out)),
        "horizons_hours": horizons,
        "min_samples": int(args.min_samples),
        "output_file": str(out_path),
        "nonzero_samples_prev_all": int((out["crowd_samples_prev_all"] > 0).sum()),
    }
    for h in horizons:
        k = f"crowd_h{h}_dir_prior_ready"
        if k in out.columns:
            summary[f"h{h}_dir_prior_ready_rate"] = float(out[k].mean())

    s_path = Path(args.summary_json)
    s_path.parent.mkdir(parents=True, exist_ok=True)
    s_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
