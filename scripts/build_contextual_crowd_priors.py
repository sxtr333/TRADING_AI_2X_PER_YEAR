#!/usr/bin/env python3
"""Build leak-safe contextual crowd priors from directional labels.

Context dimensions (cheap and robust):
- direction: long/short
- session: asia/europe/us by UTC hour
- dow_group: weekday/weekend

Leak-safe rule per horizon H:
- use only rows with timestamp <= T - H (matured observations)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def _read(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
        return path
    try:
        df.to_parquet(path, index=False)
        return path
    except Exception:
        fb = path.with_suffix(".csv")
        df.to_csv(fb, index=False)
        return fb


def _session_from_hour(h: int) -> str:
    # UTC buckets chosen for stable sample sizes.
    if 0 <= h <= 7:
        return "asia"
    if 8 <= h <= 15:
        return "europe"
    return "us"


def _ctx_key(direction: str, session: str, dow_group: str) -> str:
    return f"{direction}|{session}|{dow_group}"


def _cum_sum(arr: np.ndarray) -> np.ndarray:
    return np.cumsum(arr.astype(np.float64))


def _cum_idx_stats(
    valid: np.ndarray,
    j_idx: np.ndarray,
    c_num: np.ndarray,
    c_den: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    num = np.where(valid, c_num[j_idx], 0.0)
    den = np.where(valid, c_den[j_idx], 0.0)
    return num, den


def main() -> None:
    ap = argparse.ArgumentParser(description="Build contextual leak-safe crowd priors")
    ap.add_argument("--directional", required=True)
    ap.add_argument("--horizons-hours", default="24,72,168")
    ap.add_argument("--min-samples", type=int, default=10)
    ap.add_argument("--output", default="data/telegram/crowd_priors_contextual.parquet")
    ap.add_argument("--summary-json", default="reports/crowd_priors_contextual_summary.json")
    args = ap.parse_args()

    df = _read(Path(args.directional)).copy()
    req = {"message_id", "timestamp_utc", "direction"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"directional missing required columns: {sorted(miss)}")

    horizons: List[int] = []
    for x in str(args.horizons_hours).split(","):
        x = x.strip()
        if x:
            horizons.append(int(x))
    horizons = sorted(set(h for h in horizons if h > 0))
    if not horizons:
        raise ValueError("no positive horizons")

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df[df["timestamp_utc"].notna()].copy()
    df["direction_norm"] = df["direction"].astype(str).str.lower().str.strip()
    df = df[df["direction_norm"].isin(["long", "short"])].copy()
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    if df.empty:
        raise ValueError("no valid rows")

    ts_ns = df["timestamp_utc"].astype("int64").to_numpy()
    hour = df["timestamp_utc"].dt.hour.astype(int).to_numpy()
    dow = df["timestamp_utc"].dt.dayofweek.astype(int).to_numpy()
    session = np.array([_session_from_hour(int(h)) for h in hour], dtype=object)
    dow_group = np.where(dow >= 5, "weekend", "weekday")
    ctx = np.array(
        [
            _ctx_key(d, s, w)
            for d, s, w in zip(df["direction_norm"].to_numpy(), session, dow_group)
        ],
        dtype=object,
    )
    dir_only = df["direction_norm"].to_numpy(dtype=object)

    out = pd.DataFrame(
        {
            "message_id": df["message_id"].astype(str),
            "timestamp_utc": df["timestamp_utc"].astype(str),
            "direction": df["direction_norm"].astype(str),
            "ctx_session": session,
            "ctx_dow_group": dow_group,
            "ctx_key": ctx,
        }
    )

    all_ctx_keys: Iterable[str] = sorted(set(ctx.tolist()))
    all_dir_keys: Iterable[str] = sorted(set(dir_only.tolist()))

    for h in horizons:
        ip_col = f"is_profitable_{h}h"
        ret_col = f"signed_ret_{h}h_pct"
        err_col = f"error_reason_{h}h"
        for c in [ip_col, ret_col, err_col]:
            if c not in df.columns:
                raise ValueError(f"directional missing horizon col: {c}")

        ip = pd.to_numeric(df[ip_col], errors="coerce").fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=np.float64)
        sr = pd.to_numeric(df[ret_col], errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float64)
        err = df[err_col].fillna("unknown").astype(str).str.lower().str.strip().to_numpy(dtype=object)

        ok = np.isfinite(sr).astype(np.float64)
        sr0 = np.where(np.isfinite(sr), sr, 0.0)
        wrong = (err == "wrong_direction_or_deep_drawdown").astype(np.float64)
        nofollow = (err == "no_follow_through").astype(np.float64)
        giveback = (err == "gave_back_profit").astype(np.float64)

        # Mature index constraint: rows <= T-H.
        cut_ns = ts_ns - int(pd.Timedelta(hours=h).value)
        j = np.searchsorted(ts_ns, cut_ns, side="right") - 1
        valid = j >= 0

        # Global fallback.
        c_ok = _cum_sum(ok)
        c_ip = _cum_sum(np.where(ok > 0, ip, 0.0))
        c_sr = _cum_sum(np.where(ok > 0, sr0, 0.0))
        c_wrong = _cum_sum(np.where(ok > 0, wrong, 0.0))
        c_nofol = _cum_sum(np.where(ok > 0, nofollow, 0.0))
        c_give = _cum_sum(np.where(ok > 0, giveback, 0.0))

        g_num_ip, g_den = _cum_idx_stats(valid, j, c_ip, c_ok)
        g_num_sr, _ = _cum_idx_stats(valid, j, c_sr, c_ok)
        g_num_wrong, _ = _cum_idx_stats(valid, j, c_wrong, c_ok)
        g_num_nofol, _ = _cum_idx_stats(valid, j, c_nofol, c_ok)
        g_num_give, _ = _cum_idx_stats(valid, j, c_give, c_ok)

        eps = 1e-9
        out[f"crowd_ctx_h{h}_samples_global"] = g_den.astype(np.float32)
        out[f"crowd_ctx_h{h}_winrate_global"] = (g_num_ip / np.maximum(g_den, eps)).astype(np.float32)
        out[f"crowd_ctx_h{h}_avg_signed_ret_global"] = (g_num_sr / np.maximum(g_den, eps)).astype(np.float32)
        out[f"crowd_ctx_h{h}_err_wrong_global"] = (g_num_wrong / np.maximum(g_den, eps)).astype(np.float32)
        out[f"crowd_ctx_h{h}_err_nofollow_global"] = (g_num_nofol / np.maximum(g_den, eps)).astype(np.float32)
        out[f"crowd_ctx_h{h}_err_giveback_global"] = (g_num_give / np.maximum(g_den, eps)).astype(np.float32)

        # Direction-only fallback.
        d_samples = np.zeros(len(df), dtype=np.float64)
        d_winrate = np.zeros(len(df), dtype=np.float64)
        d_avgret = np.zeros(len(df), dtype=np.float64)
        d_wrong = np.zeros(len(df), dtype=np.float64)
        d_nofol = np.zeros(len(df), dtype=np.float64)
        d_give = np.zeros(len(df), dtype=np.float64)
        d_ready = np.zeros(len(df), dtype=np.float64)

        for dk in all_dir_keys:
            mask = (dir_only == dk).astype(np.float64)
            c_ok_d = _cum_sum(ok * mask)
            c_ip_d = _cum_sum(np.where(ok > 0, ip * mask, 0.0))
            c_sr_d = _cum_sum(np.where(ok > 0, sr0 * mask, 0.0))
            c_w_d = _cum_sum(np.where(ok > 0, wrong * mask, 0.0))
            c_n_d = _cum_sum(np.where(ok > 0, nofollow * mask, 0.0))
            c_g_d = _cum_sum(np.where(ok > 0, giveback * mask, 0.0))

            num_ip_d, den_d = _cum_idx_stats(valid, j, c_ip_d, c_ok_d)
            num_sr_d, _ = _cum_idx_stats(valid, j, c_sr_d, c_ok_d)
            num_w_d, _ = _cum_idx_stats(valid, j, c_w_d, c_ok_d)
            num_n_d, _ = _cum_idx_stats(valid, j, c_n_d, c_ok_d)
            num_g_d, _ = _cum_idx_stats(valid, j, c_g_d, c_ok_d)

            idx = np.where(dir_only == dk)[0]
            d_samples[idx] = den_d[idx]
            d_winrate[idx] = num_ip_d[idx] / np.maximum(den_d[idx], eps)
            d_avgret[idx] = num_sr_d[idx] / np.maximum(den_d[idx], eps)
            d_wrong[idx] = num_w_d[idx] / np.maximum(den_d[idx], eps)
            d_nofol[idx] = num_n_d[idx] / np.maximum(den_d[idx], eps)
            d_give[idx] = num_g_d[idx] / np.maximum(den_d[idx], eps)
            d_ready[idx] = (den_d[idx] >= float(args.min_samples)).astype(np.float64)

        # Full contextual stats.
        c_samples = np.zeros(len(df), dtype=np.float64)
        c_winrate = np.zeros(len(df), dtype=np.float64)
        c_avgret = np.zeros(len(df), dtype=np.float64)
        c_wrong_rate = np.zeros(len(df), dtype=np.float64)
        c_nofol_rate = np.zeros(len(df), dtype=np.float64)
        c_give_rate = np.zeros(len(df), dtype=np.float64)
        c_ready = np.zeros(len(df), dtype=np.float64)

        for ck in all_ctx_keys:
            mask = (ctx == ck).astype(np.float64)
            c_ok_k = _cum_sum(ok * mask)
            c_ip_k = _cum_sum(np.where(ok > 0, ip * mask, 0.0))
            c_sr_k = _cum_sum(np.where(ok > 0, sr0 * mask, 0.0))
            c_w_k = _cum_sum(np.where(ok > 0, wrong * mask, 0.0))
            c_n_k = _cum_sum(np.where(ok > 0, nofollow * mask, 0.0))
            c_g_k = _cum_sum(np.where(ok > 0, giveback * mask, 0.0))

            num_ip_k, den_k = _cum_idx_stats(valid, j, c_ip_k, c_ok_k)
            num_sr_k, _ = _cum_idx_stats(valid, j, c_sr_k, c_ok_k)
            num_w_k, _ = _cum_idx_stats(valid, j, c_w_k, c_ok_k)
            num_n_k, _ = _cum_idx_stats(valid, j, c_n_k, c_ok_k)
            num_g_k, _ = _cum_idx_stats(valid, j, c_g_k, c_ok_k)

            idx = np.where(ctx == ck)[0]
            c_samples[idx] = den_k[idx]
            c_winrate[idx] = num_ip_k[idx] / np.maximum(den_k[idx], eps)
            c_avgret[idx] = num_sr_k[idx] / np.maximum(den_k[idx], eps)
            c_wrong_rate[idx] = num_w_k[idx] / np.maximum(den_k[idx], eps)
            c_nofol_rate[idx] = num_n_k[idx] / np.maximum(den_k[idx], eps)
            c_give_rate[idx] = num_g_k[idx] / np.maximum(den_k[idx], eps)
            c_ready[idx] = (den_k[idx] >= float(args.min_samples)).astype(np.float64)

        # Blend: context if ready else dir fallback if ready else global.
        use_ctx = c_ready >= 1.0
        use_dir = (~use_ctx) & (d_ready >= 1.0)

        final_samples = np.where(use_ctx, c_samples, np.where(use_dir, d_samples, g_den))
        final_win = np.where(use_ctx, c_winrate, np.where(use_dir, d_winrate, g_num_ip / np.maximum(g_den, eps)))
        final_sr = np.where(use_ctx, c_avgret, np.where(use_dir, d_avgret, g_num_sr / np.maximum(g_den, eps)))
        final_wrong = np.where(use_ctx, c_wrong_rate, np.where(use_dir, d_wrong, g_num_wrong / np.maximum(g_den, eps)))
        final_nofol = np.where(use_ctx, c_nofol_rate, np.where(use_dir, d_nofol, g_num_nofol / np.maximum(g_den, eps)))
        final_give = np.where(use_ctx, c_give_rate, np.where(use_dir, d_give, g_num_give / np.maximum(g_den, eps)))

        src = np.where(use_ctx, "context", np.where(use_dir, "direction", "global"))

        out[f"crowd_ctx_h{h}_samples"] = final_samples.astype(np.float32)
        out[f"crowd_ctx_h{h}_dir_prior_ready"] = (final_samples >= float(args.min_samples)).astype(np.float32)
        out[f"crowd_ctx_h{h}_winrate_dir"] = final_win.astype(np.float32)
        out[f"crowd_ctx_h{h}_avg_signed_ret_dir"] = final_sr.astype(np.float32)
        out[f"crowd_ctx_h{h}_err_wrong_dir"] = final_wrong.astype(np.float32)
        out[f"crowd_ctx_h{h}_err_nofollow_dir"] = final_nofol.astype(np.float32)
        out[f"crowd_ctx_h{h}_err_giveback_dir"] = final_give.astype(np.float32)
        out[f"crowd_ctx_h{h}_source_context"] = (src == "context").astype(np.float32)
        out[f"crowd_ctx_h{h}_source_direction"] = (src == "direction").astype(np.float32)
        out[f"crowd_ctx_h{h}_source_global"] = (src == "global").astype(np.float32)

    out_path = _write(out, Path(args.output))

    summary: Dict[str, object] = {
        "rows": int(len(out)),
        "horizons_hours": horizons,
        "min_samples": int(args.min_samples),
        "output_file": str(out_path),
    }
    for h in horizons:
        c1 = f"crowd_ctx_h{h}_source_context"
        c2 = f"crowd_ctx_h{h}_source_direction"
        c3 = f"crowd_ctx_h{h}_source_global"
        if c1 in out.columns:
            summary[f"h{h}_source_context_rate"] = float(out[c1].mean())
            summary[f"h{h}_source_direction_rate"] = float(out[c2].mean())
            summary[f"h{h}_source_global_rate"] = float(out[c3].mean())

    s_path = Path(args.summary_json)
    s_path.parent.mkdir(parents=True, exist_ok=True)
    s_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
