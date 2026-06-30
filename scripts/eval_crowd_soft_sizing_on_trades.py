#!/usr/bin/env python3
"""Evaluate soft position sizing using leak-safe crowd priors on existing trades.

This does not alter entry/exit timestamps. It only rescales trade notional:
    size_mult = clamp(base + alpha * signal_strength, min_mult, max_mult)
where signal_strength is derived from crowd priors available at entry time.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def _load_trades(paths: List[str]) -> pd.DataFrame:
    parts = []
    for p in paths:
        df = pd.read_csv(p).copy()
        df["src_file"] = Path(p).name
        parts.append(df)
    tr = pd.concat(parts, ignore_index=True)
    tr["entry_ts"] = pd.to_datetime(tr["entry_ts"], utc=True, errors="coerce")
    tr["exit_ts"] = pd.to_datetime(tr["exit_ts"], utc=True, errors="coerce")
    tr = tr[tr["entry_ts"].notna() & tr["exit_ts"].notna()].copy()
    tr["direction"] = tr["direction"].astype(str).str.lower().str.strip()
    tr = tr[tr["direction"].isin(["long", "short"])].copy()
    tr = tr.sort_values("entry_ts").reset_index(drop=True)
    tr["signed_ret"] = np.where(
        tr["direction"].eq("long"),
        tr["exit_price"] / tr["entry_price"] - 1.0,
        tr["entry_price"] / tr["exit_price"] - 1.0,
    )
    if "notional" in tr.columns:
        tr["notional"] = pd.to_numeric(tr["notional"], errors="coerce").fillna(1.0)
    else:
        tr["notional"] = 1.0
    tr["pnl_usd"] = tr["notional"] * tr["signed_ret"]
    tr["win"] = (tr["signed_ret"] > 0).astype(np.int32)
    return tr


def _metrics(df: pd.DataFrame, pnl_col: str) -> Dict[str, float]:
    if df.empty:
        return {
            "trades": 0,
            "winrate": 0.0,
            "sum_pnl_usd": 0.0,
            "avg_pnl_usd": 0.0,
            "sum_ret": 0.0,
            "avg_ret": 0.0,
        }
    return {
        "trades": int(len(df)),
        "winrate": float(df["win"].mean()),
        "sum_pnl_usd": float(df[pnl_col].sum()),
        "avg_pnl_usd": float(df[pnl_col].mean()),
        "sum_ret": float(df["signed_ret"].sum()),
        "avg_ret": float(df["signed_ret"].mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Crowd-prior soft sizing sweep on existing trades")
    ap.add_argument("--trades", required=True, help="comma-separated trade CSVs")
    ap.add_argument("--crowd-priors", required=True, help="crowd priors parquet/csv")
    ap.add_argument("--horizon", type=int, default=24, choices=[24, 72, 168])
    ap.add_argument("--prior-prefix", default="crowd_h", help="feature prefix, e.g. crowd_h or crowd_ctx_h")
    ap.add_argument("--min-trades", type=int, default=30)
    ap.add_argument("--budget-neutral", action="store_true", help="normalize size multipliers to have mean 1.0")
    ap.add_argument("--output-csv", default="reports/crowd_soft_sizing_sweep.csv")
    ap.add_argument("--summary-json", default="reports/crowd_soft_sizing_summary.json")
    args = ap.parse_args()

    trade_paths = [x.strip() for x in str(args.trades).split(",") if x.strip()]
    tr = _load_trades(trade_paths)

    cp = pd.read_parquet(args.crowd_priors) if str(args.crowd_priors).endswith(".parquet") else pd.read_csv(args.crowd_priors)
    cp["timestamp_utc"] = pd.to_datetime(cp["timestamp_utc"], utc=True, errors="coerce")
    cp = cp[cp["timestamp_utc"].notna()].sort_values("timestamp_utc").reset_index(drop=True)

    h = int(args.horizon)
    pfx = str(args.prior_prefix)
    wr_col = f"{pfx}{h}_winrate_dir"
    ewrong_col = f"{pfx}{h}_err_wrong_dir"
    ready_col = f"{pfx}{h}_dir_prior_ready"
    need = [wr_col, ewrong_col]
    ready_exists = ready_col in cp.columns
    for c in need:
        if c not in cp.columns:
            raise ValueError(f"crowd priors missing column: {c}")

    m = pd.merge_asof(
        tr.sort_values("entry_ts"),
        cp[["timestamp_utc"] + [wr_col, ewrong_col] + ([ready_col] if ready_exists else [])].sort_values("timestamp_utc"),
        left_on="entry_ts",
        right_on="timestamp_utc",
        direction="backward",
        allow_exact_matches=True,
    )
    m[wr_col] = pd.to_numeric(m[wr_col], errors="coerce").fillna(0.5)
    m[ewrong_col] = pd.to_numeric(m[ewrong_col], errors="coerce").fillna(0.5)
    if ready_exists:
        m[ready_col] = pd.to_numeric(m[ready_col], errors="coerce").fillna(0.0)
    else:
        # If no explicit ready flag, consider prior "ready" when winrate is finite.
        m[ready_col] = m[wr_col].notna().astype(np.float32)

    # Strength > 0 means "slightly better than neutral crowd context".
    m["signal_strength"] = (m[wr_col] - m[ewrong_col]).clip(-1.0, 1.0)
    m.loc[m[ready_col] < 1.0, "signal_strength"] = 0.0

    baseline = _metrics(m, "pnl_usd")
    baseline["coverage_rate"] = float(m[ready_col].mean())
    baseline["horizon"] = h

    rows = []
    for alpha in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
        for min_mult in [0.5, 0.6, 0.7, 0.8, 0.9]:
            for max_mult in [1.1, 1.2, 1.3, 1.4, 1.5]:
                if max_mult <= min_mult:
                    continue
                size_mult = (1.0 + alpha * m["signal_strength"]).clip(min_mult, max_mult)
                if args.budget_neutral:
                    mu = float(size_mult.mean())
                    if mu > 1e-9:
                        size_mult = size_mult / mu
                pnl_soft = m["pnl_usd"] * size_mult
                rr = _metrics(m.assign(pnl_soft=pnl_soft), "pnl_soft")
                rr["alpha"] = float(alpha)
                rr["min_mult"] = float(min_mult)
                rr["max_mult"] = float(max_mult)
                rr["avg_size_mult"] = float(size_mult.mean())
                rr["delta_sum_pnl_usd"] = rr["sum_pnl_usd"] - baseline["sum_pnl_usd"]
                rr["delta_avg_pnl_usd"] = rr["avg_pnl_usd"] - baseline["avg_pnl_usd"]
                rr["delta_winrate"] = rr["winrate"] - baseline["winrate"]  # should stay 0 (same entries)
                rows.append(rr)

    sweep = pd.DataFrame(rows).sort_values(["delta_sum_pnl_usd", "avg_pnl_usd"], ascending=[False, False]).reset_index(drop=True)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(args.output_csv, index=False)

    best_any = sweep.iloc[0].to_dict() if len(sweep) else {}
    sweep_safe = sweep[sweep["avg_size_mult"] <= 1.05]
    best_safe = sweep_safe.iloc[0].to_dict() if len(sweep_safe) else {}
    sweep_mid = sweep[(sweep["avg_size_mult"] >= 0.95) & (sweep["avg_size_mult"] <= 1.10)]
    best_mid = sweep_mid.iloc[0].to_dict() if len(sweep_mid) else {}

    summary = {
        "trades_files": trade_paths,
        "crowd_priors_file": args.crowd_priors,
        "prior_prefix": pfx,
        "horizon": h,
        "baseline": baseline,
        "best_any": best_any,
        "best_avg_size_le_1.05": best_safe,
        "best_avg_size_0.95_1.10": best_mid,
        "rows_sweep": int(len(sweep)),
        "budget_neutral": bool(args.budget_neutral),
        "output_csv": args.output_csv,
    }
    Path(args.summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
