#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.append(str(HERE))

from build_stable_alt_universe import normalize_symbol

QUOTE_SUFFIXES = ("USDT", "USDC", "BUSD", "USD")
DEFAULT_MEME_KEYWORDS = [
    "FART",
    "GOAT",
    "MOODENG",
    "JELLY",
    "SWARMS",
    "NAORIS",
    "PUMP",
    "MEME",
]
DEFAULT_HARD_DENY = {
    "BTCDOMUSDT",
    "LONGUSDT",
    "SHORTUSDT",
    "BINANCEUSDT",
}


def symbol_base(sym: str) -> str:
    s = sym.upper()
    for q in QUOTE_SUFFIXES:
        if s.endswith(q) and len(s) > len(q):
            return s[: -len(q)]
    return s


def is_meme_like(sym: str, keywords: List[str]) -> bool:
    b = symbol_base(sym)
    return any(k in b for k in keywords)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build tradeable-universe with anti-noise policy.")
    ap.add_argument("--signals", default="data/telegram/signals_raw.parquet")
    ap.add_argument("--metrics", default="reports/symbol_universe_metrics.csv")
    ap.add_argument("--min-days", type=int, default=90)
    ap.add_argument("--min-median-quote-vol", type=float, default=1_500_000.0)
    ap.add_argument("--min-corr-btc", type=float, default=0.20)
    ap.add_argument("--max-abs-ret-p95", type=float, default=0.30)
    ap.add_argument("--allow-memes", action="store_true")
    ap.add_argument("--denylist-file", default="")
    ap.add_argument("--whitelist-txt", default="reports/tradeable_universe_whitelist.txt")
    ap.add_argument("--excluded-csv", default="reports/tradeable_universe_excluded.csv")
    ap.add_argument("--filtered-signals", default="data/telegram/signals_raw_tradeable_universe.parquet")
    ap.add_argument("--summary-json", default="reports/tradeable_universe_summary.json")
    args = ap.parse_args()

    m = pd.read_csv(args.metrics)
    s = pd.read_parquet(args.signals)
    s["symbol_norm"] = s["symbol"].map(normalize_symbol)

    ok = m[m["status"] == "ok"].copy()
    ok["reason"] = ""

    # Base quantitative filters.
    ok.loc[ok["n_days"] < args.min_days, "reason"] += "low_history;"
    ok.loc[ok["median_quote_vol_usdt"] < args.min_median_quote_vol, "reason"] += "low_liquidity;"
    ok.loc[ok["corr_btc"] < args.min_corr_btc, "reason"] += "low_btc_corr;"
    ok.loc[ok["abs_ret_p95"] > args.max_abs_ret_p95, "reason"] += "too_volatile;"

    # Hard denylist.
    deny = set(DEFAULT_HARD_DENY)
    if args.denylist_file:
        p = Path(args.denylist_file)
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                x = line.strip().upper()
                if x:
                    deny.add(x)
    ok.loc[ok["symbol"].isin(deny), "reason"] += "hard_deny;"

    # Meme-like heuristic (can be disabled).
    if not args.allow_memes:
        ok.loc[ok["symbol"].apply(lambda x: is_meme_like(str(x), DEFAULT_MEME_KEYWORDS)), "reason"] += "meme_like;"

    keep = ok[ok["reason"] == ""].copy()
    drop = ok[ok["reason"] != ""].copy()

    keep_set = set(keep["symbol"].astype(str))
    filtered = s[s["symbol_norm"].isin(keep_set)].drop(columns=["symbol_norm"])

    wl = Path(args.whitelist_txt)
    wl.parent.mkdir(parents=True, exist_ok=True)
    wl.write_text("\n".join(sorted(keep_set)) + ("\n" if keep_set else ""), encoding="utf-8")

    ex = Path(args.excluded_csv)
    ex.parent.mkdir(parents=True, exist_ok=True)
    drop = drop.sort_values(["reason", "symbol"])
    drop.to_csv(ex, index=False)

    out = Path(args.filtered_signals)
    out.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_parquet(out, index=False)

    summary = {
        "source_rows": int(len(s)),
        "source_unique_symbols_norm": int(s["symbol_norm"].nunique(dropna=True)),
        "metrics_ok_symbols": int(len(ok)),
        "kept_symbols": int(len(keep_set)),
        "excluded_symbols": int(len(drop)),
        "filtered_rows": int(len(filtered)),
        "params": {
            "min_days": args.min_days,
            "min_median_quote_vol": args.min_median_quote_vol,
            "min_corr_btc": args.min_corr_btc,
            "max_abs_ret_p95": args.max_abs_ret_p95,
            "allow_memes": args.allow_memes,
        },
        "outputs": {
            "whitelist_txt": str(wl),
            "excluded_csv": str(ex),
            "filtered_signals": str(out),
        },
    }
    Path(args.summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
