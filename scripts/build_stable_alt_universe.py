#!/usr/bin/env python3
import argparse
import json
import math
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

FUT_EXCHANGE_INFO = "https://fapi.binance.com/fapi/v1/exchangeInfo"
FUT_KLINES = "https://fapi.binance.com/fapi/v1/klines"
SPOT_KLINES = "https://api.binance.com/api/v3/klines"

BAD_SYMBOLS = {
    "LONGUSDT",
    "SHORTUSDT",
    "BINANCEUSDT",
    "SIGNALUSDT",
    "ENTRYUSDT",
}
LEVERAGED_SUFFIXES = ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT")


def http_json(url: str, timeout: int = 25) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "model6-universe-filter/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="ignore"))


def normalize_symbol(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip().upper()
    if not s:
        return None
    s = s.replace("#", "").replace("$", "")
    s = s.replace("/", "").replace("-", "").replace("_", "")
    s = s.replace("PERP", "USDT").replace("USDTP", "USDT")
    if s in {"BTC", "ETH", "BNB", "XRP", "SOL", "ADA", "DOGE", "LTC", "DOT", "LINK", "ATOM", "ALGO"}:
        s = s + "USDT"
    if not re.search(r"(USDT|BUSD|USDC|USD)$", s):
        s = s + "USDT"
    if s == "LITUSDT":
        s = "LTCUSDT"
    if s in BAD_SYMBOLS:
        return None
    if s.endswith(LEVERAGED_SUFFIXES):
        return None
    return s


def fetch_futures_usdt_symbols() -> set:
    info = http_json(FUT_EXCHANGE_INFO)
    out = set()
    for s in info.get("symbols", []):
        if s.get("status") != "TRADING":
            continue
        if s.get("quoteAsset") != "USDT":
            continue
        sym = s.get("symbol")
        if sym:
            out.add(sym.upper())
    return out


def fetch_klines(symbol: str, start_ms: int, end_ms: int, limit: int = 1500) -> List[List[Any]]:
    q = urllib.parse.urlencode(
        {"symbol": symbol, "interval": "1d", "startTime": start_ms, "endTime": end_ms, "limit": limit}
    )
    try:
        return http_json(f"{FUT_KLINES}?{q}")
    except Exception:
        return http_json(f"{SPOT_KLINES}?{q}")


def kline_to_df(kl: List[List[Any]]) -> pd.DataFrame:
    rows = []
    for r in kl:
        # Binance kline: 0 open_time, 4 close, 7 quote_volume
        rows.append(
            {
                "open_time": int(r[0]),
                "close": float(r[4]),
                "quote_vol": float(r[7]),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("open_time").drop_duplicates("open_time")
    df["ret"] = df["close"].pct_change()
    return df


def compute_metrics(symbol: str, btc_ret: pd.Series, start_ms: int, end_ms: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {"symbol": symbol}
    try:
        kl = fetch_klines(symbol, start_ms, end_ms)
        if not isinstance(kl, list) or len(kl) < 20:
            out.update({"status": "skip", "reason": "not_enough_klines"})
            return out
        df = kline_to_df(kl)
        if len(df) < 20:
            out.update({"status": "skip", "reason": "not_enough_klines"})
            return out

        rets = df["ret"].dropna()
        if len(rets) < 10:
            out.update({"status": "skip", "reason": "not_enough_returns"})
            return out

        joined = pd.DataFrame({"ret": rets.values}, index=df.loc[rets.index, "open_time"].values)
        j = joined.join(btc_ret.rename("btc_ret"), how="inner")
        corr = float(j["ret"].corr(j["btc_ret"])) if len(j) >= 10 else float("nan")

        out.update(
            {
                "status": "ok",
                "n_days": int(len(df)),
                "n_overlap_btc": int(len(j)),
                "median_quote_vol_usdt": float(df["quote_vol"].median()),
                "avg_quote_vol_usdt": float(df["quote_vol"].mean()),
                "abs_ret_p95": float(rets.abs().quantile(0.95)),
                "corr_btc": corr,
            }
        )
        return out
    except Exception as e:
        out.update({"status": "skip", "reason": str(e)[:160]})
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build stable altcoin universe from signal symbols.")
    ap.add_argument("--signals", default="data/telegram/signals_raw.parquet")
    ap.add_argument("--lookback-days", type=int, default=365)
    ap.add_argument("--min-days", type=int, default=120)
    ap.add_argument("--min-median-quote-vol", type=float, default=5_000_000.0)
    ap.add_argument("--min-corr-btc", type=float, default=0.25)
    ap.add_argument("--max-abs-ret-p95", type=float, default=0.25)
    ap.add_argument("--max-workers", type=int, default=10)
    ap.add_argument("--metrics-csv", default="reports/symbol_universe_metrics.csv")
    ap.add_argument("--whitelist-txt", default="reports/symbol_universe_whitelist.txt")
    ap.add_argument("--summary-json", default="reports/symbol_universe_summary.json")
    ap.add_argument("--filtered-signals", default="data/telegram/signals_raw_stable_universe.parquet")
    args = ap.parse_args()

    df = pd.read_parquet(args.signals)
    raw_symbols = df["symbol"].fillna("").astype(str).tolist()
    norm_symbols = sorted({s for s in (normalize_symbol(x) for x in raw_symbols) if s})

    fut_symbols = fetch_futures_usdt_symbols()
    candidates = [s for s in norm_symbols if s in fut_symbols]

    end_dt = datetime.now(tz=timezone.utc)
    start_dt = end_dt - timedelta(days=args.lookback_days)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    btc_df = kline_to_df(fetch_klines("BTCUSDT", start_ms, end_ms))
    btc_ret = pd.Series(btc_df["ret"].values, index=btc_df["open_time"].values).dropna()

    rows: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = [ex.submit(compute_metrics, s, btc_ret, start_ms, end_ms) for s in candidates]
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if i % 50 == 0:
                print(f"processed {i}/{len(candidates)}")

    mdf = pd.DataFrame(rows)
    if mdf.empty:
        raise RuntimeError("No metrics were computed.")

    ok = mdf[mdf["status"] == "ok"].copy()
    ok["is_keep"] = (
        (ok["n_days"] >= args.min_days)
        & (ok["median_quote_vol_usdt"] >= args.min_median_quote_vol)
        & (ok["corr_btc"] >= args.min_corr_btc)
        & (ok["abs_ret_p95"] <= args.max_abs_ret_p95)
    )
    keep = sorted(ok.loc[ok["is_keep"], "symbol"].tolist())

    metrics_csv = Path(args.metrics_csv)
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    mdf.sort_values(["status", "symbol"]).to_csv(metrics_csv, index=False)

    whitelist_txt = Path(args.whitelist_txt)
    whitelist_txt.parent.mkdir(parents=True, exist_ok=True)
    whitelist_txt.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")

    # Filter signals parquet by normalized symbol in keep-set.
    keep_set = set(keep)
    tmp = df.copy()
    tmp["symbol_norm"] = tmp["symbol"].map(normalize_symbol)
    fdf = tmp[tmp["symbol_norm"].isin(keep_set)].drop(columns=["symbol_norm"])
    filtered_out = Path(args.filtered_signals)
    filtered_out.parent.mkdir(parents=True, exist_ok=True)
    fdf.to_parquet(filtered_out, index=False)

    summary = {
        "signals_rows_total": int(len(df)),
        "raw_unique_symbols": int(len(set(raw_symbols))),
        "normalized_unique_symbols": int(len(norm_symbols)),
        "futures_usdt_candidates": int(len(candidates)),
        "metrics_ok_symbols": int(len(ok)),
        "whitelist_symbols": int(len(keep)),
        "filtered_signals_rows": int(len(fdf)),
        "params": {
            "lookback_days": args.lookback_days,
            "min_days": args.min_days,
            "min_median_quote_vol": args.min_median_quote_vol,
            "min_corr_btc": args.min_corr_btc,
            "max_abs_ret_p95": args.max_abs_ret_p95,
        },
        "outputs": {
            "metrics_csv": str(metrics_csv),
            "whitelist_txt": str(whitelist_txt),
            "filtered_signals": str(filtered_out),
        },
    }
    Path(args.summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

