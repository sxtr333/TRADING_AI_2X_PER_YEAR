"""
Lightweight FastAPI inference service for the Keras model.
- Loads a trained model and exposes /predict and /health endpoints.
- Request body: {
      "window": [[...],[...],...],  # shape (seq_len, n_features) in the order of default_feature_list()
      "mean": [...],                # optional, length n_features
      "std":  [...],                # optional, length n_features
  }
- If mean/std are absent, the window is used as-is (assumes pre-normalized or already normalized data).
"""

from __future__ import annotations

import os
import re
import sys
import time
import html
import copy
import smtplib
import secrets
import base64
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, validator

from trading_keras_core import default_feature_list, DropPath
from model_layers import RevIN, TSMixerBlock, ITransformerBlock, LastStep

import pandas as pd
import math
import json
import subprocess
import csv
import sqlite3
import hashlib
import hmac
import threading
import urllib.parse
import urllib.error
import urllib.request
from pathlib import Path
from email.message import EmailMessage

_TRADES_CACHE: Dict[str, Dict[str, Any]] = {}
_LIQ_FEED_CACHE: Dict[str, Dict[str, Any]] = {}
_BINANCE_KLINES_CACHE: Dict[str, Dict[str, Any]] = {}
_FEATURES_CACHE: Dict[str, Dict[str, Any]] = {}
_FEATURES_CACHE_LOCK = threading.Lock()
_NEWS_REFRESH_LOCK = threading.Lock()
_NEWS_REFRESH_LAST_TS = 0.0
_API_RESP_CACHE: Dict[str, Dict[str, Any]] = {}
_API_RESP_CACHE_LOCK = threading.Lock()
_API_CACHE_STATS: Dict[str, int] = {
    "hits": 0,
    "misses": 0,
    "sets": 0,
    "evicted": 0,
    "expired": 0,
}
_ARBITRAGE_TOP_CACHE: Dict[str, Any] = {"ts": 0.0, "coins": []}
_ARBITRAGE_SNAPSHOT_CACHE: Dict[str, Any] = {}
_BINANCE_ALL_PRICES_CACHE: Dict[str, Any] = {"ts": 0.0, "prices": {}}
_BINANCE_24H_CACHE: Dict[str, Any] = {"ts": 0.0, "rows": {}}
_MEXC_PERP_TICKER_CACHE: Dict[str, Any] = {"ts": 0.0, "rows": {}}
_MEXC_PERP_DETAIL_CACHE: Dict[str, Any] = {"ts": 0.0, "rows": {}}
_EXCHANGE_BOOK_CACHE: Dict[str, Dict[str, Any]] = {}
_LIVE_TRADE_STATE: Dict[str, Any] = {"open": None, "last_base_ts": None}
_LIVE_TRADES_PATH = Path(__file__).resolve().parent / "reports" / "trades_live.csv"
_LIVE_MODEL_TRADE_STATE: Dict[str, Dict[str, Any]] = {}
_LIVE_MODELS_THREAD_STARTED = False
_SIGNAL_QUALITY_LOG_PATH = Path(__file__).resolve().parent / "reports" / "signal_quality_runtime.jsonl"
_SIGNAL_QUALITY_LOG_LOCK = threading.Lock()
_LIVE_TRADE_DEADZONE = float(os.getenv("LIVE_TRADE_DEADZONE", "0.0005"))
_LIVE_TRADE_FORCE = os.getenv("LIVE_TRADE_FORCE", "0") == "1"
_LIVE_MODEL_LOOP_SEC = max(20, int(os.getenv("LIVE_MODEL_LOOP_SEC", "120")))

_SITE_MODEL_PROFILE: Dict[str, Dict[str, Any]] = {
    "conservative": {
        "label": "Model v7 ATR",
        "leverage": "1×",
        "risk_pct": "0.5%",
        "weights": {"5H": 0.20, "20H": 0.35, "40H": 0.45},
        "deadzone": 0.0016,
        "logic_name": "ATR consensus",
        "atr_mult": 1.25,
    },
    "aggressive": {
        "label": "Model v7 +39%",
        "leverage": "3×",
        "risk_pct": "39%",
        "weights": {"5H": 0.65, "20H": 0.25, "40H": 0.10},
        "deadzone": 0.0008,
        "logic_name": "Momentum hunter",
        "atr_mult": 1.95,
    },
    "alpha75": {
        "label": "Model v7 +75%",
        "leverage": "3×",
        "risk_pct": "39%",
        "weights": {"5H": 0.50, "20H": 0.35, "40H": 0.15},
        "deadzone": 0.0010,
        "logic_name": "Directional alpha",
        "atr_mult": 1.70,
    },
    "tg_hybrid": {
        "label": "TG Hybrid v3_2_0",
        "leverage": "3×",
        "risk_pct": "0.8%",
        "weights": {"5H": 0.35, "20H": 0.40, "40H": 0.25},
        "deadzone": 0.0012,
        "logic_name": "News + crowd hybrid",
        "atr_mult": 1.55,
    },
    "mm_r13": {
        "label": "MM Supervisor r13",
        "leverage": "3×",
        "risk_pct": "0.5%",
        "weights": {"5H": 0.30, "20H": 0.25, "40H": 0.45},
        "deadzone": 0.0011,
        "logic_name": "MM supervisor",
        "atr_mult": 1.45,
    },
    "chronos2": {
        "label": "Chronos2 stage2",
        "leverage": "3×",
        "risk_pct": "0.8%",
        "weights": {"5H": 0.20, "20H": 0.45, "40H": 0.35},
        "deadzone": 0.0012,
        "logic_name": "Time path",
        "atr_mult": 1.60,
    },
    "meta_combo_v1": {
        "label": "Meta Combo v1",
        "leverage": "3×",
        "risk_pct": "65%",
        "weights": {"5H": 0.35, "20H": 0.15, "40H": 0.50},
        "deadzone": 0.0009,
        "logic_name": "Meta combo breakout",
        "atr_mult": 1.85,
    },
}

# Precomputed leverage results (summary CSVs)
_LEV_FILES = {
    (2023, 3): "backtest_meta_newsflag_mix15_2024calib_full_2023_thr059_lev3.csv",
    (2023, 5): "backtest_meta_newsflag_mix15_2024calib_full_2023_thr059_lev5.csv",
    (2023, 10): "backtest_meta_newsflag_mix15_2024calib_full_2023_thr059_lev10.csv",
    (2024, 3): "backtest_meta_newsflag_mix15_2024calib_full_2024_thr059_lev3.csv",
    (2024, 5): "backtest_meta_newsflag_mix15_2024calib_full_2024_thr059_lev5.csv",
    (2024, 10): "backtest_meta_newsflag_mix15_2024calib_full_2024_thr059_lev10.csv",
    (2025, 3): "backtest_meta_newsflag_mix15_2024calib_full_2025_thr059_lev3.csv",
    (2025, 5): "backtest_meta_newsflag_mix15_2024calib_full_2025_thr059_lev5.csv",
    (2025, 10): "backtest_meta_newsflag_mix15_2024calib_full_2025_thr059_lev10.csv",
}

MULTI_SCALE = float(os.getenv("MULTI_SCALE", "15"))
MIN_MULTI_PCT = float(os.getenv("MIN_MULTI_PCT", "0.05"))

# API response cache TTLs (seconds)
_TTL_FORECAST_SEC = max(1, int(os.getenv("API_CACHE_FORECAST_SEC", "20")))
_TTL_FORECAST_MULTI_SEC = max(1, int(os.getenv("API_CACHE_FORECAST_MULTI_SEC", "20")))
_TTL_CANDLES_SEC = max(1, int(os.getenv("API_CACHE_CANDLES_SEC", "60")))
_TTL_NEWS_SEC = max(1, int(os.getenv("API_CACHE_NEWS_SEC", "45")))
_TTL_NEWS_AGG_SEC = max(1, int(os.getenv("API_CACHE_NEWS_AGG_SEC", "45")))
_TTL_TRADES_SEC = max(1, int(os.getenv("API_CACHE_TRADES_SEC", "60")))
_TTL_LIQUIDATIONS_SEC = max(1, int(os.getenv("API_CACHE_LIQUIDATIONS_SEC", "30")))
_TTL_RESULTS_SEC = max(1, int(os.getenv("API_CACHE_RESULTS_SEC", "120")))
_TTL_DASHBOARD_BOOTSTRAP_SEC = max(1, int(os.getenv("API_CACHE_DASHBOARD_BOOTSTRAP_SEC", "20")))
_TTL_SCREENER_BOOTSTRAP_SEC = max(1, int(os.getenv("API_CACHE_SCREENER_BOOTSTRAP_SEC", "15")))
_API_CACHE_MAX_ITEMS = max(100, int(os.getenv("API_CACHE_MAX_ITEMS", "4000")))

def _first_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return path.split(",")[0].strip()


def _feature_cols_from_stats(path: Optional[str], fallback: Optional[List[str]] = None) -> List[str]:
    path = _first_path(path)
    if not path:
        return fallback or default_feature_list()
    stats = np.load(path, allow_pickle=True)
    feature_names = stats.get("feature_names")
    if feature_names is None or len(feature_names) == 0:
        return fallback or default_feature_list()
    return [str(x) for x in feature_names]


def _ensure_feature_columns(df: pd.DataFrame, cols: List[str]) -> None:
    for col in cols:
        if col not in df.columns:
            df[col] = 0.0


def _list_trade_csvs(reports_dir: Path) -> List[str]:
    if not reports_dir.exists():
        return []
    return sorted([p.name for p in reports_dir.glob("trades_*.csv")])


def _binance_all_prices(max_age_sec: int = 20) -> Dict[str, float]:
    now = time.time()
    cached = _BINANCE_ALL_PRICES_CACHE
    if cached.get("prices") and (now - float(cached.get("ts", 0.0)) < max_age_sec):
        return cached["prices"]
    url = "https://api.binance.com/api/v3/ticker/price"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "tradeforge/1.0"},
    )
    with urllib.request.urlopen(req, timeout=8) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    out: Dict[str, float] = {}
    if isinstance(data, list):
        for row in data:
            try:
                sym = str(row.get("symbol", "")).upper()
                px = float(row.get("price", "nan"))
                if sym.endswith("USDT") and math.isfinite(px) and px > 0:
                    out[sym] = px
            except Exception:
                continue
    _BINANCE_ALL_PRICES_CACHE["ts"] = now
    _BINANCE_ALL_PRICES_CACHE["prices"] = out
    return out


def _stable_jitter(symbol: str, exchange: str, bps: float = 8.0) -> float:
    # Deterministic tiny offset (in basis points) to emulate exchange micro-diffs.
    key = f"{symbol}|{exchange}".encode("utf-8")
    h = hashlib.md5(key).digest()
    val = int.from_bytes(h[:2], "big") / 65535.0  # [0..1]
    centered = (val - 0.5) * 2.0
    return centered * (bps / 10000.0)


def _coin_name_from_symbol(sym: str) -> str:
    mapping = {
        "BTC": "Bitcoin",
        "ETH": "Ethereum",
        "SOL": "Solana",
        "XRP": "XRP",
        "BNB": "BNB",
        "DOGE": "Dogecoin",
        "ADA": "Cardano",
        "AVAX": "Avalanche",
        "TON": "Toncoin",
        "TRX": "TRON",
    }
    return mapping.get(sym, sym)


def _fetch_json_url(url: str, timeout: int = 8) -> Any:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "tradeforge/1.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _book_cache_get(exchange: str, max_age_sec: int = 8) -> Optional[Dict[str, Dict[str, float]]]:
    row = _EXCHANGE_BOOK_CACHE.get(exchange)
    now = time.time()
    if row and row.get("data") and (now - float(row.get("ts", 0.0)) < max_age_sec):
        return row["data"]
    return None


def _book_cache_get_stale(exchange: str) -> Optional[Dict[str, Dict[str, float]]]:
    row = _EXCHANGE_BOOK_CACHE.get(exchange)
    if row and row.get("data"):
        return row["data"]
    return None


def _book_cache_set(exchange: str, data: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    _EXCHANGE_BOOK_CACHE[exchange] = {"ts": time.time(), "data": data}
    return data


def _bybit_spot_books(max_age_sec: int = 8) -> Dict[str, Dict[str, float]]:
    cached = _book_cache_get("BYBIT", max_age_sec=max_age_sec)
    if cached is not None:
        return cached
    out: Dict[str, Dict[str, float]] = {}
    try:
        data = _fetch_json_url("https://api.bybit.com/v5/market/tickers?category=spot")
        rows = (((data or {}).get("result") or {}).get("list") or []) if isinstance(data, dict) else []
        for row in rows:
            try:
                sym = str(row.get("symbol", "")).upper().strip()
                bid = float(row.get("bid1Price", "nan"))
                ask = float(row.get("ask1Price", "nan"))
                if sym.endswith("USDT") and math.isfinite(bid) and math.isfinite(ask) and bid > 0 and ask > 0:
                    out[sym] = {"bid": bid, "ask": ask}
            except Exception:
                continue
    except Exception:
        out = {}
    if not out:
        return _book_cache_get_stale("BYBIT") or {}
    return _book_cache_set("BYBIT", out)


def _bitget_spot_books(max_age_sec: int = 8) -> Dict[str, Dict[str, float]]:
    cached = _book_cache_get("BITGET", max_age_sec=max_age_sec)
    if cached is not None:
        return cached
    out: Dict[str, Dict[str, float]] = {}
    try:
        data = _fetch_json_url("https://api.bitget.com/api/v2/spot/market/tickers")
        rows = (data or {}).get("data", []) if isinstance(data, dict) else []
        for row in rows:
            try:
                sym = str(row.get("symbol", "")).upper().replace("-", "").replace("_", "").strip()
                bid = float(row.get("bidPr", "nan"))
                ask = float(row.get("askPr", "nan"))
                if sym.endswith("USDT") and math.isfinite(bid) and math.isfinite(ask) and bid > 0 and ask > 0:
                    out[sym] = {"bid": bid, "ask": ask}
            except Exception:
                continue
    except Exception:
        out = {}
    if not out:
        return _book_cache_get_stale("BITGET") or {}
    return _book_cache_set("BITGET", out)


def _mexc_spot_books(max_age_sec: int = 8) -> Dict[str, Dict[str, float]]:
    cached = _book_cache_get("MEXC", max_age_sec=max_age_sec)
    if cached is not None:
        return cached
    out: Dict[str, Dict[str, float]] = {}
    try:
        data = _fetch_json_url("https://api.mexc.com/api/v3/ticker/bookTicker")
        rows = data if isinstance(data, list) else []
        for row in rows:
            try:
                sym = str(row.get("symbol", "")).upper().strip()
                bid = float(row.get("bidPrice", "nan"))
                ask = float(row.get("askPrice", "nan"))
                if sym.endswith("USDT") and math.isfinite(bid) and math.isfinite(ask) and bid > 0 and ask > 0:
                    out[sym] = {"bid": bid, "ask": ask}
            except Exception:
                continue
    except Exception:
        out = {}
    if not out:
        return _book_cache_get_stale("MEXC") or {}
    return _book_cache_set("MEXC", out)


def _binance_24h_stats(max_age_sec: int = 45) -> Dict[str, Dict[str, float]]:
    now = time.time()
    cached = _BINANCE_24H_CACHE
    if cached.get("rows") and (now - float(cached.get("ts", 0.0)) < max_age_sec):
        return cached["rows"]
    out: Dict[str, Dict[str, float]] = {}
    try:
        data = _fetch_json_url("https://api.binance.com/api/v3/ticker/24hr", timeout=10)
        rows = data if isinstance(data, list) else []
        for row in rows:
            try:
                sym = str(row.get("symbol", "")).upper().strip()
                if not sym.endswith("USDT"):
                    continue
                last = float(row.get("lastPrice", "nan"))
                change = float(row.get("priceChangePercent", "nan"))
                qv = float(row.get("quoteVolume", "nan"))
                if math.isfinite(last) and math.isfinite(change) and math.isfinite(qv):
                    out[sym] = {
                        "lastPrice": last,
                        "changePct": change,
                        "quoteVolume": qv,
                    }
            except Exception:
                continue
    except Exception:
        out = {}
    if not out and cached.get("rows"):
        return cached["rows"]
    _BINANCE_24H_CACHE["ts"] = now
    _BINANCE_24H_CACHE["rows"] = out
    return out


def _mexc_perp_tickers(max_age_sec: int = 20) -> Dict[str, Dict[str, float]]:
    now = time.time()
    cached = _MEXC_PERP_TICKER_CACHE
    if cached.get("rows") and (now - float(cached.get("ts", 0.0)) < max_age_sec):
        return cached["rows"]
    out: Dict[str, Dict[str, float]] = {}
    try:
        data = _fetch_json_url("https://contract.mexc.com/api/v1/contract/ticker", timeout=10)
        rows = ((data or {}).get("data") or []) if isinstance(data, dict) else []
        for row in rows:
            try:
                sym = str(row.get("symbol", "")).upper().strip()
                last = float(row.get("lastPrice", "nan"))
                bid = float(row.get("bid1", "nan"))
                ask = float(row.get("ask1", "nan"))
                change = float(row.get("riseFallRate", "nan")) * 100.0
                rf = row.get("riseFallRates") or {}
                quote_vol = float(row.get("amount24", "nan"))
                funding = float(row.get("fundingRate", "nan"))
                hold_vol = float(row.get("holdVol", "nan"))
                if sym.endswith("_USDT") and math.isfinite(last):
                    out[sym] = {
                        "lastPrice": last,
                        "bid": bid if math.isfinite(bid) and bid > 0 else last,
                        "ask": ask if math.isfinite(ask) and ask > 0 else last,
                        "changePct": change if math.isfinite(change) else 0.0,
                        "change24h": (float(rf.get("r", row.get("riseFallRate", 0.0))) * 100.0) if math.isfinite(float(rf.get("r", row.get("riseFallRate", 0.0)))) else 0.0,
                        "change7d": (float(rf.get("r7", 0.0)) * 100.0) if math.isfinite(float(rf.get("r7", 0.0))) else 0.0,
                        "change30d": (float(rf.get("r30", 0.0)) * 100.0) if math.isfinite(float(rf.get("r30", 0.0))) else 0.0,
                        "change90d": (float(rf.get("r90", 0.0)) * 100.0) if math.isfinite(float(rf.get("r90", 0.0))) else 0.0,
                        "quoteVolume": quote_vol if math.isfinite(quote_vol) else 0.0,
                        "fundingRate": funding if math.isfinite(funding) else 0.0,
                        "holdVol": hold_vol if math.isfinite(hold_vol) else 0.0,
                        "timestamp": float(row.get("timestamp", 0.0) or 0.0),
                    }
            except Exception:
                continue
    except Exception:
        out = {}
    if not out and cached.get("rows"):
        return cached["rows"]
    _MEXC_PERP_TICKER_CACHE["ts"] = now
    _MEXC_PERP_TICKER_CACHE["rows"] = out
    return out


def _mexc_perp_details(max_age_sec: int = 600) -> Dict[str, Dict[str, Any]]:
    now = time.time()
    cached = _MEXC_PERP_DETAIL_CACHE
    if cached.get("rows") and (now - float(cached.get("ts", 0.0)) < max_age_sec):
        return cached["rows"]
    out: Dict[str, Dict[str, Any]] = {}
    try:
        data = _fetch_json_url("https://contract.mexc.com/api/v1/contract/detail", timeout=10)
        rows = ((data or {}).get("data") or []) if isinstance(data, dict) else []
        for row in rows:
            try:
                sym = str(row.get("symbol", "")).upper().strip()
                out[sym] = {
                    "baseCoin": str(row.get("baseCoin", "")).upper().strip(),
                    "quoteCoin": str(row.get("quoteCoin", "")).upper().strip(),
                    "displayNameEn": str(row.get("displayNameEn", "")).strip(),
                    "state": int(row.get("state", 0) or 0),
                    "isHidden": bool(row.get("isHidden", False)),
                    "isNew": bool(row.get("isNew", False)),
                    "isHot": bool(row.get("isHot", False)),
                }
            except Exception:
                continue
    except Exception:
        out = {}
    if not out and cached.get("rows"):
        return cached["rows"]
    _MEXC_PERP_DETAIL_CACHE["ts"] = now
    _MEXC_PERP_DETAIL_CACHE["rows"] = out
    return out


def _major_coin(sym: str) -> bool:
    return sym in {"BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "ADA", "TRX", "TON"}


def _stable_or_quote_like(sym: str) -> bool:
    return sym in {"USDT", "USDC", "FDUSD", "BUSD", "TUSD", "USDP", "DAI", "EUR", "EURC", "PAXG", "USD1", "USDE", "USDS"}


def _mexc_window_change(row: Dict[str, Any], window: str) -> float:
    win = str(window or "24h").lower()
    if win == "7d":
        return float(row.get("change7d", row.get("changePct", 0.0)) or 0.0)
    if win == "30d":
        return float(row.get("change30d", row.get("changePct", 0.0)) or 0.0)
    if win == "90d":
        return float(row.get("change90d", row.get("changePct", 0.0)) or 0.0)
    return float(row.get("change24h", row.get("changePct", 0.0)) or 0.0)


def _mexc_pump_score(row: Dict[str, Any], meta: Dict[str, Any], window: str = "24h") -> float:
    change = _mexc_window_change(row, window)
    quote_vol = max(0.0, float(row.get("quoteVolume", 0.0) or 0.0))
    hold_vol = max(0.0, float(row.get("holdVol", 0.0) or 0.0))
    funding = float(row.get("fundingRate", 0.0) or 0.0)
    volume_component = max(0.0, math.log10(quote_vol + 1.0) - 5.0) * 10.0
    hold_component = max(0.0, math.log10(hold_vol + 1.0) - 3.0) * 6.0
    funding_component = max(0.0, funding * 10000.0) * 0.35
    hot_bonus = 5.0 if bool(meta.get("isHot")) else 0.0
    new_bonus = 2.5 if bool(meta.get("isNew")) else 0.0
    trend_component = max(0.0, change) * 1.8
    return round(trend_component + volume_component + hold_component + funding_component + hot_bonus + new_bonus, 3)


def _market_top_spot(limit: int = 500) -> Dict[str, Any]:
    stats = _binance_24h_stats(max_age_sec=45)
    coins: List[Dict[str, Any]] = []
    for pair, row in stats.items():
        if not pair.endswith("USDT"):
            continue
        base = pair[:-4]
        if len(base) < 2:
            continue
        if _stable_or_quote_like(base):
            continue
        coins.append(
            {
                "symbol": base,
                "name": _coin_name_from_symbol(base),
                "rank": 0,
                "volume24h": float(row.get("quoteVolume", 0.0) or 0.0),
                "change24h": float(row.get("changePct", 0.0) or 0.0),
                "lastPrice": float(row.get("lastPrice", 0.0) or 0.0),
            }
        )
    coins.sort(key=lambda x: float(x.get("volume24h", 0.0)), reverse=True)
    for idx, row in enumerate(coins, start=1):
        row["rank"] = idx
    return {"coins": coins[: max(1, int(limit))], "source": "binance_spot"}


def _market_top_mexc_perp(limit: int = 500, direction: str = "all", window: str = "24h") -> Dict[str, Any]:
    tickers = _mexc_perp_tickers(max_age_sec=20)
    details = _mexc_perp_details(max_age_sec=600)
    direction = str(direction or "all").lower()
    coins: List[Dict[str, Any]] = []
    for sym, row in tickers.items():
        meta = details.get(sym) or {}
        base = str(meta.get("baseCoin") or sym.replace("_USDT", "")).upper()
        quote = str(meta.get("quoteCoin") or "USDT").upper()
        if quote != "USDT":
            continue
        if _major_coin(base):
            continue
        if meta.get("isHidden") or int(meta.get("state", 0) or 0) != 0:
            continue
        change = _mexc_window_change(row, window)
        if direction == "pump" and change <= 0:
            continue
        if direction == "dump" and change >= 0:
            continue
        pump_score = _mexc_pump_score(row, meta, window)
        coins.append(
            {
                "symbol": base,
                "name": str(meta.get("displayNameEn") or f"{base} PERPETUAL"),
                "rank": 0,
                "volume24h": float(row.get("quoteVolume", 0.0) or 0.0),
                "change24h": change,
                "lastPrice": float(row.get("lastPrice", 0.0) or 0.0),
                "fundingRate": float(row.get("fundingRate", 0.0) or 0.0),
                "holdVol": float(row.get("holdVol", 0.0) or 0.0),
                "pumpScore": pump_score,
                "contract": sym,
                "venue": "MEXC",
                "marketType": "perpetual",
                "isHot": bool(meta.get("isHot")),
                "isNew": bool(meta.get("isNew")),
            }
        )
    if direction == "pump":
        coins.sort(key=lambda x: (float(x.get("pumpScore", 0.0)), float(x.get("change24h", 0.0))), reverse=True)
    elif direction == "dump":
        coins.sort(key=lambda x: abs(float(x.get("change24h", 0.0))), reverse=True)
    else:
        coins.sort(key=lambda x: (abs(float(x.get("change24h", 0.0))), float(x.get("pumpScore", 0.0))), reverse=True)
    for idx, row in enumerate(coins, start=1):
        row["rank"] = idx
    return {"coins": coins[: max(1, int(limit))], "source": "mexc_perpetual"}


def _market_snapshot_spot(symbol_list: List[str]) -> Dict[str, Any]:
    prices_all = _binance_all_prices(max_age_sec=8)
    changes_all = _binance_24h_stats(max_age_sec=45)
    exchanges = ["BYBIT", "BITGET", "MEXC", "OKX", "BINGX", "BITMART", "COINEX"]
    real_books = {
        "BYBIT": _bybit_spot_books(max_age_sec=8),
        "BITGET": _bitget_spot_books(max_age_sec=8),
        "MEXC": _mexc_spot_books(max_age_sec=8),
    }

    prices: Dict[str, Dict[str, Optional[float]]] = {ex: {} for ex in exchanges}
    book: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {ex: {} for ex in exchanges}
    momentums: Dict[str, Dict[str, Any]] = {}
    for sym in symbol_list:
        pair = f"{sym}USDT"
        pair_change = changes_all.get(pair) or {}
        change_pct = float(pair_change.get("changePct", 0.0) or 0.0)
        last_price = float(pair_change.get("lastPrice", prices_all.get(pair, 0.0)) or 0.0)
        ex_count = 0
        for ex in exchanges:
            if ex in real_books:
                row = real_books[ex].get(pair)
                if row:
                    bid = float(row.get("bid", 0.0) or 0.0)
                    ask = float(row.get("ask", 0.0) or 0.0)
                    if bid > 0 and ask > 0:
                        prices[ex][sym] = ask
                        book[ex][sym] = {"ask": ask, "bid": bid}
                        ex_count += 1
                        continue
            prices[ex][sym] = None
        momentums[sym] = {
            "grossPct": change_pct,
            "netPct": change_pct,
            "changePct": change_pct,
            "lastPrice": last_price,
            "source": "BINANCE_24H",
            "exCount": ex_count,
        }
    return {"prices": prices, "book": book, "momentums": momentums, "ts": int(time.time())}


def _market_snapshot_mexc_perp(symbol_list: List[str], window: str = "24h") -> Dict[str, Any]:
    tickers = _mexc_perp_tickers(max_age_sec=20)
    exchanges = ["BYBIT", "BITGET", "MEXC", "OKX", "BINGX", "BITMART", "COINEX"]
    prices: Dict[str, Dict[str, Optional[float]]] = {ex: {} for ex in exchanges}
    book: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {ex: {} for ex in exchanges}
    momentums: Dict[str, Dict[str, Any]] = {}
    for sym in symbol_list:
        contract = f"{sym}_USDT"
        row = tickers.get(contract) or {}
        bid = float(row.get("bid", 0.0) or 0.0)
        ask = float(row.get("ask", 0.0) or 0.0)
        last = float(row.get("lastPrice", 0.0) or 0.0)
        change = _mexc_window_change(row, window)
        funding = float(row.get("fundingRate", 0.0) or 0.0)
        hold_vol = float(row.get("holdVol", 0.0) or 0.0)
        for ex in exchanges:
            prices[ex][sym] = None
        if last > 0:
            prices["MEXC"][sym] = ask if ask > 0 else last
            book["MEXC"][sym] = {
                "ask": ask if ask > 0 else last,
                "bid": bid if bid > 0 else last,
            }
        momentums[sym] = {
            "grossPct": change,
            "netPct": change,
            "changePct": change,
            "lastPrice": last,
            "fundingRate": funding,
            "holdVol": hold_vol,
            "source": "MEXC_PERP",
            "exCount": 1 if last > 0 else 0,
        }
    return {"prices": prices, "book": book, "momentums": momentums, "ts": int(time.time())}


def _api_cache_get(ns: str, key: str, ttl_sec: int) -> Optional[Any]:
    now = time.time()
    k = f"{ns}:{key}"
    with _API_RESP_CACHE_LOCK:
        row = _API_RESP_CACHE.get(k)
        if not row:
            _API_CACHE_STATS["misses"] += 1
            return None
        if (now - float(row.get("ts", 0.0))) > ttl_sec:
            _API_RESP_CACHE.pop(k, None)
            _API_CACHE_STATS["misses"] += 1
            _API_CACHE_STATS["expired"] += 1
            return None
        _API_CACHE_STATS["hits"] += 1
        return row.get("value")


def _api_cache_set(ns: str, key: str, value: Any) -> None:
    now = time.time()
    k = f"{ns}:{key}"
    with _API_RESP_CACHE_LOCK:
        _API_RESP_CACHE[k] = {"ts": now, "value": value}
        _API_CACHE_STATS["sets"] += 1
        # Simple bounded cache eviction by oldest timestamp
        if len(_API_RESP_CACHE) > _API_CACHE_MAX_ITEMS:
            overflow = len(_API_RESP_CACHE) - _API_CACHE_MAX_ITEMS
            if overflow > 0:
                oldest = sorted(_API_RESP_CACHE.items(), key=lambda kv: float(kv[1].get("ts", 0.0)))[:overflow]
                for old_k, _ in oldest:
                    _API_RESP_CACHE.pop(old_k, None)
                _API_CACHE_STATS["evicted"] += int(overflow)


def _load_trades_csv(path: Path) -> List[Dict[str, Any]]:
    key = str(path)
    if not path.exists():
        return []
    mtime = path.stat().st_mtime
    cached = _TRADES_CACHE.get(key)
    if cached and cached.get("mtime") == mtime:
        return cached["rows"]
    df = pd.read_csv(path)
    for col in ("entry_ts", "exit_ts"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        entry_ts = r.get("entry_ts")
        exit_ts = r.get("exit_ts")
        if pd.isna(entry_ts) or pd.isna(exit_ts):
            # allow open trades (exit_ts missing) for live display
            if pd.isna(entry_ts):
                continue
            entry_ts = pd.to_datetime(entry_ts, utc=True, errors="coerce")
            if pd.isna(entry_ts):
                continue
            exit_ts = entry_ts
            entry_price = r.get("entry_price", 0.0)
            exit_price = r.get("exit_price", entry_price)
            if entry_price is None or (isinstance(entry_price, float) and math.isnan(entry_price)):
                entry_price = 0.0
            if exit_price is None or (isinstance(exit_price, float) and math.isnan(exit_price)):
                exit_price = entry_price
            rows.append(
                {
                    "entry_time": int(entry_ts.timestamp()),
                    "exit_time": int(exit_ts.timestamp()),
                    "direction": str(r.get("direction", "")),
                    "entry_price": float(entry_price),
                    "exit_price": float(exit_price),
                    "model": str(r.get("model", "")),
                    "open": True,
                }
            )
            continue
        rows.append(
            {
                "entry_time": int(entry_ts.timestamp()),
                "exit_time": int(exit_ts.timestamp()),
                "direction": str(r.get("direction", "")),
                "entry_price": float(r.get("entry_price", 0.0)),
                "exit_price": float(r.get("exit_price", 0.0)),
                "model": str(r.get("model", "")),
            }
        )
    _TRADES_CACHE[key] = {"mtime": mtime, "rows": rows}
    return rows


def _load_leverage_results(reports_dir: Path) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for (year, lev), name in _LEV_FILES.items():
        path = reports_dir / name
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "final_equity" not in df.columns or df.empty:
            continue
        val = float(df.iloc[-1]["final_equity"])
        results.setdefault(str(year), {})[str(lev)] = val
    return results


def _ensure_live_trades_header(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["entry_ts", "exit_ts", "direction", "entry_price", "exit_price", "model"])


def _parse_live_trade_ts(value: Any) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        ts = pd.to_datetime(text, utc=True, errors="coerce")
    except Exception:
        return None
    if pd.isna(ts):
        return None
    return int(ts.timestamp())


def _dedupe_live_trade_file(path: Path) -> None:
    _ensure_live_trades_header(path)
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return
    if not rows:
        return
    unique_rows: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str, str]] = set()
    for row in rows:
        key = (
            str(row.get("entry_ts") or "").strip(),
            str(row.get("exit_ts") or "").strip(),
            str(row.get("direction") or "").strip().lower(),
            str(row.get("entry_price") or "").strip(),
            str(row.get("exit_price") or "").strip(),
            str(row.get("model") or "").strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)
    if len(unique_rows) == len(rows):
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["entry_ts", "exit_ts", "direction", "entry_price", "exit_price", "model"],
        )
        writer.writeheader()
        writer.writerows(unique_rows)


def _restore_live_trade_state(path: Path) -> Dict[str, Any]:
    _dedupe_live_trade_file(path)
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return {"open": None, "last_base_ts": None}
    open_rows: Dict[tuple[str, int, str], Dict[str, Any]] = {}
    last_base_ts: Optional[int] = None
    for row in rows:
        entry_ts = _parse_live_trade_ts(row.get("entry_ts"))
        exit_ts = _parse_live_trade_ts(row.get("exit_ts"))
        direction = str(row.get("direction") or "").strip().lower()
        entry_price_text = str(row.get("entry_price") or "").strip()
        if entry_ts is not None:
            last_base_ts = max(last_base_ts or entry_ts, entry_ts)
        if exit_ts is not None:
            last_base_ts = max(last_base_ts or exit_ts, exit_ts)
        if not direction or entry_ts is None or not entry_price_text:
            continue
        key = (direction, entry_ts, entry_price_text)
        if exit_ts is None:
            open_rows[key] = {
                "direction": direction,
                "entry_ts": entry_ts,
                "entry_price": float(entry_price_text),
            }
        else:
            open_rows.pop(key, None)
    current_open = None
    if open_rows:
        current_open = max(open_rows.values(), key=lambda r: int(r.get("entry_ts") or 0))
    return {"open": current_open, "last_base_ts": last_base_ts}


def _ensure_live_trade_state_loaded() -> None:
    if _LIVE_TRADE_STATE.get("_bootstrapped"):
        return
    restored = _restore_live_trade_state(_LIVE_TRADES_PATH)
    _LIVE_TRADE_STATE["open"] = restored.get("open")
    _LIVE_TRADE_STATE["last_base_ts"] = restored.get("last_base_ts")
    _LIVE_TRADE_STATE["_bootstrapped"] = True


def _ensure_live_model_trade_state_loaded(model_key: str) -> Dict[str, Any]:
    state = _LIVE_MODEL_TRADE_STATE.get(model_key)
    if state and state.get("_bootstrapped"):
        return state
    restored = _restore_live_trade_state(_live_model_trade_path(model_key))
    restored["_bootstrapped"] = True
    _LIVE_MODEL_TRADE_STATE[model_key] = restored
    return restored


def _append_live_trade(entry_ts: int, exit_ts: Optional[int], direction: str, entry_price: float, exit_price: Optional[float]) -> None:
    _ensure_live_trades_header(_LIVE_TRADES_PATH)
    with _LIVE_TRADES_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                pd.to_datetime(entry_ts, unit="s", utc=True).isoformat(),
                "" if exit_ts is None else pd.to_datetime(exit_ts, unit="s", utc=True).isoformat(),
                direction,
                float(entry_price),
                "" if exit_price is None else float(exit_price),
                "live_model",
            ]
        )


def _append_live_trade_generic(path: Path, entry_ts: int, exit_ts: Optional[int], direction: str, entry_price: float, exit_price: Optional[float], model: str) -> None:
    _ensure_live_trades_header(path)
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                pd.to_datetime(entry_ts, unit="s", utc=True).isoformat(),
                "" if exit_ts is None else pd.to_datetime(exit_ts, unit="s", utc=True).isoformat(),
                direction,
                float(entry_price),
                "" if exit_price is None else float(exit_price),
                str(model),
            ]
        )


def _update_live_trades(status: str, base_ts: int, base_price: float) -> None:
    _ensure_live_trades_header(_LIVE_TRADES_PATH)
    _ensure_live_trade_state_loaded()
    # Avoid duplicate updates on the same candle
    if _LIVE_TRADE_STATE.get("last_base_ts") == base_ts:
        return
    _LIVE_TRADE_STATE["last_base_ts"] = base_ts

    open_trade = _LIVE_TRADE_STATE.get("open")

    if status in ("LONG", "SHORT"):
        if open_trade is None:
            _LIVE_TRADE_STATE["open"] = {
                "direction": status.lower(),
                "entry_ts": base_ts,
                "entry_price": base_price,
            }
            _append_live_trade(base_ts, None, status.lower(), base_price, None)
            return
        if open_trade["direction"] != status.lower():
            # close and flip
            _close_live_trade(
                open_trade["entry_ts"], base_ts, open_trade["direction"], open_trade["entry_price"], base_price
            )
            _LIVE_TRADE_STATE["open"] = {
                "direction": status.lower(),
                "entry_ts": base_ts,
                "entry_price": base_price,
            }
            _append_live_trade(base_ts, None, status.lower(), base_price, None)
        return

    # status == FLAT -> close if open
    if open_trade is not None:
        _close_live_trade(
            open_trade["entry_ts"], base_ts, open_trade["direction"], open_trade["entry_price"], base_price
        )
        _LIVE_TRADE_STATE["open"] = None


def _live_model_trade_path(model_key: str) -> Path:
    return Path(__file__).resolve().parent / "reports" / f"trades_live_{model_key}.csv"


def _update_live_model_trade(model_key: str, status: str, base_ts: int, base_price: float) -> None:
    state = _ensure_live_model_trade_state_loaded(model_key)
    if state.get("last_base_ts") == base_ts:
        return
    state["last_base_ts"] = base_ts
    path = _live_model_trade_path(model_key)
    open_trade = state.get("open")
    lowered = str(status or "").lower()

    if lowered in ("long", "short"):
        if open_trade is None:
            state["open"] = {
                "direction": lowered,
                "entry_ts": base_ts,
                "entry_price": base_price,
            }
            _append_live_trade_generic(path, base_ts, None, lowered, base_price, None, model_key)
            return
        if open_trade["direction"] != lowered:
            _append_live_trade_generic(
                path,
                open_trade["entry_ts"],
                base_ts,
                open_trade["direction"],
                open_trade["entry_price"],
                base_price,
                model_key,
            )
            state["open"] = {
                "direction": lowered,
                "entry_ts": base_ts,
                "entry_price": base_price,
            }
            _append_live_trade_generic(path, base_ts, None, lowered, base_price, None, model_key)
        return

    if open_trade is not None:
        _append_live_trade_generic(
            path,
            open_trade["entry_ts"],
            base_ts,
            open_trade["direction"],
            open_trade["entry_price"],
            base_price,
            model_key,
        )
        state["open"] = None


def _append_signal_quality_log(record: Dict[str, Any]) -> None:
    _SIGNAL_QUALITY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _SIGNAL_QUALITY_LOG_LOCK:
        with _SIGNAL_QUALITY_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_liq_feed(path: Path, limit: int = 600) -> List[Dict[str, Any]]:
    key = str(path)
    if not path.exists():
        return []
    mtime = path.stat().st_mtime
    cached = _LIQ_FEED_CACHE.get(key)
    if cached and cached.get("mtime") == mtime:
        return cached["rows"]
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                topic = str(msg.get("topic") or "")
                if not topic.startswith("liquidation."):
                    continue
                data = msg.get("data")
                if isinstance(data, dict):
                    data = [data]
                if not isinstance(data, list):
                    continue
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    ts = item.get("time") or item.get("T") or item.get("t")
                    if ts is None:
                        continue
                    # Bybit timestamps are ms
                    if ts > 10_000_000_000:
                        ts = int(ts / 1000)
                    side = str(item.get("side") or item.get("S") or "").lower()
                    qty = float(item.get("qty") or item.get("size") or item.get("q") or 0.0)
                    if qty <= 0:
                        continue
                    rows.append({"time": int(ts), "side": side, "qty": qty})
    except OSError:
        return []
    rows = rows[-max(10, int(limit)) :]
    _LIQ_FEED_CACHE[key] = {"mtime": mtime, "rows": rows}
    return rows


def _fetch_binance_klines(
    symbol: str,
    interval: str,
    limit: int,
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> List[Dict[str, float]]:
    allowed = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"}
    symbol = (symbol or "BTCUSDT").upper()
    interval = (interval or "15m").lower()
    if interval not in allowed:
        interval = "15m"
    # Binance endpoint limits one response to <=1500 bars.
    # Allow deeper pagination for long dashboard history requests.
    total_limit = max(50, min(int(limit or 240), 50000))
    max_limit_by_interval = {
        "1m": 1200,
        "3m": 1200,
        "5m": 1500,
        "15m": 2200,
        "30m": 2600,
        "1h": 10000,
        "2h": 7000,
        "4h": 3000,
        "6h": 2400,
        "8h": 1800,
        "12h": 1500,
        "1d": 1200,
    }
    total_limit = min(total_limit, int(max_limit_by_interval.get(interval, 2200)))
    start_ts = int(start) if start is not None else None
    end_ts = int(end) if end is not None else None
    cache_key = f"{symbol}:{interval}:{total_limit}:{start_ts or ''}:{end_ts or ''}"
    now_ts = time.time()
    cached = _BINANCE_KLINES_CACHE.get(cache_key)
    if cached and (now_ts - float(cached.get("ts", 0.0)) <= 8.0):
        rows = cached.get("rows")
        if isinstance(rows, list) and rows:
            return rows

    remaining = total_limit
    batches: List[List[Any]] = []
    tries = 0

    if start_ts is not None or end_ts is not None:
        cursor_ms: Optional[int] = int(start_ts * 1000) if start_ts is not None else None
        end_time_ms: Optional[int] = int(end_ts * 1000) if end_ts is not None else None
        while remaining > 0 and tries < 80:
            tries += 1
            page_limit = min(1000, remaining)
            params: Dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": page_limit}
            if cursor_ms is not None:
                params["startTime"] = cursor_ms
            if end_time_ms is not None:
                params["endTime"] = end_time_ms
            qs = urllib.parse.urlencode(params)
            url = f"https://api.binance.com/api/v3/klines?{qs}"
            with urllib.request.urlopen(url, timeout=8) as resp:
                page = json.loads(resp.read().decode("utf-8"))
            if not isinstance(page, list) or not page:
                break
            batches.append(page)
            remaining -= len(page)
            last = page[-1]
            if not isinstance(last, list) or len(last) < 1:
                break
            last_open_ms = int(last[0])
            cursor_ms = last_open_ms + 1
            if end_time_ms is not None and last_open_ms >= end_time_ms:
                break
            if len(page) < page_limit:
                break
    else:
        end_time_ms: Optional[int] = None
        while remaining > 0 and tries < 8:
            tries += 1
            page_limit = min(1000, remaining)
            params: Dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": page_limit}
            if end_time_ms is not None:
                params["endTime"] = end_time_ms
            qs = urllib.parse.urlencode(params)
            url = f"https://api.binance.com/api/v3/klines?{qs}"
            with urllib.request.urlopen(url, timeout=8) as resp:
                page = json.loads(resp.read().decode("utf-8"))
            if not isinstance(page, list) or not page:
                break
            batches.append(page)
            remaining -= len(page)
            first = page[0]
            if not isinstance(first, list) or len(first) < 1:
                break
            end_time_ms = int(first[0]) - 1
            if len(page) < page_limit:
                break

    raw: List[Any] = []
    for b in batches:
        raw.extend(b)
    raw.sort(key=lambda r: int(r[0]) if isinstance(r, list) and r else 0)

    out: List[Dict[str, float]] = []
    for row in raw:
        if not isinstance(row, list) or len(row) < 5:
            continue
        ts_ms = int(row[0])
        out.append(
            {
                "time": int(ts_ms // 1000),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
            }
        )
    if out:
        _BINANCE_KLINES_CACHE[cache_key] = {"ts": now_ts, "rows": out}
    return out

class PredictRequest(BaseModel):
    window: List[List[float]] = Field(..., description="Sequence window, shape (seq_len, n_features)")
    mean: Optional[List[float]] = Field(None, description="Optional mean for normalization")
    std: Optional[List[float]] = Field(None, description="Optional std for normalization")
    mc_samples: int = Field(1, description="Monte Carlo dropout samples; >1 enables stochastic forward")

    @validator("window")
    def check_window(cls, v):
        if not v or not isinstance(v, list) or not isinstance(v[0], list):
            raise ValueError("window must be a 2D list")
        return v


class AuthRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=200)
    password: str = Field(..., min_length=10, max_length=200)

class VerifyEmailRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=200)
    code: str = Field(..., min_length=4, max_length=16)


class ResendCodeRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=200)


class RequestPasswordReset(BaseModel):
    email: str = Field(..., min_length=3, max_length=200)


class ResetPasswordRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=200)
    code: str = Field(..., min_length=4, max_length=16)
    new_password: str = Field(..., min_length=10, max_length=200)


class UpdateProfileRequest(BaseModel):
    full_name: Optional[str] = Field("", max_length=200)
    telegram: Optional[str] = Field("", max_length=200)


class SubmitPaymentRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=200)
    txid: str = Field(..., min_length=8, max_length=200)
    plan: str = Field("pro50", min_length=2, max_length=50)


class CreateYooKassaPaymentRequest(BaseModel):
    plan: str = Field("pro", min_length=2, max_length=50)
    billing_period: str = Field("monthly", min_length=2, max_length=32)


class AdminActivateSubscriptionRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=200)
    plan: str = Field("pro", min_length=2, max_length=50)
    days: int = Field(30, ge=1, le=3650)
    txid: Optional[str] = Field(None, max_length=200)
    note: Optional[str] = Field(None, max_length=500)


def _load_model(model_path: str) -> tf.keras.Model:
    print(f"[load_model] path={model_path}", flush=True)
    class _PatchedInputLayer(tf.keras.layers.InputLayer):
        @classmethod
        def from_config(cls, config):
            if "batch_shape" in config and "batch_input_shape" not in config:
                config["batch_input_shape"] = config.pop("batch_shape")
            return super().from_config(config)

    use_keras3 = os.environ.get("USE_KERAS3", "0") == "1"
    if use_keras3:
        import keras as _keras  # Keras 3 loader for new-format models
        # If USE_KERAS3=1, do not silently fall back to tf.keras.
        class LastStepK3(_keras.layers.Layer):
            def call(self, x):
                return x[:, -1, :]

        class RevINK3(_keras.layers.Layer):
            def __init__(self, affine: bool = True, eps: float = 1e-5, **kwargs):
                super().__init__(**kwargs)
                self.affine = bool(affine)
                self.eps = float(eps)
                self._mu = None
                self._std = None

            def build(self, input_shape):
                feat_dim = int(input_shape[-1])
                if self.affine:
                    self.gamma = self.add_weight(
                        name="gamma", shape=(1, 1, feat_dim), initializer="ones", trainable=True
                    )
                    self.beta = self.add_weight(
                        name="beta", shape=(1, 1, feat_dim), initializer="zeros", trainable=True
                    )
                super().build(input_shape)

            def call(self, x, mode: str = "norm"):
                if mode == "norm":
                    mu = tf.reduce_mean(x, axis=1, keepdims=True)
                    var = tf.reduce_mean(tf.square(x - mu), axis=1, keepdims=True)
                    std = tf.sqrt(var + self.eps)
                    self._mu = mu
                    self._std = std
                    y = (x - mu) / std
                    if self.affine:
                        y = y * self.gamma + self.beta
                    return y
                if mode == "denorm":
                    if self._mu is None or self._std is None:
                        raise ValueError("RevIN denorm called before norm.")
                    y = x
                    if self.affine:
                        y = (y - self.beta) / (self.gamma + self.eps)
                    return y * self._std + self._mu
                raise ValueError("mode must be 'norm' or 'denorm'")

            def get_config(self):
                return {"affine": self.affine, "eps": self.eps}

        class DropPathK3(_keras.layers.Layer):
            def __init__(self, drop_prob: float = 0.0, **kwargs):
                super().__init__(**kwargs)
                self.drop_prob = float(drop_prob)

            def call(self, x, training=False):
                if not training or self.drop_prob == 0.0:
                    return x
                keep = 1.0 - self.drop_prob
                shape = (tf.shape(x)[0],) + (1,) * (len(x.shape) - 1)
                rnd = keep + tf.random.uniform(shape, dtype=x.dtype)
                mask = tf.floor(rnd)
                return (x / keep) * mask

            def get_config(self):
                return {"drop_prob": self.drop_prob}

        class TSMixerBlockK3(_keras.layers.Layer):
            def __init__(self, mlp_dim: int, dropout: float = 0.0, **kwargs):
                super().__init__(**kwargs)
                self.mlp_dim = int(mlp_dim)
                self.dropout = float(dropout)
                self.ln_time = _keras.layers.LayerNormalization(epsilon=1e-5)
                self.ln_feat = _keras.layers.LayerNormalization(epsilon=1e-5)
                self.do = _keras.layers.Dropout(self.dropout)

            def build(self, input_shape):
                seq_len = int(input_shape[1])
                feat_dim = int(input_shape[2])
                self.time_dense = _keras.layers.Dense(seq_len)
                self.ff1 = _keras.layers.Dense(self.mlp_dim, activation="gelu")
                self.ff2 = _keras.layers.Dense(feat_dim)
                super().build(input_shape)

            def call(self, x, training=False):
                y = self.ln_time(x)
                y = tf.transpose(y, [0, 2, 1])
                y = self.time_dense(y)
                y = tf.transpose(y, [0, 2, 1])
                y = self.do(y, training=training)
                x = x + y

                y = self.ln_feat(x)
                y = self.ff1(y)
                y = self.do(y, training=training)
                y = self.ff2(y)
                y = self.do(y, training=training)
                return x + y

            def get_config(self):
                return {"mlp_dim": self.mlp_dim, "dropout": self.dropout}

        class ITransformerBlockK3(_keras.layers.Layer):
            def __init__(self, seq_len: int, d_model: int, heads: int, dropout: float = 0.0, **kwargs):
                super().__init__(**kwargs)
                self.seq_len = int(seq_len)
                self.d_model = int(d_model)
                self.heads = int(heads)
                self.dropout = float(dropout)
                self.ln1 = _keras.layers.LayerNormalization(epsilon=1e-5)
                self.ln2 = _keras.layers.LayerNormalization(epsilon=1e-5)

            def build(self, input_shape):
                key_dim = max(self.d_model // max(self.heads, 1), 8)
                self.proj = _keras.layers.Dense(self.d_model)
                self.mha = _keras.layers.MultiHeadAttention(
                    num_heads=self.heads, key_dim=key_dim, dropout=self.dropout
                )
                self.ff1 = _keras.layers.Dense(self.d_model * 4, activation="gelu")
                self.ff2 = _keras.layers.Dense(self.d_model)
                self.to_time = _keras.layers.Dense(self.seq_len)
                self.do = _keras.layers.Dropout(self.dropout)
                super().build(input_shape)

            def call(self, x, training=False):
                xt = tf.transpose(x, [0, 2, 1])
                h = self.proj(xt)
                h1 = self.ln1(h)
                h1 = self.mha(h1, h1)
                h1 = self.do(h1, training=training)
                h = h + h1

                h2 = self.ln2(h)
                h2 = self.ff1(h2)
                h2 = self.do(h2, training=training)
                h2 = self.ff2(h2)
                h2 = self.do(h2, training=training)
                h = h + h2

                yt = self.to_time(h)
                y = tf.transpose(yt, [0, 2, 1])
                return x + y

            def get_config(self):
                return {
                    "seq_len": self.seq_len,
                    "d_model": self.d_model,
                    "heads": self.heads,
                    "dropout": self.dropout,
                }

        custom_objects_keras = {
            "RevIN": RevINK3,
            "model6>RevIN": RevINK3,
            "TSMixerBlock": TSMixerBlockK3,
            "model6>TSMixerBlock": TSMixerBlockK3,
            "ITransformerBlock": ITransformerBlockK3,
            "model6>ITransformerBlock": ITransformerBlockK3,
            "LastStep": LastStepK3,
            "model6>LastStep": LastStepK3,
            "DropPath": DropPathK3,
            "model6>DropPath": DropPathK3,
        }
        return _keras.models.load_model(model_path, custom_objects=custom_objects_keras, compile=False)
    custom_objects = {
        "RevIN": RevIN,
        "model6>RevIN": RevIN,
        "TSMixerBlock": TSMixerBlock,
        "model6>TSMixerBlock": TSMixerBlock,
        "ITransformerBlock": ITransformerBlock,
        "model6>ITransformerBlock": ITransformerBlock,
        "LastStep": LastStep,
        "model6>LastStep": LastStep,
        "DropPath": DropPath,
        "model6>DropPath": DropPath,
        "InputLayer": _PatchedInputLayer,
    }
    custom_objects["Functional"] = tf.keras.Model
    return tf.keras.models.load_model(
        model_path,
        custom_objects=custom_objects,
        compile=False,
        safe_mode=False,
    )


def _load_stats_meta(path: Optional[str]) -> dict:
    if not path:
        return {}
    stats = np.load(path, allow_pickle=True)
    meta = {k: stats[k] for k in stats.files}
    return meta


def _unpack_obj(val):
    if isinstance(val, np.ndarray) and val.dtype == object:
        if val.size == 1:
            return val.item()
    return val


def _get_price_head_scale(stats_meta: dict) -> dict:
    scale = _unpack_obj(stats_meta.get("price_head_scale"))
    return scale if isinstance(scale, dict) else {}


def _get_price_target_mode(stats_meta: dict) -> str:
    mode = _unpack_obj(stats_meta.get("price_target_mode"))
    if isinstance(mode, str) and mode:
        return mode
    return "cumulative"


def _parse_list(val: Optional[str]) -> List[str]:
    if not val:
        return []
    return [v.strip() for v in val.split(",") if v.strip()]


def _parse_weights(val: Optional[str], n: int) -> Optional[List[float]]:
    if not val:
        return None
    parts = [v.strip() for v in val.split(",") if v.strip()]
    if len(parts) != n:
        return None
    try:
        weights = [float(x) for x in parts]
    except Exception:
        return None
    s = float(sum(weights))
    if s <= 0:
        return None
    return [w / s for w in weights]


def _load_model_specs(
    model_paths: Optional[str],
    stats_paths: Optional[str],
    fallback_feature_cols: List[str],
    default_seq_len: int,
) -> List[dict]:
    models = _parse_list(model_paths)
    if not models:
        return []
    stats = _parse_list(stats_paths)
    if stats and len(stats) != len(models):
        raise ValueError("model/stats count mismatch (use comma-separated lists of equal length)")
    specs = []
    for i, model_path in enumerate(models):
        stats_path = stats[i] if stats else None
        stats_meta = _load_stats_meta(stats_path)
        seq_len = default_seq_len
        if "seq_len" in stats_meta:
            try:
                seq_len = int(np.asarray(stats_meta["seq_len"]).reshape(-1)[0])
            except Exception:
                seq_len = default_seq_len
        feature_cols = fallback_feature_cols
        if "feature_names" in stats_meta and len(stats_meta["feature_names"]) > 0:
            feature_cols = [str(x) for x in stats_meta["feature_names"]]
        specs.append(
            {
                "model": _load_model(model_path),
                "stats_path": stats_path,
                "stats_meta": stats_meta,
                "seq_len": seq_len,
                "feature_cols": feature_cols,
                "model_path": model_path,
            }
        )
    return specs


def _interval_to_minutes(interval: str) -> int:
    m = {
        "h20": 300,
        "h80": 1200,
        "h160": 2400,
        "h320": 4800,
        "h640": 9600,
        "15m": 15,
        "1h": 60,
        "3h": 180,
        "4h": 240,
        "6h": 360,
        "12h": 720,
        "24h": 1440,
    }
    return m.get(interval, 15)


def _load_features_df(features_path: str) -> pd.DataFrame:
    key = str(features_path)
    try:
        mtime = os.path.getmtime(key)
    except OSError:
        mtime = None
    cached = None
    if mtime is not None:
        with _FEATURES_CACHE_LOCK:
            cached = _FEATURES_CACHE.get(key)
            if cached and cached.get("mtime") == mtime:
                return cached["df"]
    df = pd.read_parquet(features_path).sort_values("timestamp")
    if "timestamp" not in df.columns:
        raise ValueError("features parquet missing 'timestamp' column")
    df = df[df["timestamp"].notna()].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if mtime is not None:
        with _FEATURES_CACHE_LOCK:
            _FEATURES_CACHE[key] = {"mtime": mtime, "df": df}
    return df


def _prepare_window(
    df: pd.DataFrame, seq_len: int, feature_cols: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    cols = ["timestamp", "close"] + [c for c in feature_cols if c not in ("timestamp", "close")]
    work = df[[c for c in cols if c in df.columns]].copy()
    for col in feature_cols:
        if col not in work.columns:
            work[col] = 0.0
    work[feature_cols] = work[feature_cols].ffill().fillna(0.0)
    if "close" in work.columns:
        work["close"] = work["close"].ffill().bfill()
    window = work.tail(seq_len)
    if len(window) < seq_len:
        raise ValueError(f"Not enough rows: have {len(window)}, need {seq_len}")
    return window, feature_cols


def _normalize_array(x: np.ndarray, stats_path: Optional[str]) -> np.ndarray:
    if not stats_path:
        return x
    stats = np.load(stats_path)
    mean = stats["mean"]
    std = stats["std"]
    std = np.where(std == 0, 1.0, std)
    if mean.shape[0] != x.shape[-1]:
        raise ValueError(f"Stats feature length mismatch: {mean.shape[0]} vs {x.shape[-1]}")
    return (x - mean) / std


def _normalize_window(window: pd.DataFrame, feature_cols: List[str], stats_path: Optional[str]) -> np.ndarray:
    x = window[feature_cols].to_numpy(dtype=np.float32)
    return _normalize_array(x, stats_path)


def _cls_prob_from_output(cls_out: np.ndarray) -> float:
    cls = np.asarray(cls_out)
    if cls.ndim == 2 and cls.shape[1] > 1:
        # softmax -> prob of last class (UP)
        return float(cls[0, -1])
    return float(np.squeeze(cls))


def _extract_price_and_cls(out, quantiles: Optional[np.ndarray], cls_key: Optional[str] = None):
    price = None
    cls = None
    if isinstance(out, dict):
        if "price_q" in out:
            price = out["price_q"]
        elif "price" in out:
            price = out["price"]
        if cls_key and cls_key in out:
            cls = out[cls_key]
        elif "cls" in out:
            cls = out["cls"]
        else:
            # fallback: any cls_* head
            for k in out.keys():
                if k.startswith("cls_"):
                    cls = out[k]
                    break
    elif isinstance(out, (list, tuple)):
        if len(out) >= 1:
            price = out[0]
        if len(out) >= 2:
            cls = out[1]
    else:
        price = out
    price_arr = np.asarray(price) if price is not None else None
    if price_arr is not None and price_arr.ndim == 2 and price_arr.shape[1] > 1:
        if quantiles is not None and len(quantiles) > 0:
            q = np.asarray(quantiles, dtype=np.float32)
            idx = int(np.argmin(np.abs(q - 0.5)))
        else:
            idx = price_arr.shape[1] // 2
        price_val = float(price_arr[0, idx])
    else:
        price_val = float(np.squeeze(price_arr)) if price_arr is not None else 0.0
    cls_val = _cls_prob_from_output(cls) if cls is not None else None
    return price_val, cls_val


def _predict_cls_prob(
    model: tf.keras.Model,
    window: pd.DataFrame,
    feature_cols: List[str],
    stats_path: Optional[str],
    cls_key: str,
) -> float:
    x = window[feature_cols].to_numpy(dtype=np.float32)[None, ...]
    x = _normalize_array(x, stats_path)
    out = model.predict(x, verbose=0)
    _, cls_val = _extract_price_and_cls(out, quantiles=None, cls_key=cls_key)
    return float(cls_val) if cls_val is not None else 0.5


def _estimate_vol(window: pd.DataFrame) -> float:
    if "rv_long" in window.columns:
        val = float(window["rv_long"].iloc[-1])
        if np.isfinite(val) and val > 0:
            return val
    if "rv_short" in window.columns:
        val = float(window["rv_short"].iloc[-1])
        if np.isfinite(val) and val > 0:
            return val
    if "close" in window.columns:
        close = window["close"].astype(float).to_numpy()
        if close.size >= 32:
            log_ret = np.diff(np.log(close))
            vol = float(np.std(log_ret))
            if np.isfinite(vol) and vol > 0:
                return vol
    return 0.0


def _last_log_return(window: pd.DataFrame) -> float:
    if "close" not in window.columns:
        return 0.0
    close = window["close"].astype(float).to_numpy()
    if close.size < 2:
        return 0.0
    return float(np.log(close[-1]) - np.log(close[-2]))


def _predict_signed_log_return(
    model: tf.keras.Model,
    window: pd.DataFrame,
    feature_cols: List[str],
    stats_path: Optional[str],
    stats_meta: dict,
    horizon: Optional[int] = None,
    cls_key: Optional[str] = None,
) -> float:
    x = window[feature_cols].to_numpy(dtype=np.float32)[None, ...]
    x = _normalize_array(x, stats_path)
    out = model.predict(x, verbose=0)
    if isinstance(out, dict) and horizon is not None:
        target_mode = _get_price_target_mode(stats_meta)
        scale_map = _get_price_head_scale(stats_meta)
        if target_mode == "segment_deltas":
            mh = _unpack_obj(stats_meta.get("price_multi_horizons"))
            if isinstance(mh, np.ndarray):
                mh = mh.tolist()
            horizons = [int(h) for h in (mh or [])]
            if not horizons:
                horizons = [horizon]
            total = 0.0
            for h in sorted(horizons):
                if h > horizon:
                    break
                key = f"price_h{h}"
                if key not in out:
                    continue
                val = float(np.squeeze(out[key]))
                if scale_map and key in scale_map:
                    val *= float(scale_map[key])
                total += val
            return float(total)
        price_key = f"price_h{horizon}"
        if price_key in out:
            val = float(np.squeeze(out[price_key]))
            if scale_map and price_key in scale_map:
                val *= float(scale_map[price_key])
            return float(val)
    quantiles = stats_meta.get("quantiles")
    price_col = stats_meta.get("price_col")
    price_val, cls_prob = _extract_price_and_cls(out, quantiles, cls_key=cls_key)
    scale_map = _get_price_head_scale(stats_meta)
    if scale_map:
        key = "price"
        if horizon is not None and f"price_h{horizon}" in scale_map:
            key = f"price_h{horizon}"
        if key in scale_map:
            price_val = float(price_val) * float(scale_map[key])
    if isinstance(price_col, np.ndarray):
        price_col = price_col.tolist()
        price_col = price_col[0] if price_col else None
    if price_col == "target_ret":
        return float(price_val)
    if price_col == "target_amp_abs":
        magnitude = abs(float(price_val))
        mh = stats_meta.get("multi_horizons")
        base_h = None
        if isinstance(mh, np.ndarray) and mh.size > 0:
            base_h = int(np.min(mh))
        if horizon and base_h and base_h > 0:
            magnitude *= math.sqrt(float(horizon) / float(base_h))
        if cls_prob is None:
            return float(magnitude)
        signed = (2.0 * float(cls_prob) - 1.0) * magnitude
        return float(signed)
    return float(price_val)


def _predict_ensemble_log_return(
    specs: List[dict],
    df: pd.DataFrame,
    horizon: Optional[int],
    cls_key: Optional[str],
) -> Tuple[float, pd.DataFrame]:
    if not specs:
        raise ValueError("No model specs provided")
    preds = []
    vol_window = None
    for spec in specs:
        window, fcols = _prepare_window(df, spec["seq_len"], spec["feature_cols"])
        if vol_window is None:
            vol_window = window
        pred = _predict_signed_log_return(
            spec["model"],
            window,
            fcols,
            spec["stats_path"],
            spec["stats_meta"],
            horizon=horizon,
            cls_key=cls_key,
        )
        preds.append(pred)
    return float(np.mean(preds)), vol_window if vol_window is not None else window


def _predict_weighted_ensemble_log_return(
    specs: List[dict],
    df: pd.DataFrame,
    horizon: Optional[int],
    cls_key: Optional[str],
    weights: Optional[List[float]] = None,
    gate_cfg: Optional[dict] = None,
) -> Tuple[float, pd.DataFrame]:
    if not specs:
        raise ValueError("No model specs provided")
    preds = []
    vol_window = None
    for spec in specs:
        window, fcols = _prepare_window(df, spec["seq_len"], spec["feature_cols"])
        if vol_window is None:
            vol_window = window
        pred = _predict_signed_log_return(
            spec["model"],
            window,
            fcols,
            spec["stats_path"],
            spec["stats_meta"],
            horizon=horizon,
            cls_key=cls_key,
        )
        preds.append(pred)
    use_weights = weights
    if gate_cfg and len(preds) == 2:
        feat = gate_cfg.get("feature")
        thresh = gate_cfg.get("threshold")
        news_feat = gate_cfg.get("news_feature")
        news_thresh = gate_cfg.get("news_threshold")
        w_hi = gate_cfg.get("w_v3_high")
        w_lo = gate_cfg.get("w_v3_low")
        div_thresh = gate_cfg.get("div_threshold")
        if feat and (w_hi is not None) and (w_lo is not None):
            v = float(vol_window.get(feat, 0.0).iloc[-1]) if vol_window is not None and feat in vol_window.columns else 0.0
            nv = float(vol_window.get(news_feat, 0.0).iloc[-1]) if (vol_window is not None and news_feat and news_feat in vol_window.columns) else 0.0
            div = abs(float(preds[1]) - float(preds[0])) if len(preds) == 2 else 0.0
            trigger = False
            if thresh is not None and v > float(thresh):
                trigger = True
            if news_feat and news_thresh is not None and nv > float(news_thresh):
                trigger = True
            if div_thresh is not None and div > float(div_thresh):
                trigger = True
            w_v3 = float(w_hi) if trigger else float(w_lo)
            w_v3 = min(max(w_v3, 0.0), 1.0)
            use_weights = [1.0 - w_v3, w_v3]
    if use_weights and len(use_weights) == len(preds):
        pred_val = float(np.sum(np.asarray(preds, dtype=np.float64) * np.asarray(use_weights, dtype=np.float64)))
    else:
        pred_val = float(np.mean(preds))
    return pred_val, vol_window if vol_window is not None else window


def _apply_bias_to_log_return(base_price: float, log_ret: float, bias_usd: float) -> float:
    if not bias_usd:
        return float(log_ret)
    price = float(base_price * math.exp(float(log_ret)))
    adj = price - float(bias_usd)
    adj = max(adj, 1e-6)
    return float(math.log(adj / float(base_price)))


def _predict_multi_horizon_log_return(
    model: tf.keras.Model,
    window: pd.DataFrame,
    feature_cols: List[str],
    stats_path: Optional[str],
    stats_meta: dict,
    horizon: int,
    cls_key: str,
) -> float:
    x = window[feature_cols].to_numpy(dtype=np.float32)[None, ...]
    x = _normalize_array(x, stats_path)
    out = model.predict(x, verbose=0)
    if isinstance(out, dict):
        scale_map = _get_price_head_scale(stats_meta)
        target_mode = _get_price_target_mode(stats_meta)
        if target_mode == "segment_deltas":
            mh = _unpack_obj(stats_meta.get("price_multi_horizons"))
            if isinstance(mh, np.ndarray):
                mh = mh.tolist()
            horizons = [int(h) for h in (mh or [])]
            if not horizons:
                horizons = [horizon]
            total = 0.0
            for h in sorted(horizons):
                if h > horizon:
                    break
                key = f"price_h{h}"
                if key not in out:
                    continue
                val = float(np.squeeze(out[key]))
                if scale_map and key in scale_map:
                    val *= float(scale_map[key])
                total += val
            return float(total)
        price_key = f"price_h{horizon}"
        if price_key in out:
            val = float(np.squeeze(out[price_key]))
            if scale_map and price_key in scale_map:
                val *= float(scale_map[price_key])
            return float(val)
    quantiles = stats_meta.get("quantiles")
    price_val, cls_prob = _extract_price_and_cls(out, quantiles, cls_key=cls_key)
    # model price head is trained on target_amp_abs for h20
    model_mag = abs(float(price_val)) * math.sqrt(float(horizon) / 20.0)
    vol_mag = _estimate_vol(window) * math.sqrt(float(horizon))
    magnitude = max(model_mag, vol_mag) * MULTI_SCALE
    magnitude = max(magnitude, MIN_MULTI_PCT)
    if cls_prob is None:
        sign = 1.0 if _last_log_return(window) >= 0 else -1.0
    else:
        sign = 1.0 if cls_prob >= 0.5 else -1.0
    return float(sign * magnitude)


def _build_noisy_trend(
    base_price: float,
    horizon: int,
    step_min: int,
    total_log_return: float,
    window: pd.DataFrame,
) -> list[float]:
    if horizon <= 0:
        return []
    rng = np.random.default_rng(42)
    vol = _estimate_vol(window)
    # cap extreme trend so we don't draw cliff-like lines
    cap = max(vol * math.sqrt(float(horizon)) * 1.5, 0.02)
    total_log_return = float(np.clip(total_log_return, -cap, cap))
    # per-step drift from total log-return
    drift = float(total_log_return) / float(horizon)
    # noise scaled by vol; cap to keep line sane
    noise_scale = max(vol * 0.5, 1e-5)
    noise = rng.normal(0.0, noise_scale, size=horizon)
    # mean-revert noise so endpoints stay close to target
    noise = noise - noise.mean()
    price = base_price
    series = []
    for i in range(horizon):
        step = drift + noise[i]
        price = float(price * math.exp(step))
        series.append(price)
    return series


def _build_piecewise_trend(
    base_price: float,
    horizon: int,
    segments: list[tuple[int, float]],
) -> list[float]:
    """
    Build piecewise log-linear path from segment log-returns.
    segments: list of (end_step, seg_log_return) where end_step is cumulative step index.
    """
    if horizon <= 0:
        return []
    price = base_price
    series: list[float] = []
    start = 0
    for end_step, seg_log in segments:
        end_step = min(int(end_step), int(horizon))
        seg_len = max(1, end_step - start)
        per_step = float(seg_log) / float(seg_len)
        for _ in range(seg_len):
            price = float(price * math.exp(per_step))
            series.append(price)
        start = end_step
        if start >= horizon:
            break
    # if segments end early, extend flat
    while len(series) < horizon:
        series.append(float(price))
    return series


def _compute_trap_market_maker(base_price: float, points: list[dict[str, Any]]) -> dict[str, Any]:
    """Estimate stop-hunt / trap risk from forecast path shape."""
    if not points or base_price <= 0:
        return {
            "risk": 50.0,
            "entry_delay_bars": 2,
            "up_prob": 34.0,
            "down_prob": 33.0,
            "flat_prob": 33.0,
            "cluster_up": float(base_price or 0.0),
            "cluster_down": float(base_price or 0.0),
        }

    vals = np.array([float(p.get("value", base_price)) for p in points], dtype=np.float64)
    vals = np.where(np.isfinite(vals), vals, float(base_price))
    if vals.size < 2:
        vals = np.array([base_price, base_price], dtype=np.float64)

    start_val = float(vals[0])
    end_val = float(vals[-1])
    total_ret = (end_val - start_val) / max(abs(base_price), 1e-9)

    rets = np.diff(vals) / np.maximum(np.abs(vals[:-1]), 1e-9)
    vol = float(np.std(rets)) if rets.size else 0.0
    trend_strength = abs(total_ret) / max(vol * math.sqrt(max(len(vals), 1)), 1e-6)

    # More sign flips => more likely a trap/noisy regime.
    signs = np.sign(rets)
    sign_flips = float(np.mean(np.abs(np.diff(signs)) > 0)) if signs.size > 2 else 0.0

    # If early move and final move disagree, trap probability increases.
    early = float((vals[min(4, len(vals) - 1)] - vals[0]) / max(abs(base_price), 1e-9))
    reversal_score = 1.0 if (early * total_ret) < 0 else 0.0

    trap_risk = 28.0 + (sign_flips * 38.0) + (reversal_score * 18.0) + max(0.0, 24.0 - trend_strength * 20.0)
    trap_risk = float(np.clip(trap_risk, 5.0, 95.0))
    entry_delay_bars = int(np.clip(round((trap_risk - 20.0) / 20.0), 0, 4))

    # Direction probabilities: trend vote minus trap penalty.
    score = float(np.tanh(total_ret / 0.002))
    up = 50.0 + score * 28.0
    down = 50.0 - score * 28.0
    trap_penalty = trap_risk * 0.22
    if score >= 0:
        up -= trap_penalty
        down += trap_penalty * 0.6
    else:
        down -= trap_penalty
        up += trap_penalty * 0.6
    flat = max(6.0, 100.0 - up - down)
    total = max(up + down + flat, 1e-6)
    up = float(np.clip(up * 100.0 / total, 2.0, 96.0))
    down = float(np.clip(down * 100.0 / total, 2.0, 96.0))
    flat = float(np.clip(100.0 - up - down, 2.0, 90.0))

    px_hi = float(np.max(vals))
    px_lo = float(np.min(vals))
    span = max(px_hi - px_lo, base_price * 0.002)
    cluster_up = float(base_price + span * (0.5 + trap_risk / 220.0))
    cluster_down = float(base_price - span * (0.5 + trap_risk / 220.0))

    return {
        "risk": float(round(trap_risk, 2)),
        "entry_delay_bars": int(entry_delay_bars),
        "up_prob": float(round(up, 2)),
        "down_prob": float(round(down, 2)),
        "flat_prob": float(round(flat, 2)),
        "cluster_up": float(round(cluster_up, 6)),
        "cluster_down": float(round(cluster_down, 6)),
    }


def _sign_deadzone(x: float, dz: float = 0.0) -> int:
    if not math.isfinite(x):
        return 0
    if x > dz:
        return 1
    if x < -dz:
        return -1
    return 0


def _legacy_status_from_raw(status_raw: str) -> str:
    s = str(status_raw or "").upper()
    if s in {"LONG", "LL"}:
        return "LL"
    if s in {"SHORT", "SS"}:
        return "SS"
    return "LX"


def _extract_forecast_stats(payload: Dict[str, Any], base_price: float) -> Optional[Dict[str, Any]]:
    pts = payload.get("points") if isinstance(payload, dict) else None
    if not isinstance(pts, list) or not pts or not math.isfinite(base_price) or base_price <= 0:
        return None
    first = pts[0].get("value", base_price)
    last = pts[-1].get("value", base_price)
    try:
        start_val = float(first)
        end_val = float(last)
    except Exception:
        return None
    delta = end_val - start_val
    delta_pct = delta / base_price
    return {
        "start_val": start_val,
        "end_val": end_val,
        "delta": delta,
        "delta_pct": delta_pct,
        "direction": _sign_deadzone(delta_pct, 0.0),
        "trap": payload.get("trap_market_maker") if isinstance(payload, dict) else None,
    }


def _derive_site_model_signal(
    model_key: str,
    forecasts: Dict[str, Dict[str, Any]],
    selected_horizon: str,
    base_price: float,
    atr_value: Optional[float],
    news_count: int,
) -> Optional[Dict[str, Any]]:
    profile = _SITE_MODEL_PROFILE.get(model_key) or _SITE_MODEL_PROFILE["conservative"]
    weights = profile["weights"]
    deadzone = float(profile["deadzone"])
    stats: Dict[str, Dict[str, Any]] = {}
    for horizon_key in ("5H", "20H", "40H"):
        extracted = _extract_forecast_stats(forecasts.get(horizon_key, {}), base_price)
        if extracted:
            stats[horizon_key] = extracted
    primary = stats.get(selected_horizon) or stats.get("5H") or stats.get("20H") or stats.get("40H")
    if not primary:
        return None

    st5 = stats.get("5H")
    st20 = stats.get("20H")
    st40 = stats.get("40H")
    d5 = float((st5 or {}).get("delta_pct", 0.0) or 0.0)
    d20 = float((st20 or {}).get("delta_pct", 0.0) or 0.0)
    d40 = float((st40 or {}).get("delta_pct", 0.0) or 0.0)
    dir5 = int((st5 or {}).get("direction", 0) or 0)
    dir20 = int((st20 or {}).get("direction", 0) or 0)
    dir40 = int((st40 or {}).get("direction", 0) or 0)

    weighted_delta = 0.0
    aligned = 0
    opposing = 0
    for horizon_key in ("5H", "20H", "40H"):
        st = stats.get(horizon_key)
        if not st:
            continue
        w = float(weights.get(horizon_key, 0.0))
        weighted_delta += float(st["delta_pct"]) * w
    signed = _sign_deadzone(weighted_delta, deadzone)
    for horizon_key in ("5H", "20H", "40H"):
        st = stats.get(horizon_key)
        if not st:
            continue
        if st["direction"] == signed and st["direction"] != 0:
            aligned += 1
        elif st["direction"] == -signed and st["direction"] != 0:
            opposing += 1

    trap = primary.get("trap") or stats.get("5H", {}).get("trap") or stats.get("20H", {}).get("trap") or stats.get("40H", {}).get("trap")
    trap_risk = float(np.clip(float((trap or {}).get("risk", 0.0) or 0.0), 0.0, 100.0))
    up_prob = float((trap or {}).get("up_prob", 0.0) or 0.0)
    down_prob = float((trap or {}).get("down_prob", 0.0) or 0.0)
    flat_prob = float((trap or {}).get("flat_prob", 0.0) or 0.0)
    agreement = f"{profile['logic_name']}: {aligned}/3 aligned"
    final_signed = signed

    if model_key == "conservative":
        core_score = 0.25 * d5 + 0.35 * d20 + 0.40 * d40
        weighted_delta = core_score
        if aligned == 3 and opposing == 0 and trap_risk < 62.0:
            final_signed = _sign_deadzone(core_score, deadzone * 0.95)
            agreement = "ATR consensus: 5H/20H/40H aligned"
        else:
            final_signed = 0
            agreement = "ATR consensus: wait / mixed or trap-heavy structure"
    elif model_key == "aggressive":
        impulse_score = 0.80 * d5 + 0.20 * d20
        weighted_delta = impulse_score
        final_signed = _sign_deadzone(impulse_score, deadzone * 0.55)
        if abs(d5) < deadzone * 0.45:
            final_signed = 0
            agreement = "Momentum hunter: 5H impulse too weak"
        else:
            agreement = "Momentum hunter: no edge after short-term scan" if final_signed == 0 else f"Momentum hunter: {'impulse up' if final_signed > 0 else 'impulse down'} from 5H"
    elif model_key == "alpha75":
        alpha_score = 0.65 * d5 + 0.35 * d20
        weighted_delta = alpha_score * (1.10 if dir5 != 0 and dir5 == dir20 else 1.0)
        if dir5 != 0 and dir20 != 0 and dir5 == dir20 and abs(d5) >= deadzone * 0.70:
            final_signed = _sign_deadzone(weighted_delta, deadzone * 0.65)
            agreement = "Directional alpha: 5H + 20H momentum stack"
        else:
            final_signed = 0
            agreement = "Directional alpha: no clean 5H/20H momentum stack"
    elif model_key == "tg_hybrid":
        hybrid_score = 0.20 * d5 + 0.45 * d20 + 0.35 * d40
        if news_count <= 0:
            weighted_delta = hybrid_score * 0.60
            final_signed = 0 if aligned < 3 else _sign_deadzone(weighted_delta, deadzone * 1.15)
            agreement = "TG hybrid: no fresh news, waiting for confirmation"
        else:
            news_boost = 1.12 if news_count >= 10 else 1.0
            weighted_delta = hybrid_score * news_boost
            final_signed = _sign_deadzone(weighted_delta, deadzone * 1.05)
            agreement = f"TG hybrid: news flow active ({news_count})"
    elif model_key == "mm_r13":
        trap_bias = 0
        if trap and trap_risk >= 52.0:
            if down_prob > max(up_prob, flat_prob) + 6.0:
                trap_bias = -1
            elif up_prob > max(down_prob, flat_prob) + 6.0:
                trap_bias = 1
        mm_score = 0.15 * d5 + 0.25 * d20 + 0.60 * d40
        weighted_delta = mm_score
        if trap_bias != 0:
            final_signed = trap_bias
            agreement = f"MM trap: {'upside sweep' if trap_bias > 0 else 'downside trap'} bias {round(max(up_prob, down_prob))}%"
        elif dir20 != 0 and dir20 == dir40 and abs(mm_score) >= deadzone * 0.85:
            final_signed = _sign_deadzone(mm_score, deadzone * 0.90)
            agreement = "MM supervisor: 20H/40H imbalance without trap conflict"
        else:
            final_signed = 0
            agreement = "MM supervisor: no clean crowd imbalance"
    elif model_key == "chronos2":
        path_score = 0.72 * d20 + 0.28 * d40
        weighted_delta = path_score
        if dir20 != 0 and dir20 == dir40:
            final_signed = _sign_deadzone(path_score, deadzone * 0.90)
            agreement = "Chronos path: continuation bias from 20H/40H"
        else:
            final_signed = 0
            agreement = "Chronos path: flat / timing unclear"
    elif model_key == "meta_combo_v1":
        long_core = max(d5, 0.0) * 0.58 + max(d40, 0.0) * 0.42
        short_core = max(-d5, 0.0) * 0.46 + max(-d20, 0.0) * 0.54
        long_bias = dir5 > 0 and dir40 > 0 and d5 >= deadzone * 0.55 and d40 >= deadzone * 0.35
        short_bias = dir5 < 0 and dir20 < 0 and abs(d5) >= deadzone * 0.55 and abs(d20) >= deadzone * 0.40
        weighted_delta = long_core - short_core
        combo_edge = abs(weighted_delta)
        if long_bias and (not short_bias or long_core >= short_core * 1.18):
            weighted_delta = long_core
            if d20 < -deadzone * 0.70:
                weighted_delta *= 0.72
            if trap_risk <= 78.0 and flat_prob < 54.0:
                final_signed = 1
                agreement = "Meta combo: 5H breakout confirmed by 40H path"
            else:
                final_signed = 0
                agreement = "Meta combo: bullish stack found, but trap / flat filter blocks entry"
        elif short_bias and (not long_bias or short_core >= long_core * 1.18):
            weighted_delta = -short_core
            if d40 > deadzone * 0.55:
                weighted_delta *= 0.70
            if trap_risk <= 82.0 and flat_prob < 56.0:
                final_signed = -1
                agreement = "Meta combo: 5H downside continuation confirmed by 20H pressure"
            else:
                final_signed = 0
                agreement = "Meta combo: bearish stack found, but trap / flat filter blocks entry"
        else:
            final_signed = 0
            weighted_delta = (long_core - short_core) * 0.55
            agreement = "Meta combo: no clean continuation stack"
        conf_base = float(np.clip((combo_edge / 0.0085) * 86.0, 22.0, 98.0))
        conf_boost = 12.0 if (long_bias or short_bias) else 0.0
        conf_penalty = 0.0 if model_key == "mm_r13" else round(trap_risk * 0.10, 2)
        confidence = float(np.clip(conf_base + conf_boost - conf_penalty, 18.0, 97.0))

    if model_key != "meta_combo_v1":
        magnitude = abs(weighted_delta)
        conf_base = float(np.clip((magnitude / 0.01) * 72.0, 16.0, 96.0))
        conf_boost = 8.0 if aligned >= 2 else 0.0
        conf_penalty = 10.0 if opposing > 0 else 0.0
        trap_penalty = 0.0 if model_key == "mm_r13" else round(trap_risk * 0.08, 2)
        confidence = float(np.clip(conf_base + conf_boost - conf_penalty - trap_penalty, 18.0, 96.0))
    status_raw = "LONG" if final_signed > 0 else "SHORT" if final_signed < 0 else "FLAT"
    invalidation = f"{base_price:,.2f} (no trade)"
    if atr_value and math.isfinite(atr_value) and atr_value > 0:
        k = float(profile.get("atr_mult", 1.6))
        if status_raw == "LONG":
            invalidation = f"{(base_price - k * atr_value):,.2f} (stop)"
        elif status_raw == "SHORT":
            invalidation = f"{(base_price + k * atr_value):,.2f} (stop)"

    vol = (atr_value / base_price) if (atr_value and base_price and base_price > 0) else None
    vol_tag = "—" if vol is None else ("LOW" if vol < 0.003 else "NORMAL" if vol < 0.008 else "HIGH")
    news_tag = f"OK ({news_count})" if news_count > 0 else "NO DATA"
    return {
        "model": model_key,
        "label": profile["label"],
        "logic_name": profile["logic_name"],
        "status": _legacy_status_from_raw(status_raw),
        "status_raw": status_raw,
        "confidence": round(confidence, 2),
        "horizon": selected_horizon,
        "leverage": profile["leverage"],
        "risk_pct": profile["risk_pct"],
        "invalidation": invalidation,
        "why": {
            "agreement": agreement,
            "volatility": vol_tag,
            "news": news_tag,
        },
        "trap": trap,
        "metrics": {
            "weighted_delta_pct": round(weighted_delta * 100.0, 4),
            "aligned": aligned,
            "opposing": opposing,
        },
    }


def create_app(
    model_h20_path: str,
    seq_len: int,
    stats_h20_path: Optional[str] = None,
    model_multi_path: Optional[str] = None,
    stats_multi_path: Optional[str] = None,
    features_path: Optional[str] = None,
    model_h80_path: Optional[str] = None,
    stats_h80_path: Optional[str] = None,
    model_h160_path: Optional[str] = None,
    stats_h160_path: Optional[str] = None,
    model_h320_path: Optional[str] = None,
    stats_h320_path: Optional[str] = None,
    model_h640_path: Optional[str] = None,
    stats_h640_path: Optional[str] = None,
):
    feature_cols = _feature_cols_from_stats(stats_h20_path, default_feature_list())
    n_features = len(feature_cols)
    model_h20_specs = _load_model_specs(model_h20_path, stats_h20_path, feature_cols, seq_len) if model_h20_path else None
    model_h80_specs = _load_model_specs(model_h80_path, stats_h80_path, feature_cols, seq_len) if model_h80_path else None
    model_h160_specs = _load_model_specs(model_h160_path, stats_h160_path, feature_cols, seq_len) if model_h160_path else None
    model_h320_specs = _load_model_specs(model_h320_path, stats_h320_path, feature_cols, seq_len) if model_h320_path else None
    model_h640_specs = _load_model_specs(model_h640_path, stats_h640_path, feature_cols, seq_len) if model_h640_path else None
    model_multi = _load_model(model_multi_path) if model_multi_path else None
    weights_h20 = _parse_weights(os.getenv("H20_WEIGHTS"), len(model_h20_specs)) if model_h20_specs else None
    weights_h80 = _parse_weights(os.getenv("H80_WEIGHTS"), len(model_h80_specs)) if model_h80_specs else None
    weights_h160 = _parse_weights(os.getenv("H160_WEIGHTS"), len(model_h160_specs)) if model_h160_specs else None
    bias_h20 = float(os.getenv("BIAS_H20_USD", "0") or 0.0)
    bias_h80 = float(os.getenv("BIAS_H80_USD", "0") or 0.0)
    bias_h160 = float(os.getenv("BIAS_H160_USD", "0") or 0.0)
    gate_h80 = {
        "feature": os.getenv("H80_GATE_FEATURE"),
        "threshold": float(os.getenv("H80_GATE_THRESHOLD", "0") or 0.0) if os.getenv("H80_GATE_THRESHOLD") else None,
        "news_feature": os.getenv("H80_GATE_NEWS_FEATURE"),
        "news_threshold": float(os.getenv("H80_GATE_NEWS_THRESHOLD", "0") or 0.0) if os.getenv("H80_GATE_NEWS_THRESHOLD") else None,
        "w_v3_high": float(os.getenv("H80_GATE_WV3_HIGH", "0") or 0.0) if os.getenv("H80_GATE_WV3_HIGH") else None,
        "w_v3_low": float(os.getenv("H80_GATE_WV3_LOW", "0") or 0.0) if os.getenv("H80_GATE_WV3_LOW") else None,
        "div_threshold": float(os.getenv("H80_GATE_DIV_THRESHOLD", "0") or 0.0) if os.getenv("H80_GATE_DIV_THRESHOLD") else None,
    }
    gate_h160 = {
        "feature": os.getenv("H160_GATE_FEATURE"),
        "threshold": float(os.getenv("H160_GATE_THRESHOLD", "0") or 0.0) if os.getenv("H160_GATE_THRESHOLD") else None,
        "news_feature": os.getenv("H160_GATE_NEWS_FEATURE"),
        "news_threshold": float(os.getenv("H160_GATE_NEWS_THRESHOLD", "0") or 0.0) if os.getenv("H160_GATE_NEWS_THRESHOLD") else None,
        "w_v3_high": float(os.getenv("H160_GATE_WV3_HIGH", "0") or 0.0) if os.getenv("H160_GATE_WV3_HIGH") else None,
        "w_v3_low": float(os.getenv("H160_GATE_WV3_LOW", "0") or 0.0) if os.getenv("H160_GATE_WV3_LOW") else None,
        "div_threshold": float(os.getenv("H160_GATE_DIV_THRESHOLD", "0") or 0.0) if os.getenv("H160_GATE_DIV_THRESHOLD") else None,
    }

    def _load_stats(path: Optional[str]):
        if not path:
            return None, None, {}
        stats = np.load(path, allow_pickle=True)
        return stats["mean"], stats["std"], {k: stats[k] for k in stats.files}

    saved_mean_h20, saved_std_h20, stats_meta_h20 = _load_stats(_first_path(stats_h20_path))
    saved_mean_multi, saved_std_multi, stats_meta_multi = _load_stats(stats_multi_path)

    _default_news_dir = os.getenv("NEWS_DIR", os.path.join(os.getcwd(), "data", "news"))
    _default_news_sent = os.getenv("NEWS_SENT", os.path.join(_default_news_dir, "news_sentiment.parquet"))
    news_path = os.getenv("NEWS_PATH", _default_news_sent)
    news_cache = {"mtime": None, "df": None}
    crowd_priors_path = os.getenv(
        "CROWD_PRIORS_PATH",
        str(Path(__file__).resolve().parent / "data" / "telegram" / "crowd_priors_contextual_strict_s3.parquet"),
    )
    crowd_cache: Dict[str, Any] = {"mtime": None, "df": None}
    crowd_cfg = {
        "enabled": os.getenv("CROWD_SIZING_ENABLED", "1") == "1",
        "prefix": os.getenv("CROWD_SIZING_PREFIX", "crowd_ctx_h"),
        "horizon": int(os.getenv("CROWD_SIZING_HORIZON", "24") or 24),
        "alpha": float(os.getenv("CROWD_SIZING_ALPHA", "2.0") or 2.0),
        "min_mult": float(os.getenv("CROWD_SIZING_MIN_MULT", "0.9") or 0.9),
        "max_mult": float(os.getenv("CROWD_SIZING_MAX_MULT", "1.5") or 1.5),
    }

    def _read_news_df() -> pd.DataFrame:
        if not news_path or not os.path.exists(news_path):
            return pd.DataFrame()
        mtime = os.path.getmtime(news_path)
        if news_cache["df"] is None or news_cache["mtime"] != mtime:
            if news_path.endswith(".jsonl"):
                df = pd.read_json(news_path, lines=True)
            else:
                df = pd.read_parquet(news_path)
            news_cache["df"] = df
            news_cache["mtime"] = mtime
        return news_cache["df"]

    def _sentiment_to_num(v) -> float:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return 0.0
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).lower().strip()
        if s in {"positive", "pos", "bull"}:
            return 1.0
        if s in {"negative", "neg", "bear"}:
            return -1.0
        return 0.0

    def _read_crowd_df() -> pd.DataFrame:
        if not crowd_cfg["enabled"]:
            return pd.DataFrame()
        if not crowd_priors_path or not os.path.exists(crowd_priors_path):
            return pd.DataFrame()
        mtime = os.path.getmtime(crowd_priors_path)
        if crowd_cache["df"] is None or crowd_cache["mtime"] != mtime:
            if crowd_priors_path.endswith(".parquet"):
                df = pd.read_parquet(crowd_priors_path)
            else:
                df = pd.read_csv(crowd_priors_path)
            if "timestamp_utc" not in df.columns:
                crowd_cache["df"] = pd.DataFrame()
                crowd_cache["mtime"] = mtime
                return crowd_cache["df"]
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
            df = df[df["timestamp_utc"].notna()].sort_values("timestamp_utc").reset_index(drop=True)
            crowd_cache["df"] = df
            crowd_cache["mtime"] = mtime
        return crowd_cache["df"]

    def _crowd_size_multiplier(direction: str, ts_utc: pd.Timestamp) -> Dict[str, Any]:
        default = {
            "enabled": bool(crowd_cfg["enabled"]),
            "ready": False,
            "multiplier": 1.0,
            "strength": 0.0,
            "winrate": None,
            "err_wrong": None,
            "source": None,
            "ts": None,
            "horizon": int(crowd_cfg["horizon"]),
            "prefix": str(crowd_cfg["prefix"]),
        }
        d = (direction or "").upper()
        if d not in {"LONG", "SHORT"}:
            return default
        cdf = _read_crowd_df()
        if cdf.empty:
            return default

        pfx = str(crowd_cfg["prefix"])
        h = int(crowd_cfg["horizon"])
        wr_col = f"{pfx}{h}_winrate_dir"
        err_col = f"{pfx}{h}_err_wrong_dir"
        ready_col = f"{pfx}{h}_dir_prior_ready"
        if wr_col not in cdf.columns or err_col not in cdf.columns:
            return default

        ts_ns = cdf["timestamp_utc"].astype("int64").to_numpy()
        j = int(np.searchsorted(ts_ns, int(pd.Timestamp(ts_utc).value), side="right") - 1)
        if j < 0:
            return default
        row = cdf.iloc[j]
        wr = float(pd.to_numeric(row.get(wr_col), errors="coerce")) if pd.notna(row.get(wr_col)) else np.nan
        er = float(pd.to_numeric(row.get(err_col), errors="coerce")) if pd.notna(row.get(err_col)) else np.nan
        if not np.isfinite(wr) or not np.isfinite(er):
            return default

        ready = True
        if ready_col in cdf.columns:
            rv = pd.to_numeric(row.get(ready_col), errors="coerce")
            ready = bool(pd.notna(rv) and float(rv) >= 0.5)

        strength = 0.0 if not ready else float(np.clip(wr - er, -1.0, 1.0))
        mult = float(np.clip(1.0 + float(crowd_cfg["alpha"]) * strength, float(crowd_cfg["min_mult"]), float(crowd_cfg["max_mult"])))
        src = None
        src_context_col = f"{pfx}{h}_source_context"
        src_direction_col = f"{pfx}{h}_source_direction"
        src_global_col = f"{pfx}{h}_source_global"
        if src_context_col in cdf.columns and float(pd.to_numeric(row.get(src_context_col), errors="coerce") or 0.0) >= 0.5:
            src = "context"
        elif src_direction_col in cdf.columns and float(pd.to_numeric(row.get(src_direction_col), errors="coerce") or 0.0) >= 0.5:
            src = "direction"
        elif src_global_col in cdf.columns and float(pd.to_numeric(row.get(src_global_col), errors="coerce") or 0.0) >= 0.5:
            src = "global"

        return {
            "enabled": True,
            "ready": bool(ready),
            "multiplier": mult,
            "strength": strength,
            "winrate": wr,
            "err_wrong": er,
            "source": src,
            "ts": pd.to_datetime(row["timestamp_utc"], utc=True).isoformat(),
            "horizon": h,
            "prefix": pfx,
        }

    app = FastAPI()
    api_key = os.environ.get("API_KEY", "").strip()

    db_path = os.environ.get(
        "AUTH_DB_PATH",
        os.path.join(os.path.dirname(__file__), "state", "users.db"),
    )
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    def _db():
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS users ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "email TEXT UNIQUE NOT NULL,"
            "salt TEXT NOT NULL,"
            "hash TEXT NOT NULL,"
            "created_at INTEGER NOT NULL,"
            "verified_at INTEGER"
            ")"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS email_verifications ("
            "email TEXT PRIMARY KEY,"
            "code_hash TEXT NOT NULL,"
            "expires_at INTEGER NOT NULL,"
            "attempts INTEGER NOT NULL DEFAULT 0,"
            "sent_at INTEGER NOT NULL"
            ")"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS password_resets ("
            "email TEXT PRIMARY KEY,"
            "code_hash TEXT NOT NULL,"
            "expires_at INTEGER NOT NULL,"
            "attempts INTEGER NOT NULL DEFAULT 0,"
            "sent_at INTEGER NOT NULL"
            ")"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS subscriptions ("
            "email TEXT PRIMARY KEY,"
            "plan TEXT NOT NULL,"
            "status TEXT NOT NULL,"
            "started_at INTEGER NOT NULL,"
            "expires_at INTEGER NOT NULL,"
            "updated_at INTEGER NOT NULL,"
            "updated_by TEXT,"
            "txid TEXT"
            ")"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS payments ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "email TEXT NOT NULL,"
            "plan TEXT,"
            "txid TEXT NOT NULL UNIQUE,"
            "status TEXT NOT NULL,"
            "created_at INTEGER NOT NULL,"
            "reviewed_at INTEGER,"
            "review_note TEXT"
            ")"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS user_profiles ("
            "email TEXT PRIMARY KEY,"
            "full_name TEXT,"
            "telegram TEXT,"
            "updated_at INTEGER NOT NULL"
            ")"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_payments_status_created ON payments(status, created_at)")
        # DB migration for old users.db
        cols = {row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
        if "verified_at" not in cols:
            conn.execute("ALTER TABLE users ADD COLUMN verified_at INTEGER")
            conn.execute("UPDATE users SET verified_at = created_at WHERE verified_at IS NULL")
        payment_cols = {row[1] for row in conn.execute("PRAGMA table_info(payments)").fetchall()}
        if "plan" not in payment_cols:
            conn.execute("ALTER TABLE payments ADD COLUMN plan TEXT")
        if "provider" not in payment_cols:
            conn.execute("ALTER TABLE payments ADD COLUMN provider TEXT")
        if "billing_period" not in payment_cols:
            conn.execute("ALTER TABLE payments ADD COLUMN billing_period TEXT")
        if "amount_rub" not in payment_cols:
            conn.execute("ALTER TABLE payments ADD COLUMN amount_rub INTEGER")
        return conn

    _db_conn = _db()
    _db_lock = threading.Lock()

    def _hash_password(password: str, salt: bytes) -> str:
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
        return dk.hex()

    def _make_salt() -> bytes:
        return os.urandom(16)

    _email_code_secret = os.environ.get("EMAIL_CODE_SECRET", "dev-change-email-code-secret")
    _admin_api_key = os.environ.get("ADMIN_API_KEY", "").strip()
    _session_secret = os.environ.get("SESSION_SECRET", "").strip() or _email_code_secret
    _session_ttl_sec = max(900, int(os.environ.get("SESSION_TTL_SEC", str(30 * 86400))))
    _yandex_client_id = os.environ.get("YANDEX_CLIENT_ID", "").strip()
    _yandex_client_secret = os.environ.get("YANDEX_CLIENT_SECRET", "").strip()
    _yandex_redirect_uri = os.environ.get("YANDEX_REDIRECT_URI", "https://api.tradeforge.art/auth/yandex/callback").strip() or "https://api.tradeforge.art/auth/yandex/callback"
    _public_web_root = os.environ.get("PUBLIC_WEB_ROOT", "https://www.tradeforge.art").strip().rstrip("/") or "https://www.tradeforge.art"
    _yookassa_shop_id = os.environ.get("YOOKASSA_SHOP_ID", "").strip()
    _yookassa_secret_key = os.environ.get("YOOKASSA_SECRET_KEY", "").strip()
    _yookassa_return_url = os.environ.get("YOOKASSA_RETURN_URL", "https://tradeforge.art/payment-success.html").strip()
    _yookassa_cancel_url = os.environ.get("YOOKASSA_CANCEL_URL", "https://tradeforge.art/payment-cancel.html").strip()
    _yookassa_payment_method = os.environ.get("YOOKASSA_PAYMENT_METHOD", "").strip().lower()
    _prodamus_secret_key = os.environ.get("PRODAMUS_SECRET_KEY", "").strip()
    _prodamus_payment_url = os.environ.get("PRODAMUS_PAYMENT_URL", "").strip()
    _prodamus_return_url = os.environ.get("PRODAMUS_RETURN_URL", "https://tradeforge.art/payment-success.html").strip()
    _prodamus_cancel_url = os.environ.get("PRODAMUS_CANCEL_URL", "https://tradeforge.art/payment-cancel.html").strip()
    _prodamus_webhook_url = os.environ.get("PRODAMUS_WEBHOOK_URL", "https://api.tradeforge.art/billing/prodamus/webhook").strip()
    _prodamus_payment_method = os.environ.get("PRODAMUS_PAYMENT_METHOD", "").strip()
    _prodamus_demo_mode = os.environ.get("PRODAMUS_DEMO_MODE", "0").strip() == "1"
    _owner_emails = {
        e.strip().lower()
        for e in os.environ.get("OWNER_EMAILS", "").split(",")
        if e.strip()
    }

    _plan_aliases = {
        "starter30": "starter",
        "pro50": "pro",
        "elite100": "elite",
        "starter": "starter",
        "pro": "pro",
        "elite": "elite",
    }
    _billing_period_aliases = {
        "month": "monthly",
        "monthly": "monthly",
        "1m": "monthly",
        "quarterly": "quarterly",
        "3m": "quarterly",
        "3months": "quarterly",
        "semiannual": "semiannual",
        "6m": "semiannual",
        "halfyear": "semiannual",
        "yearly": "yearly",
        "annual": "yearly",
        "12m": "yearly",
        "1y": "yearly",
    }
    _plan_catalog = {
        "starter": {
            "name": "Starter",
            "prices": {"monthly": 3333, "quarterly": 8990, "semiannual": 16990, "yearly": 29990},
            "days": {"monthly": 30, "quarterly": 92, "semiannual": 183, "yearly": 365},
        },
        "pro": {
            "name": "Pro",
            "prices": {"monthly": 4890, "quarterly": 13290, "semiannual": 24990, "yearly": 43990},
            "days": {"monthly": 30, "quarterly": 92, "semiannual": 183, "yearly": 365},
        },
        "elite": {
            "name": "Elite",
            "prices": {"monthly": 7890, "quarterly": 21490, "semiannual": 39990, "yearly": 69990},
            "days": {"monthly": 30, "quarterly": 92, "semiannual": 183, "yearly": 365},
        },
    }

    def _hash_email_code(email: str, code: str) -> str:
        msg = f"{email}|{code}".encode("utf-8")
        return hmac.new(_email_code_secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()

    def _b64url_encode(s: str) -> str:
        return base64.urlsafe_b64encode(s.encode("utf-8")).decode("ascii").rstrip("=")

    def _b64url_decode(s: str) -> str:
        pad = "=" * ((4 - len(s) % 4) % 4)
        return base64.urlsafe_b64decode((s + pad).encode("ascii")).decode("utf-8")

    def _issue_session_token(email: str) -> str:
        exp = int(time.time()) + _session_ttl_sec
        payload = f"{email}|{exp}"
        sig = hmac.new(_session_secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
        return f"tf1.{exp}.{sig}.{_b64url_encode(email)}"

    def _support_code(email: str) -> str:
        digest = hmac.new(_session_secret.encode("utf-8"), email.strip().lower().encode("utf-8"), hashlib.sha256).hexdigest()
        return f"TF-{digest[:10].upper()}"

    def _issue_oauth_state(provider: str, next_path: str = "") -> str:
        exp = int(time.time()) + 15 * 60
        safe_next = next_path.strip() if next_path and next_path.startswith("/") and not next_path.startswith("//") else ""
        payload = f"{provider}|{safe_next}|{exp}"
        sig = hmac.new(_session_secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
        return f"oauth1.{exp}.{sig}.{_b64url_encode(provider)}.{_b64url_encode(safe_next)}"

    def _verify_oauth_state(provider: str, state: str) -> Optional[str]:
        try:
            parts = (state or "").split(".")
            if len(parts) != 5 or parts[0] != "oauth1":
                return None
            exp = int(parts[1])
            sig = parts[2]
            got_provider = _b64url_decode(parts[3])
            next_path = _b64url_decode(parts[4])
        except Exception:
            return None
        if exp < int(time.time()):
            return None
        if got_provider != provider:
            return None
        if next_path and (not next_path.startswith("/") or next_path.startswith("//")):
            return None
        payload = f"{provider}|{next_path}|{exp}"
        expected = hmac.new(_session_secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        return next_path

    def _normalize_plan(plan: str) -> str:
        return _plan_aliases.get((plan or "").strip().lower(), "")

    def _normalize_billing_period(period: str) -> str:
        return _billing_period_aliases.get((period or "").strip().lower(), "")

    def _is_owner_email(email: str) -> bool:
        return bool(email and email.strip().lower() in _owner_emails)

    def _owner_subscription_snapshot(email: str) -> Dict[str, Any]:
        now = int(time.time())
        far_future = 4102444800  # 2100-01-01 UTC
        return {
            "active": True,
            "plan": "owner",
            "status": "active",
            "started_at": now,
            "expires_at": far_future,
            "updated_at": now,
            "txid": "owner_whitelist",
        }

    def _plan_offer(plan: str, billing_period: str) -> Dict[str, Any]:
        norm_plan = _normalize_plan(plan)
        norm_period = _normalize_billing_period(billing_period)
        if not norm_plan or norm_plan not in _plan_catalog:
            raise HTTPException(status_code=400, detail="invalid plan")
        if not norm_period:
            raise HTTPException(status_code=400, detail="invalid billing period")
        spec = _plan_catalog[norm_plan]
        amount_rub = int(spec["prices"][norm_period])
        days = int(spec["days"][norm_period])
        return {
            "plan": norm_plan,
            "plan_name": spec["name"],
            "billing_period": norm_period,
            "amount_rub": amount_rub,
            "days": days,
        }

    def _upsert_subscription(email: str, plan: str, days: int, updated_by: str, txid: Optional[str]) -> Dict[str, Any]:
        now = int(time.time())
        with _db_lock:
            existing = _db_conn.execute(
                "SELECT expires_at, status FROM subscriptions WHERE email = ?",
                (email,),
            ).fetchone()
            started_at = now
            expires_at = now + int(days) * 86400
            if existing:
                cur_exp, cur_status = existing
                if str(cur_status) == "active" and int(cur_exp or 0) > now:
                    started_at = now
                    expires_at = int(cur_exp) + int(days) * 86400
            _db_conn.execute(
                "INSERT INTO subscriptions(email, plan, status, started_at, expires_at, updated_at, updated_by, txid) "
                "VALUES (?, ?, 'active', ?, ?, ?, ?, ?) "
                "ON CONFLICT(email) DO UPDATE SET "
                "plan=excluded.plan, status='active', started_at=excluded.started_at, expires_at=excluded.expires_at, "
                "updated_at=excluded.updated_at, updated_by=excluded.updated_by, txid=excluded.txid",
                (email, plan, started_at, expires_at, now, updated_by, txid),
            )
            _db_conn.commit()
        return _subscription_snapshot(email)

    def _yookassa_auth_header() -> str:
        raw = f"{_yookassa_shop_id}:{_yookassa_secret_key}".encode("utf-8")
        return "Basic " + base64.b64encode(raw).decode("ascii")

    def _yookassa_request(method: str, path: str, payload: Optional[Dict[str, Any]] = None, *, idempotence_key: Optional[str] = None) -> Dict[str, Any]:
        if not _yookassa_shop_id or not _yookassa_secret_key:
            raise HTTPException(status_code=503, detail="yookassa is not configured")
        url = f"https://api.yookassa.ru/v3{path}"
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": _yookassa_auth_header(),
            "Content-Type": "application/json",
        }
        if idempotence_key:
            headers["Idempotence-Key"] = idempotence_key
        req = urllib.request.Request(url, data=body, headers=headers, method=method.upper())
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", "ignore")
            try:
                detail = json.loads(raw)
            except Exception:
                detail = {"detail": raw or f"yookassa http {e.code}"}
            raise HTTPException(status_code=502, detail={"yookassa_error": detail})
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"yookassa request failed: {e}")

    def _prodamus_stringify(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): _prodamus_stringify(value[k]) for k in sorted(value.keys(), key=lambda x: str(x))}
        if isinstance(value, list):
            return [_prodamus_stringify(v) for v in value]
        if value is None:
            return ""
        if isinstance(value, bool):
            return "1" if value else "0"
        return str(value)

    def _prodamus_signature(payload: Dict[str, Any]) -> str:
        normalized = _prodamus_stringify(payload)
        body = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"), sort_keys=True).replace("/", "\\/")
        return hmac.new(_prodamus_secret_key.encode("utf-8"), body.encode("utf-8"), hashlib.sha256).hexdigest()

    def _flatten_query(prefix: str, value: Any) -> List[tuple[str, str]]:
        if isinstance(value, dict):
            items: List[tuple[str, str]] = []
            for key in sorted(value.keys(), key=lambda x: str(x)):
                next_prefix = f"{prefix}[{key}]" if prefix else str(key)
                items.extend(_flatten_query(next_prefix, value[key]))
            return items
        if isinstance(value, list):
            items = []
            for idx, item in enumerate(value):
                next_prefix = f"{prefix}[{idx}]"
                items.extend(_flatten_query(next_prefix, item))
            return items
        return [(prefix, "" if value is None else str(value))]

    def _prodamus_payment_link(payload: Dict[str, Any]) -> str:
        if not _prodamus_secret_key or not _prodamus_payment_url:
            raise HTTPException(status_code=503, detail="prodamus is not configured")
        signed = dict(payload)
        signed["signature"] = _prodamus_signature(payload)
        return f"{_prodamus_payment_url}?{urllib.parse.urlencode(_flatten_query('', signed))}"

    def _parse_prodamus_body(raw: bytes, content_type: str) -> Dict[str, Any]:
        if "application/json" in (content_type or "").lower():
            data = json.loads((raw or b"{}").decode("utf-8"))
            if not isinstance(data, dict):
                raise HTTPException(status_code=400, detail="invalid prodamus json")
            return data
        parsed = urllib.parse.parse_qs((raw or b"").decode("utf-8"), keep_blank_values=True)
        flat: Dict[str, Any] = {}
        for key, value in parsed.items():
            flat[key] = value[0] if isinstance(value, list) and len(value) == 1 else value
        return flat

    def _append_query_params(url: str, params: Dict[str, Any]) -> str:
        parsed = urllib.parse.urlsplit(url or "")
        current = dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
        for key, value in params.items():
            if value is None:
                continue
            current[str(key)] = str(value)
        return urllib.parse.urlunsplit(
            (parsed.scheme, parsed.netloc, parsed.path, urllib.parse.urlencode(current), parsed.fragment)
        )

    def _verify_session_token(token: str) -> Optional[str]:
        try:
            parts = token.split(".")
            if len(parts) != 4 or parts[0] != "tf1":
                return None
            exp = int(parts[1])
            sig = parts[2]
            email = _b64url_decode(parts[3]).strip().lower()
        except Exception:
            return None
        if not email or not _valid_email(email):
            return None
        if exp < int(time.time()):
            return None
        payload = f"{email}|{exp}"
        expected = hmac.new(_session_secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        return email

    def _upsert_oauth_user(email: str, *, full_name: str = "", telegram: str = "") -> None:
        now = int(time.time())
        with _db_lock:
            row = _db_conn.execute("SELECT verified_at FROM users WHERE email = ?", (email,)).fetchone()
            if row:
                verified_at = int(row[0] or 0)
                if verified_at <= 0:
                    _db_conn.execute("UPDATE users SET verified_at = ? WHERE email = ?", (now, email))
            else:
                salt = _make_salt()
                pw_hash = _hash_password(secrets.token_urlsafe(24), salt)
                _db_conn.execute(
                    "INSERT INTO users(email, salt, hash, created_at, verified_at) VALUES (?, ?, ?, ?, ?)",
                    (email, salt.hex(), pw_hash, now, now),
                )
            if full_name or telegram:
                existing = _db_conn.execute(
                    "SELECT full_name, telegram FROM user_profiles WHERE email = ?",
                    (email,),
                ).fetchone()
                old_name = str(existing[0] or "") if existing else ""
                old_tg = str(existing[1] or "") if existing else ""
                _db_conn.execute(
                    "INSERT INTO user_profiles(email, full_name, telegram, updated_at) VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(email) DO UPDATE SET full_name=excluded.full_name, telegram=excluded.telegram, updated_at=excluded.updated_at",
                    (
                        email,
                        full_name or old_name,
                        telegram or old_tg,
                        now,
                    ),
                )
            _db_conn.commit()

    def _yandex_token_exchange(code: str) -> Dict[str, Any]:
        if not _yandex_client_id or not _yandex_client_secret:
            raise HTTPException(status_code=503, detail="yandex oauth is not configured")
        body = urllib.parse.urlencode(
            {
                "grant_type": "authorization_code",
                "code": code,
                "client_id": _yandex_client_id,
                "client_secret": _yandex_client_secret,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            "https://oauth.yandex.com/token",
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", "ignore")
            raise HTTPException(status_code=502, detail=f"yandex token exchange failed: {raw or e.code}")
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"yandex token exchange failed: {e}")

    def _yandex_userinfo(access_token: str) -> Dict[str, Any]:
        req = urllib.request.Request(
            "https://login.yandex.ru/info?format=json",
            headers={"Authorization": f"OAuth {access_token}"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", "ignore")
            raise HTTPException(status_code=502, detail=f"yandex userinfo failed: {raw or e.code}")
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"yandex userinfo failed: {e}")

    _web_session_cookie = os.environ.get("WEB_SESSION_COOKIE_NAME", "tf_session_v1").strip() or "tf_session_v1"
    _web_session_domain = os.environ.get("WEB_SESSION_COOKIE_DOMAIN", ".tradeforge.art").strip() or ".tradeforge.art"
    _web_session_secure = os.environ.get("WEB_SESSION_COOKIE_SECURE", "1") == "1"

    def _set_web_session_cookie(response: Response, token: str) -> None:
        response.set_cookie(
            key=_web_session_cookie,
            value=token,
            httponly=True,
            secure=_web_session_secure,
            samesite="lax",
            domain=_web_session_domain,
            path="/",
            max_age=_session_ttl_sec,
        )

    def _clear_web_session_cookie(response: Response) -> None:
        response.delete_cookie(
            key=_web_session_cookie,
            domain=_web_session_domain,
            path="/",
            httponly=True,
            secure=_web_session_secure,
            samesite="lax",
        )

    def _extract_bearer_token(request: Request) -> str:
        auth = request.headers.get("authorization", "").strip()
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        token = request.headers.get("x-auth-token", "").strip()
        if token:
            return token
        return request.cookies.get(_web_session_cookie, "").strip()

    def _require_user(request: Request) -> str:
        token = _extract_bearer_token(request)
        if not token:
            raise HTTPException(status_code=401, detail="missing auth token")
        email = _verify_session_token(token)
        if not email:
            raise HTTPException(status_code=401, detail="invalid auth token")
        return email

    def _require_internal_api_key(request: Request) -> None:
        if not api_key:
            raise HTTPException(status_code=503, detail="api key is not configured")
        header = request.headers.get("x-api-key", "").strip()
        if not hmac.compare_digest(header, api_key):
            raise HTTPException(status_code=401, detail="unauthorized")

    def _send_smtp_message(msg: EmailMessage) -> tuple[bool, str]:
        smtp_host = os.environ.get("SMTP_HOST", "").strip()
        smtp_port = int(os.environ.get("SMTP_PORT", "587"))
        smtp_user = os.environ.get("SMTP_USER", "").strip()
        smtp_pass = os.environ.get("SMTP_PASS", "").strip()
        smtp_tls = os.environ.get("SMTP_TLS", "1") == "1"
        smtp_ssl = os.environ.get("SMTP_SSL", "0") == "1"
        smtp_timeout = int(os.environ.get("SMTP_TIMEOUT", "15"))
        smtp_from = os.environ.get("SMTP_FROM", "").strip() or smtp_user

        if not smtp_host or not smtp_from:
            return False, "smtp_not_configured"
        if "From" not in msg:
            msg["From"] = smtp_from

        try:
            if smtp_ssl:
                with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=smtp_timeout) as s:
                    if smtp_user:
                        s.login(smtp_user, smtp_pass)
                    s.send_message(msg)
            else:
                with smtplib.SMTP(smtp_host, smtp_port, timeout=smtp_timeout) as s:
                    s.ehlo()
                    if smtp_tls:
                        s.starttls()
                        s.ehlo()
                    if smtp_user:
                        s.login(smtp_user, smtp_pass)
                    s.send_message(msg)
            return True, "sent"
        except Exception as e:
            print(f"[auth] SMTP send failed: {e}", flush=True)
            return False, "smtp_send_failed"

    def _build_auth_email_html(title: str, subtitle: str, code: str) -> str:
        safe_code = html.escape(code)
        safe_title = html.escape(title)
        safe_sub = html.escape(subtitle)
        return (
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>TradeForge</title></head>"
            "<body style='margin:0;background:#0a0c10;color:#f3f3f3;font-family:Arial,sans-serif;'>"
            "<div style='max-width:560px;margin:24px auto;padding:20px;border:1px solid #2a2f3a;"
            "border-radius:14px;background:#11151d;'>"
            "<div style='font-size:22px;font-weight:700;color:#e1c26a;margin-bottom:8px;'>TradeForge</div>"
            f"<div style='font-size:20px;font-weight:700;margin-bottom:8px;'>{safe_title}</div>"
            f"<div style='color:#b8bfcc;margin-bottom:18px;line-height:1.5;'>{safe_sub}</div>"
            "<div style='background:#0b0f16;border:1px solid #2f3a4d;border-radius:12px;"
            "padding:14px;text-align:center;margin-bottom:14px;'>"
            f"<span style='font-size:30px;letter-spacing:6px;font-weight:800;color:#9ce6d0;'>{safe_code}</span>"
            "</div>"
            "<div style='color:#9aa2b2;font-size:13px;line-height:1.5;'>"
            "Код действителен 15 минут. Если это были не вы, просто проигнорируйте письмо."
            "</div></div></body></html>"
        )

    def _send_email_code(email: str, code: str) -> tuple[bool, str]:
        smtp_user = os.environ.get("SMTP_USER", "").strip()
        smtp_from = os.environ.get("SMTP_FROM", "").strip() or smtp_user
        if not smtp_from:
            print(f"[auth] SMTP not configured; verification code for {email}: {code}", flush=True)
            return False, "smtp_not_configured"

        msg = EmailMessage()
        msg["Subject"] = "TradeForge — Код подтверждения email"
        msg["From"] = smtp_from
        msg["To"] = email
        msg.set_content(
            f"Ваш код подтверждения TradeForge: {code}\n\n"
            "Код действует 15 минут.\n"
            "Если это были не вы, проигнорируйте письмо."
        )
        msg.add_alternative(
            _build_auth_email_html(
                title="Подтверждение email",
                subtitle="Используйте код ниже, чтобы завершить регистрацию в TradeForge.",
                code=code,
            ),
            subtype="html",
        )
        return _send_smtp_message(msg)

    def _issue_email_code(email: str) -> tuple[bool, str]:
        now = int(time.time())
        code = f"{secrets.randbelow(1_000_000):06d}"
        code_hash = _hash_email_code(email, code)
        exp = now + 15 * 60
        with _db_lock:
            _db_conn.execute(
                "INSERT INTO email_verifications(email, code_hash, expires_at, attempts, sent_at) "
                "VALUES (?, ?, ?, 0, ?) "
                "ON CONFLICT(email) DO UPDATE SET code_hash=excluded.code_hash, expires_at=excluded.expires_at, attempts=0, sent_at=excluded.sent_at",
                (email, code_hash, exp, now),
            )
            _db_conn.commit()
        return _send_email_code(email, code)

    def _send_password_reset_code(email: str, code: str) -> tuple[bool, str]:
        smtp_user = os.environ.get("SMTP_USER", "").strip()
        smtp_from = os.environ.get("SMTP_FROM", "").strip() or smtp_user
        if not smtp_from:
            print(f"[auth] SMTP not configured; reset code for {email}: {code}", flush=True)
            return False, "smtp_not_configured"

        msg = EmailMessage()
        msg["Subject"] = "TradeForge — Код сброса пароля"
        msg["From"] = smtp_from
        msg["To"] = email
        msg.set_content(
            f"Ваш код для сброса пароля TradeForge: {code}\n\n"
            "Код действует 15 минут.\n"
            "Если это были не вы, проигнорируйте письмо."
        )
        msg.add_alternative(
            _build_auth_email_html(
                title="Сброс пароля",
                subtitle="Введите код ниже на странице входа, чтобы установить новый пароль.",
                code=code,
            ),
            subtype="html",
        )
        return _send_smtp_message(msg)

    def _issue_password_reset_code(email: str) -> tuple[bool, str]:
        now = int(time.time())
        code = f"{secrets.randbelow(1_000_000):06d}"
        code_hash = _hash_email_code(email, code)
        exp = now + 15 * 60
        with _db_lock:
            _db_conn.execute(
                "INSERT INTO password_resets(email, code_hash, expires_at, attempts, sent_at) "
                "VALUES (?, ?, ?, 0, ?) "
                "ON CONFLICT(email) DO UPDATE SET code_hash=excluded.code_hash, expires_at=excluded.expires_at, attempts=0, sent_at=excluded.sent_at",
                (email, code_hash, exp, now),
            )
            _db_conn.commit()
        return _send_password_reset_code(email, code)

    def _subscription_snapshot(email: str) -> Dict[str, Any]:
        if _is_owner_email(email):
            return _owner_subscription_snapshot(email)
        now = int(time.time())
        with _db_lock:
            row = _db_conn.execute(
                "SELECT plan, status, started_at, expires_at, updated_at, txid FROM subscriptions WHERE email = ?",
                (email,),
            ).fetchone()
        if not row:
            return {
                "active": False,
                "plan": None,
                "status": "none",
                "started_at": None,
                "expires_at": None,
                "updated_at": None,
                "txid": None,
            }
        plan, status, started_at, expires_at, updated_at, txid = row
        active = (status == "active") and int(expires_at) > now
        return {
            "active": bool(active),
            "plan": plan,
            "status": status,
            "started_at": int(started_at),
            "expires_at": int(expires_at),
            "updated_at": int(updated_at),
            "txid": txid,
        }

    def _ensure_trial_subscription(email: str) -> Dict[str, Any]:
        """
        Grant one-time 3-day trial only if user has never had a subscription row.
        """
        if _is_owner_email(email):
            return _owner_subscription_snapshot(email)
        now = int(time.time())
        with _db_lock:
            row = _db_conn.execute(
                "SELECT plan, status, started_at, expires_at, updated_at, txid FROM subscriptions WHERE email = ?",
                (email,),
            ).fetchone()
            if row:
                plan, status, started_at, expires_at, updated_at, txid = row
                active = (status == "active") and int(expires_at) > now
                return {
                    "active": bool(active),
                    "plan": plan,
                    "status": status,
                    "started_at": int(started_at),
                    "expires_at": int(expires_at),
                    "updated_at": int(updated_at),
                    "txid": txid,
                }

            trial_expires = now + 3 * 86400
            _db_conn.execute(
                "INSERT INTO subscriptions(email, plan, status, started_at, expires_at, updated_at, updated_by, txid) "
                "VALUES (?, 'trial3', 'active', ?, ?, ?, 'system_trial', NULL)",
                (email, now, trial_expires, now),
            )
            _db_conn.commit()
            return {
                "active": True,
                "plan": "trial3",
                "status": "active",
                "started_at": now,
                "expires_at": trial_expires,
                "updated_at": now,
                "txid": None,
            }

    def _require_admin(request: Request) -> None:
        if not _admin_api_key:
            raise HTTPException(status_code=503, detail="admin api key is not configured")
        got = request.headers.get("x-admin-key", "")
        if not got or not hmac.compare_digest(got, _admin_api_key):
            raise HTTPException(status_code=401, detail="unauthorized")
    allow_origins = os.environ.get(
        "ALLOW_ORIGINS",
        "http://localhost:8080,http://127.0.0.1:8080,"
        "http://localhost:8090,http://127.0.0.1:8090,"
        "http://localhost:8091,http://127.0.0.1:8091,"
        "http://localhost:8088,http://127.0.0.1:8088,"
        "https://tradeforge.art,https://www.tradeforge.art",
    )
    allow_origins = [o.strip() for o in allow_origins.split(",") if o.strip()]
    if "*" in allow_origins:
        allow_credentials = False
        allow_origins = ["*"]
    else:
        allow_credentials = True
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(
        GZipMiddleware,
        minimum_size=1024,
        compresslevel=6,
    )

    api_key_protected_paths = tuple(
        p.strip() for p in os.environ.get("API_KEY_PROTECTED_PATHS", "/signal_quality/log").split(",") if p.strip()
    )
    strict_api_key = os.environ.get("REQUIRE_API_KEY", "1") == "1"

    @app.middleware("http")
    async def require_api_key(request: Request, call_next):
        if request.method == "OPTIONS" or request.url.path == "/health":
            return await call_next(request)
        path = request.url.path
        is_protected = any(path == p or path.startswith(p + "/") for p in api_key_protected_paths)
        if not is_protected:
            return await call_next(request)
        if not api_key:
            if strict_api_key:
                return JSONResponse(status_code=503, content={"detail": "api key is not configured"})
            return await call_next(request)
        header = request.headers.get("x-api-key", "").strip()
        if not hmac.compare_digest(header, api_key):
            return JSONResponse(status_code=401, content={"detail": "unauthorized"})
        return await call_next(request)

    @app.get("/health")
    def health():
        return {"status": "ok", "features": feature_cols}

    @app.post("/signal_quality/log")
    async def signal_quality_log(payload: Dict[str, Any], request: Request):
        rec = dict(payload or {})
        rec["server_ts"] = pd.Timestamp.utcnow().isoformat()
        rec["client_ip"] = request.client.host if request.client else None
        _append_signal_quality_log(rec)
        return {"ok": True}

    def _valid_email(email: str) -> bool:
        return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email))

    @app.post("/auth/register")
    def auth_register(req: AuthRequest):
        email = req.email.strip().lower()
        if not _valid_email(email):
            raise HTTPException(status_code=400, detail="invalid email")
        if len(req.password) < 10:
            raise HTTPException(status_code=400, detail="password too short")
        salt = _make_salt()
        pw_hash = _hash_password(req.password, salt)
        with _db_lock:
            try:
                _db_conn.execute(
                    "INSERT INTO users(email, salt, hash, created_at, verified_at) VALUES (?, ?, ?, ?, NULL)",
                    (email, salt.hex(), pw_hash, int(time.time())),
                )
                _db_conn.commit()
            except sqlite3.IntegrityError:
                cur = _db_conn.execute("SELECT verified_at FROM users WHERE email = ?", (email,))
                row = cur.fetchone()
                if not row:
                    raise HTTPException(status_code=409, detail="email already registered")
                if row[0]:
                    raise HTTPException(status_code=409, detail="email already registered")
                # User exists but still unverified: refresh password and resend verification
                _db_conn.execute(
                    "UPDATE users SET salt = ?, hash = ? WHERE email = ?",
                    (salt.hex(), pw_hash, email),
                )
                _db_conn.commit()
        sent, status = _issue_email_code(email)
        return {"ok": True, "verification_required": True, "email_sent": sent, "status": status}

    @app.post("/auth/verify-email")
    def auth_verify_email(req: VerifyEmailRequest):
        email = req.email.strip().lower()
        code = req.code.strip()
        if not _valid_email(email):
            raise HTTPException(status_code=400, detail="invalid email")
        if not re.match(r"^\d{4,8}$", code):
            raise HTTPException(status_code=400, detail="invalid code format")
        with _db_lock:
            cur = _db_conn.execute(
                "SELECT code_hash, expires_at, attempts FROM email_verifications WHERE email = ?",
                (email,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="verification code not found")
            code_hash, expires_at, attempts = row
            now = int(time.time())
            if int(attempts or 0) >= 10:
                raise HTTPException(status_code=429, detail="too many attempts")
            if now > int(expires_at):
                raise HTTPException(status_code=410, detail="verification code expired")
            calc = _hash_email_code(email, code)
            if not hmac.compare_digest(calc, code_hash):
                _db_conn.execute(
                    "UPDATE email_verifications SET attempts = attempts + 1 WHERE email = ?",
                    (email,),
                )
                _db_conn.commit()
                raise HTTPException(status_code=401, detail="invalid verification code")
            _db_conn.execute("UPDATE users SET verified_at = ? WHERE email = ?", (now, email))
            _db_conn.execute("DELETE FROM email_verifications WHERE email = ?", (email,))
            _db_conn.commit()
        sub = _ensure_trial_subscription(email)
        token = _issue_session_token(email)
        return {"ok": True, "verified": True, "subscription": sub, "token": token}

    @app.post("/auth/resend-code")
    def auth_resend_code(req: ResendCodeRequest):
        email = req.email.strip().lower()
        if not _valid_email(email):
            raise HTTPException(status_code=400, detail="invalid email")
        with _db_lock:
            cur = _db_conn.execute("SELECT id, verified_at FROM users WHERE email = ?", (email,))
            row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="user not found")
        if row[1]:
            return {"ok": True, "already_verified": True}
        sent, status = _issue_email_code(email)
        return {"ok": True, "email_sent": sent, "status": status}

    @app.post("/auth/login")
    def auth_login(req: AuthRequest):
        email = req.email.strip().lower()
        if not _valid_email(email):
            raise HTTPException(status_code=400, detail="invalid email")
        with _db_lock:
            cur = _db_conn.execute("SELECT salt, hash, verified_at FROM users WHERE email = ?", (email,))
            row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=401, detail="invalid credentials")
        salt_hex, stored_hash, verified_at = row
        try:
            salt = bytes.fromhex(salt_hex)
        except Exception:
            raise HTTPException(status_code=500, detail="user data corrupted")
        calc = _hash_password(req.password, salt)
        if not hmac.compare_digest(calc, stored_hash):
            raise HTTPException(status_code=401, detail="invalid credentials")
        if not verified_at:
            raise HTTPException(status_code=403, detail="email not verified")
        sub = _ensure_trial_subscription(email)
        token = _issue_session_token(email)
        return {"ok": True, "subscription": sub, "token": token}

    @app.post("/auth/web-login")
    def auth_web_login(req: AuthRequest):
        payload = auth_login(req)
        token = str(payload.get("token") or "").strip()
        resp = JSONResponse(payload)
        if token:
            _set_web_session_cookie(resp, token)
        return resp

    @app.get("/auth/yandex/start")
    def auth_yandex_start(next: str = ""):
        if not _yandex_client_id or not _yandex_client_secret:
            raise HTTPException(status_code=503, detail="yandex oauth is not configured")
        safe_next = next.strip() if next and next.startswith("/") and not next.startswith("//") else ""
        state = _issue_oauth_state("yandex", safe_next)
        params = {
            "response_type": "code",
            "client_id": _yandex_client_id,
            "redirect_uri": _yandex_redirect_uri,
            "state": state,
        }
        return RedirectResponse(f"https://oauth.yandex.com/authorize?{urllib.parse.urlencode(params)}", status_code=302)

    @app.get("/auth/yandex/callback")
    def auth_yandex_callback(code: str = "", state: str = "", error: str = "", error_description: str = ""):
        next_path = _verify_oauth_state("yandex", state)
        if next_path is None:
            raise HTTPException(status_code=400, detail="invalid oauth state")
        if error:
            detail = error_description or error
            target = _append_query_params(f"{_public_web_root}/", {"oauth_error": detail, "next": next_path or None})
            return RedirectResponse(target, status_code=302)
        if not code:
            raise HTTPException(status_code=400, detail="missing oauth code")
        token_payload = _yandex_token_exchange(code)
        access_token = str(token_payload.get("access_token") or "").strip()
        if not access_token:
            raise HTTPException(status_code=502, detail="yandex access token missing")
        profile = _yandex_userinfo(access_token)
        email = (
            str(profile.get("default_email") or "").strip().lower()
            or str(profile.get("email") or "").strip().lower()
        )
        if not email:
            emails = profile.get("emails") or []
            if isinstance(emails, list) and emails:
                email = str(emails[0] or "").strip().lower()
        if not _valid_email(email):
            target = _append_query_params(f"{_public_web_root}/", {
                "oauth_error": "Yandex не вернул email. Разреши доступ к email в приложении OAuth.",
                "next": next_path or None,
            })
            return RedirectResponse(target, status_code=302)
        first_name = str(profile.get("first_name") or "").strip()
        last_name = str(profile.get("last_name") or "").strip()
        display_name = str(profile.get("real_name") or profile.get("display_name") or "").strip()
        full_name = " ".join(part for part in [first_name, last_name] if part).strip() or display_name
        _upsert_oauth_user(email, full_name=full_name)
        sub = _ensure_trial_subscription(email)
        token = _issue_session_token(email)
        target = _append_query_params(
            f"{_public_web_root}/",
            {
                "oauth": "yandex",
                "next": next_path or None,
            },
        )
        resp = RedirectResponse(target, status_code=302)
        _set_web_session_cookie(resp, token)
        return resp

    @app.post("/auth/web-logout")
    def auth_web_logout():
        resp = JSONResponse({"ok": True})
        _clear_web_session_cookie(resp)
        return resp

    @app.get("/auth/nginx-check")
    def auth_nginx_check(request: Request):
        _require_user(request)
        return Response(status_code=204)

    @app.get("/auth/me")
    def auth_me(request: Request):
        email = _require_user(request)
        with _db_lock:
            user_row = _db_conn.execute(
                "SELECT id FROM users WHERE email = ?",
                (email,),
            ).fetchone()
            row = _db_conn.execute(
                "SELECT full_name, telegram, updated_at FROM user_profiles WHERE email = ?",
                (email,),
            ).fetchone()
            payment_row = _db_conn.execute(
                "SELECT txid, status, provider, billing_period, amount_rub, created_at, reviewed_at "
                "FROM payments WHERE email = ? ORDER BY created_at DESC LIMIT 1",
                (email,),
            ).fetchone()
        profile = {
            "full_name": (row[0] or "") if row else "",
            "telegram": (row[1] or "") if row else "",
            "updated_at": int(row[2] or 0) if row and row[2] else 0,
        }
        latest_payment = None
        if payment_row:
            latest_payment = {
                "txid": str(payment_row[0] or ""),
                "status": str(payment_row[1] or ""),
                "provider": str(payment_row[2] or ""),
                "billing_period": str(payment_row[3] or ""),
                "amount_rub": int(payment_row[4] or 0),
                "created_at": int(payment_row[5] or 0),
                "reviewed_at": int(payment_row[6] or 0) if payment_row[6] else 0,
            }
        subscription = _subscription_snapshot(email)
        return {
            "ok": True,
            "email": email,
            "subscription": subscription,
            "profile": profile,
            "support": {
                "user_id": int(user_row[0]) if user_row else 0,
                "support_code": _support_code(email),
                "subscription_txid": str(subscription.get("txid") or ""),
                "latest_payment": latest_payment,
            },
        }

    @app.get("/auth/session-bridge")
    def auth_session_bridge(request: Request):
        email = _require_user(request)
        token = _extract_bearer_token(request)
        return {"ok": True, "email": email, "token": token, "subscription": _subscription_snapshot(email)}

    @app.post("/auth/profile")
    def auth_update_profile(req: UpdateProfileRequest, request: Request):
        email = _require_user(request)
        full_name = str(req.full_name or "").strip()
        telegram = str(req.telegram or "").strip()
        if len(full_name) > 200:
            raise HTTPException(status_code=400, detail="full_name too long")
        if len(telegram) > 200:
            raise HTTPException(status_code=400, detail="telegram too long")
        now = int(time.time())
        with _db_lock:
            _db_conn.execute(
                "INSERT INTO user_profiles(email, full_name, telegram, updated_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(email) DO UPDATE SET full_name=excluded.full_name, telegram=excluded.telegram, updated_at=excluded.updated_at",
                (email, full_name, telegram, now),
            )
            _db_conn.commit()
        return {
            "ok": True,
            "profile": {
                "full_name": full_name,
                "telegram": telegram,
                "updated_at": now,
            },
        }

    @app.post("/auth/request-password-reset")
    def auth_request_password_reset(req: RequestPasswordReset):
        email = req.email.strip().lower()
        # Do not reveal if user exists.
        if not _valid_email(email):
            return {"ok": True, "status": "accepted"}
        with _db_lock:
            row = _db_conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
        if row:
            _issue_password_reset_code(email)
        return {"ok": True, "status": "accepted"}

    @app.post("/auth/reset-password")
    def auth_reset_password(req: ResetPasswordRequest):
        email = req.email.strip().lower()
        code = req.code.strip()
        new_password = req.new_password
        if not _valid_email(email):
            raise HTTPException(status_code=400, detail="invalid email")
        if not re.match(r"^\d{4,8}$", code):
            raise HTTPException(status_code=400, detail="invalid code format")
        if len(new_password) < 10:
            raise HTTPException(status_code=400, detail="password too short")
        with _db_lock:
            row = _db_conn.execute(
                "SELECT code_hash, expires_at, attempts FROM password_resets WHERE email = ?",
                (email,),
            ).fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="reset code not found")
            code_hash, expires_at, attempts = row
            now = int(time.time())
            if int(attempts or 0) >= 10:
                raise HTTPException(status_code=429, detail="too many attempts")
            if now > int(expires_at):
                raise HTTPException(status_code=410, detail="reset code expired")
            calc = _hash_email_code(email, code)
            if not hmac.compare_digest(calc, code_hash):
                _db_conn.execute("UPDATE password_resets SET attempts = attempts + 1 WHERE email = ?", (email,))
                _db_conn.commit()
                raise HTTPException(status_code=401, detail="invalid reset code")

            user = _db_conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
            if not user:
                raise HTTPException(status_code=404, detail="user not found")
            salt = _make_salt()
            pw_hash = _hash_password(new_password, salt)
            _db_conn.execute("UPDATE users SET salt = ?, hash = ? WHERE email = ?", (salt.hex(), pw_hash, email))
            _db_conn.execute("DELETE FROM password_resets WHERE email = ?", (email,))
            _db_conn.commit()
        return {"ok": True}

    @app.get("/subscription/status")
    def subscription_status(request: Request):
        email = _require_user(request)
        return {"ok": True, "subscription": _subscription_snapshot(email)}

    @app.get("/bot/subscription/status")
    def bot_subscription_status(email: str, request: Request):
        _require_internal_api_key(request)
        user_email = (email or "").strip().lower()
        if not _valid_email(user_email):
            raise HTTPException(status_code=400, detail="invalid email")
        return {"ok": True, "subscription": _subscription_snapshot(user_email)}

    @app.post("/billing/submit-tx")
    def billing_submit_tx(req: SubmitPaymentRequest, request: Request):
        token_email = _require_user(request)
        email = req.email.strip().lower()
        txid = req.txid.strip()
        plan = (req.plan or "pro50").strip().lower()
        if not _valid_email(email):
            raise HTTPException(status_code=400, detail="invalid email")
        if email != token_email:
            raise HTTPException(status_code=403, detail="email/token mismatch")
        if len(txid) < 8:
            raise HTTPException(status_code=400, detail="invalid txid")
        allowed_plans = {"starter30", "pro50", "elite100"}
        if plan not in allowed_plans:
            raise HTTPException(status_code=400, detail="invalid plan")
        now = int(time.time())
        with _db_lock:
            user = _db_conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
            if not user:
                raise HTTPException(status_code=404, detail="user not found")
            try:
                _db_conn.execute(
                    "INSERT INTO payments(email, plan, txid, status, created_at) VALUES (?, ?, ?, 'pending', ?)",
                    (email, plan, txid, now),
                )
                _db_conn.commit()
            except sqlite3.IntegrityError:
                return {"ok": True, "status": "already_submitted"}
        return {"ok": True, "status": "pending_review", "plan": plan}

    @app.post("/billing/yookassa/create-payment")
    def billing_yookassa_create_payment(req: CreateYooKassaPaymentRequest, request: Request):
        email = _require_user(request)
        offer = _plan_offer(req.plan, req.billing_period)
        amount_rub = offer["amount_rub"]
        description = f"TradeForge {offer['plan_name']} — {offer['billing_period']}"
        confirmation_return_url = _append_query_params(
            _yookassa_return_url,
            {
                "plan": offer["plan"],
                "period": offer["billing_period"],
                "amount": int(amount_rub),
            },
        )
        payload: Dict[str, Any] = {
            "amount": {"value": f"{amount_rub:.2f}", "currency": "RUB"},
            "capture": True,
            "confirmation": {
                "type": "redirect",
                "return_url": confirmation_return_url,
            },
            "description": description,
            "metadata": {
                "email": email,
                "plan": offer["plan"],
                "billing_period": offer["billing_period"],
                "days": str(offer["days"]),
            },
        }
        if _yookassa_payment_method:
            payload["payment_method_data"] = {"type": _yookassa_payment_method}
        data = _yookassa_request(
            "POST",
            "/payments",
            payload,
            idempotence_key=f"tf-{email}-{offer['plan']}-{offer['billing_period']}-{secrets.token_hex(8)}",
        )
        payment_id = str(data.get("id") or "").strip()
        if not payment_id:
            raise HTTPException(status_code=502, detail="yookassa returned empty payment id")
        confirmation_url = (
            ((data.get("confirmation") or {}) if isinstance(data.get("confirmation"), dict) else {})
            .get("confirmation_url", "")
            .strip()
        )
        now = int(time.time())
        with _db_lock:
            _db_conn.execute(
                "INSERT INTO payments(email, plan, txid, status, created_at, provider, billing_period, amount_rub) "
                "VALUES (?, ?, ?, ?, ?, 'yookassa', ?, ?) "
                "ON CONFLICT(txid) DO UPDATE SET "
                "email=excluded.email, plan=excluded.plan, status=excluded.status, created_at=excluded.created_at, "
                "provider='yookassa', billing_period=excluded.billing_period, amount_rub=excluded.amount_rub",
                (email, offer["plan"], payment_id, str(data.get("status") or "pending"), now, offer["billing_period"], amount_rub),
            )
            _db_conn.commit()
        return {
            "ok": True,
            "provider": "yookassa",
            "payment_id": payment_id,
            "status": str(data.get("status") or "pending"),
            "amount_rub": amount_rub,
            "plan": offer["plan"],
            "billing_period": offer["billing_period"],
            "confirmation_url": confirmation_url,
        }

    @app.post("/billing/prodamus/create-payment")
    def billing_prodamus_create_payment(req: CreateYooKassaPaymentRequest, request: Request):
        email = _require_user(request)
        offer = _plan_offer(req.plan, req.billing_period)
        amount_rub = offer["amount_rub"]
        order_id = f"tf-{int(time.time())}-{secrets.token_hex(5)}"
        success_url = _append_query_params(
            _prodamus_return_url,
            {
                "provider": "prodamus",
                "order_id": order_id,
                "plan": offer["plan"],
                "period": offer["billing_period"],
                "amount": int(amount_rub),
            },
        )
        payload: Dict[str, Any] = {
            "order_id": order_id,
            "customer_email": email,
            "products": [
                {
                    "name": f"TradeForge {offer['plan_name']} — {offer['billing_period']}",
                    "price": str(amount_rub),
                    "quantity": "1",
                }
            ],
            "currency": "rub",
            "do": "pay",
            "urlReturn": _prodamus_cancel_url,
            "urlSuccess": success_url,
            "urlNotification": _prodamus_webhook_url,
            "callbackType": "json",
            "paid_content": "Оплата принята. Доступ к TradeForge будет активирован автоматически после webhook.",
        }
        if _prodamus_payment_method:
            payload["payment_method"] = _prodamus_payment_method
        if _prodamus_demo_mode:
            payload["demo_mode"] = "1"
        confirmation_url = _prodamus_payment_link(payload)
        now = int(time.time())
        with _db_lock:
            _db_conn.execute(
                "INSERT INTO payments(email, plan, txid, status, created_at, provider, billing_period, amount_rub) "
                "VALUES (?, ?, ?, ?, ?, 'prodamus', ?, ?) "
                "ON CONFLICT(txid) DO UPDATE SET "
                "email=excluded.email, plan=excluded.plan, status=excluded.status, created_at=excluded.created_at, "
                "provider='prodamus', billing_period=excluded.billing_period, amount_rub=excluded.amount_rub",
                (email, offer["plan"], order_id, "pending", now, offer["billing_period"], amount_rub),
            )
            _db_conn.commit()
        return {
            "ok": True,
            "provider": "prodamus",
            "payment_id": order_id,
            "status": "pending",
            "amount_rub": amount_rub,
            "plan": offer["plan"],
            "billing_period": offer["billing_period"],
            "confirmation_url": confirmation_url,
        }

    @app.post("/billing/yookassa/webhook")
    async def billing_yookassa_webhook(request: Request):
        try:
            payload = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="invalid json")
        obj = payload.get("object") if isinstance(payload, dict) else None
        if not isinstance(obj, dict):
            raise HTTPException(status_code=400, detail="invalid payload")
        payment_id = str(obj.get("id") or "").strip()
        status = str(obj.get("status") or "").strip().lower()
        metadata = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
        email = str(metadata.get("email") or "").strip().lower()
        plan = str(metadata.get("plan") or "").strip().lower()
        billing_period = str(metadata.get("billing_period") or "").strip().lower()
        if not payment_id:
            raise HTTPException(status_code=400, detail="missing payment id")
        review_note = json.dumps({"event": payload.get("event"), "status": status}, ensure_ascii=False)
        now = int(time.time())
        with _db_lock:
            _db_conn.execute(
                "INSERT INTO payments(email, plan, txid, status, created_at, reviewed_at, review_note, provider, billing_period, amount_rub) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'yookassa', ?, ?) "
                "ON CONFLICT(txid) DO UPDATE SET "
                "status=excluded.status, reviewed_at=excluded.reviewed_at, review_note=excluded.review_note, "
                "provider='yookassa', billing_period=COALESCE(excluded.billing_period, payments.billing_period), "
                "amount_rub=COALESCE(excluded.amount_rub, payments.amount_rub), plan=COALESCE(excluded.plan, payments.plan), "
                "email=COALESCE(excluded.email, payments.email)",
                (
                    email or "",
                    _normalize_plan(plan) or None,
                    payment_id,
                    status or "pending",
                    now,
                    now,
                    review_note,
                    _normalize_billing_period(billing_period) or None,
                    int(float(((obj.get("amount") or {}) if isinstance(obj.get("amount"), dict) else {}).get("value", 0.0) or 0.0)),
                ),
            )
            _db_conn.commit()
        if status == "succeeded" and email and _valid_email(email):
            try:
                offer = _plan_offer(plan, billing_period)
                sub = _upsert_subscription(email, offer["plan"], offer["days"], "yookassa", payment_id)
                return {"ok": True, "subscription": sub}
            except Exception as e:
                print(f"[billing] yookassa activation failed: {e}", flush=True)
        return {"ok": True}

    @app.post("/billing/prodamus/webhook")
    async def billing_prodamus_webhook(request: Request):
        raw = await request.body()
        signature = request.headers.get("Sign", "").strip()
        if not signature:
            raise HTTPException(status_code=400, detail="missing prodamus signature")
        payload = _parse_prodamus_body(raw, request.headers.get("content-type", ""))
        if not _prodamus_secret_key:
            raise HTTPException(status_code=503, detail="prodamus is not configured")
        expected = _prodamus_signature(payload)
        if not hmac.compare_digest(signature, expected):
            raise HTTPException(status_code=401, detail="invalid prodamus signature")

        order_id = str(payload.get("order_id") or payload.get("order_num") or "").strip()
        email = str(payload.get("customer_email") or "").strip().lower()
        review_note = json.dumps(payload, ensure_ascii=False)
        paid_markers = {
            str(payload.get("paid") or "").strip().lower(),
            str(payload.get("payment_status") or "").strip().lower(),
            str(payload.get("status") or "").strip().lower(),
        }
        is_paid = any(marker in {"1", "true", "paid", "success", "succeeded"} for marker in paid_markers)
        if not order_id:
            raise HTTPException(status_code=400, detail="missing order id")

        now = int(time.time())
        with _db_lock:
            existing = _db_conn.execute(
                "SELECT email, plan, billing_period, amount_rub FROM payments WHERE txid = ?",
                (order_id,),
            ).fetchone()
            existing_email = str(existing[0] or "").strip().lower() if existing else ""
            plan = str(existing[1] or "").strip().lower() if existing else ""
            billing_period = str(existing[2] or "").strip().lower() if existing else ""
            amount_rub = int(existing[3] or 0) if existing else 0
            final_email = email or existing_email
            _db_conn.execute(
                "INSERT INTO payments(email, plan, txid, status, created_at, reviewed_at, review_note, provider, billing_period, amount_rub) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'prodamus', ?, ?) "
                "ON CONFLICT(txid) DO UPDATE SET "
                "email=COALESCE(excluded.email, payments.email), "
                "plan=COALESCE(excluded.plan, payments.plan), "
                "status=excluded.status, reviewed_at=excluded.reviewed_at, review_note=excluded.review_note, "
                "provider='prodamus', billing_period=COALESCE(excluded.billing_period, payments.billing_period), "
                "amount_rub=CASE WHEN excluded.amount_rub > 0 THEN excluded.amount_rub ELSE payments.amount_rub END",
                (final_email or None, plan or None, order_id, "paid" if is_paid else "pending", now, now, review_note, billing_period or None, amount_rub),
            )
            if is_paid and final_email and plan and billing_period:
                offer = _plan_offer(plan, billing_period)
                _upsert_subscription(final_email, offer["plan"], offer["days"], "prodamus", order_id)
            _db_conn.commit()
        return {"ok": True}

    @app.post("/admin/subscription/activate")
    def admin_subscription_activate(req: AdminActivateSubscriptionRequest, request: Request):
        _require_admin(request)
        email = req.email.strip().lower()
        if not _valid_email(email):
            raise HTTPException(status_code=400, detail="invalid email")
        now = int(time.time())
        exp = now + int(req.days) * 86400
        with _db_lock:
            user = _db_conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
            if not user:
                raise HTTPException(status_code=404, detail="user not found")
            _db_conn.execute(
                "INSERT INTO subscriptions(email, plan, status, started_at, expires_at, updated_at, updated_by, txid) "
                "VALUES (?, ?, 'active', ?, ?, ?, 'admin', ?) "
                "ON CONFLICT(email) DO UPDATE SET "
                "plan=excluded.plan, status='active', started_at=excluded.started_at, expires_at=excluded.expires_at, "
                "updated_at=excluded.updated_at, updated_by='admin', txid=excluded.txid",
                (email, req.plan, now, exp, now, req.txid),
            )
            if req.txid:
                _db_conn.execute(
                    "UPDATE payments SET status='approved', reviewed_at=?, review_note=? WHERE txid = ?",
                    (now, req.note or "approved", req.txid),
                )
            _db_conn.commit()
        return {"ok": True, "subscription": _subscription_snapshot(email)}

    @app.get("/admin/payments/pending")
    def admin_payments_pending(request: Request, limit: int = 100):
        _require_admin(request)
        lim = max(1, min(int(limit), 500))
        with _db_lock:
            rows = _db_conn.execute(
                "SELECT id, email, plan, txid, status, created_at FROM payments WHERE status='pending' ORDER BY created_at DESC LIMIT ?",
                (lim,),
            ).fetchall()
        items = [
            {
                "id": int(r[0]),
                "email": r[1],
                "plan": r[2] or "pro50",
                "txid": r[3],
                "status": r[4],
                "created_at": int(r[5]),
            }
            for r in rows
        ]
        return {"ok": True, "items": items}

    @app.post("/predict")
    def predict(req: PredictRequest):
        window = np.array(req.window, dtype=np.float32)
        if window.shape != (seq_len, n_features):
            raise HTTPException(status_code=400, detail=f"window shape must be ({seq_len}, {n_features})")
        # choose stats: request-level overrides saved stats; otherwise raw
        mean = None
        std = None
        if req.mean is not None and req.std is not None:
            mean = np.array(req.mean, dtype=np.float32)
            std = np.array(req.std, dtype=np.float32)
            if mean.shape[0] != n_features or std.shape[0] != n_features:
                raise HTTPException(status_code=400, detail="mean/std length mismatch")
        elif saved_mean_h20 is not None and saved_std_h20 is not None:
            mean = saved_mean_h20.astype(np.float32)
            std = saved_std_h20.astype(np.float32)

        if mean is not None and std is not None:
            std = np.where(std == 0, 1.0, std)
            window = (window - mean) / std

        x = window[np.newaxis, ...]  # (1, seq, features)
        if req.mc_samples > 1:
            samples = []
            for _ in range(req.mc_samples):
                samples.append(model_h20(x, training=True).numpy().squeeze().item())
            pred_mean = float(np.mean(samples))
            pred_std = float(np.std(samples))
            return {"prediction": pred_mean, "pred_std": pred_std}
        else:
            pred = model_h20_specs[0]["model"](x, training=False).numpy().squeeze().item()
            return {"prediction": pred}

    @app.post("/predict_batch")
    def predict_batch(reqs: List[PredictRequest]):
        if not reqs:
            raise HTTPException(status_code=400, detail="Empty batch")
        windows = []
        mc_samples = max(r.mc_samples for r in reqs)
        for req in reqs:
            window = np.array(req.window, dtype=np.float32)
            if window.shape != (seq_len, n_features):
                raise HTTPException(status_code=400, detail=f"window shape must be ({seq_len}, {n_features})")
            mean = None
            std = None
            if req.mean is not None and req.std is not None:
                mean = np.array(req.mean, dtype=np.float32)
                std = np.array(req.std, dtype=np.float32)
                if mean.shape[0] != n_features or std.shape[0] != n_features:
                    raise HTTPException(status_code=400, detail="mean/std length mismatch")
            elif saved_mean_h20 is not None and saved_std_h20 is not None:
                mean = saved_mean_h20.astype(np.float32)
                std = saved_std_h20.astype(np.float32)
            if mean is not None and std is not None:
                std = np.where(std == 0, 1.0, std)
                window = (window - mean) / std
            windows.append(window)
        x = np.stack(windows, axis=0)
        if mc_samples > 1:
            samples = []
            for _ in range(mc_samples):
                samples.append(model_h20(x, training=True).numpy().squeeze())
            samples = np.stack(samples, axis=0)
            preds = samples.mean(axis=0).tolist()
            preds_std = samples.std(axis=0).tolist()
            return {"predictions": preds, "pred_std": preds_std}
        else:
            preds = model_h20_specs[0]["model"](x, training=False).numpy().squeeze().tolist()
            return {"predictions": preds}

    @app.get("/forecast")
    def forecast(interval: str = "h20", hours: int = 24, debug: bool = False):
        cache_key = f"i={interval}|h={int(hours)}|d={1 if debug else 0}"
        if not debug:
            cached = _api_cache_get("forecast", cache_key, _TTL_FORECAST_SEC)
            if cached is not None:
                return cached
        if not features_path:
            raise HTTPException(status_code=400, detail="features_path not configured on server")
        horizon_steps = {"h20": 20, "h80": 80, "h160": 160, "h320": 320, "h640": 640}.get(interval)
        if interval in ("h80", "h160", "h320", "h640") and not (
            model_h80_specs or model_h160_specs or model_h320_specs or model_h640_specs or model_multi
        ):
            raise HTTPException(status_code=400, detail="multi-horizon model not configured")

        # Always keep 15m spacing on chart
        step_min = 15
        df = _load_features_df(features_path)
        noisy_series = None
        debug_info = {}
        if interval == "h20":
            pred_log, window = _predict_weighted_ensemble_log_return(
                model_h20_specs, df, horizon_steps, cls_key="cls", weights=weights_h20
            )
            base_price = float(window["close"].iloc[-1])
            pred_log = _apply_bias_to_log_return(base_price, pred_log, bias_h20)
        elif interval == "h80" and model_h80_specs:
            pred_log, window = _predict_weighted_ensemble_log_return(
                model_h80_specs, df, horizon_steps, cls_key="cls", weights=weights_h80, gate_cfg=gate_h80
            )
            base_price = float(window["close"].iloc[-1])
            pred_log = _apply_bias_to_log_return(base_price, pred_log, bias_h80)
            if model_h20_specs:
                r20, _ = _predict_weighted_ensemble_log_return(model_h20_specs, df, 20, cls_key="cls", weights=weights_h20)
                r20 = _apply_bias_to_log_return(base_price, r20, bias_h20)
                r80 = pred_log
                segs = [(20, float(r20)), (80, float(r80 - r20))]
                noisy_series = _build_piecewise_trend(base_price, horizon_steps, segs)
                debug_info["piecewise"] = True
                debug_info["r20"] = float(r20)
                debug_info["r80"] = float(r80)
        elif interval == "h160" and model_h160_specs:
            pred_log, window = _predict_weighted_ensemble_log_return(
                model_h160_specs, df, horizon_steps, cls_key="cls", weights=weights_h160, gate_cfg=gate_h160
            )
            base_price = float(window["close"].iloc[-1])
            pred_log = _apply_bias_to_log_return(base_price, pred_log, bias_h160)
            if model_h20_specs and model_h80_specs:
                r20, _ = _predict_weighted_ensemble_log_return(model_h20_specs, df, 20, cls_key="cls", weights=weights_h20)
                r80, _ = _predict_weighted_ensemble_log_return(model_h80_specs, df, 80, cls_key="cls", weights=weights_h80)
                r20 = _apply_bias_to_log_return(base_price, r20, bias_h20)
                r80 = _apply_bias_to_log_return(base_price, r80, bias_h80)
                r160 = pred_log
                segs = [(20, float(r20)), (80, float(r80 - r20)), (160, float(r160 - r80))]
                noisy_series = _build_piecewise_trend(base_price, horizon_steps, segs)
                debug_info["piecewise"] = True
                debug_info["r20"] = float(r20)
                debug_info["r80"] = float(r80)
                debug_info["r160"] = float(r160)
        elif interval == "h320" and model_h320_specs:
            pred_log, window = _predict_ensemble_log_return(model_h320_specs, df, horizon_steps, cls_key="cls")
        elif interval == "h640" and model_h640_specs:
            pred_log, window = _predict_ensemble_log_return(model_h640_specs, df, horizon_steps, cls_key="cls")
        else:
            stats_path = stats_multi_path
            window, fcols = _prepare_window(df, seq_len, _feature_cols_from_stats(stats_path, default_feature_list()))
            pred_log = _predict_signed_log_return(
                model_multi, window, fcols, stats_path, stats_meta_multi, horizon=horizon_steps, cls_key=f"cls_h{horizon_steps}"
            )
        last_ts = pd.to_datetime(window["timestamp"].iloc[-1], utc=True)
        last_close = float(window["close"].iloc[-1])
        steps = max(1, int((hours * 60) // step_min))
        if horizon_steps:
            steps = horizon_steps
        times = [last_ts + pd.Timedelta(minutes=step_min * (i + 1)) for i in range(steps)]

        price = last_close
        points = []
        for i in range(steps):
            if noisy_series:
                price = float(noisy_series[i])
            else:
                if horizon_steps:
                    per_step = float(pred_log) / float(horizon_steps)
                else:
                    per_step = float(pred_log)
                price = float(price * math.exp(per_step))
            points.append({"time": int(times[i].timestamp()), "value": price})
            # minimal window shift to avoid repeated timestamp
            last_row = window.iloc[-1].copy()
            last_row["open"] = price
            last_row["high"] = price
            last_row["low"] = price
            last_row["close"] = price
            last_row["vwap"] = price
            window = pd.concat([window.iloc[1:], last_row.to_frame().T], ignore_index=True)

        # Live trade logging (signal from forecast slope)
        crowd_info = {"enabled": bool(crowd_cfg["enabled"]), "ready": False, "multiplier": 1.0}
        if points:
            end_val = points[-1]["value"]
            delta_pct = (end_val - last_close) / last_close if last_close else 0.0
            status = "FLAT"
            if delta_pct > _LIVE_TRADE_DEADZONE:
                status = "LONG"
            elif delta_pct < -_LIVE_TRADE_DEADZONE:
                status = "SHORT"
            elif _LIVE_TRADE_FORCE:
                status = "LONG" if delta_pct >= 0 else "SHORT"
            crowd_info = _crowd_size_multiplier(status, last_ts)
            _update_live_trades(status, int(last_ts.timestamp()), float(last_close))

        res = {
            "interval": interval,
            "step_min": step_min,
            "base_time": int(last_ts.timestamp()),
            "base_price": float(last_close),
            "points": points,
            "crowd_sizing": crowd_info,
            "trap_market_maker": _compute_trap_market_maker(float(last_close), points),
        }
        if debug:
            res["debug"] = {
                "last_ts": str(last_ts),
                "last_close": last_close,
                "horizon_steps": horizon_steps,
                **debug_info,
            }
        if not debug:
            _api_cache_set("forecast", cache_key, res)
        return res

    @app.get("/forecast_multi")
    def forecast_multi(interval: str = "h20", debug: bool = False):
        cache_key = f"i={interval}|d={1 if debug else 0}"
        if not debug:
            cached = _api_cache_get("forecast_multi", cache_key, _TTL_FORECAST_MULTI_SEC)
            if cached is not None:
                return cached
        if not features_path:
            raise HTTPException(status_code=400, detail="features_path not configured on server")
        keys = {"h20": [20], "h80": [80], "h160": [160], "h320": [320], "h640": [640]}
        if interval not in keys:
            raise HTTPException(status_code=400, detail="interval must be h20/h80/h160/h320/h640")
        df = _load_features_df(features_path)
        horizon = keys[interval][0]
        step_min = 15
        step_sec = 15 * 60
        points = []
        debug_info = {}
        if interval == "h20":
            pred_log, window = _predict_weighted_ensemble_log_return(model_h20_specs, df, horizon, cls_key="cls", weights=weights_h20)
            base_price = float(window["close"].iloc[-1])
            pred_log = _apply_bias_to_log_return(base_price, pred_log, bias_h20)
            noisy_series = None
        elif interval == "h80" and model_h80_specs:
            pred_log, window = _predict_weighted_ensemble_log_return(model_h80_specs, df, horizon, cls_key="cls", weights=weights_h80, gate_cfg=gate_h80)
            base_price = float(window["close"].iloc[-1])
            pred_log = _apply_bias_to_log_return(base_price, pred_log, bias_h80)
            # piecewise: use h20 and h80 if available
            if model_h20_specs:
                r20, _ = _predict_weighted_ensemble_log_return(model_h20_specs, df, 20, cls_key="cls", weights=weights_h20)
                r20 = _apply_bias_to_log_return(base_price, r20, bias_h20)
                r80 = pred_log
                segs = [(20, float(r20)), (80, float(r80 - r20))]
                noisy_series = _build_piecewise_trend(base_price, horizon, segs)
                debug_info["piecewise"] = True
                debug_info["r20"] = float(r20)
                debug_info["r80"] = float(r80)
            else:
                noisy_series = _build_noisy_trend(base_price, horizon, step_min, pred_log, window)
        elif interval == "h160" and model_h160_specs:
            pred_log, window = _predict_weighted_ensemble_log_return(model_h160_specs, df, horizon, cls_key="cls", weights=weights_h160, gate_cfg=gate_h160)
            base_price = float(window["close"].iloc[-1])
            pred_log = _apply_bias_to_log_return(base_price, pred_log, bias_h160)
            # piecewise: use h20+h80+h160 if available
            if model_h20_specs and model_h80_specs:
                r20, _ = _predict_weighted_ensemble_log_return(model_h20_specs, df, 20, cls_key="cls", weights=weights_h20)
                r80, _ = _predict_weighted_ensemble_log_return(model_h80_specs, df, 80, cls_key="cls", weights=weights_h80)
                r20 = _apply_bias_to_log_return(base_price, r20, bias_h20)
                r80 = _apply_bias_to_log_return(base_price, r80, bias_h80)
                r160 = pred_log
                segs = [(20, float(r20)), (80, float(r80 - r20)), (160, float(r160 - r80))]
                noisy_series = _build_piecewise_trend(base_price, horizon, segs)
                debug_info["piecewise"] = True
                debug_info["r20"] = float(r20)
                debug_info["r80"] = float(r80)
                debug_info["r160"] = float(r160)
            else:
                noisy_series = _build_noisy_trend(base_price, horizon, step_min, pred_log, window)
        elif interval == "h320" and model_h320_specs:
            pred_log, window = _predict_ensemble_log_return(model_h320_specs, df, horizon, cls_key="cls")
            base_price = float(window["close"].iloc[-1])
            noisy_series = _build_noisy_trend(base_price, horizon, step_min, pred_log, window)
        elif interval == "h640" and model_h640_specs:
            pred_log, window = _predict_ensemble_log_return(model_h640_specs, df, horizon, cls_key="cls")
            base_price = float(window["close"].iloc[-1])
            noisy_series = _build_noisy_trend(base_price, horizon, step_min, pred_log, window)
        elif model_multi is not None:
            window, fcols = _prepare_window(df, seq_len, _feature_cols_from_stats(stats_multi_path, default_feature_list()))
            if horizon > 20:
                pred_log = _predict_multi_horizon_log_return(
                    model_multi,
                    window,
                    fcols,
                    stats_multi_path,
                    stats_meta_multi,
                    horizon=horizon,
                    cls_key=f"cls_h{horizon}",
                )
                debug_info["multi_scale"] = MULTI_SCALE
                noisy_series = _build_noisy_trend(base_price, horizon, step_min, pred_log, window)
            else:
                pred_log = _predict_signed_log_return(
                    model_multi,
                    window,
                    fcols,
                    stats_multi_path,
                    stats_meta_multi,
                    horizon=horizon,
                    cls_key=f"cls_h{horizon}",
                )
                noisy_series = None
        else:
            raise HTTPException(status_code=400, detail="multi-horizon model not configured")

        last_ts = pd.to_datetime(window["timestamp"].iloc[-1], utc=True)
        base_price = float(window["close"].iloc[-1])
        price = base_price
        for i in range(horizon):
            if noisy_series:
                price = float(noisy_series[i])
            else:
                per_step = float(pred_log) / float(horizon)
                price = float(price * math.exp(per_step))
            points.append({"time": int((last_ts + pd.Timedelta(minutes=step_min * (i + 1))).timestamp()), "value": price})
        res = {
            "interval": interval,
            "horizon": horizon,
            "step_min": step_min,
            "base_time": int(last_ts.timestamp()),
            "base_price": float(base_price),
            "points": points,
            "trap_market_maker": _compute_trap_market_maker(float(base_price), points),
        }
        if debug:
            debug_info.update(
                {
                    "last_ts": str(last_ts),
                    "last_close": base_price,
                    "pred_log": float(pred_log),
                    "per_step": float(pred_log / float(horizon)),
                    "min_multi_pct": MIN_MULTI_PCT,
                }
            )
            res["debug"] = debug_info
        if not debug:
            _api_cache_set("forecast_multi", cache_key, res)
        return res

    def _compute_site_model_signals(selected_horizon: str = "5H", persist: bool = False) -> Dict[str, Any]:
        horizon_norm = str(selected_horizon or "5H").upper()
        if horizon_norm not in {"5H", "20H", "40H"}:
            horizon_norm = "5H"

        forecasts = {
            "5H": forecast(interval="h20", debug=False),
            "20H": forecast_multi(interval="h80", debug=False),
            "40H": forecast_multi(interval="h160", debug=False),
        }
        selected_payload = forecasts[horizon_norm]
        base_price = float(selected_payload.get("base_price") or 0.0)
        base_ts = int(selected_payload.get("base_time") or 0)

        atr_value: Optional[float] = None
        try:
            if features_path and os.path.exists(features_path):
                dfx = _load_features_df(features_path)
                if not dfx.empty:
                    if "atr" in dfx.columns:
                        atr_raw = pd.to_numeric(dfx["atr"].iloc[-1], errors="coerce")
                        if pd.notna(atr_raw):
                            atr_value = float(atr_raw)
                    if (atr_value is None or not math.isfinite(atr_value)) and {"high", "low", "close"}.issubset(set(dfx.columns)):
                        tail = dfx.tail(24).copy()
                        highs = pd.to_numeric(tail["high"], errors="coerce").to_numpy(dtype=float)
                        lows = pd.to_numeric(tail["low"], errors="coerce").to_numpy(dtype=float)
                        closes = pd.to_numeric(tail["close"], errors="coerce").to_numpy(dtype=float)
                        if len(closes) >= 2:
                            trs = []
                            for i in range(1, len(closes)):
                                tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
                                if math.isfinite(tr):
                                    trs.append(tr)
                            if trs:
                                atr_value = float(np.mean(trs))
        except Exception:
            atr_value = None

        news_count = 0
        try:
            news_df = _read_news_df()
            if not news_df.empty and "published_at" in news_df.columns:
                ts = pd.to_datetime(news_df["published_at"], utc=True, errors="coerce")
                cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=24)
                news_count = int((ts >= cutoff).sum())
            elif not news_df.empty:
                news_count = int(min(len(news_df), 50))
        except Exception:
            news_count = 0

        out_models: Dict[str, Any] = {}
        for model_key in _SITE_MODEL_PROFILE.keys():
            sig = _derive_site_model_signal(model_key, forecasts, horizon_norm, base_price, atr_value, news_count)
            if not sig:
                continue
            if persist and base_ts > 0 and base_price > 0:
                _update_live_model_trade(model_key, sig["status_raw"], base_ts, base_price)
            state = _LIVE_MODEL_TRADE_STATE.get(model_key, {})
            open_trade = state.get("open")
            sig["trade_state"] = {
                "has_open_trade": bool(open_trade),
                "direction": open_trade.get("direction") if isinstance(open_trade, dict) else None,
                "entry_ts": open_trade.get("entry_ts") if isinstance(open_trade, dict) else None,
                "entry_price": open_trade.get("entry_price") if isinstance(open_trade, dict) else None,
            }
            out_models[model_key] = sig

        return {
            "selected_horizon": horizon_norm,
            "base_time": base_ts,
            "base_price": base_price,
            "news_count_24h": news_count,
            "models": out_models,
        }

    @app.get("/model_signals")
    def model_signals(horizon: str = "5H", persist: bool = True):
        cache_key = f"h={horizon}|p={1 if persist else 0}"
        if persist:
            # Persisted states should stay fresh, but still cache briefly to avoid burst recompute.
            cached = _api_cache_get("model_signals", cache_key, 10)
            if cached is not None:
                return cached
        else:
            cached = _api_cache_get("model_signals", cache_key, 5)
            if cached is not None:
                return cached
        payload = _compute_site_model_signals(selected_horizon=horizon, persist=bool(persist))
        _api_cache_set("model_signals", cache_key, payload)
        return payload

    def _start_live_model_loop_once() -> None:
        global _LIVE_MODELS_THREAD_STARTED
        if _LIVE_MODELS_THREAD_STARTED:
            return
        _LIVE_MODELS_THREAD_STARTED = True

        def _loop() -> None:
            while True:
                try:
                    _compute_site_model_signals(selected_horizon="5H", persist=True)
                except Exception:
                    pass
                time.sleep(_LIVE_MODEL_LOOP_SEC)

        threading.Thread(target=_loop, name="tradeforge-live-model-loop", daemon=True).start()

    _start_live_model_loop_once()

    @app.get("/liquidations")
    def liquidations(limit: int = 600, source: Optional[str] = None):
        cache_key = f"l={int(limit)}|s={source or ''}"
        cached = _api_cache_get("liquidations", cache_key, _TTL_LIQUIDATIONS_SEC)
        if cached is not None:
            return cached
        liq_feed = os.getenv("LIQ_FEED_JSONL")
        if source in ("feed", "bybit") and liq_feed:
            feed_points = _load_liq_feed(Path(liq_feed), limit=limit)
            payload = {"points": feed_points, "source": "feed"}
            _api_cache_set("liquidations", cache_key, payload)
            return payload
        if liq_feed:
            feed_points = _load_liq_feed(Path(liq_feed), limit=limit)
            if feed_points:
                payload = {"points": feed_points, "source": "feed"}
                _api_cache_set("liquidations", cache_key, payload)
                return payload
        if not features_path:
            raise HTTPException(status_code=400, detail="features_path not configured on server")
        df = pd.read_parquet(features_path).sort_values("timestamp")
        if "timestamp" not in df.columns:
            raise HTTPException(status_code=400, detail="features parquet missing timestamp")
        if "liq_long" not in df.columns and "liq_short" not in df.columns:
            return {"points": []}
        df = df.tail(max(10, int(limit)))
        df["liq_long"] = df.get("liq_long", 0.0).fillna(0.0)
        df["liq_short"] = df.get("liq_short", 0.0).fillna(0.0)
        points = []
        for _, row in df.iterrows():
            ts = pd.to_datetime(row["timestamp"], utc=True)
            long_v = float(row["liq_long"])
            short_v = float(row["liq_short"])
            if long_v <= 0 and short_v <= 0:
                continue
            points.append(
                {
                    "time": int(ts.timestamp()),
                    "long": max(long_v, 0.0),
                    "short": max(short_v, 0.0),
                }
            )
        payload = {"points": points, "source": "features"}
        _api_cache_set("liquidations", cache_key, payload)
        return payload

    @app.get("/candles")
    def candles(
        start: Optional[int] = None,
        end: Optional[int] = None,
        limit: int = 20000,
        symbol: Optional[str] = None,
        interval: Optional[str] = None,
    ):
        cache_key = (
            f"sym={symbol or ''}|i={interval or ''}|s={start if start is not None else ''}|"
            f"e={end if end is not None else ''}|l={int(limit) if limit is not None else ''}"
        )
        cached = _api_cache_get("candles", cache_key, _TTL_CANDLES_SEC)
        if cached is not None:
            return cached
        # Live mode for dashboard: /candles?symbol=BTCUSDT&interval=15m&limit=...
        if symbol:
            try:
                live_points = _fetch_binance_klines(
                    symbol=symbol,
                    interval=(interval or "15m"),
                    limit=limit,
                    start=start,
                    end=end,
                )
                if live_points:
                    payload = {"candles": live_points, "source": "binance"}
                    _api_cache_set("candles", cache_key, payload)
                    return payload
            except Exception:
                # Silent fallback to parquet source below
                pass

        if not features_path:
            raise HTTPException(status_code=400, detail="features_path not configured on server")
        df = _load_features_df(features_path)
        if start is not None:
            df = df[df["timestamp"] >= pd.to_datetime(int(start), unit="s", utc=True)]
        if end is not None:
            df = df[df["timestamp"] <= pd.to_datetime(int(end), unit="s", utc=True)]
        if limit:
            df = df.tail(int(limit))
        cols = ["timestamp", "open", "high", "low", "close"]
        for c in cols:
            if c not in df.columns:
                raise HTTPException(status_code=400, detail=f"features parquet missing {c}")
        points = []
        for _, row in df.iterrows():
            ts = pd.to_datetime(row["timestamp"], utc=True)
            points.append(
                {
                    "time": int(ts.timestamp()),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                }
            )
        payload = {"candles": points}
        _api_cache_set("candles", cache_key, payload)
        return payload

    @app.get("/trades")
    def trades(file: str = "trades_meta_best_2025.csv", start: Optional[int] = None, end: Optional[int] = None):
        cache_key = f"f={file}|s={start if start is not None else ''}|e={end if end is not None else ''}"
        cached = _api_cache_get("trades", cache_key, _TTL_TRADES_SEC)
        if cached is not None:
            return cached
        reports_dir = Path(__file__).resolve().parent / "reports"
        allowed = set(_list_trade_csvs(reports_dir))
        if file not in allowed:
            raise HTTPException(status_code=400, detail=f"Unknown trades file. Allowed: {sorted(allowed)}")
        rows = _load_trades_csv(reports_dir / file)
        if start is not None:
            rows = [r for r in rows if r["exit_time"] >= int(start)]
        if end is not None:
            rows = [r for r in rows if r["entry_time"] <= int(end)]
        payload = {"trades": rows, "file": file}
        _api_cache_set("trades", cache_key, payload)
        return payload

    @app.get("/trades/files")
    def trades_files():
        cached = _api_cache_get("trades_files", "all", _TTL_TRADES_SEC)
        if cached is not None:
            return cached
        reports_dir = Path(__file__).resolve().parent / "reports"
        files = _list_trade_csvs(reports_dir)
        payload = {"files": files}
        _api_cache_set("trades_files", "all", payload)
        return payload

    @app.get("/results/leverage")
    def leverage_results():
        cached = _api_cache_get("results_leverage", "all", _TTL_RESULTS_SEC)
        if cached is not None:
            return cached
        reports_dir = Path(__file__).resolve().parent / "reports"
        payload = {"thr": 0.59, "results": _load_leverage_results(reports_dir)}
        _api_cache_set("results_leverage", "all", payload)
        return payload

    @app.get("/dashboard/bootstrap")
    def dashboard_bootstrap(
        interval: str = "15m",
        candles_limit: int = 320,
        horizon: str = "5H",
        model: str = "conservative",
        news_limit: int = 30,
    ):
        interval_norm = str(interval or "15m").lower()
        if interval_norm not in {"15m", "1h", "4h", "1d"}:
            interval_norm = "15m"
        horizon_norm = str(horizon or "5H").upper()
        if horizon_norm not in {"5H", "20H", "40H"}:
            horizon_norm = "5H"
        model_norm = str(model or "conservative").strip().lower()
        if model_norm not in _SITE_MODEL_PROFILE:
            model_norm = "conservative"
        news_limit = max(1, min(int(news_limit), 50))
        candles_limit = max(60, min(int(candles_limit), 2000))

        cache_key = (
            f"i={interval_norm}|cl={candles_limit}|h={horizon_norm}|"
            f"m={model_norm}|nl={news_limit}"
        )
        cached = _api_cache_get("dashboard_bootstrap", cache_key, _TTL_DASHBOARD_BOOTSTRAP_SEC)
        if cached is not None:
            return cached

        candles_payload = candles(symbol="BTCUSDT", interval=interval_norm, limit=candles_limit)
        leverage_payload = leverage_results()
        news_payload = news(limit=news_limit)
        model_payload = _compute_site_model_signals(selected_horizon=horizon_norm, persist=False)
        selected_signal = (model_payload.get("models") or {}).get(model_norm)

        payload = {
            "interval": interval_norm,
            "horizon": horizon_norm,
            "model": model_norm,
            "base_time": model_payload.get("base_time"),
            "base_price": model_payload.get("base_price"),
            "candles": candles_payload,
            "signal": selected_signal,
            "model_signals": model_payload,
            "news": news_payload,
            "leverage": leverage_payload,
        }
        _api_cache_set("dashboard_bootstrap", cache_key, payload)
        return payload

    @app.get("/screener/bootstrap")
    def screener_bootstrap(
        limit: int = 180,
        mode: str = "spot",
        direction: str = "all",
        window: str = "24h",
    ):
        mode_norm = str(mode or "spot").lower()
        if mode_norm not in {"spot", "mexc_perp"}:
            mode_norm = "spot"
        direction_norm = str(direction or "all").lower()
        if direction_norm not in {"all", "pump", "dump"}:
            direction_norm = "all"
        window_norm = str(window or "24h").lower()
        if window_norm not in {"24h", "7d", "30d", "90d"}:
            window_norm = "24h"
        max_limit = 120 if mode_norm == "mexc_perp" else 180
        limit_norm = max(10, min(int(limit), max_limit))

        cache_key = (
            f"m={mode_norm}|d={direction_norm}|w={window_norm}|l={limit_norm}"
        )
        cached = _api_cache_get("screener_bootstrap", cache_key, _TTL_SCREENER_BOOTSTRAP_SEC)
        if cached is not None:
            return cached

        try:
            top_payload = (
                _market_top_mexc_perp(limit=limit_norm, direction=direction_norm, window=window_norm)
                if mode_norm == "mexc_perp"
                else _market_top_spot(limit=limit_norm)
            )
            coins = list(top_payload.get("coins") or [])[:limit_norm]
            symbol_list = [
                str(row.get("symbol") or "").strip().upper()
                for row in coins
                if str(row.get("symbol") or "").strip()
            ]
            snapshot_payload = (
                _market_snapshot_mexc_perp(symbol_list, window=window_norm)
                if mode_norm == "mexc_perp"
                else _market_snapshot_spot(symbol_list)
            )
            payload = {
                "feed_mode": mode_norm,
                "direction": direction_norm,
                "window": window_norm,
                "limit": limit_norm,
                "mode": "api",
                "source": top_payload.get("source", "api"),
                "coins": coins,
                "prices": snapshot_payload.get("prices", {}),
                "book": snapshot_payload.get("book", {}),
                "momentums": snapshot_payload.get("momentums", {}),
                "ts": snapshot_payload.get("ts", int(time.time())),
            }
            _api_cache_set("screener_bootstrap", cache_key, payload)
            return payload
        except Exception:
            payload = {
                "feed_mode": mode_norm,
                "direction": direction_norm,
                "window": window_norm,
                "limit": limit_norm,
                "mode": "demo",
                "source": "fallback",
                "coins": [{"symbol": "BTC", "name": "Bitcoin", "rank": 1, "volume24h": 1.0}],
                "prices": {},
                "book": {},
                "momentums": {},
                "ts": int(time.time()),
            }
            _api_cache_set("screener_bootstrap", cache_key, payload)
            return payload

    @app.get("/news")
    def news(limit: int = 50, source: Optional[str] = None):
        cache_key = f"l={int(limit)}|s={source or ''}"
        cached = _api_cache_get("news", cache_key, _TTL_NEWS_SEC)
        if cached is not None:
            return cached
        df = _read_news_df()
        if df.empty:
            payload = {"items": []}
            _api_cache_set("news", cache_key, payload)
            return payload
        df = df.copy()
        if source:
            df = df[df.get("source", "").astype(str).str.lower() == source.lower()]
        if "published_at" in df.columns:
            ts = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
            df["published_ts"] = ts
        df = df.dropna(subset=["published_ts"]).sort_values("published_ts", ascending=False)
        df = df.head(max(1, int(limit)))
        items = []
        for _, row in df.iterrows():
            ts = row.get("published_ts")
            items.append(
                {
                    "id": row.get("id", ""),
                    "source": row.get("source", ""),
                    "title": row.get("title", ""),
                    "url": row.get("url", ""),
                    "published_at": ts.isoformat() if isinstance(ts, pd.Timestamp) else str(ts),
                    "published_ts": int(ts.timestamp()) if isinstance(ts, pd.Timestamp) else None,
                    "sentiment": row.get("sentiment", None),
                    "impact": row.get("impact", None),
                    "category": row.get("category", None),
                    "currency": row.get("currency", None),
                }
            )
        payload = {"items": items}
        _api_cache_set("news", cache_key, payload)
        return payload

    @app.post("/news_refresh")
    def news_refresh(request: Request):
        _require_user(request)
        global _NEWS_REFRESH_LAST_TS
        if not _NEWS_REFRESH_LOCK.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="news refresh already running")
        root_dir = Path(__file__).resolve().parent
        news_dir = Path(os.getenv("NEWS_DIR", "/mnt/data/news"))
        news_dir.mkdir(parents=True, exist_ok=True)
        raw_news = Path(os.getenv("RAW_NEWS", str(news_dir / "news_raw.parquet")))
        dedup_news = Path(os.getenv("DEDUP_NEWS", str(news_dir / "news_dedup.parquet")))
        news_cache = Path(os.getenv("NEWS_CACHE", str(news_dir / "news_sentiment_cache.parquet")))
        news_sent = Path(os.getenv("NEWS_SENT", str(news_dir / "news_sentiment.parquet")))
        news_max_items = os.getenv("NEWS_MAX_ITEMS", "200")
        ledger_model = os.getenv("LEDGER_MODEL_PATH", "ExponentialScience/LedgerBERT-Market-Sentiment")
        xlmr_model = os.getenv("XLMR_MODEL_PATH", "cardiffnlp/twitter-xlm-roberta-base-sentiment")
        device = os.getenv("NEWS_DEVICE", "cpu")
        min_interval = max(0, int(os.getenv("NEWS_REFRESH_MIN_INTERVAL_SEC", "120")))

        try:
            now_ts = time.time()
            if min_interval and (_NEWS_REFRESH_LAST_TS > 0) and ((now_ts - _NEWS_REFRESH_LAST_TS) < min_interval):
                wait_sec = int(min_interval - (now_ts - _NEWS_REFRESH_LAST_TS))
                return {"status": "skipped", "reason": "cooldown", "retry_after_sec": max(1, wait_sec)}

            def run_checked(cmd: List[str], name: str) -> None:
                p = subprocess.run(cmd, check=False, capture_output=True, text=True)
                if p.returncode != 0:
                    tail = (p.stderr or p.stdout or "").strip().splitlines()[-8:]
                    raise RuntimeError(f"{name} failed rc={p.returncode}: {' | '.join(tail)}")

            if raw_news.suffix:
                tmp = raw_news.with_name(raw_news.stem + ".tmp" + raw_news.suffix)
            else:
                tmp = raw_news.with_name(raw_news.name + ".tmp.parquet")
            ingest_cmd = [
                sys.executable,
                str(root_dir / "scripts" / "news_ingest.py"),
                "--out",
                str(tmp),
                "--currency",
                "BTC",
                "--max-items",
                str(news_max_items),
            ]
            run_checked(ingest_cmd, "news_ingest")
            if tmp.exists():
                if raw_news.exists():
                    dedup_cmd = [
                        sys.executable,
                        str(root_dir / "scripts" / "news_dedup.py"),
                        "--inputs",
                        str(raw_news),
                        str(tmp),
                        "--output",
                        str(dedup_news),
                    ]
                    run_checked(dedup_cmd, "news_dedup")
                else:
                    dedup_news.write_bytes(tmp.read_bytes())
                tmp.unlink(missing_ok=True)
                if dedup_news.exists():
                    raw_news.write_bytes(dedup_news.read_bytes())
            if raw_news.exists():
                sent_cmd = [
                    sys.executable,
                    str(root_dir / "scripts" / "news_sentiment_hf.py"),
                    "--input",
                    str(raw_news),
                    "--output",
                    str(news_sent),
                    "--cache",
                    str(news_cache),
                    "--device",
                    str(device),
                    "--batch-size",
                    str(os.getenv("NEWS_BATCH_SIZE", "16")),
                    "--max-length",
                    str(os.getenv("NEWS_MAX_LENGTH", "512")),
                    "--save-every",
                    str(os.getenv("NEWS_SAVE_EVERY", "1000")),
                    "--model-ledger",
                    str(ledger_model),
                    "--model-xlm",
                    str(xlmr_model),
                    "--require-xlmr",
                ]
                run_checked(sent_cmd, "news_sentiment_hf")
            _NEWS_REFRESH_LAST_TS = time.time()
            return {"status": "ok"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            _NEWS_REFRESH_LOCK.release()

    @app.get("/news_agg")
    def news_agg(tf_min: int = 15, window: int = 96):
        cache_key = f"tf={int(tf_min)}|w={int(window)}"
        cached = _api_cache_get("news_agg", cache_key, _TTL_NEWS_AGG_SEC)
        if cached is not None:
            return cached
        df = _read_news_df()
        if df.empty:
            payload = {"points": []}
            _api_cache_set("news_agg", cache_key, payload)
            return payload
        df = df.copy()
        if "published_at" not in df.columns:
            payload = {"points": []}
            _api_cache_set("news_agg", cache_key, payload)
            return payload
        df["published_ts"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
        df = df.dropna(subset=["published_ts"]).sort_values("published_ts")
        df["sentiment_num"] = df.get("sentiment", 0.0).apply(_sentiment_to_num)
        df = df.set_index("published_ts")
        freq = f"{max(1, int(tf_min))}min"
        grouped = df.groupby(pd.Grouper(freq=freq))
        points = []
        for ts, g in grouped:
            if g.empty:
                continue
            points.append(
                {
                    "time": int(ts.timestamp()),
                    "count": int(len(g)),
                    "sentiment": float(g["sentiment_num"].mean()) if "sentiment_num" in g else 0.0,
                }
            )
        if window:
            points = points[-int(window) :]
        payload = {"points": points, "tf_min": int(tf_min)}
        _api_cache_set("news_agg", cache_key, payload)
        return payload

    @app.get("/arbitrage/top")
    def arbitrage_top(limit: int = 500, source: Optional[str] = None):
        try:
            now = time.time()
            cache_age = now - float(_ARBITRAGE_TOP_CACHE.get("ts", 0.0))
            if cache_age < 60 and _ARBITRAGE_TOP_CACHE.get("coins"):
                coins = _ARBITRAGE_TOP_CACHE["coins"]
                return {"coins": coins[: max(1, int(limit))], "source": "cache"}

            prices = _binance_all_prices(max_age_sec=20)
            coins = []
            rank = 1
            for sym in sorted(prices.keys()):
                if not sym.endswith("USDT"):
                    continue
                base = sym[:-4]
                # approximate liquidity ordering: major tickers first via custom priority
                major_bonus = 10_000_000 if base in {"BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "ADA"} else 0
                pseudo_vol = float(prices[sym]) * 1000.0 + major_bonus
                coins.append(
                    {
                        "symbol": base,
                        "name": _coin_name_from_symbol(base),
                        "rank": rank,
                        "volume24h": pseudo_vol,
                    }
                )
                rank += 1
            coins.sort(key=lambda x: float(x.get("volume24h", 0.0)), reverse=True)
            for idx, row in enumerate(coins, start=1):
                row["rank"] = idx
            _ARBITRAGE_TOP_CACHE["ts"] = now
            _ARBITRAGE_TOP_CACHE["coins"] = coins
            return {"coins": coins[: max(1, int(limit))], "source": "binance"}
        except Exception:
            # Hard fallback to keep UI alive
            demo = [{"symbol": "BTC", "name": "Bitcoin", "rank": 1, "volume24h": 1.0}]
            return {"coins": demo[: max(1, int(limit))], "source": "fallback"}

    @app.get("/market/top")
    def market_top(
        limit: int = 500,
        mode: str = "spot",
        direction: str = "all",
        window: str = "24h",
    ):
        try:
            mode = str(mode or "spot").lower()
            direction = str(direction or "all").lower()
            if mode == "mexc_perp":
                return _market_top_mexc_perp(limit=limit, direction=direction, window=window)
            return _market_top_spot(limit=limit)
        except Exception:
            demo = [{"symbol": "BTC", "name": "Bitcoin", "rank": 1, "volume24h": 1.0, "change24h": 0.0}]
            return {"coins": demo[: max(1, int(limit))], "source": "fallback"}

    @app.get("/arbitrage/snapshot")
    def arbitrage_snapshot(symbols: str):
        symbol_list = [
            s.strip().upper()
            for s in str(symbols or "").split(",")
            if s and s.strip()
        ]
        if not symbol_list:
            return {"prices": {}, "book": {}, "momentums": {}, "ts": int(time.time())}
        symbol_list = symbol_list[:1200]
        cache_key = ",".join(symbol_list)
        cache = _ARBITRAGE_SNAPSHOT_CACHE.get(cache_key)
        now = time.time()
        if cache and (now - float(cache.get("ts", 0.0)) < 4):
            return cache["payload"]

        prices_all = _binance_all_prices(max_age_sec=8)
        exchanges = ["BYBIT", "BITGET", "MEXC", "OKX", "BINGX", "BITMART", "COINEX"]
        real_books = {
            "BYBIT": _bybit_spot_books(max_age_sec=8),
            "BITGET": _bitget_spot_books(max_age_sec=8),
            "MEXC": _mexc_spot_books(max_age_sec=8),
        }
        active = {ex for ex, rows in real_books.items() if rows}

        prices: Dict[str, Dict[str, Optional[float]]] = {ex: {} for ex in exchanges}
        book: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {ex: {} for ex in exchanges}
        momentums: Dict[str, Dict[str, Any]] = {}

        taker_fee_pct = 0.06  # ~0.06% per leg, conservative default

        for sym in symbol_list:
            pair = f"{sym}USDT"
            px = float(prices_all.get(pair, 0.0) or 0.0)
            if px <= 0:
                for ex in exchanges:
                    prices[ex][sym] = None
                continue

            best_ask = None
            best_bid = None
            best_ask_ex = None
            best_bid_ex = None

            for ex in exchanges:
                if ex in real_books:
                    row = real_books[ex].get(pair)
                    if not row:
                        prices[ex][sym] = None
                        continue
                    bid = float(row.get("bid", 0.0) or 0.0)
                    ask = float(row.get("ask", 0.0) or 0.0)
                    if bid <= 0 or ask <= 0:
                        prices[ex][sym] = None
                        continue
                    prices[ex][sym] = float(ask)
                    book[ex][sym] = {"ask": float(ask), "bid": float(bid)}
                else:
                    prices[ex][sym] = None
                    continue

                if best_ask is None or ask < best_ask:
                    best_ask = ask
                    best_ask_ex = ex
                if best_bid is None or bid > best_bid:
                    best_bid = bid
                    best_bid_ex = ex

            if best_ask and best_bid and best_ask > 0:
                gross_pct = ((best_bid - best_ask) / best_ask) * 100.0
                net_pct = gross_pct - (2.0 * taker_fee_pct)
                momentums[sym] = {
                    "grossPct": float(gross_pct),
                    "netPct": float(net_pct),
                    "buyEx": best_ask_ex,
                    "sellEx": best_bid_ex,
                    "buyAsk": float(best_ask),
                    "sellBid": float(best_bid),
                    "exCount": len(active),
                }

        payload = {
            "prices": prices,
            "book": book,
            "momentums": momentums,
            "ts": int(now),
        }
        _ARBITRAGE_SNAPSHOT_CACHE[cache_key] = {"ts": now, "payload": payload}
        return payload

    @app.get("/market/snapshot")
    def market_snapshot(symbols: str, mode: str = "spot", window: str = "24h"):
        symbol_list = [
            s.strip().upper()
            for s in str(symbols or "").split(",")
            if s and s.strip()
        ]
        if not symbol_list:
            return {"prices": {}, "book": {}, "momentums": {}, "ts": int(time.time())}
        symbol_list = symbol_list[:1200]
        mode = str(mode or "spot").lower()
        cache_key = f"{mode}|{window}|{','.join(symbol_list)}"
        cache = _ARBITRAGE_SNAPSHOT_CACHE.get(cache_key)
        now = time.time()
        if cache and (now - float(cache.get("ts", 0.0)) < 4):
            return cache["payload"]
        payload = _market_snapshot_mexc_perp(symbol_list, window=window) if mode == "mexc_perp" else _market_snapshot_spot(symbol_list)
        _ARBITRAGE_SNAPSHOT_CACHE[cache_key] = {"ts": now, "payload": payload}
        return payload

    @app.get("/arbitrage/icon/{symbol}")
    def arbitrage_icon(symbol: str):
        s = (symbol or "?").strip().upper()[:8]
        letter = html.escape(s[:1] or "?")
        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32">
<rect width="32" height="32" rx="8" ry="8" fill="#1b1e22"/>
<text x="16" y="21" text-anchor="middle" font-size="14" font-family="Arial" fill="#cfcfcf">{letter}</text>
</svg>"""
        return Response(content=svg, media_type="image/svg+xml")

    @app.get("/market/icon/{symbol}")
    def market_icon(symbol: str):
        return arbitrage_icon(symbol)

    @app.get("/cache/stats")
    def cache_stats(request: Request):
        _require_admin(request)
        with _API_RESP_CACHE_LOCK:
            stats = dict(_API_CACHE_STATS)
            size = len(_API_RESP_CACHE)
            keys_sample = list(_API_RESP_CACHE.keys())[:50]
        return {
            "ok": True,
            "cache": {
                "size": size,
                "max_items": _API_CACHE_MAX_ITEMS,
                "stats": stats,
                "ttl_sec": {
                    "forecast": _TTL_FORECAST_SEC,
                    "forecast_multi": _TTL_FORECAST_MULTI_SEC,
                    "candles": _TTL_CANDLES_SEC,
                    "news": _TTL_NEWS_SEC,
                    "news_agg": _TTL_NEWS_AGG_SEC,
                    "trades": _TTL_TRADES_SEC,
                    "liquidations": _TTL_LIQUIDATIONS_SEC,
                    "results_leverage": _TTL_RESULTS_SEC,
                },
                "keys_sample": keys_sample,
            },
        }

    @app.post("/cache/clear")
    def cache_clear(request: Request):
        _require_admin(request)
        with _API_RESP_CACHE_LOCK:
            before = len(_API_RESP_CACHE)
            _API_RESP_CACHE.clear()
            for k in list(_API_CACHE_STATS.keys()):
                _API_CACHE_STATS[k] = 0
        return {"ok": True, "cleared": before}

    @app.get("/forecast.csv")
    def forecast_csv(interval: str = "1h", hours: int = 24):
        res = forecast(interval=interval, hours=hours)
        rows = ["timestamp,pred_close"]
        for p in res["points"]:
            ts = pd.to_datetime(p["time"], unit="s", utc=True)
            rows.append(f"{ts.isoformat()},{p['value']}")
        return "\n".join(rows)

    return app


def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Serve Keras model via FastAPI.")
    parser.add_argument(
        "--model-h20",
        default="model_15m_itransformer_price_h20.keras",
        help="Path(s) to h20 model(s). Comma-separated for ensemble.",
    )
    parser.add_argument(
        "--model-multi",
        default=None,
        help="Path to legacy multi-horizon model (h80/h160).",
    )
    parser.add_argument(
        "--model-h80",
        default="model_15m_itransformer_price_h80.keras",
        help="Path(s) to h80 model(s). Comma-separated for ensemble.",
    )
    parser.add_argument(
        "--model-h160",
        default="model_15m_itransformer_price_h160_e24.keras",
        help="Path(s) to h160 model(s). Comma-separated for ensemble.",
    )
    parser.add_argument(
        "--model-h320",
        default="model_15m_itransformer_price_h320.keras",
        help="Path(s) to h320 model(s). Comma-separated for ensemble.",
    )
    parser.add_argument(
        "--model-h640",
        default="model_15m_itransformer_price_h640.keras",
        help="Path(s) to h640 model(s). Comma-separated for ensemble.",
    )
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length used during training")
    parser.add_argument(
        "--stats-h20",
        default="norm_stats_15m_itransformer_price_h20.npz",
        help="Path(s) to stats for h20 model(s). Comma-separated for ensemble.",
    )
    parser.add_argument(
        "--stats-multi",
        default=None,
        help="Path to stats for legacy multi-horizon model",
    )
    parser.add_argument(
        "--stats-h80",
        default="norm_stats_15m_itransformer_price_h80.npz",
        help="Path(s) to stats for h80 model(s). Comma-separated for ensemble.",
    )
    parser.add_argument(
        "--stats-h160",
        default="norm_stats_15m_itransformer_price_h160_e24.npz",
        help="Path(s) to stats for h160 model(s). Comma-separated for ensemble.",
    )
    parser.add_argument(
        "--stats-h320",
        default="norm_stats_15m_itransformer_price_h320.npz",
        help="Path(s) to stats for h320 model(s). Comma-separated for ensemble.",
    )
    parser.add_argument(
        "--stats-h640",
        default="norm_stats_15m_itransformer_price_h640.npz",
        help="Path(s) to stats for h640 model(s). Comma-separated for ensemble.",
    )
    parser.add_argument("--features", help="Features parquet for forecast endpoint")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    args = parser.parse_args()

    app = create_app(
        args.model_h20,
        seq_len=args.seq_len,
        stats_h20_path=args.stats_h20,
        model_multi_path=args.model_multi,
        stats_multi_path=args.stats_multi,
        features_path=args.features,
        model_h80_path=args.model_h80,
        stats_h80_path=args.stats_h80,
        model_h160_path=args.model_h160,
        stats_h160_path=args.stats_h160,
        model_h320_path=args.model_h320,
        stats_h320_path=args.stats_h320,
        model_h640_path=args.model_h640,
        stats_h640_path=args.stats_h640,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
def _close_live_trade(entry_ts: int, exit_ts: int, direction: str, entry_price: float, exit_price: float) -> None:
    if not _LIVE_TRADES_PATH.exists():
        return
    df = pd.read_csv(_LIVE_TRADES_PATH)
    if df.empty:
        return
    mask = df["exit_ts"].isna() | (df["exit_ts"].astype(str).str.len() == 0)
    if not mask.any():
        return
    idx = df[mask].index[-1]
    df.at[idx, "exit_ts"] = pd.to_datetime(exit_ts, unit="s", utc=True).isoformat()
    df.at[idx, "exit_price"] = float(exit_price)
    df.to_csv(_LIVE_TRADES_PATH, index=False)
