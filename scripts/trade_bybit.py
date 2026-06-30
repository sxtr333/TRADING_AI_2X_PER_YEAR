#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import time
import math
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

try:
    import ccxt
except Exception as exc:  # pragma: no cover
    raise SystemExit("ccxt not installed. Run: pip install ccxt") from exc

ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "state"
REPORTS_DIR = ROOT / "reports"
CONFIG_PATH = ROOT / "configs" / "bybit_trade_accounts.json"

from scripts.backtest_trade_combo_meta import (
    _load_stats,
    _load_model,
    _pred_meta_probs,
    _unpack_obj,
)


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _now_ts() -> int:
    return int(pd.Timestamp.utcnow().timestamp())


def _read_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_state(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    tmp.replace(path)


def _today_key() -> str:
    return pd.Timestamp.utcnow().strftime("%Y-%m-%d")


def _update_daily_pnl(state: Dict[str, Any], pnl: float) -> None:
    key = _today_key()
    daily = state.get("daily_pnl") or {}
    cur = float(daily.get(key, 0.0))
    daily[key] = cur + float(pnl)
    state["daily_pnl"] = daily


def _daily_loss_exceeded(state: Dict[str, Any], equity: float, max_daily_loss_pct: float) -> bool:
    key = _today_key()
    daily = state.get("daily_pnl") or {}
    pnl = float(daily.get(key, 0.0))
    return pnl <= -(equity * max_daily_loss_pct / 100.0)


def _ensure_reports_header(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "timestamp,account,symbol,action,side,qty,price,meta_prob,horizon,reason\n",
        encoding="utf-8",
    )


def _append_trade_log(path: Path, row: Dict[str, Any]) -> None:
    _ensure_reports_header(path)
    line = (
        f"{row['timestamp']},{row['account']},{row['symbol']},{row['action']},{row['side']},"
        f"{row['qty']},{row['price']},{row['meta_prob']},{row['horizon']},{row['reason']}\n"
    )
    with path.open("a", encoding="utf-8") as f:
        f.write(line)


def _init_exchange(key: str, secret: str) -> "ccxt.bybit":
    ex = ccxt.bybit({
        "apiKey": key,
        "secret": secret,
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })
    ex.load_markets()
    return ex


def _set_position_mode(ex, symbol: str, mode: str) -> None:
    # mode: oneway or hedge
    if mode != "oneway":
        return
    try:
        ex.set_position_mode(False, symbol)
    except Exception:
        pass


def _set_leverage(ex, symbol: str, leverage: int) -> None:
    try:
        ex.set_leverage(leverage, symbol)
    except Exception:
        pass


def _fetch_leverage(ex, symbol: str) -> Optional[float]:
    try:
        positions = ex.fetch_positions([symbol])
    except Exception:
        return None
    for p in positions:
        if p.get("symbol") == symbol:
            lev = p.get("leverage") or p.get("leverageRate")
            if lev is not None:
                try:
                    return float(lev)
                except Exception:
                    return None
    return None


def _fetch_price(ex, symbol: str) -> float:
    ticker = ex.fetch_ticker(symbol)
    return float(ticker["last"]) if ticker and ticker.get("last") else float(ticker["close"])


def _fetch_position(ex, symbol: str) -> Optional[Dict[str, Any]]:
    try:
        positions = ex.fetch_positions([symbol])
    except Exception:
        positions = []
    for p in positions:
        if p.get("symbol") == symbol and abs(p.get("contracts") or p.get("positionAmt") or 0) > 0:
            return p
    return None


def _close_position(ex, symbol: str) -> None:
    pos = _fetch_position(ex, symbol)
    if not pos:
        return
    side = "sell" if (pos.get("side") == "long" or (pos.get("contracts") or 0) > 0) else "buy"
    amount = abs(pos.get("contracts") or pos.get("positionAmt") or 0)
    if amount:
        ex.create_order(symbol, "market", side, amount, None, {"reduce_only": True})


def _market_order(ex, symbol: str, side: str, qty: float) -> float:
    order = ex.create_order(symbol, "market", side, qty, None, {})
    if order and order.get("average"):
        return float(order["average"])
    return _fetch_price(ex, symbol)


def _calc_qty(ex, symbol: str, equity: float, trade_frac: float, leverage: int, price: float) -> float:
    notional = equity * trade_frac * leverage
    qty = notional / price if price else 0.0
    # precision
    market = ex.market(symbol)
    step = market.get("precision", {}).get("amount")
    if step is not None:
        return float(ex.amount_to_precision(symbol, qty))
    return float(qty)


def _load_meta_models(meta_dir: Path):
    meta_specs = {
        "h20_long": (meta_dir / "meta_h20_long.keras", meta_dir / "meta_h20_long_stats.npz"),
        "h20_short": (meta_dir / "meta_h20_short.keras", meta_dir / "meta_h20_short_stats.npz"),
        "h80_short_v2": (meta_dir / "meta_h80_short_v2.keras", meta_dir / "meta_h80_short_v2_stats.npz"),
        "h160_long_v2": (meta_dir / "meta_h160_long_v2.keras", meta_dir / "meta_h160_long_v2_stats.npz"),
    }
    loaded = {}
    for name, (m, s) in meta_specs.items():
        model = _load_model(str(m))
        stats = _load_stats(str(s))
        loaded[name] = {"model": model, "stats": stats}
    return loaded


def _load_meta_features(meta_path: Path, feature_names: list) -> pd.DataFrame:
    cols = list(dict.fromkeys(["timestamp"] + feature_names))
    df = pd.read_parquet(meta_path, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _compute_meta_probs(meta_df: pd.DataFrame, meta_models: Dict[str, Any]) -> Dict[str, np.ndarray]:
    probs = {}
    for name, obj in meta_models.items():
        stats = obj["stats"]
        model = obj["model"]
        probs[name] = _pred_meta_probs(model, stats, meta_df, batch_size=512)
    return probs


def _pick_signal(meta_df: pd.DataFrame, meta_probs: Dict[str, np.ndarray], meta_thr: float) -> Dict[str, Any]:
    last_i = len(meta_df) - 1
    if last_i < 0:
        return {"side": "FLAT"}

    candidates = []
    for name, prob_arr in meta_probs.items():
        prob = float(prob_arr[last_i]) if not math.isnan(float(prob_arr[last_i])) else 0.0
        signal_col = f"meta_signal_{name}"
        if signal_col not in meta_df.columns:
            continue
        signal = float(meta_df[signal_col].iloc[last_i]) > 0.0
        if not signal or prob < meta_thr:
            continue
        if name in ("h20_long", "h160_long_v2"):
            side = "LONG"
        else:
            side = "SHORT"
        horizon = 20 if "h20" in name else 80 if "h80" in name else 160
        candidates.append({"name": name, "side": side, "prob": prob, "horizon": horizon})

    if not candidates:
        return {"side": "FLAT"}

    long_any = [c for c in candidates if c["side"] == "LONG"]
    short_any = [c for c in candidates if c["side"] == "SHORT"]
    if long_any and short_any:
        return {"side": "FLAT"}

    picked = max(candidates, key=lambda x: x["prob"])
    return picked


def _load_atr(meta_df: pd.DataFrame) -> float:
    if "atr" in meta_df.columns:
        val = float(meta_df["atr"].iloc[-1])
        return max(val, 0.0)
    return 0.0


def _equity(ex) -> float:
    bal = ex.fetch_balance()
    usdt = bal.get("USDT") or bal.get("USDC") or bal.get("total", {})
    if isinstance(usdt, dict):
        return float(usdt.get("total") or usdt.get("free") or 0.0)
    return float(usdt or 0.0)


def run_account(
    cfg: Dict[str, Any],
    account: Dict[str, Any],
    meta_models: Dict[str, Any],
    meta_feature_names: list,
    meta_df_cache: Dict[str, Any],
    dry_run: bool,
):
    key = os.getenv(account["env_key"], "").strip()
    secret = os.getenv(account["env_secret"], "").strip()
    if not key or not secret:
        print(f"[skip] missing keys for {account['name']}")
        return

    ex = _init_exchange(key, secret)
    symbol = cfg["symbol"]
    if not dry_run:
        _set_position_mode(ex, symbol, cfg.get("position_mode", "oneway"))
        _set_leverage(ex, symbol, int(account["leverage"]))

        max_lev = float(cfg.get("risk", {}).get("max_leverage", account["leverage"]))
        cur_lev = _fetch_leverage(ex, symbol)
        if cur_lev is not None and cur_lev > max_lev + 1e-6:
            print(f"[halt] leverage too high on {account['name']}: {cur_lev} > {max_lev}")
            return

    state_path = STATE_DIR / f"bybit_{account['name']}.json"
    state = _read_state(state_path)

    meta_path = Path(cfg["model"]["meta_features"]).expanduser()
    meta_mtime = meta_path.stat().st_mtime if meta_path.exists() else 0
    if meta_df_cache.get("mtime") != meta_mtime:
        meta_df_cache["df"] = _load_meta_features(meta_path, meta_feature_names)
        meta_df_cache["probs"] = _compute_meta_probs(meta_df_cache["df"], meta_models)
        meta_df_cache["mtime"] = meta_mtime

    meta_df = meta_df_cache["df"]
    probs = meta_df_cache["probs"]

    # stale guard: skip if data older than 2 bars
    last_ts = pd.to_datetime(meta_df["timestamp"].iloc[-1], utc=True)
    if (pd.Timestamp.utcnow() - last_ts) > pd.Timedelta(minutes=30):
        print(f"[{account['name']}] stale meta data: {last_ts}")
        return

    signal = _pick_signal(meta_df, probs, float(cfg["model"]["meta_prob_thr"]))
    side = signal.get("side", "FLAT")

    current_price = _fetch_price(ex, symbol)
    if dry_run:
        open_pos = state.get("open_position")
    else:
        open_pos = _fetch_position(ex, symbol)

    # handle open position exit
    if open_pos:
        if dry_run:
            entry_price = float(open_pos.get("entry_price") or state.get("entry_price") or current_price)
            side_name = open_pos.get("side")
            contracts = open_pos.get("qty")
        else:
            entry_price = float(open_pos.get("entryPrice") or open_pos.get("averagePrice") or current_price)
            side_name = open_pos.get("side")
            contracts = open_pos.get("contracts")
        entry_ts = int(state.get("entry_ts") or 0)
        horizon = int(state.get("horizon") or 20)
        stop_price = float(state.get("stop_price") or 0.0)
        if stop_price and ((side_name == "long" and current_price <= stop_price) or (side_name == "short" and current_price >= stop_price)):
            if not dry_run:
                _close_position(ex, symbol)
            _append_trade_log(REPORTS_DIR / f"live_trades_{account['name']}.csv", {
                "timestamp": _now_ts(),
                "account": account["name"],
                "symbol": symbol,
                "action": "close",
                "side": side_name,
                "qty": contracts,
                "price": current_price,
                "meta_prob": state.get("meta_prob"),
                "horizon": horizon,
                "reason": "stop_loss",
            })
            pnl = (current_price - entry_price) * (1 if side_name == "long" else -1) * float(contracts or 0.0)
            _update_daily_pnl(state, pnl)
            state.clear()
            _write_state(state_path, state)
            return

        if entry_ts:
            elapsed = _now_ts() - entry_ts
            if elapsed >= horizon * 15 * 60:
                if not dry_run:
                    _close_position(ex, symbol)
                _append_trade_log(REPORTS_DIR / f"live_trades_{account['name']}.csv", {
                    "timestamp": _now_ts(),
                    "account": account["name"],
                    "symbol": symbol,
                    "action": "close",
                    "side": side_name,
                    "qty": contracts,
                    "price": current_price,
                    "meta_prob": state.get("meta_prob"),
                    "horizon": horizon,
                    "reason": "time_exit",
                })
                pnl = (current_price - entry_price) * (1 if side_name == "long" else -1) * float(contracts or 0.0)
                _update_daily_pnl(state, pnl)
                state.clear()
                _write_state(state_path, state)
                return

        # if signal flips, close
        if side in ("LONG", "SHORT") and ((side_name == "long" and side == "SHORT") or (side_name == "short" and side == "LONG")):
            if not dry_run:
                _close_position(ex, symbol)
            _append_trade_log(REPORTS_DIR / f"live_trades_{account['name']}.csv", {
                "timestamp": _now_ts(),
                "account": account["name"],
                "symbol": symbol,
                "action": "close",
                "side": side_name,
                "qty": contracts,
                "price": current_price,
                "meta_prob": state.get("meta_prob"),
                "horizon": horizon,
                "reason": "flip",
            })
            pnl = (current_price - entry_price) * (1 if side_name == "long" else -1) * float(contracts or 0.0)
            _update_daily_pnl(state, pnl)
            state.clear()
            _write_state(state_path, state)
            return

        return

    # no open position: open if signal
    if side == "FLAT":
        return

    equity = _equity(ex)
    if equity <= 0:
        print(f"[{account['name']}] equity=0")
        return

    max_daily_loss = float(cfg.get("risk", {}).get("max_daily_loss_pct", 0.0))
    if max_daily_loss > 0 and _daily_loss_exceeded(state, equity, max_daily_loss):
        print(f"[halt] daily loss exceeded for {account['name']}")
        _write_state(state_path, state)
        return

    qty = _calc_qty(ex, symbol, equity, float(account["trade_frac"]), int(account["leverage"]), current_price)
    if qty <= 0:
        return

    order_side = "buy" if side == "LONG" else "sell"
    if dry_run:
        fill_price = current_price
    else:
        fill_price = _market_order(ex, symbol, order_side, qty)

    atr = _load_atr(meta_df)
    stop_mult = float(cfg["risk"].get("stop_atr_mult", 2.5))
    if atr > 0:
        stop_price = fill_price - stop_mult * atr if side == "LONG" else fill_price + stop_mult * atr
    else:
        stop_price = 0.0

    _append_trade_log(REPORTS_DIR / f"live_trades_{account['name']}.csv", {
        "timestamp": _now_ts(),
        "account": account["name"],
        "symbol": symbol,
        "action": "open",
        "side": side.lower(),
        "qty": qty,
        "price": fill_price,
        "meta_prob": signal.get("prob"),
        "horizon": signal.get("horizon"),
        "reason": signal.get("name"),
    })

    state.update({
        "entry_ts": _now_ts(),
        "entry_price": fill_price,
        "horizon": signal.get("horizon"),
        "meta_prob": signal.get("prob"),
        "stop_price": stop_price,
    })
    if dry_run:
        state["open_position"] = {
            "side": side.lower(),
            "qty": qty,
            "entry_price": fill_price,
        }
    _write_state(state_path, state)


def main():
    ap = argparse.ArgumentParser(description="Bybit live trader (two accounts).")
    ap.add_argument("--dry-run", action="store_true", help="Simulate trades without sending orders.")
    args = ap.parse_args()
    cfg = _load_config(CONFIG_PATH)
    meta_dir = Path(cfg["model"]["meta_model_dir"]).expanduser()
    meta_models = _load_meta_models(meta_dir)

    # feature names for meta dataset
    any_stats = next(iter(meta_models.values()))["stats"]
    meta_feature_names = list(_unpack_obj(any_stats["feature_names"]))

    cache = {"mtime": None, "df": None, "probs": None}

    while True:
        for account in cfg.get("accounts", []):
            run_account(cfg, account, meta_models, meta_feature_names, cache, args.dry_run)
        time.sleep(int(cfg.get("poll_seconds", 900)))


if __name__ == "__main__":
    main()
