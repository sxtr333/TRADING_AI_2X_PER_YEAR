"""
Bybit WebSocket realtime loop that feeds the Keras predictor.
- Subscribes to public linear kline channel (1m) for BTCUSDT
- Maintains rolling indicators (Bollinger, ATR, realized vol) on the fly
- Builds feature rows matching trading_keras.py and prints predictions as bars close

Requires: websockets (`pip install websockets`) and TensorFlow.
This is a minimal scaffold; extend with real OI/funding/CVD/orderbook data if needed.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf

from trading_keras_core import OnlineNormalizer, RealTimePredictor, default_feature_list

WS_URL = "wss://stream.bybit.com/v5/public/linear"


class RollingIndicators:
    """Maintains rolling windows for Bollinger, ATR, and realized vol."""

    def __init__(self, bb_window: int = 20, bb_std: float = 2.0, atr_window: int = 14, rv_window: int = 30):
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.atr_window = atr_window
        self.rv_window = rv_window
        self.closes: Deque[float] = deque(maxlen=max(bb_window, rv_window))
        self.highs: Deque[float] = deque(maxlen=atr_window + 1)
        self.lows: Deque[float] = deque(maxlen=atr_window + 1)
        self.prev_close: Optional[float] = None
        self.trs: Deque[float] = deque(maxlen=atr_window)

    def update(self, o: float, h: float, l: float, c: float) -> Dict[str, float]:
        self.closes.append(c)
        self.highs.append(h)
        self.lows.append(l)
        if self.prev_close is not None:
            tr = max(h - l, abs(h - self.prev_close), abs(l - self.prev_close))
            self.trs.append(tr)
        self.prev_close = c

        features = {
            "bollinger_upper": 0.0,
            "bollinger_lower": 0.0,
            "bollinger_bandwidth": 0.0,
            "atr": 0.0,
            "rv": 0.0,
            "iv": 0.0,
            "iv_rv_spread": 0.0,
        }

        if len(self.closes) >= self.bb_window:
            arr = np.array(self.closes, dtype=np.float64)
            ma = arr[-self.bb_window :].mean()
            std = arr[-self.bb_window :].std(ddof=0)
            upper = ma + self.bb_std * std
            lower = ma - self.bb_std * std
            features["bollinger_upper"] = upper
            features["bollinger_lower"] = lower
            features["bollinger_bandwidth"] = (upper - lower) / ma if ma != 0 else 0.0

        if len(self.trs) == self.atr_window:
            features["atr"] = float(np.mean(self.trs))

        if len(self.closes) >= self.rv_window:
            closes = np.array(self.closes, dtype=np.float64)
            log_ret = np.diff(np.log(closes[-self.rv_window :]))
            rv = log_ret.std(ddof=0) * math.sqrt(len(log_ret))
            features["rv"] = float(rv)
            features["iv"] = float(rv)  # placeholder
            features["iv_rv_spread"] = 0.0

        return features


class MarketState:
    """
    Holds latest auxiliary signals (OI, funding, liquidations delta, orderbook imbalance).
    Values are updated asynchronously and applied to feature rows.
    """

    def __init__(self):
        self.open_interest: float = 0.0
        self.funding_rate: float = 0.0
        self.liq_long: float = 0.0
        self.liq_short: float = 0.0
        self.orderbook_imb: Dict[str, float] = {"ob_imb_01": 0.0, "ob_imb_025": 0.0, "ob_imb_05": 0.0, "ob_imb_1": 0.0}
        self.tick_buy_volume: float = 0.0
        self.tick_sell_volume: float = 0.0
        self.delta: float = 0.0
        self.buy_sell_ratio: float = 0.0

    def update_ticker(self, data: Dict) -> None:
        # Bybit ticker may include openInterest and fundingRate
        self.open_interest = float(data.get("openInterest", self.open_interest))
        self.funding_rate = float(data.get("fundingRate", self.funding_rate))

    def update_liquidation(self, data: Dict) -> None:
        side = data.get("side", "").lower()
        qty = float(data.get("size", 0.0))
        if side == "buy":  # shorts liquidated
            self.liq_short += qty
        elif side == "sell":  # longs liquidated
            self.liq_long += qty

    def update_orderbook(self, data: Dict) -> None:
        """
        Expect 'b' and 'a' arrays: [[price, size], ...].
        Compute imbalance within percentage bands around mid.
        """
        bids = data.get("b", [])
        asks = data.get("a", [])
        if not bids or not asks:
            return
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        mid = (best_bid + best_ask) / 2.0
        levels = {"ob_imb_01": 0.001, "ob_imb_025": 0.0025, "ob_imb_05": 0.005, "ob_imb_1": 0.01}
        for key, pct in levels.items():
            bid_vol = sum(float(sz) for p, sz in bids if float(p) >= mid * (1 - pct))
            ask_vol = sum(float(sz) for p, sz in asks if float(p) <= mid * (1 + pct))
            total = bid_vol + ask_vol
            self.orderbook_imb[key] = (bid_vol - ask_vol) / total if total > 0 else 0.0

    def update_trades(self, trades: List[Dict]) -> None:
        """
        Accumulate aggressive buy/sell volume and delta for the bar window.
        """
        buy = 0.0
        sell = 0.0
        for t in trades:
            side = t.get("S") or t.get("side", "").upper()
            size = float(t.get("v") or t.get("size") or 0.0)
            if side == "Buy" or side == "BUY":
                buy += size
            elif side == "Sell" or side == "SELL":
                sell += size
        total = buy + sell
        self.tick_buy_volume = buy
        self.tick_sell_volume = sell
        self.delta = buy - sell
        self.buy_sell_ratio = buy / total if total > 0 else 0.0

    def reset_per_bar(self):
        """
        Reset per-bar accumulators (liquidations and trade volumes) after each bar close.
        """
        self.liq_long = 0.0
        self.liq_short = 0.0
        self.tick_buy_volume = 0.0
        self.tick_sell_volume = 0.0
        self.delta = 0.0
        self.buy_sell_ratio = 0.0


async def subscribe_kline(symbol: str = "BTCUSDT", interval: str = "1") -> asyncio.Queue:
    """
    Subscribes to Bybit public kline stream.
    Returns a queue with bar dicts (as delivered by Bybit).
    """
    q: asyncio.Queue = asyncio.Queue()

    async def reader():
        async for ws in _connect():
            try:
                sub = {
                    "op": "subscribe",
                    "args": [f"kline.{interval}.{symbol}"],
                }
                await ws.send(json.dumps(sub))
                async for msg in ws:
                    data = json.loads(msg)
                    if data.get("topic", "").startswith("kline"):
                        for item in data.get("data", []):
                            await q.put(item)
            except Exception as e:
                print(f"WebSocket error: {e}, reconnecting...")
                await asyncio.sleep(1.0)
                continue

    asyncio.create_task(reader())
    return q


async def _connect():
    import websockets  # type: ignore

    while True:
        try:
            ws = await websockets.connect(WS_URL, ping_interval=20, ping_timeout=20)
            print("Connected to Bybit WS")
            yield ws
        except Exception as e:
            print(f"WS connect failed: {e}, retrying...")
            await asyncio.sleep(1.0)


async def subscribe_ticker(state: MarketState, symbol: str = "BTCUSDT"):
    """
    Subscribes to ticker stream to update funding/open interest.
    """
    async for ws in _connect():
        try:
            sub = {"op": "subscribe", "args": [f"tickers.{symbol}"]}
            await ws.send(json.dumps(sub))
            async for msg in ws:
                data = json.loads(msg)
                if data.get("topic", "").startswith("tickers"):
                    for item in data.get("data", []):
                        state.update_ticker(item)
        except Exception as e:
            print(f"Ticker stream error: {e}, reconnecting...")
            await asyncio.sleep(1.0)
            continue


async def subscribe_liquidation(state: MarketState, symbol: str = "BTCUSDT"):
    """
    Subscribes to liquidation stream; accumulates per bar.
    """
    async for ws in _connect():
        try:
            sub = {"op": "subscribe", "args": [f"liquidation.{symbol}"]}
            await ws.send(json.dumps(sub))
            async for msg in ws:
                data = json.loads(msg)
                if data.get("topic", "").startswith("liquidation"):
                    for item in data.get("data", []):
                        state.update_liquidation(item)
        except Exception as e:
            print(f"Liquidation stream error: {e}, reconnecting...")
            await asyncio.sleep(1.0)
            continue


async def subscribe_orderbook(state: MarketState, symbol: str = "BTCUSDT", depth: int = 50):
    """
    Subscribes to orderbook snapshot/updates to compute imbalance.
    """
    async for ws in _connect():
        try:
            sub = {"op": "subscribe", "args": [f"orderbook.{depth}.{symbol}"]}
            await ws.send(json.dumps(sub))
            async for msg in ws:
                data = json.loads(msg)
                if data.get("topic", "").startswith("orderbook"):
                    if "data" in data:
                        state.update_orderbook(data["data"])
        except Exception as e:
            print(f"Orderbook stream error: {e}, reconnecting...")
            await asyncio.sleep(1.0)
            continue


def build_row(bar: Dict, indicators: RollingIndicators, state: MarketState) -> Dict[str, float]:
    """
    Build feature row from Bybit kline bar.
    bar fields: start, end, interval, open, close, high, low, volume, turnover, confirm...
    """
    o = float(bar["open"])
    h = float(bar["high"])
    l = float(bar["low"])
    c = float(bar["close"])
    v = float(bar["volume"])
    to = float(bar.get("turnover", 0.0))
    vwap = to / v if v != 0 else c

    feats = {
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "volume": v,
        "vwap": vwap,
        # state-driven features
        "open_interest": state.open_interest,
        "funding_rate": state.funding_rate,
        "cvd": state.delta,  # reuse delta as simple cvd proxy
        "liq_long": state.liq_long,
        "liq_short": state.liq_short,
        "basis": 0.0,  # placeholder (needs spot vs perp mid)
        "tick_buy_volume": state.tick_buy_volume,
        "tick_sell_volume": state.tick_sell_volume,
        "delta": state.delta,
        "buy_sell_ratio": state.buy_sell_ratio,
        "ob_imb_01": state.orderbook_imb["ob_imb_01"],
        "ob_imb_025": state.orderbook_imb["ob_imb_025"],
        "ob_imb_05": state.orderbook_imb["ob_imb_05"],
        "ob_imb_1": state.orderbook_imb["ob_imb_1"],
        # time features will be filled below
        "dow": 0.0,
        "hour": 0.0,
        "minute": 0.0,
    }

    ind_feats = indicators.update(o, h, l, c)
    feats.update(ind_feats)

    # time features from start time (ms)
    start_ms = int(bar["start"])
    ts = np.datetime64(start_ms, "ms").astype("datetime64[ns]")
    dt = pd.Timestamp(ts, tz="UTC")
    feats["dow"] = dt.dayofweek / 6.0
    feats["hour"] = dt.hour / 23.0
    feats["minute"] = dt.minute / 59.0

    return feats


async def run_realtime(model_path: str, seq_len: int = 256, symbol: str = "BTCUSDT", log_file: Optional[str] = None, drift_threshold: float = 3.0, ema_alpha: float = 0.2, mc_samples: int = 1):
    feature_cols = default_feature_list()
    model = tf.keras.models.load_model(model_path)
    norm = OnlineNormalizer(n_features=len(feature_cols))
    predictor = RealTimePredictor(model, norm, feature_cols, seq_len)
    indicators = RollingIndicators()
    state = MarketState()
    prev_close: Optional[float] = None
    ret_window: Deque[float] = deque(maxlen=500)
    log_records: List[Dict[str, float]] = []
    ema_pred: Optional[float] = None

    q = await subscribe_kline(symbol=symbol, interval="1")
    # spin background listeners
    asyncio.create_task(subscribe_ticker(state, symbol=symbol))
    asyncio.create_task(subscribe_liquidation(state, symbol=symbol))
    asyncio.create_task(subscribe_orderbook(state, symbol=symbol, depth=50))

    print("Streaming... waiting for bars to confirm.")

    while True:
        bar = await q.get()
        # Bybit sends interim bars; confirm==True means closed bar
        if not bar.get("confirm", False):
            continue
        row = build_row(bar, indicators, state)
        pred = predictor.step(row)
        if pred is not None and mc_samples > 1:
            # MC dropout averaging
            preds = []
            x = np.stack(list(predictor.buffer), axis=0)[np.newaxis, ...]
            for _ in range(mc_samples):
                preds.append(model(x, training=True).numpy().squeeze().item())
            pred = float(np.mean(preds))
        if pred is not None:
            ts = int(bar["start"])
            close_price = row["close"]
            # drift detection on realized return
            if prev_close:
                ret = (close_price / prev_close) - 1.0
                ret_window.append(ret)
                if len(ret_window) > 20:
                    mu = np.mean(ret_window)
                    sigma = np.std(ret_window) + 1e-9
                    z = abs(ret - mu) / sigma
                    if z > drift_threshold:
                        print(f"Drift alert z={z:.2f} ret={ret:.5f}")
            prev_close = close_price

            if ema_pred is None:
                ema_pred = pred
            else:
                ema_pred = ema_alpha * pred + (1 - ema_alpha) * ema_pred

            print(f"{pd.to_datetime(ts, unit='ms', utc=True)} pred={pred:.6f} ema_pred={ema_pred:.6f} close={close_price}")
            if log_file:
                log_records.append({"timestamp": pd.to_datetime(ts, unit="ms", utc=True), "prediction": pred, "ema_prediction": ema_pred, "close": close_price})
                if len(log_records) >= 100:
                    pd.DataFrame(log_records).to_csv(log_file, mode="a", header=not Path(log_file).exists(), index=False)
                    log_records.clear()
        state.reset_per_bar()


if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="Realtime Bybit predictor using Keras model.")
    parser.add_argument("--model", required=True, help="Path to trained Keras model (.keras)")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length used in training")
    parser.add_argument("--symbol", default="BTCUSDT", help="Symbol (e.g., BTCUSDT)")
    parser.add_argument("--log-file", help="Optional CSV to log timestamp,prediction,close")
    parser.add_argument("--drift-threshold", type=float, default=3.0, help="Std dev threshold for return drift alert")
    parser.add_argument("--ema-alpha", type=float, default=0.2, help="EMA smoothing factor for predictions")
    parser.add_argument("--mc-samples", type=int, default=1, help="MC dropout samples for uncertainty; >1 enables stochastic forward")
    args = parser.parse_args()

    asyncio.run(
        run_realtime(
            args.model,
            seq_len=args.seq_len,
            symbol=args.symbol,
            log_file=args.log_file,
            drift_threshold=args.drift_threshold,
            ema_alpha=args.ema_alpha,
            mc_samples=args.mc_samples,
        )
    )
