"""
Multi-timeframe feature builder.
- Loads base timeframe OHLCV (e.g., 1m) plus higher TF files (5m/15m/1h/4h/1d)
- Computes per-TF indicators (EMA, Bollinger, ATR, RV) and merges into a single feature set
- Adds time features and target on the base timeframe
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from trading_keras_core import add_time_features, build_target, default_feature_list


def load_tf(path: Path, suffix: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.rename(columns={c: f"{c}_{suffix}" for c in ["open", "high", "low", "close", "volume", "turnover", "vwap"] if c in df.columns})
    return df


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def add_tf_indicators(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    close = df[f"close_{suffix}"]
    high = df[f"high_{suffix}"]
    low = df[f"low_{suffix}"]
    vol = df[f"volume_{suffix}"]

    df[f"ema_fast_{suffix}"] = ema(close, span=10)
    df[f"ema_slow_{suffix}"] = ema(close, span=30)
    df[f"atr_{suffix}"] = (pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)).rolling(14).mean()
    ma = close.rolling(20).mean()
    std = close.rolling(20).std()
    df[f"bb_upper_{suffix}"] = ma + 2 * std
    df[f"bb_lower_{suffix}"] = ma - 2 * std
    df[f"bb_bandwidth_{suffix}"] = (df[f"bb_upper_{suffix}"] - df[f"bb_lower_{suffix}"]) / ma
    log_ret = np.log(close) - np.log(close.shift(1))
    df[f"rv_{suffix}"] = log_ret.rolling(30).std() * np.sqrt(len(log_ret))
    df[f"volatility_to_volume_{suffix}"] = df[f"rv_{suffix}"] / (vol.rolling(20).mean() + 1e-6)
    return df


def infer_suffix_from_name(path: Path) -> str:
    name = path.stem
    # try to grab last token like BTCUSDT_15m -> 15m
    if "_" in name:
        return name.split("_")[-1]
    return "base"


def merge_timeframes(base_path: Path, tf_paths: Dict[str, Path], horizon: int, target_mode: str, out_path: Path, base_suffix: str | None = None) -> None:
    base = pd.read_parquet(base_path).sort_values("timestamp").reset_index(drop=True)
    base = add_time_features(base)
    base = build_target(base, horizon=horizon, mode=target_mode)

    # compute base-level indicators too
    base_suffix = base_suffix or infer_suffix_from_name(base_path)
    base = base.rename(columns={c: f"{c}_{base_suffix}" for c in ["open", "high", "low", "close", "volume", "turnover", "vwap"] if c in base.columns})
    base = add_tf_indicators(base, suffix=base_suffix)

    for suffix, path in tf_paths.items():
        df_tf = load_tf(path, suffix)
        df_tf = add_tf_indicators(df_tf, suffix=suffix)
        # align on timestamp (asof backward, then ffill)
        base = pd.merge_asof(
            base.sort_values("timestamp"),
            df_tf.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )
        base = base.ffill()

    base = base.dropna().reset_index(drop=True)
    base.to_parquet(out_path, index=False)
    print(f"Saved multi-TF features to {out_path}, rows={len(base)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build multi-timeframe features.")
    parser.add_argument("--base", required=True, help="Base timeframe Parquet (e.g., data/BTCUSDT_1m.parquet)")
    parser.add_argument("--tf", nargs="+", help="Additional TF files in format suffix=path, e.g., 5m=data/BTCUSDT_5m.parquet 1h=data/BTCUSDT_1h.parquet")
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon")
    parser.add_argument("--target-mode", choices=["log_return", "price"], default="log_return")
    parser.add_argument("--out", required=True, help="Output Parquet path")
    parser.add_argument("--base-suffix", default=None, help="Suffix for base timeframe (auto from filename if not set)")
    args = parser.parse_args()

    tf_paths: Dict[str, Path] = {}
    if args.tf:
        for item in args.tf:
            if "=" not in item:
                continue
            suffix, path = item.split("=", 1)
            tf_paths[suffix] = Path(path)

    merge_timeframes(Path(args.base), tf_paths, horizon=args.horizon, target_mode=args.target_mode, out_path=Path(args.out), base_suffix=args.base_suffix)


if __name__ == "__main__":
    main()
