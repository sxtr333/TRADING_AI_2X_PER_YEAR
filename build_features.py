"""
Feature builder for Bybit BTCUSDT klines (aggregated from public trades).
- Reads OHLCV Parquet (e.g., from bybit_public_trades.py outputs)
- Computes basic indicators: Bollinger Bands, ATR, simple realized vol, band width
- Adds time features and target
- Saves processed Parquet ready for model training (Keras pipeline)
"""

from __future__ import annotations

import argparse
import math
import numpy as np
import pandas as pd
from pathlib import Path

from trading_keras_core import add_time_features, build_target, default_feature_list


def dbg(tag: str, df: pd.DataFrame) -> None:
    ts_min = df["timestamp"].min() if "timestamp" in df else None
    ts_max = df["timestamp"].max() if "timestamp" in df else None
    print(f"[{tag}] rows={len(df)} range={ts_min} .. {ts_max}")


def bollinger(df: pd.DataFrame, close_col: str = "close", window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    ma = df[close_col].rolling(window=window, min_periods=window).mean()
    std = df[close_col].rolling(window=window, min_periods=window).std()
    df["bollinger_upper"] = ma + num_std * std
    df["bollinger_lower"] = ma - num_std * std
    df["bollinger_bandwidth"] = (df["bollinger_upper"] - df["bollinger_lower"]) / ma
    return df


def atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.rolling(window=window, min_periods=window).mean()
    return df


def realized_vol(df: pd.DataFrame, close_col: str = "close", window: int = 30) -> pd.DataFrame:
    log_ret = np.log(df[close_col]) - np.log(df[close_col].shift(1))
    # scale by window length (not full series length) to keep rv stable
    rv = log_ret.rolling(window=window, min_periods=window).std() * np.sqrt(window)
    df["rv"] = rv
    # Placeholder for implied vol; set equal to rv for now
    df["iv"] = rv
    df["iv_rv_spread"] = df["iv"] - df["rv"]
    return df


def rsi(df: pd.DataFrame, close_col: str = "close", window: int = 14) -> pd.DataFrame:
    delta = df[close_col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi14"] = 100 - (100 / (1 + rs))
    return df


def macd(df: pd.DataFrame, close_col: str = "close", fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = df[close_col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[close_col].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    df["macd_line"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = macd_line - signal_line
    return df


def adx(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr_components = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    )
    tr = tr_components.max(axis=1)
    atr_val = tr.rolling(window=window, min_periods=window).mean()
    plus_di = 100 * (plus_dm.rolling(window=window, min_periods=window).mean() / (atr_val + 1e-9))
    minus_di = 100 * (minus_dm.rolling(window=window, min_periods=window).mean() / (atr_val + 1e-9))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
    df["adx14"] = dx.rolling(window=window, min_periods=window).mean()
    return df


def stoch_rsi(df: pd.DataFrame, rsi_col: str = "rsi14", window: int = 14) -> pd.DataFrame:
    rsi_series = df[rsi_col]
    min_rsi = rsi_series.rolling(window=window, min_periods=window).min()
    max_rsi = rsi_series.rolling(window=window, min_periods=window).max()
    df["stoch_rsi14"] = (rsi_series - min_rsi) / (max_rsi - min_rsi + 1e-9)
    return df


def obv(df: pd.DataFrame, close_col: str = "close", vol_col: str = "volume") -> pd.DataFrame:
    direction = np.sign(df[close_col].diff().fillna(0))
    df["obv"] = (direction * df[vol_col]).cumsum()
    return df


def cmf(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"] + 1e-9)
    mfv = mfm * df["volume"]
    df["cmf"] = mfv.rolling(window=period, min_periods=period).sum() / (
        df["volume"].rolling(window=period, min_periods=period).sum() + 1e-9
    )
    return df


def add_regime_features(df: pd.DataFrame, close_col: str = "close", base_minutes: int = 1) -> pd.DataFrame:
    # short/long realized vol (using log returns) with timeframe-aware windows
    log_ret = np.log(df[close_col]) - np.log(df[close_col].shift(1))
    win_short = max(4, int((24 * 60) / base_minutes))  # ~24h
    win_long = max(8, int((7 * 24 * 60) / base_minutes))  # ~7d
    df["rv_short"] = log_ret.rolling(window=win_short, min_periods=win_short).std()
    df["rv_long"] = log_ret.rolling(window=win_long, min_periods=win_long).std()
    df["rv_ratio"] = df["rv_short"] / (df["rv_long"] + 1e-9)
    rv_short_mean = df["rv_short"].rolling(window=win_long, min_periods=win_long).mean()
    rv_short_std = df["rv_short"].rolling(window=win_long, min_periods=win_long).std()
    df["rv_short_z"] = (df["rv_short"] - rv_short_mean) / (rv_short_std + 1e-9)
    # OI deltas and zscore
    if "open_interest" in df.columns:
        df["oi_delta"] = df["open_interest"].diff()
        df["oi_zscore"] = (df["oi_delta"] - df["oi_delta"].rolling(win_short, min_periods=win_short).mean()) / (
            df["oi_delta"].rolling(win_short, min_periods=win_short).std() + 1e-9
        )
    else:
        df["oi_delta"] = 0.0
        df["oi_zscore"] = 0.0
    if "funding_rate" in df.columns:
        df["funding_delta"] = df["funding_rate"].diff()
    else:
        df["funding_delta"] = 0.0
    # Range/structure features
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    df["range_norm"] = rng / (df[close_col] + 1e-9)
    df["range_percent"] = df["range_norm"]
    df["wick_up"] = (df["high"] - df["close"]) / (rng + 1e-9)
    df["wick_down"] = (df["close"] - df["low"]) / (rng + 1e-9)
    return df


def triple_barrier_labels(
    df: pd.DataFrame,
    price_col: str = "close",
    sigma_col: str = "rv_long",
    H: int = 48,
    k: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Симметричный triple-barrier:
      up, если logP достигает +B раньше -B;
      down, если -B раньше +B;
      neutral, если ни один барьер за H не достигнут.
    Возвращает (label, tth) где label: 0=down, 1=neutral, 2=up; tth=время до срабатывания (0 если neutral).
    """
    logp = np.log(df[price_col].astype(float).values)
    sigma = df[sigma_col].astype(float).values if sigma_col in df.columns else np.zeros_like(logp)
    sigma = np.nan_to_num(sigma, nan=0.0)
    sigma = np.maximum(sigma, 1e-8)

    B = k * sigma * np.sqrt(H)
    n = len(logp)
    label = np.ones(n, dtype=np.int8)  # neutral
    tth = np.zeros(n, dtype=np.int16)

    last = n - H - 1
    for i in range(max(0, last)):
        up_thr = logp[i] + B[i]
        dn_thr = logp[i] - B[i]
        future = logp[i + 1 : i + H + 1]
        hit_up = np.where(future >= up_thr)[0]
        hit_dn = np.where(future <= dn_thr)[0]
        if hit_up.size == 0 and hit_dn.size == 0:
            continue
        j_up = hit_up[0] if hit_up.size else 10**9
        j_dn = hit_dn[0] if hit_dn.size else 10**9
        if j_up < j_dn:
            label[i] = 2
            tth[i] = j_up + 1
        else:
            label[i] = 0
            tth[i] = j_dn + 1
    return label, tth


def add_dummy_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure core derivative columns exist; avoid adding placeholder features that stay zero.
    """
    for col in [
        "open_interest",
        "open_interest_value",
        "oi_zscore",
        "oi_delta",
        "funding_rate",
        "funding_delta",
        "funding_event",
        "buy_sell_ratio",
        "volume_delta",
        "close_delta",
        "log_return_1m",
        "taker_long_short_vol_ratio",
        "toptrader_long_short_ratio",
        "long_short_ratio",
        "liq_long",
        "liq_short",
        "liq_imbalance",
        "liq_total",
        "basis",
        "cvd",
        "tick_buy_volume",
        "tick_sell_volume",
        "delta",
        "ob_imb_01",
        "ob_imb_025",
        "ob_imb_05",
        "ob_imb_1",
    ]:
        if col not in df.columns:
            df[col] = 0.0
    return df


def maybe_merge_aux(df: pd.DataFrame, aux_path: Path, base_minutes: int) -> pd.DataFrame:
    """
    Merge auxiliary data (oi/funding/liquidations/basis/etc) if provided.
    Expected columns in aux: timestamp plus any of [open_interest, funding_rate, liq_long, liq_short, basis, cvd, tick_buy_volume, tick_sell_volume, delta, buy_sell_ratio, ob_imb_01, ob_imb_025, ob_imb_05, ob_imb_1]
    """
    if aux_path is None or not aux_path.exists():
        return df
    if aux_path.suffix.lower() in [".csv"]:
        aux = pd.read_csv(aux_path, parse_dates=["timestamp"])
    else:
        aux = pd.read_parquet(aux_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    aux["timestamp"] = pd.to_datetime(aux["timestamp"], utc=True)
    df = df.sort_values("timestamp")
    # привести aux к сетке базового таймфрейма: последнее известное значение, затем ffill
    resample_rule = f"{base_minutes}min"
    aux = aux.sort_values("timestamp").set_index("timestamp").resample(resample_rule).last().ffill().reset_index()

    aux_cols = [
        "open_interest",
        "open_interest_value",
        "funding_rate",
        "liq_long",
        "liq_short",
        "basis",
        "cvd",
        "tick_buy_volume",
        "tick_sell_volume",
        "delta",
        "buy_sell_ratio",
        "ob_imb_01",
        "ob_imb_025",
        "ob_imb_05",
        "ob_imb_1",
        "taker_long_short_vol_ratio",
        "toptrader_long_short_ratio",
        "long_short_ratio",
    ]
    has_oi_value = "open_interest_value" in aux.columns
    # drop conflicting cols to avoid _x/_y and stale zeros
    to_drop = [c for c in aux_cols if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)

    keep = ["timestamp"] + [c for c in aux_cols if c in aux.columns]
    aux = aux[keep]

    df = pd.merge_asof(
        df,
        aux,
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta("2H"),
    )
    # coverage diagnostics before ffill
    pre_nan_share = {}
    for c in ["open_interest_value", "open_interest"]:
        if c in df.columns:
            pre_nan_share[c] = df[c].isna().mean()
    if pre_nan_share:
        print(f"aux pre-ffill NaN share: {pre_nan_share}")

    level_cols = [c for c in ["open_interest", "open_interest_value", "funding_rate"] if c in df.columns]
    if level_cols:
        df[level_cols] = df[level_cols].ffill()
        # обрезаем только до первого валидного значения по ключу
        first = df["open_interest_value"].first_valid_index() if "open_interest_value" in df.columns else None
        if first is not None:
            df = df.loc[first:]

    if "open_interest_value" in df.columns:
        zero_share = (df["open_interest_value"] == 0).mean()
        print(f"after merge_asof: rows={len(df)}, zero_share={zero_share:.3f}, nan_share={df['open_interest_value'].isna().mean():.3f}")
        if has_oi_value and zero_share > 0.5:
            print("[warn] open_interest_value is mostly zeros after merge")
    dbg("after_merge_asof", df)
    return df


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


def _parse_window_list(windows: str, base_minutes: int):
    if not windows:
        return []
    out = []
    for raw in windows.split(","):
        w = raw.strip().lower()
        if not w:
            continue
        mult = 1
        if w.endswith("h"):
            mult = 60
            w = w[:-1]
        elif w.endswith("d"):
            mult = 1440
            w = w[:-1]
        elif w.endswith("m"):
            mult = 1
            w = w[:-1]
        try:
            minutes = int(float(w) * mult)
        except ValueError:
            continue
        if minutes <= 0:
            continue
        bars = max(1, int(round(minutes / base_minutes)))
        label = raw.strip()
        out.append((label, bars))
    return out


def maybe_merge_news(
    df: pd.DataFrame,
    news_path: Path,
    base_minutes: int,
    news_windows: str | None = None,
    news_ewm: str | None = None,
    news_clip_pct: float = 0.99,
    news_count_cap: int | None = None,
) -> pd.DataFrame:
    """
    Merge news signals. Expected columns in news: timestamp or published_at + optional
    [sentiment, votes, impact]. Produces:
      - news_count, news_sentiment, news_shock, news_votes
    """
    if news_path is None or not news_path.exists():
        return df
    if news_path.suffix.lower() in [".jsonl"]:
        news = pd.read_json(news_path, lines=True)
    elif news_path.suffix.lower() in [".csv"]:
        news = pd.read_csv(news_path)
    else:
        news = pd.read_parquet(news_path)

    if "timestamp" in news.columns:
        news["timestamp"] = pd.to_datetime(news["timestamp"], utc=True)
    elif "published_at" in news.columns:
        news["timestamp"] = pd.to_datetime(news["published_at"], utc=True, errors="coerce")
    else:
        return df

    news = news.dropna(subset=["timestamp"]).sort_values("timestamp")
    if "news_count" not in news.columns:
        news["news_count"] = 1.0
    if "news_sentiment" not in news.columns:
        if "sentiment" in news.columns:
            news["news_sentiment"] = news["sentiment"].apply(_sentiment_to_num)
        else:
            news["news_sentiment"] = 0.0
    if "news_votes" not in news.columns:
        if "votes" in news.columns:
            news["news_votes"] = pd.to_numeric(news["votes"], errors="coerce").fillna(0.0)
        else:
            news["news_votes"] = 0.0
    if "news_shock" not in news.columns:
        news["news_shock"] = (news["news_sentiment"].abs() >= 0.5).astype(float)

    resample_rule = f"{base_minutes}min"
    news = (
        news.set_index("timestamp")
        .resample(resample_rule)
        .agg(
            {
                "news_count": "sum",
                "news_sentiment": "mean",
                "news_shock": "sum",
                "news_votes": "sum",
            }
        )
        .fillna(0.0)
    )

    # Cap per-window news density to match training distribution
    if news_count_cap is not None and "news_count" in news.columns:
        news["news_count"] = news["news_count"].clip(upper=float(news_count_cap))

    # Stabilize heavy tails when news volume spikes
    if news_clip_pct:
        clip_cols = ["news_count", "news_shock", "news_votes"]
        for col in clip_cols:
            if col in news.columns:
                cap = news[col].quantile(news_clip_pct)
                news[col] = news[col].clip(upper=cap)
        if "news_count" in news.columns:
            news["news_count"] = np.log1p(news["news_count"])

    windows = _parse_window_list(news_windows or "", base_minutes)
    for label, bars in windows:
        news[f"news_count_{label}"] = news["news_count"].rolling(bars, min_periods=1).sum()
        news[f"news_shock_{label}"] = news["news_shock"].rolling(bars, min_periods=1).sum()
        news[f"news_votes_{label}"] = news["news_votes"].rolling(bars, min_periods=1).sum()
        news[f"news_sentiment_{label}"] = news["news_sentiment"].rolling(bars, min_periods=1).mean()

    ewm_windows = _parse_window_list(news_ewm or "", base_minutes)
    for label, bars in ewm_windows:
        news[f"news_count_ewm_{label}"] = news["news_count"].ewm(halflife=bars, adjust=False).mean()
        news[f"news_votes_ewm_{label}"] = news["news_votes"].ewm(halflife=bars, adjust=False).mean()
        news[f"news_sentiment_ewm_{label}"] = news["news_sentiment"].ewm(halflife=bars, adjust=False).mean()

    df_ts = pd.to_datetime(df["timestamp"], utc=True)
    news = news.reindex(df_ts)
    news_missing = news.isna().any(axis=1).astype(float)
    news["news_missing"] = news_missing
    news = news.reset_index().rename(columns={"index": "timestamp"})
    df = df.merge(news, on="timestamp", how="left")
    for col in news.columns:
        if col in df.columns and col != "news_missing":
            df[col] = df[col].fillna(0.0)
    if "news_missing" in df.columns:
        df["news_missing"] = df["news_missing"].fillna(1.0)
    return df


def _read_daily_df(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in [".csv"]:
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)

    # locate date-like column
    date_col = None
    for c in ("date", "Date", "timestamp", "Timestamp", "observation_date"):
        if c in df.columns:
            date_col = c
            break
    if date_col is None and "Unnamed: 0" in df.columns:
        date_col = "Unnamed: 0"

    if date_col is not None:
        ts = pd.to_datetime(df[date_col], utc=True, errors="coerce")
        df = df.drop(columns=[date_col])
    else:
        ts = pd.to_datetime(df.index, utc=True, errors="coerce")
        if isinstance(ts, pd.DatetimeIndex):
            ts = pd.Series(ts.to_numpy())
        else:
            ts = pd.Series(ts.to_numpy())
        df = df.reset_index(drop=True)
    ts = pd.Series(pd.to_datetime(ts, utc=True, errors="coerce").to_numpy())

    df = df.copy()
    df["__date__"] = ts.dt.normalize()
    df = df.dropna(subset=["__date__"])

    # coerce numeric where possible, keep numeric/bool only
    for c in df.columns:
        if c == "__date__":
            continue
        df[c] = pd.to_numeric(df[c], errors="ignore")
    keep = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    keep.append("__date__")
    out = df[keep].rename(columns={"__date__": "date"}).reset_index(drop=True)
    out = out.sort_values("date").drop_duplicates("date", keep="last")
    return out


def merge_daily_features(
    df_15m: pd.DataFrame,
    daily_paths: list[str],
    join_on: str = "timestamp",
    shift_days: int = 1,
    ffill_daily: bool = True,
) -> pd.DataFrame:
    """
    Anti-leakage merge for daily factors.
    - shift daily values by N days (default 1) so daily D appears only on D+1
    - optional daily ffill AFTER shift to cover weekends
    - join by date_key = floor(timestamp, 'D')
    """
    if not daily_paths:
        return df_15m

    df = df_15m.copy()
    if join_on not in df.columns:
        raise ValueError(f"merge_daily_features: missing '{join_on}' column in df_15m")
    df[join_on] = pd.to_datetime(df[join_on], utc=True)
    df = df.sort_values(join_on)
    df["date_key"] = df[join_on].dt.floor("D")

    added_cols: list[str] = []
    for raw_path in daily_paths:
        if not raw_path:
            continue
        path = Path(raw_path)
        if not path.exists():
            print(f"[warn] daily path not found: {path}")
            continue
        daily = _read_daily_df(path)
        daily = daily.sort_values("date")
        daily["date"] = pd.to_datetime(daily["date"], utc=True)
        daily = daily.drop_duplicates("date", keep="last")
        daily["date"] = daily["date"] + pd.Timedelta(days=int(shift_days))
        if ffill_daily:
            daily = daily.set_index("date").ffill().reset_index()
        # avoid column collisions
        overlap = [c for c in daily.columns if c != "date" and c in df.columns]
        if overlap:
            print(f"[warn] daily cols already exist, skipping: {overlap}")
            daily = daily.drop(columns=overlap)
        if len(daily.columns) <= 1:
            continue
        added_cols.extend([c for c in daily.columns if c != "date"])
        df = df.merge(daily, how="left", left_on="date_key", right_on="date", suffixes=("", "_daily"))
        df = df.drop(columns=["date"])

    if not added_cols:
        df = df.drop(columns=["date_key"])
        return df

    # forward-fill within day for safety
    df[added_cols] = df[added_cols].ffill()

    # log NaN percentage per added col
    nan_pct = df[added_cols].isna().mean().sort_values(ascending=False)
    print("daily merge NaN shares (top 10):")
    print(nan_pct.head(10))

    # add missing flag + neutral-fill remaining NaN
    all_na_cols = [c for c in added_cols if df[c].isna().all()]
    if all_na_cols:
        df[all_na_cols] = 0.0
    miss_cols = [c for c in added_cols if c not in all_na_cols]
    if "daily_missing_any" not in df.columns:
        df["daily_missing_any"] = 0.0
    miss = df[miss_cols].isna().any(axis=1) if miss_cols else pd.Series(False, index=df.index)
    if miss.any():
        df.loc[miss, "daily_missing_any"] = 1.0
        df[miss_cols] = df[miss_cols].fillna(0.0)

    # validation: values should be constant within each day (post-fill)
    grouped = df.groupby("date_key", sort=False)
    for c in added_cols:
        nunique_max = int(grouped[c].nunique(dropna=False).max())
        assert nunique_max <= 1, f"daily leakage risk: {c} varies within a day (max nunique={nunique_max})"

    df = df.drop(columns=["date_key"])
    return df


def maybe_add_basis(df: pd.DataFrame, spot_path: Path) -> pd.DataFrame:
    """
    Compute basis = (perp_close - spot_close)/spot_close if spot data provided.
    Expects spot parquet/csv with timestamp and close.
    """
    if spot_path is None or not spot_path.exists():
        return df
    if spot_path.suffix.lower() == ".csv":
        spot = pd.read_csv(spot_path, parse_dates=["timestamp"])
    else:
        spot = pd.read_parquet(spot_path)
    spot = spot[["timestamp", "close"]].rename(columns={"close": "close_spot"}).sort_values("timestamp")
    df = df.merge(spot, on="timestamp", how="left").sort_values("timestamp")
    df[["close_spot"]] = df[["close_spot"]].ffill()
    df["basis"] = (df["close"] - df["close_spot"]) / df["close_spot"]
    df = df.drop(columns=["close_spot"])
    return df


def infer_base_minutes(df: pd.DataFrame) -> int:
    if "timestamp" not in df.columns or len(df) < 3:
        return 1
    ts = pd.to_datetime(df["timestamp"], utc=True).sort_values()
    deltas = ts.diff().dropna().dt.total_seconds().values
    if deltas.size == 0:
        return 1
    med = float(np.median(deltas))
    if med <= 0:
        return 1
    return max(1, int(round(med / 60.0)))


def add_engineered_features(df: pd.DataFrame, base_minutes: int) -> pd.DataFrame:
    # basic per-bar deltas
    df["close_delta"] = df["close"].diff().fillna(0.0)
    df["volume_delta"] = df["volume"].diff().fillna(0.0)
    df["log_return_1m"] = (np.log(df["close"]) - np.log(df["close"].shift(1))).fillna(0.0)

    # open interest value (always recompute if open_interest exists)
    if "open_interest" in df.columns:
        df["open_interest_value"] = df["open_interest"] * df["close"]

    # liquidation imbalance/total
    if "liq_long" in df.columns and "liq_short" in df.columns:
        df["liq_total"] = df["liq_long"].fillna(0.0) + df["liq_short"].fillna(0.0)
        df["liq_imbalance"] = df["liq_long"].fillna(0.0) - df["liq_short"].fillna(0.0)

    # buy/sell ratio from tick volumes if available
    if "tick_buy_volume" in df.columns and "tick_sell_volume" in df.columns:
        df["buy_sell_ratio"] = df["tick_buy_volume"].fillna(0.0) / (df["tick_sell_volume"].fillna(0.0) + 1e-9)

    # normalize open interest deltas if possible
    if "open_interest" in df.columns:
        df["oi_delta"] = df["open_interest"].diff().fillna(0.0)
        win_short = max(4, int((24 * 60) / base_minutes))
        roll = df["oi_delta"].rolling(win_short, min_periods=win_short)
        df["oi_zscore"] = (df["oi_delta"] - roll.mean()) / (roll.std() + 1e-9)

    return df


def build_features(
    in_path: Path,
    out_path: Path,
    serve_out_path: Path | None = None,
    aux_path: Path = None,
    spot_path: Path = None,
    news_path: Path = None,
    macro_daily_path: Path | None = None,
    fed_daily_path: Path | None = None,
    inst_daily_path: Path | None = None,
    news_windows: str | None = None,
    news_ewm: str | None = None,
    news_clip_pct: float = 0.99,
    news_count_cap: int | None = None,
    horizon: int = 1,
    target_mode: str = "log_return",
    base_tf_min: int | None = None,
    vf_k: float = 0.5,
    vf_col: str = "rv_long",
    vf2_k: float | None = None,
    tb_horizon: int | None = None,
    multi_horizons: list[int] | None = None,
    tb_k: float = 0.8,
    tb_sigma_col: str = "rv_long",
) -> None:
    df = pd.read_parquet(in_path)
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    dbg("loaded_ohlcv", df)
    base_minutes = base_tf_min or infer_base_minutes(df)
    print(f"[info] base_tf_minutes={base_minutes}")

    df = bollinger(df)
    df = atr(df)
    df = realized_vol(df)
    df = rsi(df)
    df = macd(df)
    df = adx(df)
    df = stoch_rsi(df)
    df = obv(df)
    df = cmf(df)
    df = add_regime_features(df, base_minutes=base_minutes)
    df = add_dummy_features(df)
    if aux_path:
        df = maybe_merge_aux(df, aux_path, base_minutes=base_minutes)
    daily_paths: list[str] = []
    if macro_daily_path:
        daily_paths.append(str(macro_daily_path))
    if fed_daily_path:
        daily_paths.append(str(fed_daily_path))
    if inst_daily_path:
        daily_paths.append(str(inst_daily_path))
    if daily_paths:
        df = merge_daily_features(
            df_15m=df,
            daily_paths=daily_paths,
            join_on="timestamp",
            shift_days=1,
            ffill_daily=True,
        )
    if news_path:
        df = maybe_merge_news(
            df,
            news_path,
            base_minutes=base_minutes,
            news_windows=news_windows,
            news_ewm=news_ewm,
            news_clip_pct=news_clip_pct,
            news_count_cap=news_count_cap,
        )
    if spot_path:
        df = maybe_add_basis(df, spot_path)
    df = add_engineered_features(df, base_minutes=base_minutes)
    dbg("after_features_before_time", df)

    # funding: заполняем, добавляем delta/флаг изменения
    if "funding_rate" in df.columns:
        df["funding_rate"] = df["funding_rate"].ffill().fillna(0.0)
        df["funding_delta"] = df["funding_rate"].diff().fillna(0.0)
        df["funding_event"] = (df["funding_delta"].abs() > 0).astype(np.int32)
    df = add_time_features(df)
    df = build_target(df, price_col="close", horizon=horizon, mode=target_mode)
    # unified return target and direction (no future leakage beyond horizon)
    if target_mode == "log_return":
        df["target_ret"] = df["target_next_close"]
    else:
        df["target_ret"] = np.log(df["close"].shift(-horizon)) - np.log(df["close"])
    df["target_dir"] = (df["target_ret"] > 0).astype(np.int8)

    # Volatility-filtered direction target with FLAT zone
    vf_sigma = df[vf_col] if vf_col in df.columns else df["rv_long"]
    vf_thr = vf_sigma * np.sqrt(horizon) * float(vf_k)
    df["target_dir_vf"] = 0
    df.loc[df["target_ret"] > vf_thr, "target_dir_vf"] = 1
    df.loc[df["target_ret"] < -vf_thr, "target_dir_vf"] = -1
    df["label_3cls_vf"] = df["target_dir_vf"]
    if vf2_k is not None:
        vf2_thr = vf_sigma * np.sqrt(horizon) * float(vf2_k)
        df["target_dir_vf2"] = 0
        df.loc[df["target_ret"] > vf2_thr, "target_dir_vf2"] = 1
        df.loc[df["target_ret"] < -vf2_thr, "target_dir_vf2"] = -1
        df["label_3cls_vf2"] = df["target_dir_vf2"]

    # амплитуда будущего движения и будущий диапазон (для aux-регрессии)
    df["target_amp_abs"] = df["target_next_close"].abs()
    # будущий high/low на окне горизонта
    future_high = df["high"].shift(-(horizon - 1)).rolling(horizon, min_periods=horizon).max()
    future_low = df["low"].shift(-(horizon - 1)).rolling(horizon, min_periods=horizon).min()
    df["target_range"] = np.log(future_high / future_low)

    # Triple-barrier разметка для направления (0=down,1=neutral,2=up)
    tb_H = tb_horizon if tb_horizon is not None else int(horizon * 4)
    tb_label, tb_tth = triple_barrier_labels(
        df, price_col="close", sigma_col=tb_sigma_col, H=tb_H, k=float(tb_k)
    )
    df["tb_label"] = tb_label
    df["tb_tth"] = tb_tth
    # для совместимости с существующим кодом: label_3cls = -1/0/1
    df["label_3cls"] = 0
    df.loc[df["tb_label"] == 2, "label_3cls"] = 1
    df.loc[df["tb_label"] == 0, "label_3cls"] = -1

    # Multi-horizon triple-barrier labels (optional)
    if multi_horizons:
        for h in sorted({int(x) for x in multi_horizons if int(x) > 0}):
            df[f"target_ret_h{h}"] = np.log(df["close"].shift(-h)) - np.log(df["close"])
            df[f"target_dir_h{h}"] = (df[f"target_ret_h{h}"] > 0).astype(np.int8)
            # Volatility-filtered direction (per-horizon)
            vf_sigma_h = df[vf_col] if vf_col in df.columns else df["rv_long"]
            vf_thr_h = vf_sigma_h * np.sqrt(h) * float(vf_k)
            df[f"target_dir_vf_h{h}"] = 0
            df.loc[df[f"target_ret_h{h}"] > vf_thr_h, f"target_dir_vf_h{h}"] = 1
            df.loc[df[f"target_ret_h{h}"] < -vf_thr_h, f"target_dir_vf_h{h}"] = -1
            df[f"label_3cls_vf_h{h}"] = df[f"target_dir_vf_h{h}"]
            # If a base tb_horizon is provided, scale it proportionally to each horizon
            # so multi-horizon labels are not identical.
            if tb_horizon is not None and horizon > 0:
                scale = tb_horizon / float(horizon)
                tb_H_h = max(1, int(round(h * scale)))
            else:
                tb_H_h = int(h * 4)
            tb_label_h, tb_tth_h = triple_barrier_labels(
                df, price_col="close", sigma_col=tb_sigma_col, H=tb_H_h, k=float(tb_k)
            )
            df[f"tb_label_h{h}"] = tb_label_h
            df[f"tb_tth_h{h}"] = tb_tth_h
            df[f"label_3cls_h{h}"] = 0
            df.loc[df[f"tb_label_h{h}"] == 2, f"label_3cls_h{h}"] = 1
            df.loc[df[f"tb_label_h{h}"] == 0, f"label_3cls_h{h}"] = -1
    # vol-adjusted target (оставляем, если нужно где-то ещё)
    log_ret_1 = np.log(df["close"]) - np.log(df["close"].shift(1))
    vol_window = max(4, int((24 * 60) / base_minutes))  # past ~24h
    sigma = log_ret_1.rolling(window=vol_window, min_periods=vol_window).std()
    df["target_next_close_vol"] = df["target_next_close"] / (sigma * np.sqrt(horizon) + 1e-9)

    # recompute deltas/zscores after aux is merged and ffilled
    if "open_interest" in df.columns:
        df["oi_delta"] = df["open_interest"].diff().fillna(0)
        roll_win = max(8, int((7 * 24 * 60) / base_minutes))  # ~7d
        roll = df["oi_delta"].rolling(roll_win, min_periods=max(8, roll_win // 10))
        df["oi_zscore"] = (df["oi_delta"] - roll.mean()) / (roll.std() + 1e-9)

    # warmup: обрезаем только до первого валидного значения самой “длинной” фичи (rv_long)
    warm_col = "rv_long" if "rv_long" in df.columns else None
    if warm_col:
        first = df[warm_col].first_valid_index()
        if first is not None:
            df = df.loc[first:].copy()

    dbg("after_warmup", df)

    # CORE sanity: проверим наличие и NaN, но не режем всё подряд
    core = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vwap",
        "open_interest",
        "open_interest_value",
    ]
    missing = [c for c in core if c not in df.columns]
    assert not missing, f"Missing core cols: {missing}"
    core_na = df[core].isna().mean()
    print("core NaN shares:")
    print(core_na)
    if core_na.max() > 0:
        first_core = df[core].dropna().index.min()
        if first_core is not None:
            df = df.loc[first_core:].copy()
    dbg("after_core_check", df)

    # Keep only required columns
    feature_cols = default_feature_list()
    keep_cols = feature_cols + [
        "target_next_close",
        "target_ret",
        "target_dir",
        "target_dir_vf",
        "target_next_close_vol",
        "target_amp_abs",
        "target_range",
        "tb_label",
        "tb_tth",
        "label_3cls",
        "label_3cls_vf",
        "timestamp",
    ]
    if "target_dir_vf2" in df.columns:
        keep_cols.append("target_dir_vf2")
    if "label_3cls_vf2" in df.columns:
        keep_cols.append("label_3cls_vf2")
    if multi_horizons:
        for h in sorted({int(x) for x in multi_horizons if int(x) > 0}):
            for col in (
                f"target_ret_h{h}",
                f"target_dir_h{h}",
                f"target_dir_vf_h{h}",
                f"tb_label_h{h}",
                f"tb_tth_h{h}",
                f"label_3cls_h{h}",
                f"label_3cls_vf_h{h}",
            ):
                if col in df.columns:
                    keep_cols.append(col)

    # filter lists to existing columns (aux fields may be missing)
    feature_cols = [c for c in feature_cols if c in df.columns]
    keep_cols = [c for c in keep_cols if c in df.columns]

    if serve_out_path is not None:
        serve_cols = list(feature_cols)
        for col in ("timestamp", "close"):
            if col in df.columns and col not in serve_cols:
                serve_cols.append(col)
        df[serve_cols].to_parquet(serve_out_path, index=False)
        print(f"Saved serve features: {len(df)} rows to {serve_out_path}")

    # диагностика NaN по keep_cols
    na_share = df[keep_cols].isna().mean().sort_values(ascending=False).head(10)
    print("Top NaN shares (keep_cols):")
    print(na_share)
    mask_all = df[keep_cols].notna().all(axis=1)
    print(f"rows with all keep_cols non-NaN: {mask_all.sum()}")
    if mask_all.any():
        ts_min = df.loc[mask_all, "timestamp"].min()
        ts_max = df.loc[mask_all, "timestamp"].max()
        print(f"range(all-nonNaN): {ts_min} .. {ts_max}")

    # опционально удаляем строки, где NaN в таргетах
    drop_targets = [
        "target_next_close",
        "target_ret",
        "target_dir",
        "target_dir_vf",
        "target_amp_abs",
        "target_range",
        "label_3cls",
        "label_3cls_vf",
    ]
    if "target_dir_vf2" in df.columns:
        drop_targets.append("target_dir_vf2")
    if "label_3cls_vf2" in df.columns:
        drop_targets.append("label_3cls_vf2")
    if multi_horizons:
        for h in sorted({int(x) for x in multi_horizons if int(x) > 0}):
            for col in (
                f"target_ret_h{h}",
                f"target_dir_h{h}",
                f"target_dir_vf_h{h}",
                f"tb_label_h{h}",
                f"tb_tth_h{h}",
                f"label_3cls_h{h}",
                f"label_3cls_vf_h{h}",
            ):
                if col in df.columns:
                    drop_targets.append(col)
    drop_targets = [c for c in drop_targets if c in df.columns]
    df = df.dropna(subset=drop_targets)

    # проверки на NaN во всех фичах — parquet терпит NaN, но Keras нет
    feature_na = df[feature_cols].isna().mean().sort_values(ascending=False)
    print("Top NaN shares (feature_cols):")
    print(feature_na.head(10))
    if feature_na.max() > 0:
        df = df.dropna(subset=feature_cols).copy()
        feature_na = df[feature_cols].isna().mean().sort_values(ascending=False)
    max_feat_na = float(feature_na.max())
    assert max_feat_na == 0.0, f"Features still contain NaN after drop (max share {max_feat_na:.3f})"

    df[keep_cols].to_parquet(out_path, index=False)

    # sanity checks: nonzero share and std
    def check_nonzero(col, max_zero=0.05):
        if col not in df.columns:
            return
        zero_share = (df[col] == 0).mean()
        std = df[col].std()
        print(f"{col}: zero_share={zero_share:.3f}, std={std:.6f}")
        assert zero_share < max_zero, f"{col} has too many zeros: {zero_share:.3f}"
        assert std > 1e-8, f"{col} has near-zero std"

    for c in ["open_interest", "open_interest_value"]:
        # OI may be unavailable in some aux sources (e.g., Binance vision). Warn instead of assert.
        if c in df.columns:
            zero_share = (df[c] == 0).mean()
            std = df[c].std()
            print(f"{c}: zero_share={zero_share:.3f}, std={std:.6f} (info)")
    for c in ["taker_long_short_vol_ratio", "toptrader_long_short_ratio", "long_short_ratio"]:
        if c in df.columns and df[c].std() == 0:
            print(f"[warn] {c} is constant; skipping nonzero check")
        else:
            check_nonzero(c, max_zero=0.2)
    if "funding_rate" in df.columns:
        zero_share = (df["funding_rate"] == 0).mean()
        std = df["funding_rate"].std()
        print(f"funding_rate (info): zero_share={zero_share:.3f}, std={std:.6f} (not asserted)")

    print(f"Saved features: {len(df)} rows to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build features for Keras model.")
    parser.add_argument("--input", required=True, help="Input Parquet file with OHLCV (e.g., data/BTCUSDT_1m.parquet)")
    parser.add_argument("--output", required=True, help="Output Parquet for features")
    parser.add_argument("--serve-output", default=None, help="Optional Parquet for serving (features only)")
    parser.add_argument("--aux", help="Optional auxiliary Parquet/CSV with columns like open_interest,funding_rate,liq_long,liq_short,basis,... aligned by timestamp")
    parser.add_argument("--spot", help="Optional spot Parquet/CSV (timestamp, close) to compute basis")
    parser.add_argument(
        "--news",
        help="Optional news Parquet/JSONL/CSV to add news signals (news_count/news_sentiment/news_shock/news_votes)",
    )
    parser.add_argument(
        "--news-windows",
        default="",
        help="Comma-separated rolling windows for news features (e.g. '1h,4h,12h').",
    )
    parser.add_argument(
        "--news-ewm",
        default="",
        help="Comma-separated EWM halflife windows for news features (e.g. '4h,12h').",
    )
    parser.add_argument(
        "--news-clip-pct",
        type=float,
        default=0.99,
        help="Clip news_count/news_shock/news_votes at this quantile, then log1p news_count.",
    )
    parser.add_argument(
        "--news-count-cap",
        type=int,
        default=None,
        help="Hard cap for per-window news_count before clipping/log1p (e.g., 65 to match training).",
    )
    parser.add_argument("--macro-daily-path", default=None, help="Macro/risk daily parquet/csv (DXY/SPX/QQQ/VIX)")
    parser.add_argument("--fed-daily-path", default=None, help="FRED daily rates parquet/csv (DFF/DGS2/DFII5)")
    parser.add_argument("--inst-daily-path", default=None, help="Institutional daily parquet/csv (ETF/Stablecoins/Premium)")
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon (number of steps ahead)")
    parser.add_argument("--target-mode", choices=["log_return", "price"], default="log_return", help="Target type")
    parser.add_argument("--base-tf-min", type=int, default=None, help="Base timeframe in minutes (auto if not set)")
    parser.add_argument("--vf-k", type=float, default=0.5, help="Volatility filter k for flat zone (target_dir_vf)")
    parser.add_argument("--vf2-k", type=float, default=None, help="Optional second k for target_dir_vf2")
    parser.add_argument("--vf-col", type=str, default="rv_long", help="Volatility column for vf target")
    parser.add_argument("--tb-horizon", type=int, default=None, help="Triple-barrier horizon (bars), default=horizon*4")
    parser.add_argument("--tb-k", type=float, default=0.8, help="Triple-barrier k multiplier")
    parser.add_argument("--tb-sigma-col", type=str, default="rv_long", help="Triple-barrier sigma column")
    parser.add_argument(
        "--multi-horizons",
        type=str,
        default=None,
        help="Comma-separated list of horizons for multi-horizon triple-barrier labels (e.g., 20,80,160)",
    )
    args = parser.parse_args()

    multi_horizons = None
    if args.multi_horizons:
        multi_horizons = [int(x.strip()) for x in args.multi_horizons.split(",") if x.strip()]
        if not multi_horizons:
            multi_horizons = None

    build_features(
        Path(args.input),
        Path(args.output),
        serve_out_path=Path(args.serve_output) if args.serve_output else None,
        aux_path=Path(args.aux) if args.aux else None,
        spot_path=Path(args.spot) if args.spot else None,
        news_path=Path(args.news) if args.news else None,
        macro_daily_path=Path(args.macro_daily_path) if args.macro_daily_path else None,
        fed_daily_path=Path(args.fed_daily_path) if args.fed_daily_path else None,
        inst_daily_path=Path(args.inst_daily_path) if args.inst_daily_path else None,
        news_windows=args.news_windows,
        news_ewm=args.news_ewm,
        news_clip_pct=args.news_clip_pct,
        news_count_cap=args.news_count_cap,
        horizon=args.horizon,
        target_mode=args.target_mode,
        base_tf_min=args.base_tf_min,
        vf_k=args.vf_k,
        vf_col=args.vf_col,
        vf2_k=args.vf2_k,
        tb_horizon=args.tb_horizon,
        tb_k=args.tb_k,
        tb_sigma_col=args.tb_sigma_col,
        multi_horizons=multi_horizons,
    )


if __name__ == "__main__":
    main()
