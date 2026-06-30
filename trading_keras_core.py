"""
Core Keras utilities and model definition.
- Feature schema helpers
- Feature engineering helpers (time/target)
- tf.data window builder
- TCN + Transformer sequence model
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf

# Feature schema

BASE_PRICE_FEATURES = ["open", "high", "low", "close", "volume", "vwap"]

DERIV_FEATURES = [
    "open_interest",
    "open_interest_value",
    "oi_zscore",
    "funding_rate",
    "funding_delta",
    "funding_event",
    "cvd",
    "liq_long",
    "liq_short",
    "liq_imbalance",
    "liq_total",
    "basis",
    "tick_buy_volume",
    "tick_sell_volume",
    "delta",
    "buy_sell_ratio",
    "oi_delta",
    "volume_delta",
    "close_delta",
    "log_return_1m",
    "range_norm",
    "wick_up",
    "wick_down",
]

ORDERBOOK_FEATURES = ["ob_imb_01", "ob_imb_025", "ob_imb_05", "ob_imb_1"]

VOL_FEATURES = [
    "bollinger_upper",
    "bollinger_lower",
    "bollinger_bandwidth",
    "atr",
    "rv",
    "rv_short",
    "rv_long",
    "rv_ratio",
    "iv",
    "iv_rv_spread",
]

TA_FEATURES = [
    "rsi14",
    "macd_line",
    "macd_signal",
    "macd_hist",
    "adx14",
    "stoch_rsi14",
    "obv",
    "cmf",
]

TIME_FEATURES = ["dow", "hour", "minute"]
NEWS_FEATURES = ["news_count", "news_sentiment", "news_shock", "news_votes", "news_missing"]
DAILY_FEATURES = [
    # Institutional / on-chain / ETF
    "stable_usdt_circulating_usd",
    "stable_usdc_circulating_usd",
    "stable_usdt_circulating_usd_chg_1d",
    "stable_usdc_circulating_usd_chg_1d",
    "etf_flow_total_usdm",
    "etf_flow_ibit_usdm",
    "etf_flow_fbtc_usdm",
    "etf_flow_bitb_usdm",
    "etf_flow_arkb_usdm",
    "etf_flow_btco_usdm",
    "etf_flow_ezbc_usdm",
    "etf_flow_brrr_usdm",
    "etf_flow_hodl_usdm",
    "etf_flow_btcw_usdm",
    "etf_flow_gbtc_usdm",
    "etf_flow_btc_usdm",
    "etf_flow_total_usdm_5d",
    "etf_flow_total_usdm_20d",
    "etf_flow_total_abs_usdm",
    "etf_flow_total_abs_usdm_20d_max",
    "coinbase_premium_usd",
    "coinbase_premium_pct",
    "is_month_end",
    "is_quarter_end",
    "days_to_month_end",
    "days_to_quarter_end",
    "days_to_fut_expiry",
    "is_roll_window_5d",
    # Macro / risk (FRED + Yahoo)
    "vix",
    "hy_oas",
    "ig_oas",
    "hy_oas_chg_1d",
    "ig_oas_chg_1d",
    "vix_chg_1d",
    "dxy",
    "spx",
    "qqq",
    "dxy_ret_1d",
    "spx_ret_1d",
    "qqq_ret_1d",
    "risk_on_flag",
    "rrp_overnight_bil_usd",
    "tga_mil_usd",
    "fed_total_assets_mil_usd",
    "stl_fsi",
    "nfci",
    "cpi_sa",
    "pcepi_sa",
    "pcepi_core_sa",
    "effr",
    "ust_2y_yield",
    "curve_10y_minus_2y",
    "hy_oas_pct",
    "ig_oas_pct",
    "m2_bil_usd_sa",
    "ip_manufacturing_sa",
    "ust_2y_yield_chg_1d",
    "global_macro_shock_flag",
    "fed_funds_effective",
    "tips_real_yield_5y",
    "daily_missing_any",
]


def default_feature_list() -> List[str]:
    return (
        BASE_PRICE_FEATURES
        + DERIV_FEATURES
        + ORDERBOOK_FEATURES
        + VOL_FEATURES
        + TA_FEATURES
        + TIME_FEATURES
        + DAILY_FEATURES
        + NEWS_FEATURES
    )


def add_time_features(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    ts = pd.to_datetime(df[ts_col], utc=True)
    df["dow"] = ts.dt.dayofweek / 6.0
    df["hour"] = ts.dt.hour / 23.0
    df["minute"] = ts.dt.minute / 59.0
    return df


def build_target(df: pd.DataFrame, price_col: str = "close", horizon: int = 1, mode: str = "log_return") -> pd.DataFrame:
    if mode == "log_return":
        df["target_next_close"] = np.log(df[price_col].shift(-horizon)) - np.log(df[price_col])
    elif mode == "price":
        df["target_next_close"] = df[price_col].shift(-horizon)
    else:
        raise ValueError("mode must be 'log_return' or 'price'")
    return df


def make_tf_dataset(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    seq_len: int = 256,
    batch_size: int = 64,
    stride: int = 1,
    shuffle: bool = True,
) -> tf.data.Dataset:
    feats = df[feature_cols].to_numpy(dtype=np.float32)
    targets = df[target_col].to_numpy(dtype=np.float32)
    total = len(feats)
    if total < seq_len + 1:
        raise ValueError("Not enough rows for the requested sequence length.")

    ds = tf.data.Dataset.from_tensor_slices((feats, targets))
    ds = ds.window(seq_len + 1, shift=stride, drop_remainder=True)
    ds = ds.flat_map(lambda x, y: tf.data.Dataset.zip((x.batch(seq_len + 1), y.batch(seq_len + 1))))
    ds = ds.map(lambda x, y: (tf.cast(x[:-1], tf.float32), tf.cast(y[-1], tf.float32)))
    if shuffle:
        ds = ds.shuffle(2048)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


class TemporalBlock(tf.keras.layers.Layer):
    def __init__(self, filters: int, kernel_size: int, dilation_rate: int, dropout: float, use_depthwise: bool = False):
        super().__init__()
        if use_depthwise:
            self.conv1 = tf.keras.layers.SeparableConv1D(filters, kernel_size, padding="causal", dilation_rate=dilation_rate)
            self.conv2 = tf.keras.layers.SeparableConv1D(filters, kernel_size, padding="causal", dilation_rate=dilation_rate)
        else:
            self.conv1 = tf.keras.layers.Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation_rate)
            self.conv2 = tf.keras.layers.Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation_rate)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.act1 = tf.keras.layers.ReLU()
        self.do1 = tf.keras.layers.Dropout(dropout)

        self.bn2 = tf.keras.layers.BatchNormalization()
        self.act2 = tf.keras.layers.ReLU()
        self.do2 = tf.keras.layers.Dropout(dropout)

        self.downsample = None
        self.add = tf.keras.layers.Add()
        self.out_act = tf.keras.layers.ReLU()

    def build(self, input_shape):
        if input_shape[-1] != self.conv1.filters:
            self.downsample = tf.keras.layers.Conv1D(self.conv1.filters, 1, padding="same")
        super().build(input_shape)

    def call(self, x, training=False):
        y = self.conv1(x)
        y = self.bn1(y, training=training)
        y = self.act1(y)
        y = self.do1(y, training=training)

        y = self.conv2(y)
        y = self.bn2(y, training=training)
        y = self.act2(y)
        y = self.do2(y, training=training)

        res = x if self.downsample is None else self.downsample(x)
        return self.out_act(self.add([y, res]))


class SqueezeExcite(tf.keras.layers.Layer):
    def __init__(self, reduction: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction

    def build(self, input_shape):
        channels = input_shape[-1]
        reduced = max(1, channels // self.reduction)
        self.pool = tf.keras.layers.GlobalAveragePooling1D()
        self.fc1 = tf.keras.layers.Dense(reduced, activation="relu")
        self.fc2 = tf.keras.layers.Dense(channels, activation="sigmoid")
        super().build(input_shape)

    def call(self, x):
        se = self.pool(x)
        se = self.fc1(se)
        se = self.fc2(se)
        se = tf.expand_dims(se, axis=1)
        return x * se


class DropPath(tf.keras.layers.Layer):
    def __init__(self, drop_prob: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x, training=False):
        if (not training) or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        random_tensor = keep_prob + tf.random.uniform(tf.shape(x)[:1], dtype=x.dtype)
        binary_tensor = tf.floor(random_tensor)
        return x / keep_prob * tf.reshape(binary_tensor, [-1, 1, 1])


class LayerScale(tf.keras.layers.Layer):
    def __init__(self, dim: int, init: float = 1e-4):
        super().__init__()
        self.dim = dim
        self.init = init

    def build(self, input_shape):
        self.gamma = self.add_weight("gamma", shape=[self.dim], initializer=tf.keras.initializers.Constant(self.init), trainable=True)
        super().build(input_shape)

    def call(self, x):
        return x * self.gamma


class CLSToken(tf.keras.layers.Layer):
    def __init__(self, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model

    def build(self, input_shape):
        self.token = self.add_weight("cls", shape=(1, 1, self.d_model), initializer="zeros", trainable=True)
        super().build(input_shape)

    def call(self, x):
        batch_size = tf.shape(x)[0]
        cls = tf.tile(self.token, [batch_size, 1, 1])
        return tf.concat([cls, x], axis=1)


class AttentionPooling(tf.keras.layers.Layer):
    def __init__(self, d_model: int):
        super().__init__()
        self.query = tf.keras.layers.Dense(d_model)
        self.score = tf.keras.layers.Dense(1)

    def call(self, x):
        q = self.query(tf.reduce_mean(x, axis=1, keepdims=True))
        scores = self.score(tf.nn.tanh(x + q))
        weights = tf.nn.softmax(scores, axis=1)
        return tf.reduce_sum(weights * x, axis=1)


def positional_encoding(seq_len: int, d_model: int) -> tf.Tensor:
    positions = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
    dims = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
    angle_rates = 1 / tf.pow(10000.0, (2 * (dims // 2)) / tf.cast(d_model, tf.float32))
    angle_rads = positions * angle_rates
    sin = tf.sin(angle_rads[:, 0::2])
    cos = tf.cos(angle_rads[:, 1::2])
    pos = tf.reshape(tf.stack([sin, cos], axis=-1), (seq_len, d_model))
    return pos


def build_sequence_model(
    seq_len: int,
    n_features: int,
    d_model: int = 256,
    tcn_channels: Optional[List[int]] = None,
    n_heads: int = 4,
    num_transformer_layers: int = 2,
    dropout: float = 0.1,
    mlp_hidden: int = 128,
    use_positional_encoding: bool = True,
    feature_dropout: float = 0.0,
    use_glu: bool = True,
    gmlp_dim: Optional[int] = None,
    directional_head: bool = False,
    pooling: str = "last",
    use_se: bool = False,
    se_reduction: int = 4,
    drop_path_rate: float = 0.0,
    use_layerscale: bool = False,
    use_depthwise: bool = False,
    uncertainty_head: bool = False,
) -> tf.keras.Model:
    if tcn_channels is None:
        tcn_channels = [d_model // 2, d_model]

    inputs = tf.keras.Input(shape=(seq_len, n_features), dtype=tf.float32)
    x = inputs
    if feature_dropout > 0:
        x = tf.keras.layers.Dropout(feature_dropout)(x)
    x = tf.keras.layers.Dense(d_model)(x)

    if use_positional_encoding:
        pos_enc = positional_encoding(seq_len, d_model)
        x = x + pos_enc

    if pooling == "cls":
        x = CLSToken(d_model)(x)

    dilation = 1
    for ch in tcn_channels:
        x = TemporalBlock(filters=ch, kernel_size=3, dilation_rate=dilation, dropout=dropout, use_depthwise=use_depthwise)(x)
        dilation *= 2
    if use_se:
        x = SqueezeExcite(reduction=se_reduction)(x)

    drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else None

    for _ in range(num_transformer_layers):
        attn = tf.keras.layers.MultiHeadAttention(num_heads=n_heads, key_dim=d_model)(x, x)
        attn = tf.keras.layers.Dropout(dropout)(attn)
        if use_layerscale:
            attn = LayerScale(d_model)(attn)
        attn_out = x + (drop_path(attn) if drop_path else attn)
        x = tf.keras.layers.LayerNormalization()(attn_out)

        if gmlp_dim:
            ff1 = tf.keras.layers.Dense(gmlp_dim, activation="gelu")(x)
            gate = tf.keras.layers.Dense(gmlp_dim, activation="sigmoid")(x) if use_glu else None
            if gate is not None:
                ff1 = ff1 * gate
            ff1 = tf.keras.layers.Dropout(dropout)(ff1)
            ff1 = tf.keras.layers.Dense(d_model)(ff1)
            if use_layerscale:
                ff1 = LayerScale(d_model)(ff1)
            ff_out = x + (drop_path(ff1) if drop_path else ff1)
            x = tf.keras.layers.LayerNormalization()(ff_out)
        else:
            ff = tf.keras.layers.Dense(d_model * 2, activation="relu")(x)
            ff = tf.keras.layers.Dropout(dropout)(ff)
            ff = tf.keras.layers.Dense(d_model)(ff)
            if use_layerscale:
                ff = LayerScale(d_model)(ff)
            ff_out = x + (drop_path(ff) if drop_path else ff)
            x = tf.keras.layers.LayerNormalization()(ff_out)

    if pooling == "multi":
        last = tf.keras.layers.Lambda(lambda t: t[:, -1, :])(x)
        mean_pool = tf.keras.layers.Lambda(lambda t: tf.reduce_mean(t, axis=1))(x)
        max_pool = tf.keras.layers.Lambda(lambda t: tf.reduce_max(t, axis=1))(x)
        pooled = tf.keras.layers.Concatenate()([last, mean_pool, max_pool])
        pooled = tf.keras.layers.Dense(d_model, activation="relu")(pooled)
    elif pooling == "attn":
        pooled = AttentionPooling(d_model)(x)
    elif pooling == "mean":
        pooled = tf.keras.layers.Lambda(lambda t: tf.reduce_mean(t, axis=1))(x)
    elif pooling == "cls":
        pooled = tf.keras.layers.Lambda(lambda t: t[:, 0, :])(x)
    else:
        pooled = tf.keras.layers.Lambda(lambda t: t[:, -1, :])(x)

    h = tf.keras.layers.Dense(mlp_hidden, activation="relu")(pooled)
    h = tf.keras.layers.Dropout(dropout)(h)
    price_out = tf.keras.layers.Dense(1, name="price")(h)

    outputs = [price_out]
    if uncertainty_head:
        log_var = tf.keras.layers.Dense(1, name="log_var")(h)
        outputs = [price_out, log_var]
    if directional_head:
        dir_logit = tf.keras.layers.Dense(1, name="direction")(h)
        outputs.append(dir_logit)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    if directional_head and not uncertainty_head:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
            loss={"price": tf.keras.losses.Huber(), "direction": tf.keras.losses.BinaryCrossentropy(from_logits=True)},
            loss_weights={"price": 1.0, "direction": 0.2},
        )
    elif not directional_head and not uncertainty_head:
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4), loss=tf.keras.losses.Huber())
    return model


def build_patchtst_model(
    seq_len: int,
    n_features: int,
    d_model: int = 256,
    patch_size: int = 16,
    patch_stride: int = 16,
    n_heads: int = 4,
    num_transformer_layers: int = 2,
    dropout: float = 0.1,
    mlp_hidden: int = 128,
    directional_head: bool = False,
) -> tf.keras.Model:
    """
    PatchTST-style model: Conv1D patching -> Transformer over patches.
    """
    inputs = tf.keras.Input(shape=(seq_len, n_features), dtype=tf.float32)

    # Patch embedding
    x = tf.keras.layers.Conv1D(
        filters=d_model,
        kernel_size=patch_size,
        strides=patch_stride,
        padding="same",
        name="patch_embed",
    )(inputs)

    # Positional embedding (trainable)
    num_patches = int(math.ceil(seq_len / patch_stride))
    pos = tf.keras.layers.Embedding(input_dim=num_patches, output_dim=d_model, name="patch_pos")
    positions = tf.range(num_patches)
    x = x + pos(positions)

    for _ in range(num_transformer_layers):
        h = tf.keras.layers.LayerNormalization(epsilon=1e-5)(x)
        h = tf.keras.layers.MultiHeadAttention(num_heads=n_heads, key_dim=max(d_model // n_heads, 8), dropout=dropout)(h, h)
        h = tf.keras.layers.Dropout(dropout)(h)
        x = tf.keras.layers.Add()([x, h])

        h = tf.keras.layers.LayerNormalization(epsilon=1e-5)(x)
        h = tf.keras.layers.Dense(d_model * 4, activation="gelu")(h)
        h = tf.keras.layers.Dropout(dropout)(h)
        h = tf.keras.layers.Dense(d_model)(h)
        h = tf.keras.layers.Dropout(dropout)(h)
        x = tf.keras.layers.Add()([x, h])

    pooled = tf.keras.layers.GlobalAveragePooling1D()(x)
    h = tf.keras.layers.Dense(mlp_hidden, activation="relu")(pooled)
    h = tf.keras.layers.Dropout(dropout)(h)
    price_out = tf.keras.layers.Dense(1, name="price")(h)

    outputs = [price_out]
    if directional_head:
        dir_logit = tf.keras.layers.Dense(1, name="direction")(h)
        outputs.append(dir_logit)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="patchtst")
    if directional_head:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
            loss={"price": tf.keras.losses.Huber(), "direction": tf.keras.losses.BinaryCrossentropy(from_logits=True)},
            loss_weights={"price": 1.0, "direction": 0.2},
        )
    else:
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4), loss=tf.keras.losses.Huber())
    return model


class OnlineNormalizer:
    def __init__(self, n_features: int, eps: float = 1e-6):
        self.eps = eps
        self.count = 0
        self.mean = np.zeros(n_features, dtype=np.float64)
        self.m2 = np.zeros(n_features, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        if x.ndim == 1:
            x = x[np.newaxis, :]
        for row in x:
            self.count += 1
            delta = row - self.mean
            self.mean += delta / self.count
            delta2 = row - self.mean
            self.m2 += delta * delta2

    def stats(self):
        var = self.m2 / max(self.count - 1, 1)
        std = np.sqrt(var + self.eps)
        return self.mean, std

    def normalize(self, x: np.ndarray) -> np.ndarray:
        mu, std = self.stats()
        return (x - mu) / std


class RealTimePredictor:
    def __init__(self, model: tf.keras.Model, normalizer: OnlineNormalizer, feature_cols: List[str], seq_len: int):
        from collections import deque
        self.model = model
        self.normalizer = normalizer
        self.feature_cols = feature_cols
        self.seq_len = seq_len
        self.buffer = deque(maxlen=seq_len)

    def warm_start(self, rows):
        for row in rows:
            vec = np.array([row[c] for c in self.feature_cols], dtype=np.float32)
            self.normalizer.update(vec)
            normed = self.normalizer.normalize(vec)
            self.buffer.append(normed)

    def step(self, row):
        vec = np.array([row[c] for c in self.feature_cols], dtype=np.float32)
        self.normalizer.update(vec)
        normed = self.normalizer.normalize(vec)
        self.buffer.append(normed)
        if len(self.buffer) < self.seq_len:
            return None
        x = np.stack(self.buffer, axis=0)[np.newaxis, ...]
        pred = self.model(x, training=False)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        return float(np.asarray(pred).squeeze())
