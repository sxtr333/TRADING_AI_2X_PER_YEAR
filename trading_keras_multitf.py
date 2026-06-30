"""
Alternative model with explicit multi-timeframe fusion:
- Base branch for 1m (seq_len_base, feat_base)
- Higher-TF branch (seq_len_ctx, feat_ctx) processed via TCN+Transformer
- Cross-attention from base to context, then pooled
Usage: build_multitf_model(...) and feed two inputs.
"""

from __future__ import annotations

import tensorflow as tf
from typing import List, Optional

from trading_keras_core import (
    TemporalBlock,
    CLSToken,
    AttentionPooling,
    DropPath,
    LayerScale,
    SqueezeExcite,
    positional_encoding,
)


def encoder_stack(x, d_model, n_heads, layers, dropout, gmlp_dim=None, use_glu=True, use_layerscale=False, drop_path_rate=0.0):
    drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else None
    for _ in range(layers):
        attn = tf.keras.layers.MultiHeadAttention(num_heads=n_heads, key_dim=d_model)(x, x)
        attn = tf.keras.layers.Dropout(dropout)(attn)
        if use_layerscale:
            attn = LayerScale(d_model)(attn)
        x = tf.keras.layers.LayerNormalization()(x + (drop_path(attn) if drop_path else attn))
        if gmlp_dim:
            ff = tf.keras.layers.Dense(gmlp_dim, activation="gelu")(x)
            if use_glu:
                gate = tf.keras.layers.Dense(gmlp_dim, activation="sigmoid")(x)
                ff = ff * gate
            ff = tf.keras.layers.Dropout(dropout)(ff)
            ff = tf.keras.layers.Dense(d_model)(ff)
        else:
            ff = tf.keras.layers.Dense(d_model * 2, activation="relu")(x)
            ff = tf.keras.layers.Dropout(dropout)(ff)
            ff = tf.keras.layers.Dense(d_model)(ff)
        if use_layerscale:
            ff = LayerScale(d_model)(ff)
        x = tf.keras.layers.LayerNormalization()(x + (drop_path(ff) if drop_path else ff))
    return x


def build_multitf_model(
    seq_len_base: int,
    n_features_base: int,
    seq_len_ctx: int,
    n_features_ctx: int,
    d_model: int = 256,
    dropout: float = 0.1,
    n_heads: int = 4,
    layers_base: int = 2,
    layers_ctx: int = 2,
    gmlp_dim: Optional[int] = None,
    use_glu: bool = True,
    pooling: str = "attn",
) -> tf.keras.Model:
    inp_base = tf.keras.Input(shape=(seq_len_base, n_features_base), name="base")
    inp_ctx = tf.keras.Input(shape=(seq_len_ctx, n_features_ctx), name="ctx")

    x_base = tf.keras.layers.Dense(d_model)(inp_base)
    x_ctx = tf.keras.layers.Dense(d_model)(inp_ctx)
    x_base = x_base + positional_encoding(seq_len_base, d_model)
    x_ctx = x_ctx + positional_encoding(seq_len_ctx, d_model)

    # Context encoder
    x_ctx = encoder_stack(x_ctx, d_model, n_heads, layers_ctx, dropout, gmlp_dim=gmlp_dim, use_glu=use_glu)

    # Cross-attention: base attends to context
    cross = tf.keras.layers.MultiHeadAttention(num_heads=n_heads, key_dim=d_model)(x_base, x_ctx)
    x = tf.keras.layers.LayerNormalization()(x_base + cross)
    x = encoder_stack(x, d_model, n_heads, layers_base, dropout, gmlp_dim=gmlp_dim, use_glu=use_glu)

    if pooling == "attn":
        pooled = AttentionPooling(d_model)(x)
    elif pooling == "mean":
        pooled = tf.keras.layers.GlobalAveragePooling1D()(x)
    else:
        pooled = x[:, -1, :]

    h = tf.keras.layers.Dense(d_model, activation="relu")(pooled)
    out = tf.keras.layers.Dense(1, name="price")(h)

    model = tf.keras.Model(inputs=[inp_base, inp_ctx], outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4), loss=tf.keras.losses.Huber())
    return model
