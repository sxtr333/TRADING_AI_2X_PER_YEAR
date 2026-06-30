#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf


@tf.keras.utils.register_keras_serializable("model6")
class LastStep(tf.keras.layers.Layer):
    def call(self, x):
        return x[:, -1, :]


@tf.keras.utils.register_keras_serializable("model6")
class RevIN(tf.keras.layers.Layer):
    """
    Reversible Instance Normalization (RevIN).
    Normalizes each sample per feature across the time axis.
    """
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


@tf.keras.utils.register_keras_serializable("model6")
class DropPath(tf.keras.layers.Layer):
    """Stochastic depth for residual paths."""
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


@tf.keras.utils.register_keras_serializable("model6")
class TSMixerBlock(tf.keras.layers.Layer):
    def __init__(self, mlp_dim: int, dropout: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.mlp_dim = int(mlp_dim)
        self.dropout = float(dropout)
        self.ln_time = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.ln_feat = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.do = tf.keras.layers.Dropout(self.dropout)

    def build(self, input_shape):
        seq_len = int(input_shape[1])
        feat_dim = int(input_shape[2])
        self.time_dense = tf.keras.layers.Dense(seq_len)
        self.ff1 = tf.keras.layers.Dense(self.mlp_dim, activation="gelu")
        self.ff2 = tf.keras.layers.Dense(feat_dim)
        super().build(input_shape)

    def call(self, x, training=False):
        # time mixing
        y = self.ln_time(x)
        y = tf.transpose(y, [0, 2, 1])  # (B, F, T)
        y = self.time_dense(y)
        y = tf.transpose(y, [0, 2, 1])
        y = self.do(y, training=training)
        x = x + y

        # feature mixing
        y = self.ln_feat(x)
        y = self.ff1(y)
        y = self.do(y, training=training)
        y = self.ff2(y)
        y = self.do(y, training=training)
        return x + y

    def get_config(self):
        return {"mlp_dim": self.mlp_dim, "dropout": self.dropout}


@tf.keras.utils.register_keras_serializable("model6")
class ITransformerBlock(tf.keras.layers.Layer):
    """
    Inverted Transformer block: attend over variates (features as tokens).
    """
    def __init__(self, seq_len: int, d_model: int, heads: int, dropout: float = 0.0, drop_path: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = int(seq_len)
        self.d_model = int(d_model)
        self.heads = int(heads)
        self.dropout = float(dropout)
        self.drop_path = float(drop_path)
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)

    def build(self, input_shape):
        key_dim = max(self.d_model // max(self.heads, 1), 8)
        self.proj = tf.keras.layers.Dense(self.d_model)
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=self.heads, key_dim=key_dim, dropout=self.dropout
        )
        self.ff1 = tf.keras.layers.Dense(self.d_model * 4, activation="gelu")
        self.ff2 = tf.keras.layers.Dense(self.d_model)
        self.to_time = tf.keras.layers.Dense(self.seq_len)
        self.do = tf.keras.layers.Dropout(self.dropout)
        self.dp = DropPath(self.drop_path)
        super().build(input_shape)

    def call(self, x, training=False):
        # x: (B, T, F) -> (B, F, T)
        xt = tf.transpose(x, [0, 2, 1])
        h = self.proj(xt)
        h1 = self.ln1(h)
        h1 = self.mha(h1, h1)
        h1 = self.do(h1, training=training)
        h1 = self.dp(h1, training=training)
        h = h + h1

        h2 = self.ln2(h)
        h2 = self.ff1(h2)
        h2 = self.do(h2, training=training)
        h2 = self.ff2(h2)
        h2 = self.do(h2, training=training)
        h2 = self.dp(h2, training=training)
        h = h + h2

        # project back to time length and restore original shape
        yt = self.to_time(h)
        y = tf.transpose(yt, [0, 2, 1])
        return x + y

    def get_config(self):
        return {
            "seq_len": self.seq_len,
            "d_model": self.d_model,
            "heads": self.heads,
            "dropout": self.dropout,
            "drop_path": self.drop_path,
        }
