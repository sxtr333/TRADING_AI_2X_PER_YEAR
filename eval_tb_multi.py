import tensorflow as tf
import pandas as pd
import numpy as np

from train_keras import (
    make_window_dataset_multi,
    pick_feature_cols,
    compute_norm_stats,
    apply_norm,
    DropPath,
)
from model_layers import RevIN, ITransformerBlock

df = pd.read_parquet("data/BTCUSDT_15m_features_tb_multi.parquet")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values("timestamp").reset_index(drop=True)

train_end = pd.Timestamp("2024-01-01T00:00:00Z")
val_end   = pd.Timestamp("2025-01-01T00:00:00Z")

n_train = int(df.index[df["timestamp"] <= train_end].max()) + 1
val_end_idx = int(df.index[df["timestamp"] <= val_end].max()) + 1

seq_len = 256
hgap = 160
min_end = seq_len - 1

train_max = max(min_end, n_train - hgap)
val_max   = max(min_end, val_end_idx - hgap)
test_max  = max(min_end, len(df) - hgap)

purge = 20
val_start  = max(n_train + purge, min_end)
test_start = max(val_end_idx + purge, min_end)

test_ends = np.arange(test_start, test_max, dtype=np.int64)

feature_cols = pick_feature_cols(df)
X_raw = df[feature_cols].to_numpy(np.float32)
stats = compute_norm_stats(X_raw[:n_train], 0.001, 0.999)
X = apply_norm(X_raw, stats)

y_cls_dict = {}
for h in [20, 80, 160]:
    y_cls_dict[f"cls_h{h}"] = ((df[f"label_3cls_h{h}"].astype(int) == 1).astype(np.int32)).to_numpy()

y_price = df["target_amp_abs"].to_numpy(np.float32)

ds_test = make_window_dataset_multi(
    X, y_price, y_cls_dict, test_ends,
    seq_len=seq_len, batch_size=128,
    shuffle=False, sample_weight=None, price_key="price"
)

custom = {
    "DropPath": DropPath,
    "RevIN": RevIN,
    "ITransformerBlock": ITransformerBlock,
}

model = tf.keras.models.load_model(
    "model_15m_itransformer_tb_multi.keras",
    custom_objects=custom
)

res = model.evaluate(ds_test, return_dict=True, verbose=0)
for k in sorted(res.keys()):
    print(k, res[k])
