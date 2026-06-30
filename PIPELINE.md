# Pipeline (Leakage‑safe) and Model Benchmark

This document shows the recommended end‑to‑end pipeline with a leakage‑safe split and an apples‑to‑apples comparison of **PatchTST vs TSMixer vs iTransformer** on the same dataset and split.

## 1) Build features with a correct target

We build:
- `target_ret` = log return over horizon
- `target_dir` = 1 if `target_ret > 0` else 0

Example (15m horizon = 20 steps = 5 hours):

```bash
source .venv/bin/activate
python build_features.py \
  --input data/BTCUSDT_15m.parquet \
  --aux data/BTCUSDT_1h_aux_merged.parquet \
  --horizon 20 --target-mode log_return --base-tf-min 15 \
  --output data/BTCUSDT_15m_features_h20_v2.parquet
```

## 2) Walk‑forward split (leakage‑safe)

We use explicit time boundaries:
- Train: <= 2024‑01‑01
- Val:   <= 2025‑01‑01
- Test:  >  2025‑01‑01

We also set `--target-horizon 20` to prevent any training sample whose label reaches into the val/test window.

## 3) Model comparison (same split, same features)

Use the helper script:

```bash
./scripts/benchmark_compare.sh
```

It runs PatchTST, TSMixer, iTransformer with identical settings (epochs, batch size, split).

## 4) Walk‑forward PnL evaluation (monthly)

Example (on the best model after benc h):

```bash
python walk_forward_eval.py \
  --features data/BTCUSDT_15m_features_h20_v2.parquet \
  --model model_bench_patchtst.keras \
  --stats norm_stats_bench_patchtst.npz \
  --train-end 2024-01-01T00:00:00Z \
  --val-end 2025-01-01T00:00:00Z \
  --purge-gap 20 --target-horizon 20 \
  --threshold 0.55 --fee-bps 2.0
```

---

## Notes on leakage
- `target_ret` uses `close[t + horizon]` and is a label only; it is **never** part of features.
- `--target-horizon` prevents train/val/test crossing with the label window.
- Normalization uses **train‑only** statistics.
