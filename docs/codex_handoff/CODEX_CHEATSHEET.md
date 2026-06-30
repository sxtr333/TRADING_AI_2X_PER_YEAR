# Codex quick cheat‑sheet (model6)

## 1) Realtime update (every N minutes)

```bash
LIVE_UPDATE_INTERVAL_SEC=60 bash /home/vitamind/my_project/model6/scripts/live_update_loop.sh
```

Single cycle:

```bash
bash /home/vitamind/my_project/model6/scripts/live_update_once.sh
```

## 2) Start site + APIs

```bash
bash /home/vitamind/my_project/model6/scripts/run_servers.sh
```

- Main UI: http://localhost:8080/aladin_from_image.html
- Arb UI:  http://localhost:8090/arbitration.html
- Arb API: http://localhost:8091

## 3) GPU‑safe training (host)

```bash
scripts/run_train_gpu_cpulimit70.sh -- bash /home/vitamind/my_project/model6/scripts/train_news_xlmr_cosmic_v6_dr_scale_8nodes.sh
```

Custom command:

```bash
scripts/run_train_gpu_cpulimit70.sh -- .venv/bin/python train_keras_v7.py --help
```

## 4) GPU‑safe training (docker)

```bash
scripts/run_train_gpu_cpulimit70.sh --docker tensorflow/tensorflow:nightly-gpu -- \
  bash /work/scripts/train_news_xlmr_cosmic_v6_dr_scale_8nodes.sh
```

## 5) Build meta dataset

```bash
.venv/bin/python scripts/build_meta_dataset.py \
  --features data/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes_with_news_pruned.parquet \
  --output data/meta/meta_dataset_pruned.parquet \
  --cost-bps 15.0
```

## 6) Train 4 meta models (cls‑only)

```bash
mkdir -p new_models/meta_YYYY-MM-DD logs

CUDA_VISIBLE_DEVICES=0 .venv/bin/python train_keras_v7.py \
  --features data/meta/meta_dataset_pruned.parquet \
  --seq-len 256 --batch-size 128 --epochs 10 --lr 3e-4 \
  --arch itransformer --d-model 128 --var-layers 2 --time-layers 2 \
  --dropout 0.1 --drop-path 0.05 --time-pos learned --pos-dropout 0.02 \
  --no-price-head --cls-weight 1.0 --num-classes 1 \
  --target-col meta_label_h20_long \
  --model-out new_models/meta_YYYY-MM-DD/meta_h20_long.keras \
  --stats-out new_models/meta_YYYY-MM-DD/meta_h20_long_stats.npz \
  2>&1 | tee logs/meta_h20_long_train.log
```

(Repeat for `meta_label_h20_short`, `meta_label_h80_short_v2`, `meta_label_h160_long_v2`.)

## 7) Meta backtest sweep (2025)

```bash
.venv/bin/python scripts/backtest_trade_combo_meta.py \
  --features data/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes_with_news_pruned.parquet \
  --meta-features data/meta/meta_dataset_pruned.parquet \
  --start 2025-01-01T00:00:00+00:00 \
  --end 2026-01-01T00:00:00+00:00 \
  --meta-model-dir new_models/meta_YYYY-MM-DD \
  --out-csv reports/backtest_trade_combo_meta_sweep_YYYY-MM-DD.csv
```

## 8) Best known results (as of 2026‑01‑20)

- Base combo grid best: `reports/backtest_trade_combo_grid.csv` → ~+0.7 PnL
- Meta baseline: `reports/best_result_meta_2026-01-19.txt` → +18.18
- Meta boost: `reports/best_result_meta_boost_2026-01-19.txt` → +26.05
- Best local tweak: `reports/backtest_trade_combo_meta_step1_h160_tradefrac.csv` → ~+29.9

## 9) Key datasets

- Features (new): `data/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes_with_news_pruned.parquet`
- Features (serve): `data/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes_serve.parquet`
- Features (old v5): `data/BTCUSDT_15m_features_h20_v2_news_xlmr_v3.parquet`
- Meta: `data/meta/meta_dataset_pruned.parquet`

