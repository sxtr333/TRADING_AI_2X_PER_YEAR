#!/usr/bin/env bash
set -euo pipefail
PY=/home/vitamind/my_project/model6/.venv/bin/python
FEATURES=/home/vitamind/my_project/model6/data/BTCUSDT_15m_features_h20_v2_news_xlmr_full_2020_2026_dailyfill.parquet
META=/home/vitamind/my_project/model6/data/meta/meta_dataset_pruned_newsflag_mix15_2024calib_2026-01-29.parquet
META_DIR=/home/vitamind/my_project/model6/new_models/meta_2026-01-29_newsflag_mix15_2024calib
BEST_H20=/home/vitamind/my_project/model6/reports/backtest_v7_long_short_sweep_newsflag_mix15_2024_best_2026-01-29.csv
BEST_V2=/home/vitamind/my_project/model6/reports/backtest_v7_long_short_v2_sweep_newsflag_mix15_2024_best_2026-01-29.csv
REPORTS=/home/vitamind/my_project/model6/reports
LOGS=/home/vitamind/my_project/model6/logs

thr_list=(0.50 0.53 0.55 0.57 0.59 0.61 0.63 0.65 0.67)
frac=0.30

for thr in "${thr_list[@]}"; do
  label="thr${thr}_frac${frac}"
  out="$REPORTS/backtest_meta_sweep_3x_${label}_2025.csv"
  log="$LOGS/backtest_meta_sweep_3x_${label}_2025.log"
  if [ -s "$out" ]; then
    echo "[skip] $label"
    continue
  fi
  echo "[run] $label"
  TF_FORCE_GPU_ALLOW_GROWTH=1 TF_ENABLE_ONEDNN_OPTS=0 \
  $PY /home/vitamind/my_project/model6/scripts/backtest_trade_combo_meta.py \
    --features "$FEATURES" \
    --meta-features "$META" \
    --meta-model-dir "$META_DIR" \
    --best-h20 "$BEST_H20" \
    --best-v2 "$BEST_V2" \
    --start 2025-01-01T00:00:00+00:00 --end 2026-01-01T00:00:00+00:00 \
    --meta-prob-thr "$thr" \
    --trade-frac "$frac" --max-concurrent 1 \
    --leverage 3 \
    --out-csv "$out" \
    > "$log" 2>&1
  sleep 5
 done
