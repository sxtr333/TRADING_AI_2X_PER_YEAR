#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/vitamind/my_project/model6"
PY="$ROOT/.venv/bin/python"
FEATURES="$ROOT/data/BTCUSDT_15m_features_h20_v2_news_xlmr_full_2023_2026.parquet"
META="$ROOT/data/meta/meta_dataset_pruned_newsflag_mix15_2024calib_full.parquet"
META_DIR="$ROOT/new_models/meta_2026-01-24_newsflag_mix15_2024calib_full"
BEST_H20="$ROOT/reports/backtest_v7_long_short_sweep_newsflag_mix15_2024_best.csv"
BEST_V2="$ROOT/reports/backtest_v7_long_short_v2_sweep_newsflag_mix15_2024_best.csv"

export TF_XLA_FLAGS=--tf_xla_auto_jit=0
export TF_GPU_ALLOCATOR=cuda_malloc_async
export CUDA_VISIBLE_DEVICES=""

declare -A STARTS
declare -A ENDS
STARTS[2023]="2023-01-01T00:00:00+00:00"
ENDS[2023]="2024-01-01T00:00:00+00:00"
STARTS[2024]="2024-01-01T00:00:00+00:00"
ENDS[2024]="2025-01-01T00:00:00+00:00"
STARTS[2025]="2025-01-01T00:00:00+00:00"
ENDS[2025]="2026-01-01T00:00:00+00:00"

for year in 2023 2024 2025; do
  for lev in 3 5 10; do
    out="$ROOT/reports/backtest_meta_newsflag_mix15_2024calib_full_${year}_thr059_lev${lev}.csv"
    log="$ROOT/logs/backtest_meta_newsflag_mix15_2024calib_full_${year}_thr059_lev${lev}.log"
    $PY "$ROOT/scripts/backtest_trade_combo_meta.py" \
      --features "$FEATURES" \
      --meta-features "$META" \
      --meta-model-dir "$META_DIR" \
      --best-h20 "$BEST_H20" \
      --best-v2 "$BEST_V2" \
      --meta-prob-thr 0.59 \
      --leverage "$lev" \
      --start "${STARTS[$year]}" --end "${ENDS[$year]}" \
      --batch-size 16 \
      --out-csv "$out" \
      > "$log" 2>&1
  done
done

echo "Done."
