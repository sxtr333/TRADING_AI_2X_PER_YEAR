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

run_year () {
  local label="$1"; local start="$2"; local end="$3"; local lev="$4"
  local out="$REPORTS/backtest_meta_newsflag_mix15_thr059_allin_${label}_lev${lev}_2026-01-29.csv"
  local log="$LOGS/backtest_meta_thr059_allin_${label}_lev${lev}.log"
  $PY /home/vitamind/my_project/model6/scripts/backtest_trade_combo_meta.py \
    --features "$FEATURES" \
    --meta-features "$META" \
    --meta-model-dir "$META_DIR" \
    --best-h20 "$BEST_H20" \
    --best-v2 "$BEST_V2" \
    --start "$start" --end "$end" \
    --meta-prob-thr 0.59 \
    --trade-frac 1.0 \
    --max-concurrent 1 \
    --leverage "$lev" \
    --out-csv "$out" \
    > "$log" 2>&1
}

years=(
  "2020Q4|2020-10-30T00:00:00+00:00|2021-01-01T00:00:00+00:00"
  "2021|2021-01-01T00:00:00+00:00|2022-01-01T00:00:00+00:00"
  "2022|2022-01-01T00:00:00+00:00|2023-01-01T00:00:00+00:00"
  "2023|2023-01-01T00:00:00+00:00|2024-01-01T00:00:00+00:00"
  "2024|2024-01-01T00:00:00+00:00|2025-01-01T00:00:00+00:00"
  "2025|2025-01-01T00:00:00+00:00|2026-01-01T00:00:00+00:00"
)

for lev in 3 5; do
  for item in "${years[@]}"; do
    IFS='|' read -r label start end <<<"$item"
    echo "[run] $label lev=$lev"
    run_year "$label" "$start" "$end" "$lev"
  done
done
