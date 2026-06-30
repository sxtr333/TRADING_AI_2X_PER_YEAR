#!/usr/bin/env bash
set -euo pipefail
BASE_FEATURES="/mnt/oldssd/model6/data/BTCUSDT_15m_features_h20_v2_news_xlmr_full_2020_2026_dailyfill.parquet"
META_FEATURES="/home/vitamind/my_project/model6/data/meta/meta_dataset_pruned_newsflag_mix15_2024calib_2026-01-29.parquet"
META_DIR="/home/vitamind/my_project/model6/new_models/meta_2026-01-29_newsflag_mix15_2024calib"
OUT_DIR="/home/vitamind/my_project/model6/reports"
LOG_DIR="/home/vitamind/my_project/model6/logs"
PY="/home/vitamind/my_project/model6/.venv/bin/python"
SCRIPT="/home/vitamind/my_project/model6/scripts/backtest_trade_combo_meta.py"

run_one() {
  local label="$1" start="$2" end="$3"
  echo "== ${label} =="
  $PY $SCRIPT \
    --features "$BASE_FEATURES" \
    --meta-features "$META_FEATURES" \
    --start "$start" --end "$end" \
    --meta-model-dir "$META_DIR" \
    --meta-prob-thr 0.50 --trade-frac 0.33 --leverage 5 \
    --exit-mode atr --exit-atr-mult 3.0 --exit-max-hold-mult 4.0 \
    --out-csv "$OUT_DIR/bt_${label}_exit_atr_trade33_5x.csv" \
    > "$LOG_DIR/bt_${label}_exit_atr_trade33_5x.log" 2>&1
}

run_one 2020Q4 2020-10-30T00:00:00+00:00 2021-01-01T00:00:00+00:00
run_one 2021   2021-01-01T00:00:00+00:00 2022-01-01T00:00:00+00:00
run_one 2022   2022-01-01T00:00:00+00:00 2023-01-01T00:00:00+00:00
run_one 2023   2023-01-01T00:00:00+00:00 2024-01-01T00:00:00+00:00
run_one 2024   2024-01-01T00:00:00+00:00 2025-01-01T00:00:00+00:00
run_one 2025   2025-01-01T00:00:00+00:00 2026-01-01T00:00:00+00:00
