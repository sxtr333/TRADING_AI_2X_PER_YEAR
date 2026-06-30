#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/vitamind/my_project/model6"
PY="$ROOT/.venv/bin/python"
SCRIPT="$ROOT/scripts/backtest_trade_combo_meta.py"
FEATURES="$ROOT/data/BTCUSDT_15m_features_h20_v2_news_xlmr_full_2020_2026_dailyfill.parquet"
META="$ROOT/data/meta/meta_dataset_pruned_newsflag_mix15_2024calib_2026-01-29.parquet"
META_DIR="$ROOT/new_models/meta_2026-01-29_newsflag_mix15_2024calib"
BEST_H20="$ROOT/reports/backtest_v7_long_short_sweep_newsflag_mix15_2024_best_2026-01-29.csv"
BEST_V2="$ROOT/reports/backtest_v7_long_short_v2_sweep_newsflag_mix15_2024_best_2026-01-29.csv"

mkdir -p "$ROOT/reports" "$ROOT/logs"

run_year () {
  local start="$1"
  local end="$2"
  local label="$3"
  local out_csv="$ROOT/reports/backtest_meta_atr3x_${label}.csv"
  local out_trades="$ROOT/reports/trades_meta_atr3x_${label}.csv"

  $PY "$SCRIPT" \
    --features "$FEATURES" \
    --meta-features "$META" \
    --meta-model-dir "$META_DIR" \
    --best-h20 "$BEST_H20" \
    --best-v2 "$BEST_V2" \
    --start "$start" \
    --end "$end" \
    --meta-prob-thr 0.50 \
    --trade-frac 0.30 \
    --leverage 3 \
    --exit-mode atr \
    --exit-atr-mult 3.0 \
    --exit-max-hold-mult 4.0 \
    --out-csv "$out_csv" \
    --out-trades-csv "$out_trades"
}

run_year "2020-01-01T00:00:00+00:00" "2021-01-01T00:00:00+00:00" "2020"
run_year "2021-01-01T00:00:00+00:00" "2022-01-01T00:00:00+00:00" "2021"
run_year "2022-01-01T00:00:00+00:00" "2023-01-01T00:00:00+00:00" "2022"
run_year "2023-01-01T00:00:00+00:00" "2024-01-01T00:00:00+00:00" "2023"
run_year "2024-01-01T00:00:00+00:00" "2025-01-01T00:00:00+00:00" "2024"
run_year "2025-01-01T00:00:00+00:00" "2026-01-01T00:00:00+00:00" "2025"

echo "Done."
