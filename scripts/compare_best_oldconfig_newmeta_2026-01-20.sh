#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/vitamind/my_project/model6"
OUT="$ROOT/reports/backtest_trade_combo_meta_best_oldconfig_newmeta_2026-01-20.csv"

. "$ROOT/.venv/bin/activate"

python "$ROOT/scripts/backtest_trade_combo_meta.py" \
  --features "$ROOT/data/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes_with_news_pruned.parquet" \
  --meta-features "$ROOT/data/meta/meta_dataset_pruned.parquet" \
  --start "2025-01-01T00:00:00+00:00" \
  --end "2026-01-01T00:00:00+00:00" \
  --meta-model-dir "$ROOT/new_models/meta_2026-01-20" \
  --cost-rt 0.0015 \
  --trade-frac 0.55 \
  --cooldown-steps 32 \
  --max-concurrent 3 \
  --threshold-bump-sigma 0.4 \
  --meta-prob-per-model "h20_long=0.8,h20_short=0.75,h80_short_v2=0.9,h160_long_v2=0.45" \
  --out-csv "$OUT"

echo "Wrote: $OUT"
