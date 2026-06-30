#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/vitamind/my_project/model6"
PY="$ROOT/.venv/bin/python"
EVAL="$ROOT/scripts/eval_price_quality_v7.py"
LOG="$ROOT/logs"

for H in 80 160; do
  for DIR in long short; do
    MODEL_DIR="$ROOT/new_models/2026-01-18_v7_h${H}_${DIR}_v2"
    MODEL="$MODEL_DIR/model_15m_itransformer_v7_h${H}_${DIR}_v2.keras"
    STATS="$MODEL_DIR/norm_stats_v7_h${H}_${DIR}_v2.npz"

    OUT_LOG="$LOG/eval_v7_h${H}_${DIR}_v2_2026-01-18.log"

    $PY "$EVAL" \
      --features "$ROOT/data/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes_with_news_pruned.parquet" \
      --model "$MODEL" \
      --stats "$STATS" \
      --horizons "$H" \
      --batch-size 256 \
      --line-metrics \
      2>&1 | tee "$OUT_LOG"
  done
 done

