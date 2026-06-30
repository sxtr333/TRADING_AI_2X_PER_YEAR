#!/usr/bin/env bash
set -euo pipefail

FEATURES=/home/vitamind/my_project/model6/data/BTCUSDT_15m_features_h20_v2_news_xlmr_v3.parquet
MODEL=/home/vitamind/my_project/model6/new_models/2026-01-14_news_xlmr_v5_dr_scale/model_15m_itransformer_price_multi_h20_h80_h160_news_xlmr_v5_dr_scale_e14_b32.keras
STATS=/home/vitamind/my_project/model6/new_models/2026-01-14_news_xlmr_v5_dr_scale/norm_stats_15m_itransformer_price_multi_h20_h80_h160_news_xlmr_v5_dr_scale_e14_b32.npz
LOG_DIR=/home/vitamind/my_project/model6/logs

mkdir -p "$LOG_DIR"

# Baseline eval (no bias shift)
/home/vitamind/my_project/model6/.venv/bin/python \
  /home/vitamind/my_project/model6/scripts/eval_price_quality.py \
  --features "$FEATURES" \
  --model "$MODEL" \
  --stats "$STATS" \
  --horizons 20,80,160 \
  --batch-size 256 \
  --line-metrics \
  --start 2025-07-08T21:15:00+00:00 \
  2>&1 | tee "$LOG_DIR/eval_news_xlmr_v5_dr_scale_line.log"

# Bias-shifted eval (median error on val window)
/home/vitamind/my_project/model6/.venv/bin/python \
  /home/vitamind/my_project/model6/scripts/eval_price_quality.py \
  --features "$FEATURES" \
  --model "$MODEL" \
  --stats "$STATS" \
  --horizons 20,80,160 \
  --batch-size 256 \
  --line-metrics \
  --start 2025-07-08T21:15:00+00:00 \
  --bias-shift val \
  2>&1 | tee "$LOG_DIR/eval_news_xlmr_v5_dr_scale_line_bias_shift.log"
