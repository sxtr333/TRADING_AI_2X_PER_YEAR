#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/new_models/2026-01-14_news_xlmr_v5_dr_scale"
mkdir -p "${OUT_DIR}"

FEATURES="${ROOT_DIR}/data/BTCUSDT_15m_features_h20_v2_news_xlmr_v3.parquet"
MODEL_OUT="${OUT_DIR}/model_15m_itransformer_price_multi_h20_h80_h160_news_xlmr_v5_dr_scale_e14_b32.keras"
STATS_OUT="${OUT_DIR}/norm_stats_15m_itransformer_price_multi_h20_h80_h160_news_xlmr_v5_dr_scale_e14_b32.npz"
LOG_OUT="${ROOT_DIR}/logs/train_news_xlmr_v5_dr_scale_multi.log"

echo "[train] multi h20/h80/h160 -> ${MODEL_OUT}"

python3 "${ROOT_DIR}/train_keras.py" \
  --features "${FEATURES}" \
  --seq-len 512 \
  --batch-size 32 \
  --epochs 14 \
  --lr 2e-4 \
  --arch itransformer \
  --d-model 192 \
  --heads 6 \
  --layers 3 \
  --var-layers 2 \
  --time-layers 2 \
  --pooling multi \
  --revin \
  --revin-affine \
  --feature-dropout 0.05 \
  --dropout 0.05 \
  --drop-path 0.05 \
  --cosine \
  --warmup-steps 500 \
  --price-weight 1.0 \
  --cls-weight 0.0 \
  --price-loss huber \
  --huber-delta 0.01 \
  --price-clip-q-low 0.001 \
  --price-clip-q-high 0.999 \
  --price-multi-horizons 20,80,160 \
  --price-segment-deltas \
  --price-head-scale std \
  --model-out "${MODEL_OUT}" \
  --stats-out "${STATS_OUT}" \
  2>&1 | tee "${LOG_OUT}"

echo "[done] logs: ${LOG_OUT}"
