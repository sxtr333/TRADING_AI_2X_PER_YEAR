#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FEATURES="${ROOT_DIR}/data/BTCUSDT_15m_features_h20_v2_news_xlmr_v3.parquet"
OUT_DIR="${ROOT_DIR}/new_models/2026-01-14_news_xlmr_v4_priceweight"
LOG_DIR="${ROOT_DIR}/logs"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

CORES="$(nproc || echo 8)"
LIMIT=$(( (CORES*70 + 99) / 100 ))
export OMP_NUM_THREADS="${LIMIT}"
export TF_NUM_INTRAOP_THREADS="${LIMIT}"
export TF_NUM_INTEROP_THREADS="${LIMIT}"

common_args=(
  --features "${FEATURES}"
  --seq-len 512
  --batch-size 32
  --epochs 14
  --lr 2e-4
  --arch itransformer
  --d-model 192
  --heads 6
  --layers 3
  --var-layers 2
  --time-layers 2
  --pooling multi
  --revin --revin-affine
  --feature-dropout 0.05
  --dropout 0.05
  --drop-path 0.05
  --cosine --warmup-steps 500
  --price-weight 1.0
  --cls-weight 0.0
  --price-loss huber
  --huber-delta 0.01
  --price-clip-q-low 0.001
  --price-clip-q-high 0.999
  --price-weight-mode close
  --price-weight-power 1.0
  --price-weight-clip 3.0
)

run_one () {
  local h="$1"
  local tag="$2"
  local model_out="${OUT_DIR}/model_15m_itransformer_price_h${h}_news_xlmr_v4_pw_${tag}.keras"
  local stats_out="${OUT_DIR}/norm_stats_15m_itransformer_price_h${h}_news_xlmr_v4_pw_${tag}.npz"
  local log_out="${LOG_DIR}/train_news_xlmr_v4_pw_h${h}.log"

  echo "[train] h=${h} -> ${model_out}"
  python3 "${ROOT_DIR}/train_keras.py" \
    "${common_args[@]}" \
    --target-horizon "${h}" \
    --price-col "target_ret_h${h}" \
    --model-out "${model_out}" \
    --stats-out "${stats_out}" \
    2>&1 | tee "${log_out}"
}

run_one 20 "e14_b32"
run_one 80 "e14_b32"
run_one 160 "e14_b32"

echo "Logs:"
echo "  ${LOG_DIR}/train_news_xlmr_v4_pw_h20.log"
echo "  ${LOG_DIR}/train_news_xlmr_v4_pw_h80.log"
echo "  ${LOG_DIR}/train_news_xlmr_v4_pw_h160.log"
