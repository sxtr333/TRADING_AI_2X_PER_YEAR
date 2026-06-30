#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/new_models/2026-01-14_news_xlmr_v6_dr_scale_8nodes"
mkdir -p "${OUT_DIR}"

if [ ! -d "${ROOT_DIR}/.venv" ]; then
  echo "ERROR: .venv not found at ${ROOT_DIR}/.venv"
  exit 1
fi

source "${ROOT_DIR}/.venv/bin/activate"

FEATURES="${FEATURES:-${ROOT_DIR}/data/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes_daily.parquet}"
MODEL_OUT="${OUT_DIR}/model_15m_itransformer_price_multi_h20_h40_h60_h80_h100_h120_h140_h160_news_xlmr_v6_dr_scale_8nodes_e14_b32.keras"
STATS_OUT="${OUT_DIR}/norm_stats_15m_itransformer_price_multi_h20_h40_h60_h80_h100_h120_h140_h160_news_xlmr_v6_dr_scale_8nodes_e14_b32.npz"
LOG_OUT="${ROOT_DIR}/logs/train_news_xlmr_v6_dr_scale_8nodes.log"

CPU_TOTAL="$(nproc)"
CPU_LIMIT="$(( (CPU_TOTAL * 70 + 99) / 100 ))"
export OMP_NUM_THREADS="${CPU_LIMIT}"
export TF_NUM_INTRAOP_THREADS="${CPU_LIMIT}"
export TF_NUM_INTEROP_THREADS="$((CPU_LIMIT/4))"
if [ "${TF_NUM_INTEROP_THREADS}" -lt 1 ]; then
  export TF_NUM_INTEROP_THREADS=1
fi
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0"
export TF_XLA_FLAGS="--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false"

echo "[train] multi h20..h160 (8 nodes) -> ${MODEL_OUT}"
echo "[train] features=${FEATURES}"
echo "[train] CPU threads limit ~70%: OMP_NUM_THREADS=${OMP_NUM_THREADS}, TF_NUM_INTRAOP_THREADS=${TF_NUM_INTRAOP_THREADS}, TF_NUM_INTEROP_THREADS=${TF_NUM_INTEROP_THREADS}"
echo "[train] XLA_FLAGS=${XLA_FLAGS}"
echo "[train] TF_XLA_FLAGS=${TF_XLA_FLAGS}"

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
  --price-multi-horizons 20,40,60,80,100,120,140,160 \
  --price-segment-deltas \
  --price-head-scale std \
  --model-out "${MODEL_OUT}" \
  --stats-out "${STATS_OUT}" \
  2>&1 | tee "${LOG_OUT}"

echo "[done] logs: ${LOG_OUT}"
