#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/vitamind/my_project"
ROOT_DIR="/home/vitamind/my_project/model6"
IMAGE="nvcr.io/nvidia/tensorflow:25.02-tf2-py3"
FEATURES="${ROOT_DIR}/data/BTCUSDT_15m_features_h20_v2_news.parquet"

# 70% CPU
CPUS="$(awk -v c="$(nproc)" 'BEGIN{printf "%.2f", c*0.70}')"
THREADS="$(awk -v c="$(nproc)" 'BEGIN{t=int(c*0.70); if(t<1)t=1; print t}')"

run_train() {
  local horizon="$1"
  local price_col="$2"
  local model_out="$3"
  local stats_out="$4"
  local log_out="$5"

  docker run --rm \
    --gpus all --runtime=nvidia \
    --network host --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --cpus="${CPUS}" \
    -e OMP_NUM_THREADS="${THREADS}" \
    -e MKL_NUM_THREADS="${THREADS}" \
    -e OPENBLAS_NUM_THREADS="${THREADS}" \
    -e NUMEXPR_MAX_THREADS="${THREADS}" \
    -e TF_NUM_INTRAOP_THREADS="${THREADS}" \
    -e TF_NUM_INTEROP_THREADS=2 \
    -e TF_FORCE_GPU_ALLOW_GROWTH=true \
    -v "${PROJECT_DIR}:/workspace" -w /workspace \
    "${IMAGE}" bash -lc "
      pip install -q --no-cache-dir pandas pyarrow numpy && \
      python model6/train_keras.py \
        --features model6/data/BTCUSDT_15m_features_h20_v2_news.parquet \
        --seq-len 512 --batch-size 16 --epochs 12 --lr 3e-4 \
        --model-out model6/${model_out} \
        --stats-out model6/${stats_out} \
        --price-col ${price_col} --price-weight 1.0 \
        --cls-weight 0.0 --num-classes 1 \
        --target-horizon ${horizon} \
        --arch itransformer --feature-dropout 0.05 --drop-path 0.05 \
        --patience 3 --min-delta 1e-4
    " > "${log_out}" 2>&1
}

mkdir -p "${ROOT_DIR}/logs"

run_train 20 target_ret \
  model_15m_itransformer_price_h20_s512_news_e12_b16.keras \
  norm_stats_15m_itransformer_price_h20_s512_news_e12_b16.npz \
  "${ROOT_DIR}/logs/train_news_h20.log"

run_train 80 target_ret_h80 \
  model_15m_itransformer_price_h80_s512_news_e12_b16.keras \
  norm_stats_15m_itransformer_price_h80_s512_news_e12_b16.npz \
  "${ROOT_DIR}/logs/train_news_h80.log"

run_train 160 target_ret_h160 \
  model_15m_itransformer_price_h160_s512_news_e12_b16.keras \
  norm_stats_15m_itransformer_price_h160_s512_news_e12_b16.npz \
  "${ROOT_DIR}/logs/train_news_h160.log"

echo "Done. Logs:"
echo "  ${ROOT_DIR}/logs/train_news_h20.log"
echo "  ${ROOT_DIR}/logs/train_news_h80.log"
echo "  ${ROOT_DIR}/logs/train_news_h160.log"
