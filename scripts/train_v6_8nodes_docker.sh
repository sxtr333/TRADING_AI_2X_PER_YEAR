#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/vitamind/my_project"
IMAGE="nvcr.io/nvidia/tensorflow:25.02-tf2-py3"

# 70% CPU
CPUS="$(awk -v c="$(nproc)" 'BEGIN{printf "%.2f", c*0.70}')"
THREADS="$(awk -v c="$(nproc)" 'BEGIN{t=int(c*0.70); if(t<1)t=1; print t}')"

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
    python /workspace/model6/train_keras.py \
      --features /workspace/model6/data/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes.parquet \
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
      --model-out /workspace/model6/new_models/2026-01-14_news_xlmr_v6_dr_scale_8nodes/model_15m_itransformer_price_multi_h20_h40_h60_h80_h100_h120_h140_h160_news_xlmr_v6_dr_scale_8nodes_e14_b32.keras \
      --stats-out /workspace/model6/new_models/2026-01-14_news_xlmr_v6_dr_scale_8nodes/norm_stats_15m_itransformer_price_multi_h20_h40_h60_h80_h100_h120_h140_h160_news_xlmr_v6_dr_scale_8nodes_e14_b32.npz \
      2>&1 | tee /workspace/model6/logs/train_news_xlmr_v6_dr_scale_8nodes.log
  "
