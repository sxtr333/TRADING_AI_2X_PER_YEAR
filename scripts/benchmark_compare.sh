#!/usr/bin/env bash
set -euo pipefail

FEATURES="data/BTCUSDT_15m_features_h20_v2.parquet"
TRAIN_END="2024-01-01T00:00:00Z"
VAL_END="2025-01-01T00:00:00Z"
HORIZON=20
SEQ=256
BS=128
EPOCHS=6

echo "[bench] features=${FEATURES}"

run_model () {
  local ARCH="$1"
  local OUT="$2"
  local STATS="$3"
  local LOG="$4"
  echo "[bench] ${ARCH} -> ${OUT}"
  docker run --rm --runtime=nvidia --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$PWD":/workspace -w /workspace \
    nvcr.io/nvidia/tensorflow:25.02-tf2-py3 \
    bash -lc "taskset -c 0-10 python train_keras.py \
      --features ${FEATURES} --seq-len ${SEQ} --batch-size ${BS} --epochs ${EPOCHS} \
      --arch ${ARCH} --target-col target_dir --price-col target_ret \
      --cls-loss focal --focal-alpha 0.25 --focal-gamma 2.0 \
      --pos-weight 2.5 --price-weight 0.1 --cls-weight 1.0 \
      --train-end ${TRAIN_END} --val-end ${VAL_END} \
      --purge-gap ${HORIZON} --target-horizon ${HORIZON} \
      --model-out ${OUT} --stats-out ${STATS}" \
    > "${LOG}" 2>&1
}

run_model patchtst model_bench_patchtst.keras norm_stats_bench_patchtst.npz train_bench_patchtst.log
run_model tsmixer  model_bench_tsmixer.keras  norm_stats_bench_tsmixer.npz  train_bench_tsmixer.log
run_model itransformer model_bench_itransformer.keras norm_stats_bench_itransformer.npz train_bench_itransformer.log

echo "[bench] done. Logs: train_bench_patchtst.log, train_bench_tsmixer.log, train_bench_itransformer.log"
