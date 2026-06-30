#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
export OMP_NUM_THREADS=12
export TF_NUM_INTRAOP_THREADS=12
export TF_NUM_INTEROP_THREADS=3

ROOT="/home/vitamind/my_project/model6"
PY="$ROOT/.venv/bin/python"
TRAIN="$ROOT/train_keras_v7.py"
DATA="$ROOT/data/long_short_mix15"
OUT="$ROOT/new_models"
LOG="$ROOT/logs"
TAG="2026-01-29_v7_newsflag_mix15"

mkdir -p "$OUT/${TAG}_h80_long_v2" \
         "$OUT/${TAG}_h80_short_v2" \
         "$OUT/${TAG}_h160_long_v2" \
         "$OUT/${TAG}_h160_short_v2"

common_args=(
  --seq-len 512
  --batch-size 32
  --epochs 14
  --patience 3
  --arch itransformer
  --d-model 128
  --var-layers 2
  --time-layers 2
  --heads 4
  --revin
  --revin-affine
  --feature-dropout 0.05
  --dropout 0.10
  --drop-path 0.05
  --var-drop-path 0.05
  --time-pos learned
  --pos-dropout 0.02
  --head-mlp
  --head-mlp-dim 256
  --price-weight 1.0
  --cls-weight 0.0
  --price-head-scale std
)

mkdir -p "$LOG"

echo "[1/4] h80 long v2"
$PY "$TRAIN" \
  --features "$DATA/features_mix15_h80_long.parquet" \
  --price-multi-horizons 80 \
  --model-out "$OUT/${TAG}_h80_long_v2/model_15m_itransformer_v7_h80_long_v2.keras" \
  --stats-out "$OUT/${TAG}_h80_long_v2/norm_stats_v7_h80_long_v2.npz" \
  "${common_args[@]}" \
  > "$LOG/train_v7_h80_long_v2_${TAG}.log" 2>&1

echo "[2/4] h80 short v2"
$PY "$TRAIN" \
  --features "$DATA/features_mix15_h80_short.parquet" \
  --price-multi-horizons 80 \
  --model-out "$OUT/${TAG}_h80_short_v2/model_15m_itransformer_v7_h80_short_v2.keras" \
  --stats-out "$OUT/${TAG}_h80_short_v2/norm_stats_v7_h80_short_v2.npz" \
  "${common_args[@]}" \
  > "$LOG/train_v7_h80_short_v2_${TAG}.log" 2>&1

echo "[3/4] h160 long v2"
$PY "$TRAIN" \
  --features "$DATA/features_mix15_h160_long.parquet" \
  --price-multi-horizons 160 \
  --model-out "$OUT/${TAG}_h160_long_v2/model_15m_itransformer_v7_h160_long_v2.keras" \
  --stats-out "$OUT/${TAG}_h160_long_v2/norm_stats_v7_h160_long_v2.npz" \
  "${common_args[@]}" \
  > "$LOG/train_v7_h160_long_v2_${TAG}.log" 2>&1

echo "[4/4] h160 short v2"
$PY "$TRAIN" \
  --features "$DATA/features_mix15_h160_short.parquet" \
  --price-multi-horizons 160 \
  --model-out "$OUT/${TAG}_h160_short_v2/model_15m_itransformer_v7_h160_short_v2.keras" \
  --stats-out "$OUT/${TAG}_h160_short_v2/norm_stats_v7_h160_short_v2.npz" \
  "${common_args[@]}" \
  > "$LOG/train_v7_h160_short_v2_${TAG}.log" 2>&1

echo "Done."
