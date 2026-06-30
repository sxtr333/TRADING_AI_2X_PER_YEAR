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

mkdir -p "$OUT/${TAG}_h20_long" \
         "$OUT/${TAG}_h20_short" \
         "$OUT/${TAG}_h80_long" \
         "$OUT/${TAG}_h80_short" \
         "$OUT/${TAG}_h160_long" \
         "$OUT/${TAG}_h160_short"

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

echo "[1/6] h20 long"
$PY "$TRAIN" \
  --features "$DATA/features_mix15_h20_long.parquet" \
  --price-multi-horizons 20 \
  --model-out "$OUT/${TAG}_h20_long/model_15m_itransformer_v7_h20_long.keras" \
  --stats-out "$OUT/${TAG}_h20_long/norm_stats_v7_h20_long.npz" \
  "${common_args[@]}" \
  > "$LOG/train_v7_h20_long_${TAG}.log" 2>&1

echo "[2/6] h20 short"
$PY "$TRAIN" \
  --features "$DATA/features_mix15_h20_short.parquet" \
  --price-multi-horizons 20 \
  --model-out "$OUT/${TAG}_h20_short/model_15m_itransformer_v7_h20_short.keras" \
  --stats-out "$OUT/${TAG}_h20_short/norm_stats_v7_h20_short.npz" \
  "${common_args[@]}" \
  > "$LOG/train_v7_h20_short_${TAG}.log" 2>&1

echo "[3/6] h80 long"
$PY "$TRAIN" \
  --features "$DATA/features_mix15_h80_long.parquet" \
  --price-multi-horizons 80 \
  --model-out "$OUT/${TAG}_h80_long/model_15m_itransformer_v7_h80_long.keras" \
  --stats-out "$OUT/${TAG}_h80_long/norm_stats_v7_h80_long.npz" \
  "${common_args[@]}" \
  > "$LOG/train_v7_h80_long_${TAG}.log" 2>&1

echo "[4/6] h80 short"
$PY "$TRAIN" \
  --features "$DATA/features_mix15_h80_short.parquet" \
  --price-multi-horizons 80 \
  --model-out "$OUT/${TAG}_h80_short/model_15m_itransformer_v7_h80_short.keras" \
  --stats-out "$OUT/${TAG}_h80_short/norm_stats_v7_h80_short.npz" \
  "${common_args[@]}" \
  > "$LOG/train_v7_h80_short_${TAG}.log" 2>&1

echo "[5/6] h160 long"
$PY "$TRAIN" \
  --features "$DATA/features_mix15_h160_long.parquet" \
  --price-multi-horizons 160 \
  --model-out "$OUT/${TAG}_h160_long/model_15m_itransformer_v7_h160_long.keras" \
  --stats-out "$OUT/${TAG}_h160_long/norm_stats_v7_h160_long.npz" \
  "${common_args[@]}" \
  > "$LOG/train_v7_h160_long_${TAG}.log" 2>&1

echo "[6/6] h160 short"
$PY "$TRAIN" \
  --features "$DATA/features_mix15_h160_short.parquet" \
  --price-multi-horizons 160 \
  --model-out "$OUT/${TAG}_h160_short/model_15m_itransformer_v7_h160_short.keras" \
  --stats-out "$OUT/${TAG}_h160_short/norm_stats_v7_h160_short.npz" \
  "${common_args[@]}" \
  > "$LOG/train_v7_h160_short_${TAG}.log" 2>&1

echo "Done."
