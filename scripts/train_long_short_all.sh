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
DATA="$ROOT/data/long_short"
OUT="$ROOT/new_models"
LOG="$ROOT/logs"

mkdir -p "$OUT/2026-01-18_v7_h20_short" \
         "$OUT/2026-01-18_v7_h80_long" \
         "$OUT/2026-01-18_v7_h80_short" \
         "$OUT/2026-01-18_v7_h160_long" \
         "$OUT/2026-01-18_v7_h160_short"

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

echo "[1/5] h20 short"
$PY "$TRAIN" \
  --features "$DATA/features_pruned_h20_short.parquet" \
  --price-multi-horizons 20 \
  --model-out "$OUT/2026-01-18_v7_h20_short/model_15m_itransformer_v7_h20_short.keras" \
  --stats-out "$OUT/2026-01-18_v7_h20_short/norm_stats_v7_h20_short.npz" \
  "${common_args[@]}" \
  > "$LOG/train_v7_h20_short_2026-01-18.log" 2>&1

echo "[2/5] h80 long"
$PY "$TRAIN" \
  --features "$DATA/features_pruned_h80_long.parquet" \
  --price-multi-horizons 80 \
  --model-out "$OUT/2026-01-18_v7_h80_long/model_15m_itransformer_v7_h80_long.keras" \
  --stats-out "$OUT/2026-01-18_v7_h80_long/norm_stats_v7_h80_long.npz" \
  "${common_args[@]}" \
  > "$LOG/train_v7_h80_long_2026-01-18.log" 2>&1

echo "[3/5] h80 short"
$PY "$TRAIN" \
  --features "$DATA/features_pruned_h80_short.parquet" \
  --price-multi-horizons 80 \
  --model-out "$OUT/2026-01-18_v7_h80_short/model_15m_itransformer_v7_h80_short.keras" \
  --stats-out "$OUT/2026-01-18_v7_h80_short/norm_stats_v7_h80_short.npz" \
  "${common_args[@]}" \
  > "$LOG/train_v7_h80_short_2026-01-18.log" 2>&1

echo "[4/5] h160 long"
$PY "$TRAIN" \
  --features "$DATA/features_pruned_h160_long.parquet" \
  --price-multi-horizons 160 \
  --model-out "$OUT/2026-01-18_v7_h160_long/model_15m_itransformer_v7_h160_long.keras" \
  --stats-out "$OUT/2026-01-18_v7_h160_long/norm_stats_v7_h160_long.npz" \
  "${common_args[@]}" \
  > "$LOG/train_v7_h160_long_2026-01-18.log" 2>&1

echo "[5/5] h160 short"
$PY "$TRAIN" \
  --features "$DATA/features_pruned_h160_short.parquet" \
  --price-multi-horizons 160 \
  --model-out "$OUT/2026-01-18_v7_h160_short/model_15m_itransformer_v7_h160_short.keras" \
  --stats-out "$OUT/2026-01-18_v7_h160_short/norm_stats_v7_h160_short.npz" \
  "${common_args[@]}" \
  > "$LOG/train_v7_h160_short_2026-01-18.log" 2>&1

echo "Done."
