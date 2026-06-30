#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/vitamind/my_project/model6"
PY="$ROOT/.venv/bin/python"
FEATURES="$ROOT/data/BTCUSDT_15m_features_h20_v2_news_xlmr_full_2020_2026_dailyfill.parquet"
REPORTS="$ROOT/reports"
MODEL_TAG="2026-01-29_v7_newsflag_mix15"
META_TAG="meta_2026-01-29_newsflag_mix15_2024calib"
META_OUT="$ROOT/data/meta/meta_dataset_pruned_newsflag_mix15_2024calib_2026-01-29.parquet"

REQ_MODELS=(
  "$ROOT/new_models/${MODEL_TAG}_h20_long/model_15m_itransformer_v7_h20_long.keras"
  "$ROOT/new_models/${MODEL_TAG}_h20_short/model_15m_itransformer_v7_h20_short.keras"
  "$ROOT/new_models/${MODEL_TAG}_h80_long/model_15m_itransformer_v7_h80_long.keras"
  "$ROOT/new_models/${MODEL_TAG}_h80_short/model_15m_itransformer_v7_h80_short.keras"
  "$ROOT/new_models/${MODEL_TAG}_h160_long/model_15m_itransformer_v7_h160_long.keras"
  "$ROOT/new_models/${MODEL_TAG}_h160_short/model_15m_itransformer_v7_h160_short.keras"
  "$ROOT/new_models/${MODEL_TAG}_h80_long_v2/model_15m_itransformer_v7_h80_long_v2.keras"
  "$ROOT/new_models/${MODEL_TAG}_h80_short_v2/model_15m_itransformer_v7_h80_short_v2.keras"
  "$ROOT/new_models/${MODEL_TAG}_h160_long_v2/model_15m_itransformer_v7_h160_long_v2.keras"
  "$ROOT/new_models/${MODEL_TAG}_h160_short_v2/model_15m_itransformer_v7_h160_short_v2.keras"
)

echo "[wait] checking trained models..."
while true; do
  missing=0
  for f in "${REQ_MODELS[@]}"; do
    if [[ ! -s "$f" ]]; then
      missing=1
      break
    fi
  done
  if [[ "$missing" -eq 0 ]]; then
    break
  fi
  sleep 60
done
echo "[ok] all base models present."

mkdir -p "$REPORTS" "$ROOT/data/meta" "$ROOT/new_models/$META_TAG" "$ROOT/logs"

echo "[1/5] sweep v7 (2024 window)"
$PY "$ROOT/scripts/backtest_long_short_v7_sweep.py" \
  --features "$FEATURES" \
  --model-tag "$MODEL_TAG" \
  --out-csv "$REPORTS/backtest_v7_long_short_sweep_newsflag_mix15_2024_2026-01-29.csv" \
  --start "2024-01-01T00:00:00+00:00" \
  --end "2025-01-01T00:00:00+00:00"

echo "[1b] best v7"
$PY - <<'PY'
import pandas as pd
src = "/home/vitamind/my_project/model6/reports/backtest_v7_long_short_sweep_newsflag_mix15_2024_2026-01-29.csv"
dst = "/home/vitamind/my_project/model6/reports/backtest_v7_long_short_sweep_newsflag_mix15_2024_best_2026-01-29.csv"
df = pd.read_csv(src)
best = df.sort_values("total", ascending=False).groupby(["horizon","direction"], as_index=False).head(1)
best = best.sort_values(["horizon","direction"]).reset_index(drop=True)
best.to_csv(dst, index=False)
print("Saved:", dst)
PY

echo "[2/5] sweep v2 (2024 window)"
$PY "$ROOT/scripts/backtest_long_short_v2_sweep.py" \
  --features "$FEATURES" \
  --model-tag "$MODEL_TAG" \
  --out-csv "$REPORTS/backtest_v7_long_short_v2_sweep_newsflag_mix15_2024_2026-01-29.csv" \
  --start "2024-01-01T00:00:00+00:00" \
  --end "2025-01-01T00:00:00+00:00"

echo "[2b] best v2"
$PY - <<'PY'
import pandas as pd
src = "/home/vitamind/my_project/model6/reports/backtest_v7_long_short_v2_sweep_newsflag_mix15_2024_2026-01-29.csv"
dst = "/home/vitamind/my_project/model6/reports/backtest_v7_long_short_v2_sweep_newsflag_mix15_2024_best_2026-01-29.csv"
df = pd.read_csv(src)
best = df.sort_values("total", ascending=False).groupby(["horizon","direction"], as_index=False).head(1)
best = best.sort_values(["horizon","direction"]).reset_index(drop=True)
best.to_csv(dst, index=False)
print("Saved:", dst)
PY

echo "[3/5] build meta dataset (2024 bias window)"
$PY "$ROOT/scripts/build_meta_dataset.py" \
  --features "$FEATURES" \
  --output "$META_OUT" \
  --model-tag "$MODEL_TAG" \
  --best-h20 "$REPORTS/backtest_v7_long_short_sweep_newsflag_mix15_2024_best_2026-01-29.csv" \
  --best-v2 "$REPORTS/backtest_v7_long_short_v2_sweep_newsflag_mix15_2024_best_2026-01-29.csv" \
  --bias-start "2024-01-01T00:00:00+00:00" \
  --bias-end "2025-01-01T00:00:00+00:00"

echo "[4/5] train meta models"
train_meta () {
  local target="$1"
  local out="$ROOT/new_models/$META_TAG/meta_${target}.keras"
  local stats="$ROOT/new_models/$META_TAG/meta_${target}_stats.npz"
  local log="$ROOT/logs/train_meta_${target}_${META_TAG}.log"
  $PY "$ROOT/train_keras_v7.py" \
    --features "$META_OUT" \
    --target-col "meta_label_${target}" \
    --price-col "target_amp_abs" \
    --seq-len 256 \
    --batch-size 64 \
    --epochs 10 \
    --patience 3 \
    --arch itransformer \
    --d-model 128 \
    --var-layers 2 \
    --time-layers 2 \
    --heads 4 \
    --revin \
    --revin-affine \
    --feature-dropout 0.05 \
    --dropout 0.10 \
    --drop-path 0.05 \
    --var-drop-path 0.05 \
    --time-pos learned \
    --pos-dropout 0.02 \
    --head-mlp \
    --head-mlp-dim 256 \
    --cls-weight 1.0 \
    --price-weight 0.0 \
    --model-out "$out" \
    --stats-out "$stats" \
    > "$log" 2>&1
}

train_meta "h20_long"
train_meta "h20_short"
train_meta "h80_short_v2"
train_meta "h160_long_v2"

echo "[5/5] backtest 2025 with agreement gate"
$PY "$ROOT/scripts/backtest_trade_combo_meta.py" \
  --features "$FEATURES" \
  --meta-features "$META_OUT" \
  --meta-model-dir "$ROOT/new_models/$META_TAG" \
  --best-h20 "$REPORTS/backtest_v7_long_short_sweep_newsflag_mix15_2024_best_2026-01-29.csv" \
  --best-v2 "$REPORTS/backtest_v7_long_short_v2_sweep_newsflag_mix15_2024_best_2026-01-29.csv" \
  --agreement-gate \
  --out-csv "$REPORTS/backtest_meta_newsflag_mix15_2024calib_2025_agree_2026-01-29.csv"

echo "Done."
