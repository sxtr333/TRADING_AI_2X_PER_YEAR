#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT_DIR/.venv/bin/python"

INPUT_ROOT="${1:-/home/vitamind/TELEGRAM_DOWNLOAD1}"
OHLCV_PATH="${2:-$ROOT_DIR/data/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes_yrnorm_repl.parquet}"

cd "$ROOT_DIR"

$PY scripts/extract_signals_from_export.py \
  --input-root "$INPUT_ROOT" \
  --output data/telegram/signals_raw.parquet \
  --summary-json reports/signal_extraction_quality.json \
  --summary-csv reports/signal_extraction_quality.csv

$PY scripts/label_signals_with_ohlcv.py \
  --signals data/telegram/signals_raw.parquet \
  --ohlcv "$OHLCV_PATH" \
  --output data/telegram/signals_labeled.parquet \
  --summary-json reports/signal_label_quality.json

$PY scripts/build_training_dataset.py \
  --labeled data/telegram/signals_labeled.parquet \
  --include-outcomes win,loss,insufficient,invalid \
  --output-dir data/telegram/train_dataset \
  --summary-json reports/signal_dataset_summary.json

echo "[pipeline] done"
