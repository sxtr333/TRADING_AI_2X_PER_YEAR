#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data"

OHLCV_PATH="${DATA_DIR}/BTCUSDT_15m.parquet"
AUX_PATH="${DATA_DIR}/BTCUSDT_15m_aux.parquet"
NEWS_PATH="${NEWS_PATH:-/mnt/data/news/news.parquet}"

FEATURES_OUT="${DATA_DIR}/BTCUSDT_15m_features_h20_v2_news_xlmr_v3.parquet"
SERVE_OUT="${DATA_DIR}/BTCUSDT_15m_features_h20_v2_news_xlmr_v3_serve.parquet"

if [ ! -d "${ROOT_DIR}/.venv" ]; then
  echo "ERROR: .venv not found at ${ROOT_DIR}/.venv"
  exit 1
fi

source "${ROOT_DIR}/.venv/bin/activate"

NEWS_ARG=()
if [ -f "${NEWS_PATH}" ]; then
  NEWS_ARG=(--news "${NEWS_PATH}")
fi

python3 "${ROOT_DIR}/build_features.py" \
  --input "${OHLCV_PATH}" \
  --aux "${AUX_PATH}" \
  --output "${FEATURES_OUT}" \
  --serve-output "${SERVE_OUT}" \
  --horizon 20 \
  --multi-horizons 20,80,160 \
  --news-windows "1h,4h,12h,24h" \
  --news-ewm "4h,12h,24h" \
  "${NEWS_ARG[@]}"

echo "Saved: ${FEATURES_OUT}"
