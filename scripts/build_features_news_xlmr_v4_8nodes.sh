#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data"

OHLCV_PATH="${DATA_DIR}/BTCUSDT_15m.parquet"
AUX_PATH="${DATA_DIR}/BTCUSDT_15m_aux.parquet"
NEWS_PATH="${NEWS_PATH:-/mnt/data/news/news_sentiment.parquet}"
MACRO_DAILY_PATH="${MACRO_DAILY_PATH:-${ROOT_DIR}/macro and liquidity/data/macro_daily.parquet}"
FED_DAILY_PATH="${FED_DAILY_PATH:-${ROOT_DIR}/macro and liquidity/data/fed_rates_daily.parquet}"
INST_DAILY_PATH="${INST_DAILY_PATH:-${ROOT_DIR}/institutional flows/data/institutional_daily.parquet}"

FEATURES_OUT="${DATA_DIR}/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes.parquet"
SERVE_OUT="${DATA_DIR}/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes_serve.parquet"

if [ ! -d "${ROOT_DIR}/.venv" ]; then
  echo "ERROR: .venv not found at ${ROOT_DIR}/.venv"
  exit 1
fi

source "${ROOT_DIR}/.venv/bin/activate"

NEWS_ARG=()
if [ -f "${NEWS_PATH}" ]; then
  NEWS_ARG=(--news "${NEWS_PATH}")
fi
DAILY_ARGS=()
if [ -f "${MACRO_DAILY_PATH}" ]; then
  DAILY_ARGS+=(--macro-daily-path "${MACRO_DAILY_PATH}")
fi
if [ -f "${FED_DAILY_PATH}" ]; then
  DAILY_ARGS+=(--fed-daily-path "${FED_DAILY_PATH}")
fi
if [ -f "${INST_DAILY_PATH}" ]; then
  DAILY_ARGS+=(--inst-daily-path "${INST_DAILY_PATH}")
fi

python3 "${ROOT_DIR}/build_features.py" \
  --input "${OHLCV_PATH}" \
  --aux "${AUX_PATH}" \
  --output "${FEATURES_OUT}" \
  --serve-output "${SERVE_OUT}" \
  --horizon 20 \
  --multi-horizons 20,40,60,80,100,120,140,160 \
  --news-windows "1h,4h,12h,24h" \
  --news-ewm "4h,12h,24h" \
  "${NEWS_ARG[@]}" \
  "${DAILY_ARGS[@]}"

echo "Saved: ${FEATURES_OUT}"
