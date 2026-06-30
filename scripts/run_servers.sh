#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"

mkdir -p "${LOG_DIR}"

if [ ! -d "${ROOT_DIR}/.venv" ]; then
  echo "ERROR: .venv not found at ${ROOT_DIR}/.venv"
  exit 1
fi

if [ -f "${ROOT_DIR}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
fi

NEWS_PATH="${NEWS_PATH:-/mnt/data/news/news.parquet}"
export NEWS_PATH

if [ -f "${LOG_DIR}/fastapi_v6.pid" ]; then
  kill "$(cat "${LOG_DIR}/fastapi_v6.pid")" >/dev/null 2>&1 || true
  rm -f "${LOG_DIR}/fastapi_v6.pid"
fi

if [ -f "${LOG_DIR}/fastapi_v5.pid" ]; then
  kill "$(cat "${LOG_DIR}/fastapi_v5.pid")" >/dev/null 2>&1 || true
  rm -f "${LOG_DIR}/fastapi_v5.pid"
fi

if [ -f "${LOG_DIR}/http_8080.pid" ]; then
  kill "$(cat "${LOG_DIR}/http_8080.pid")" >/dev/null 2>&1 || true
  rm -f "${LOG_DIR}/http_8080.pid"
fi

if [ -f "${LOG_DIR}/arb_api.pid" ]; then
  kill "$(cat "${LOG_DIR}/arb_api.pid")" >/dev/null 2>&1 || true
  rm -f "${LOG_DIR}/arb_api.pid"
fi

if [ -f "${LOG_DIR}/arb_http.pid" ]; then
  kill "$(cat "${LOG_DIR}/arb_http.pid")" >/dev/null 2>&1 || true
  rm -f "${LOG_DIR}/arb_http.pid"
fi

source "${ROOT_DIR}/.venv/bin/activate"
export TF_USE_LEGACY_KERAS=0
export CUDA_VISIBLE_DEVICES=""
export USE_KERAS3=1
export MULTI_SCALE="${MULTI_SCALE:-15}"
export MIN_MULTI_PCT="${MIN_MULTI_PCT:-0.05}"
export H80_WEIGHTS="${H80_WEIGHTS:-0.6,0.4}"
export H160_WEIGHTS="${H160_WEIGHTS:-0.7,0.3}"
export BIAS_H80_USD="${BIAS_H80_USD:-16.76}"
export BIAS_H160_USD="${BIAS_H160_USD:-170.30}"
export H80_GATE_FEATURE="${H80_GATE_FEATURE:-rv_ratio}"
export H80_GATE_THRESHOLD="${H80_GATE_THRESHOLD:-1.2}"
export H80_GATE_NEWS_FEATURE="${H80_GATE_NEWS_FEATURE:-news_shock_4h}"
export H80_GATE_NEWS_THRESHOLD="${H80_GATE_NEWS_THRESHOLD:-0.0}"
export H80_GATE_WV3_HIGH="${H80_GATE_WV3_HIGH:-0.55}"
export H80_GATE_WV3_LOW="${H80_GATE_WV3_LOW:-0.25}"
export H80_GATE_DIV_THRESHOLD="${H80_GATE_DIV_THRESHOLD:-0.015}"
export H160_GATE_FEATURE="${H160_GATE_FEATURE:-rv_ratio}"
export H160_GATE_THRESHOLD="${H160_GATE_THRESHOLD:-1.2}"
export H160_GATE_NEWS_FEATURE="${H160_GATE_NEWS_FEATURE:-news_shock_4h}"
export H160_GATE_NEWS_THRESHOLD="${H160_GATE_NEWS_THRESHOLD:-0.0}"
export H160_GATE_WV3_HIGH="${H160_GATE_WV3_HIGH:-0.55}"
export H160_GATE_WV3_LOW="${H160_GATE_WV3_LOW:-0.25}"
export H160_GATE_DIV_THRESHOLD="${H160_GATE_DIV_THRESHOLD:-0.02}"

FEATURES_PATH_V5="${ROOT_DIR}/data/BTCUSDT_15m_features_h20_v2_news_xlmr_v3_serve.parquet"
FEATURES_PATH_V6="${ROOT_DIR}/data/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes_serve.parquet"
SERVE_FEATURES_PATH="${ROOT_DIR}/data/BTCUSDT_15m_features_h20_v2_serve.parquet"
SERVE_FEATURES_PATH_H640="${ROOT_DIR}/data/BTCUSDT_15m_features_h20_v2_h640_serve.parquet"
SERVE_FEATURES_PATH_NEWS="${ROOT_DIR}/data/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes_serve.parquet"
if [ ! -f "${FEATURES_PATH_V5}" ] && [ -f "${SERVE_FEATURES_PATH_NEWS}" ]; then
  FEATURES_PATH_V5="${SERVE_FEATURES_PATH_NEWS}"
elif [ ! -f "${FEATURES_PATH_V5}" ] && [ -f "${SERVE_FEATURES_PATH_H640}" ]; then
  FEATURES_PATH_V5="${SERVE_FEATURES_PATH_H640}"
elif [ ! -f "${FEATURES_PATH_V5}" ] && [ -f "${SERVE_FEATURES_PATH}" ]; then
  FEATURES_PATH_V5="${SERVE_FEATURES_PATH}"
fi

V6_MODEL="${ROOT_DIR}/new_models/2026-01-14_news_xlmr_v6_dr_scale_8nodes/model_15m_itransformer_price_multi_h20_h40_h60_h80_h100_h120_h140_h160_news_xlmr_v6_dr_scale_8nodes_e14_b32.keras"
V6_STATS="${ROOT_DIR}/new_models/2026-01-14_news_xlmr_v6_dr_scale_8nodes/norm_stats_15m_itransformer_price_multi_h20_h40_h60_h80_h100_h120_h140_h160_news_xlmr_v6_dr_scale_8nodes_e14_b32.npz"

V5_MODEL="${ROOT_DIR}/new_models/2026-01-14_news_xlmr_v5_dr_scale/model_15m_itransformer_price_multi_h20_h80_h160_news_xlmr_v5_dr_scale_e14_b32.keras"
V5_STATS="${ROOT_DIR}/new_models/2026-01-14_news_xlmr_v5_dr_scale/norm_stats_15m_itransformer_price_multi_h20_h80_h160_news_xlmr_v5_dr_scale_e14_b32.npz"

nohup python3 "${ROOT_DIR}/serve_fastapi.py" \
  --model-h20 "${V6_MODEL}" \
  --stats-h20 "${V6_STATS}" \
  --model-h80 "${V6_MODEL}" \
  --stats-h80 "${V6_STATS}" \
  --model-h160 "${V6_MODEL}" \
  --stats-h160 "${V6_STATS}" \
  --model-h320 "" \
  --stats-h320 "" \
  --model-h640 "" \
  --stats-h640 "" \
  --features "${FEATURES_PATH_V6}" \
  --seq-len 256 \
  --host 0.0.0.0 \
  --port 8000 \
  > "${LOG_DIR}/fastapi_v6.log" 2>&1 & echo $! > "${LOG_DIR}/fastapi_v6.pid"

nohup python3 "${ROOT_DIR}/serve_fastapi.py" \
  --model-h20 "${V5_MODEL}" \
  --stats-h20 "${V5_STATS}" \
  --model-h80 "${V5_MODEL}" \
  --stats-h80 "${V5_STATS}" \
  --model-h160 "${V5_MODEL}" \
  --stats-h160 "${V5_STATS}" \
  --model-h320 "" \
  --stats-h320 "" \
  --model-h640 "" \
  --stats-h640 "" \
  --features "${FEATURES_PATH_V5}" \
  --seq-len 512 \
  --host 0.0.0.0 \
  --port 8001 \
  > "${LOG_DIR}/fastapi_v5.log" 2>&1 & echo $! > "${LOG_DIR}/fastapi_v5.pid"

nohup python3 -m http.server 8080 --directory "${ROOT_DIR}/html" \
  > "${LOG_DIR}/http_8080.log" 2>&1 & echo $! > "${LOG_DIR}/http_8080.pid"

# Arbitration services
ARB_DIR="/home/vitamind/my_project/arbitration"
if [ -d "${ARB_DIR}" ]; then
  nohup bash "${ARB_DIR}/scripts/run_api.sh" \
    > "${LOG_DIR}/arb_api.log" 2>&1 & echo $! > "${LOG_DIR}/arb_api.pid"
  nohup python3 -m http.server 8090 --directory "${ARB_DIR}" \
    > "${LOG_DIR}/arb_http.log" 2>&1 & echo $! > "${LOG_DIR}/arb_http.pid"
fi

echo "FastAPI v6: http://localhost:8000"
echo "FastAPI v5: http://localhost:8001"
echo "UI:      http://localhost:8080/aladin_from_image.html"
echo "Arb UI:  http://localhost:8090/arbitration.html"
echo "Arb API: http://localhost:8091"
