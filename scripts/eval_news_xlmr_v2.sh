#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FEATURES="${ROOT_DIR}/data/BTCUSDT_15m_features_h20_v2_news_xlmr_v2.parquet"
START_TS="2025-07-08T21:15:00Z"

NEW_DIR="${ROOT_DIR}/new_models/2026-01-13_news_xlmr_v2"
OLD_PREFIX="${ROOT_DIR}"

eval_one () {
  local h="$1"
  local model="$2"
  local stats="$3"
  echo "[eval] h=${h} model=${model##*/}"
  python3 "${ROOT_DIR}/scripts/eval_price_quality.py" \
    --features "${FEATURES}" \
    --model "${model}" \
    --stats "${stats}" \
    --horizons "${h}" \
    --single-horizon "${h}" \
    --start "${START_TS}"
}

echo "== NEW (v2 rolling-news) =="
eval_one 20 "${NEW_DIR}/model_15m_itransformer_price_h20_news_xlmr_v2_e14_b32.keras" \
           "${NEW_DIR}/norm_stats_15m_itransformer_price_h20_news_xlmr_v2_e14_b32.npz"
eval_one 80 "${NEW_DIR}/model_15m_itransformer_price_h80_news_xlmr_v2_e14_b32.keras" \
           "${NEW_DIR}/norm_stats_15m_itransformer_price_h80_news_xlmr_v2_e14_b32.npz"
eval_one 160 "${NEW_DIR}/model_15m_itransformer_price_h160_news_xlmr_v2_e14_b32.keras" \
           "${NEW_DIR}/norm_stats_15m_itransformer_price_h160_news_xlmr_v2_e14_b32.npz"

echo "== OLD (news_xlmr_e12_b16) =="
eval_one 20 "${OLD_PREFIX}/model_15m_itransformer_price_h20_s512_news_xlmr_e12_b16.keras" \
           "${OLD_PREFIX}/norm_stats_15m_itransformer_price_h20_s512_news_xlmr_e12_b16.npz"
eval_one 80 "${OLD_PREFIX}/model_15m_itransformer_price_h80_s512_news_xlmr_e12_b16.keras" \
           "${OLD_PREFIX}/norm_stats_15m_itransformer_price_h80_s512_news_xlmr_e12_b16.npz"
eval_one 160 "${OLD_PREFIX}/model_15m_itransformer_price_h160_s512_news_xlmr_e12_b16.keras" \
           "${OLD_PREFIX}/norm_stats_15m_itransformer_price_h160_s512_news_xlmr_e12_b16.npz"
