#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data"
NEWS_DIR="${NEWS_DIR:-/mnt/data/news}"

RAW_NEWS="${RAW_NEWS:-${NEWS_DIR}/news_raw.parquet}"
DEDUP_NEWS="${DEDUP_NEWS:-${NEWS_DIR}/news_dedup.parquet}"
NEWS_CACHE="${NEWS_CACHE:-${NEWS_DIR}/news_sentiment_cache.parquet}"
NEWS_SENT="${NEWS_SENT:-${NEWS_DIR}/news_sentiment.parquet}"
MACRO_DAILY_PATH="${MACRO_DAILY_PATH:-${ROOT_DIR}/macro and liquidity/data/macro_daily.parquet}"
FED_DAILY_PATH="${FED_DAILY_PATH:-${ROOT_DIR}/macro and liquidity/data/fed_rates_daily.parquet}"
INST_DAILY_PATH="${INST_DAILY_PATH:-${ROOT_DIR}/institutional flows/data/institutional_daily.parquet}"

OHLCV_PATH="${DATA_DIR}/BTCUSDT_15m.parquet"
AUX_PATH="${DATA_DIR}/BTCUSDT_15m_aux.parquet"
FEATURES_OUT="${DATA_DIR}/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes.parquet"
SERVE_OUT="${DATA_DIR}/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes_serve.parquet"
FEATURES_OUT_V3="${DATA_DIR}/BTCUSDT_15m_features_h20_v2_news_xlmr_v3.parquet"
SERVE_OUT_V3="${DATA_DIR}/BTCUSDT_15m_features_h20_v2_news_xlmr_v3_serve.parquet"

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

source "${ROOT_DIR}/.venv/bin/activate"

mkdir -p "${NEWS_DIR}"

TMP_NEWS="${RAW_NEWS%.*}.tmp.${RAW_NEWS##*.}"

echo "[news] ingest..."
python3 "${ROOT_DIR}/scripts/news_ingest.py" \
  --out "${TMP_NEWS}" \
  --currency BTC \
  --max-items "${NEWS_MAX_ITEMS:-70}" \
  || echo "[news] ingest failed"

if [ -f "${TMP_NEWS}" ]; then
  if [ -f "${RAW_NEWS}" ]; then
    python3 "${ROOT_DIR}/scripts/news_dedup.py" \
      --inputs "${RAW_NEWS}" "${TMP_NEWS}" \
      --output "${DEDUP_NEWS}"
  else
    mv "${TMP_NEWS}" "${DEDUP_NEWS}"
  fi
  rm -f "${TMP_NEWS}"
else
  if [ -f "${RAW_NEWS}" ]; then
    cp -f "${RAW_NEWS}" "${DEDUP_NEWS}"
  fi
fi

if [ -f "${DEDUP_NEWS}" ]; then
  mv -f "${DEDUP_NEWS}" "${RAW_NEWS}"
fi

if [ -f "${RAW_NEWS}" ]; then
  echo "[news] sentiment..."
  LEDGER_MODEL="${LEDGER_MODEL_PATH:-ExponentialScience/LedgerBERT-Market-Sentiment}"
  XLMR_MODEL="${XLMR_MODEL_PATH:-cardiffnlp/twitter-xlm-roberta-base-sentiment}"
  if [ "${NEWS_SKIP_SENTIMENT:-0}" = "1" ]; then
    echo "[news] sentiment skipped (NEWS_SKIP_SENTIMENT=1)"
  else
    python3 "${ROOT_DIR}/scripts/news_sentiment_hf.py" \
      --input "${RAW_NEWS}" \
      --output "${NEWS_SENT}" \
      --cache "${NEWS_CACHE}" \
      --device "${NEWS_DEVICE:-cuda}" \
      --batch-size "${NEWS_BATCH_SIZE:-16}" \
      --max-length "${NEWS_MAX_LENGTH:-512}" \
      --save-every "${NEWS_SAVE_EVERY:-1000}" \
      --model-ledger "${LEDGER_MODEL}" \
      --model-xlm "${XLMR_MODEL}" \
      --require-xlmr \
      || echo "[news] sentiment failed"
  fi
fi

read -r START_ISO END_ISO < <(python3 - <<'PY'
from pathlib import Path
import datetime as dt
import pandas as pd

path = Path("data/BTCUSDT_15m.parquet")
end = dt.datetime.now(dt.timezone.utc).isoformat()
if path.exists():
    df = pd.read_parquet(path, columns=["timestamp"])
    last_ts = pd.to_datetime(df["timestamp"], utc=True).max()
    if pd.isna(last_ts):
        start = "2023-01-01T00:00:00Z"
    else:
        start = (last_ts - pd.Timedelta(days=7)).isoformat()
else:
    start = "2023-01-01T00:00:00Z"
print(start, end)
PY
)

echo "[ohlcv] update from ${START_ISO} to ${END_ISO}"
python3 "${ROOT_DIR}/bybit_data.py" \
  --symbol BTCUSDT \
  --timeframes 15m \
  --start "${START_ISO}" \
  --end "${END_ISO}" \
  --out "${DATA_DIR}"

echo "[features] build..."
NEWS_ARG=()
if [ -f "${NEWS_SENT}" ]; then
  NEWS_ARG=(--news "${NEWS_SENT}")
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
if [ -f "${AUX_PATH}" ]; then
  python3 "${ROOT_DIR}/build_features.py" \
    --input "${OHLCV_PATH}" \
    --aux "${AUX_PATH}" \
    --output "${FEATURES_OUT}" \
    --serve-output "${SERVE_OUT}" \
    --horizon 20 \
    --multi-horizons 20,40,60,80,100,120,140,160 \
    --news-windows "1h,4h,12h,24h" \
    --news-ewm "4h,12h,24h" \
    --news-count-cap "${NEWS_COUNT_CAP:-65}" \
    "${NEWS_ARG[@]}" \
    "${DAILY_ARGS[@]}"
else
  python3 "${ROOT_DIR}/build_features.py" \
    --input "${OHLCV_PATH}" \
    --output "${FEATURES_OUT}" \
    --serve-output "${SERVE_OUT}" \
    --horizon 20 \
    --multi-horizons 20,40,60,80,100,120,140,160 \
    --news-windows "1h,4h,12h,24h" \
    --news-ewm "4h,12h,24h" \
    --news-count-cap "${NEWS_COUNT_CAP:-65}" \
    "${NEWS_ARG[@]}" \
    "${DAILY_ARGS[@]}"
fi

echo "[features] build v3 (for v5 model)..."
if [ -f "${AUX_PATH}" ]; then
  python3 "${ROOT_DIR}/build_features.py" \
    --input "${OHLCV_PATH}" \
    --aux "${AUX_PATH}" \
    --output "${FEATURES_OUT_V3}" \
    --serve-output "${SERVE_OUT_V3}" \
    --horizon 20 \
    --multi-horizons 20,80,160 \
    --news-windows "1h,4h,12h,24h" \
    --news-ewm "4h,12h,24h" \
    --news-count-cap "${NEWS_COUNT_CAP:-65}" \
    "${NEWS_ARG[@]}"
else
  python3 "${ROOT_DIR}/build_features.py" \
    --input "${OHLCV_PATH}" \
    --output "${FEATURES_OUT_V3}" \
    --serve-output "${SERVE_OUT_V3}" \
    --horizon 20 \
    --multi-horizons 20,80,160 \
    --news-windows "1h,4h,12h,24h" \
    --news-ewm "4h,12h,24h" \
    --news-count-cap "${NEWS_COUNT_CAP:-65}" \
    "${NEWS_ARG[@]}"
fi

python3 - <<'PY'
import pandas as pd
df = pd.read_parquet("data/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes_serve.parquet", columns=["timestamp", "close"]).sort_values("timestamp")
print("[features] last_ts:", df["timestamp"].iloc[-1])
print("[features] last_close:", float(df["close"].iloc[-1]))
PY
