#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data"
OHLCV_PATH="${DATA_DIR}/BTCUSDT_15m.parquet"
AUX_PATH="${DATA_DIR}/BTCUSDT_15m_aux.parquet"
FEATURES_OUT="${DATA_DIR}/BTCUSDT_15m_features_h20_v2.parquet"
SERVE_OUT="${DATA_DIR}/BTCUSDT_15m_features_h20_v2_serve.parquet"

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

NEWS_PATH="${NEWS_PATH:-/mnt/data/news/news.parquet}"
python3 "${ROOT_DIR}/scripts/news_ingest.py" --out "${NEWS_PATH}" --currency BTC --max-items 200 || echo "[news] ingest failed"

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

echo "Updating OHLCV 15m from ${START_ISO} to ${END_ISO}"
python3 "${ROOT_DIR}/bybit_data.py" \
  --symbol BTCUSDT \
  --timeframes 15m \
  --start "${START_ISO}" \
  --end "${END_ISO}" \
  --out "${DATA_DIR}"

echo "Building features to ${FEATURES_OUT}"
NEWS_ARG=()
if [ -f "${NEWS_PATH}" ]; then
  NEWS_ARG=(--news "${NEWS_PATH}")
fi
if [ -f "${AUX_PATH}" ]; then
  python3 "${ROOT_DIR}/build_features.py" \
    --input "${OHLCV_PATH}" \
    --aux "${AUX_PATH}" \
    --output "${FEATURES_OUT}" \
    --serve-output "${SERVE_OUT}" \
    --horizon 20 \
    --multi-horizons 20,80,160 \
    "${NEWS_ARG[@]}"
else
  python3 "${ROOT_DIR}/build_features.py" \
    --input "${OHLCV_PATH}" \
    --output "${FEATURES_OUT}" \
    --serve-output "${SERVE_OUT}" \
    --horizon 20 \
    --multi-horizons 20,80,160 \
    "${NEWS_ARG[@]}"
fi

python3 - <<'PY'
import pandas as pd
df = pd.read_parquet("data/BTCUSDT_15m_features_h20_v2_serve.parquet", columns=["timestamp", "close"]).sort_values("timestamp")
print("features_last_ts:", df["timestamp"].iloc[-1])
print("features_last_close:", float(df["close"].iloc[-1]))
PY

echo "Done."
