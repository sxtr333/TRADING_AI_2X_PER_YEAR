#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   run_ccnews_shard_loop.sh SHARD START_MONTH END_MONTH OUT_DIR CHECKPOINT LOG_FILE
# Example:
#   run_ccnews_shard_loop.sh shard1 2023-01 2023-12 /mnt/data/cc-news-2023 /mnt/data/cc-news/checkpoints/processed_2023.txt /mnt/data/cc-news-2023/run_2023.log

if [ "$#" -ne 6 ]; then
  echo "usage: $0 SHARD START_MONTH END_MONTH OUT_DIR CHECKPOINT LOG_FILE" >&2
  exit 1
fi

SHARD="$1"
START_MONTH="$2"
END_MONTH="$3"
OUT_DIR="$4"
CHECKPOINT="$5"
LOG_FILE="$6"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PIPELINE="${ROOT_DIR}/scripts/cc_news_pipeline.py"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"

if [ ! -x "${PYTHON_BIN}" ]; then
  PYTHON_BIN="$(command -v python3 || true)"
fi

if [ -z "${PYTHON_BIN}" ]; then
  echo "python3 not found" >&2
  exit 1
fi

MAX_WARC="${MAX_WARC:-100}"
WARC_SAMPLE="${WARC_SAMPLE:-random}"
WARC_SEED="${WARC_SEED:-42}"
FLUSH="${FLUSH:-80}"
SLEEP_SEC="${SLEEP_SEC:-180}"
LOCK_FILE="${LOCK_FILE:-/tmp/ccnews_${SHARD}.lock}"

mkdir -p "${OUT_DIR}" "$(dirname "${CHECKPOINT}")" "$(dirname "${LOG_FILE}")"

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "[$(date -Is)] shard=${SHARD} lock busy, another instance is running" | tee -a "${LOG_FILE}"
  exit 1
fi

echo "[$(date -Is)] shard=${SHARD} start ${START_MONTH}..${END_MONTH}" | tee -a "${LOG_FILE}"
echo "[$(date -Is)] out=${OUT_DIR} checkpoint=${CHECKPOINT} max_warc=${MAX_WARC}" | tee -a "${LOG_FILE}"

while true; do
  echo "[$(date -Is)] shard=${SHARD} cycle begin" >> "${LOG_FILE}"

  "${PYTHON_BIN}" "${PIPELINE}" \
    --start-month "${START_MONTH}" \
    --end-month "${END_MONTH}" \
    --out-dir "${OUT_DIR}" \
    --checkpoint "${CHECKPOINT}" \
    --max-warc "${MAX_WARC}" \
    --warc-sample "${WARC_SAMPLE}" \
    --warc-seed "${WARC_SEED}" \
    --flush "${FLUSH}" \
    >> "${LOG_FILE}" 2>&1 || true

  echo "[$(date -Is)] shard=${SHARD} cycle end; sleep ${SLEEP_SEC}s" >> "${LOG_FILE}"
  sleep "${SLEEP_SEC}"
done
