#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/vitamind/my_project/model6"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
PIPELINE="${ROOT_DIR}/scripts/cc_news_pipeline.py"

OUT_DIR="${OUT_DIR:-/mnt/data/cc-news-2021-2023}"
CHECKPOINT="${CHECKPOINT:-/mnt/data/cc-news/checkpoints/processed_2021_2023.txt}"
LOG_FILE="${LOG_FILE:-${OUT_DIR}/run_2021_2023.log}"
LOCK_FILE="${LOCK_FILE:-/tmp/ccnews_2021_2023.lock}"

START_MONTH="${START_MONTH:-2021-01}"
END_MONTH="${END_MONTH:-2023-12}"
MAX_WARC="${MAX_WARC:-100}"
WARC_SAMPLE="${WARC_SAMPLE:-random}"
WARC_SEED="${WARC_SEED:-42}"
FLUSH="${FLUSH:-80}"
SLEEP_SEC="${SLEEP_SEC:-300}"

mkdir -p "${OUT_DIR}" "$(dirname "${CHECKPOINT}")"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "ERROR: python not found in ${PYTHON_BIN}" >&2
  exit 1
fi

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "Another 2021-2023 loop is already running (lock: ${LOCK_FILE})" >&2
  exit 1
fi

echo "[$(date -Is)] START ccnews 2021-2023 infinite loop" | tee -a "${LOG_FILE}"
echo "[$(date -Is)] out_dir=${OUT_DIR} checkpoint=${CHECKPOINT}" | tee -a "${LOG_FILE}"

while true; do
  echo "[$(date -Is)] cycle begin" >> "${LOG_FILE}"

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

  echo "[$(date -Is)] cycle end, sleep ${SLEEP_SEC}s" >> "${LOG_FILE}"
  sleep "${SLEEP_SEC}"
done
