#!/usr/bin/env bash
set -euo pipefail

START_MONTH=${1:?"usage: run_ccnews_range.sh START_MONTH END_MONTH OUT_DIR CHECKPOINT"}
END_MONTH=${2:?"usage: run_ccnews_range.sh START_MONTH END_MONTH OUT_DIR CHECKPOINT"}
OUT_DIR=${3:?"usage: run_ccnews_range.sh START_MONTH END_MONTH OUT_DIR CHECKPOINT"}
CHECKPOINT=${4:?"usage: run_ccnews_range.sh START_MONTH END_MONTH OUT_DIR CHECKPOINT"}

MAX_WARC=${MAX_WARC:-30}
WARC_SAMPLE=${WARC_SAMPLE:-random}
WARC_SEED=${WARC_SEED:-42}
FLUSH=${FLUSH:-70}
CPUSET=${CPUSET:-""}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
PY=${PYTHON:-"${ROOT_DIR}/.venv/bin/python"}

mkdir -p "${OUT_DIR}" "$(dirname "${CHECKPOINT}")"

if [[ -n "${CPUSET}" ]]; then
  exec taskset -c "${CPUSET}" "${PY}" "${ROOT_DIR}/scripts/cc_news_pipeline.py" \
    --start-month "${START_MONTH}" \
    --end-month "${END_MONTH}" \
    --out-dir "${OUT_DIR}" \
    --checkpoint "${CHECKPOINT}" \
    --max-warc "${MAX_WARC}" \
    --warc-sample "${WARC_SAMPLE}" \
    --warc-seed "${WARC_SEED}" \
    --flush "${FLUSH}"
else
  exec "${PY}" "${ROOT_DIR}/scripts/cc_news_pipeline.py" \
    --start-month "${START_MONTH}" \
    --end-month "${END_MONTH}" \
    --out-dir "${OUT_DIR}" \
    --checkpoint "${CHECKPOINT}" \
    --max-warc "${MAX_WARC}" \
    --warc-sample "${WARC_SAMPLE}" \
    --warc-seed "${WARC_SEED}" \
    --flush "${FLUSH}"
fi
