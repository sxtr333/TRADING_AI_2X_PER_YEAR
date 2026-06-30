#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"

mkdir -p "${LOG_DIR}"

if [ -f "${ROOT_DIR}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
fi

if [ -f "${LOG_DIR}/live_update.pid" ]; then
  kill "$(cat "${LOG_DIR}/live_update.pid")" >/dev/null 2>&1 || true
  rm -f "${LOG_DIR}/live_update.pid"
fi

nohup "${ROOT_DIR}/scripts/live_update_loop.sh" \
  > "${LOG_DIR}/live_update_stdout.log" 2>&1 & echo $! > "${LOG_DIR}/live_update.pid"

bash "${ROOT_DIR}/scripts/run_servers.sh"
