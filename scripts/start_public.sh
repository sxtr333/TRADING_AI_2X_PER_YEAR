#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
BIN_DIR="${ROOT_DIR}/bin"
CLOUDFLARED="${BIN_DIR}/cloudflared"

mkdir -p "${LOG_DIR}" "${BIN_DIR}"

if [ -f "${ROOT_DIR}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
fi

if [ ! -x "${CLOUDFLARED}" ]; then
  echo "Downloading cloudflared..."
  curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o "${CLOUDFLARED}"
  chmod +x "${CLOUDFLARED}"
fi

if [ -f "${LOG_DIR}/cloudflared.pid" ]; then
  kill "$(cat "${LOG_DIR}/cloudflared.pid")" >/dev/null 2>&1 || true
  rm -f "${LOG_DIR}/cloudflared.pid"
fi
if [ -f "${LOG_DIR}/cloudflared_api.pid" ]; then
  kill "$(cat "${LOG_DIR}/cloudflared_api.pid")" >/dev/null 2>&1 || true
  rm -f "${LOG_DIR}/cloudflared_api.pid"
fi

NEWS_OUT="${NEWS_PATH:-/mnt/data/news/news.parquet}"
python3 "${ROOT_DIR}/scripts/news_ingest.py" --out "${NEWS_OUT}" --currency BTC --max-items 200 || echo "[news] ingest failed"

ALLOW_ORIGINS="*" bash "${ROOT_DIR}/scripts/run_servers.sh"

nohup "${CLOUDFLARED}" tunnel --protocol http2 --edge-ip-version 4 --url http://localhost:8080 \
  > "${LOG_DIR}/cloudflared.log" 2>&1 & echo $! > "${LOG_DIR}/cloudflared.pid"
nohup "${CLOUDFLARED}" tunnel --protocol http2 --edge-ip-version 4 --url http://localhost:8000 \
  > "${LOG_DIR}/cloudflared_api.log" 2>&1 & echo $! > "${LOG_DIR}/cloudflared_api.pid"

ui_url=""
api_url=""
for _ in {1..20}; do
  if [ -z "${ui_url}" ]; then
    ui_url="$(grep -ao "https://[a-z0-9.-]*trycloudflare.com" "${LOG_DIR}/cloudflared.log" | head -n 1 || true)"
  fi
  if [ -z "${api_url}" ]; then
    api_url="$(grep -ao "https://[a-z0-9.-]*trycloudflare.com" "${LOG_DIR}/cloudflared_api.log" | head -n 1 || true)"
  fi
  if [ -n "${ui_url}" ] && [ -n "${api_url}" ]; then
    break
  fi
  sleep 1
done

if [ -z "${ui_url}" ] || [ -z "${api_url}" ]; then
  echo "Cloudflared failed to produce URLs. Falling back to localtunnel..."
  if [ -f "${LOG_DIR}/cloudflared.pid" ]; then
    kill "$(cat "${LOG_DIR}/cloudflared.pid")" >/dev/null 2>&1 || true
    rm -f "${LOG_DIR}/cloudflared.pid"
  fi
  if [ -f "${LOG_DIR}/cloudflared_api.pid" ]; then
    kill "$(cat "${LOG_DIR}/cloudflared_api.pid")" >/dev/null 2>&1 || true
    rm -f "${LOG_DIR}/cloudflared_api.pid"
  fi

  nohup npx localtunnel --port 8080 --local-host localhost \
    > "${LOG_DIR}/localtunnel.log" 2>&1 & echo $! > "${LOG_DIR}/localtunnel.pid"
  nohup npx localtunnel --port 8000 --local-host localhost \
    > "${LOG_DIR}/localtunnel_api.log" 2>&1 & echo $! > "${LOG_DIR}/localtunnel_api.pid"

  ui_url=""
  api_url=""
  for _ in {1..30}; do
    if [ -z "${ui_url}" ]; then
      ui_url="$(grep -ao "https://[a-z0-9.-]*\\.loca.lt" "${LOG_DIR}/localtunnel.log" | head -n 1 || true)"
    fi
    if [ -z "${api_url}" ]; then
      api_url="$(grep -ao "https://[a-z0-9.-]*\\.loca.lt" "${LOG_DIR}/localtunnel_api.log" | head -n 1 || true)"
    fi
    if [ -n "${ui_url}" ] && [ -n "${api_url}" ]; then
      break
    fi
    sleep 1
  done
fi

if [ -z "${ui_url}" ] || [ -z "${api_url}" ]; then
  echo "ERROR: failed to get tunnel URLs. Check logs in ${LOG_DIR}."
  exit 1
fi

API_URL="${api_url}" ROOT_DIR="${ROOT_DIR}" python3 - <<'PY'
from pathlib import Path
import os

root_dir = os.environ.get("ROOT_DIR", "").strip()
if not root_dir:
    raise SystemExit("ROOT_DIR env var missing")
path = Path(root_dir) / "html" / "aladin_from_image.html"
text = path.read_text(encoding="utf-8")
old = 'const API_BASE = "'
start = text.find(old)
if start == -1:
    raise SystemExit("API_BASE not found in aladin_from_image.html")
end = text.find('";', start)
if end == -1:
    raise SystemExit("API_BASE line malformed in aladin_from_image.html")
api_url = os.environ.get("API_URL", "").strip()
if not api_url:
    raise SystemExit("API_URL env var missing")
new = text[:start] + f'const API_BASE = "{api_url}";' + text[end+2:]
path.write_text(new, encoding="utf-8")
PY

echo "Public UI:  ${ui_url}/aladin_from_image.html"
echo "Public API: ${api_url}"
