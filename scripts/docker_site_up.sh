#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$ROOT/docker-compose.site.yml"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker not found"
  exit 1
fi

( cd "$ROOT" && docker compose -f "$COMPOSE_FILE" up -d --build )

echo "[docker_site] waiting for public URLs..."
WEB_URL=""
API_URL=""

for _ in $(seq 1 60); do
  if [ -z "$WEB_URL" ]; then
    WEB_URL=$(docker compose -f "$COMPOSE_FILE" logs --no-color cloudflared_web 2>/dev/null | grep -Eo 'https://[^ ]+\.trycloudflare\.com' | head -n 1 || true)
  fi
  if [ -z "$API_URL" ]; then
    API_URL=$(docker compose -f "$COMPOSE_FILE" logs --no-color cloudflared_api 2>/dev/null | grep -Eo 'https://[^ ]+\.trycloudflare\.com' | head -n 1 || true)
  fi
  if [ -n "$WEB_URL" ] && [ -n "$API_URL" ]; then
    break
  fi
  sleep 1
done

if [ -n "$API_URL" ]; then
  API_KEY=""
  if [ -f "$ROOT/.api_key" ]; then
    API_KEY=$(tr -d '\n' < "$ROOT/.api_key")
  fi
  cat > "$ROOT/html/config.json" <<JSON
{
  "api_base": "${API_URL}",
  "api_key": "${API_KEY}"
}
JSON
  echo "[docker_site] wrote html/config.json"
fi

if [ -n "$WEB_URL" ]; then
  echo "[docker_site] web url: ${WEB_URL}"
  echo "[docker_site] open: ${WEB_URL}/index.html"
else
  echo "[docker_site] web url not found yet; check: docker compose -f $COMPOSE_FILE logs cloudflared_web"
fi

if [ -n "$API_URL" ]; then
  echo "[docker_site] api url: ${API_URL}"
else
  echo "[docker_site] api url not found yet; check: docker compose -f $COMPOSE_FILE logs cloudflared_api"
fi
