#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ ! -f ".env.site" ]; then
  if [ -f ".env.site.example" ]; then
    cp .env.site.example .env.site
    echo "[deploy] created .env.site from template (.env.site.example)"
    echo "[deploy] fill SMTP_PASS / EMAIL_CODE_SECRET / ADMIN_API_KEY in .env.site"
  else
    echo "[deploy] ERROR: .env.site not found and template missing"
    exit 1
  fi
fi

if [ ! -f ".api_key" ]; then
  python3 - <<'PY' > .api_key
import secrets
print(secrets.token_urlsafe(32))
PY
  chmod 600 .api_key
  echo "[deploy] generated .api_key"
fi

set -a
source .env.site
set +a

export API_KEY="$(cat .api_key)"

# Always publish MONEY as default landing/index
if [ -f "html/MONEY.html" ]; then
  cp -f "html/MONEY.html" "html/index.html"
fi

cat > html/config.json <<JSON
{
  "api_base": "https://api.tradeforge.art",
  "api_key": "${API_KEY}"
}
JSON

echo "[deploy] starting docker stack (web + api)"
docker compose --env-file .env.site -f docker-compose.site.yml down || true
pkill -f "http.server 8088" || true
pkill -f "serve_fastapi.py.*--port 8000" || true
docker compose --env-file .env.site -f docker-compose.site.yml up -d --build api web

echo "[deploy] local health checks"
curl -sS --max-time 10 http://127.0.0.1:8088/ | head -n 1 || true
for _ in $(seq 1 20); do
  if curl -sS --max-time 3 http://127.0.0.1:8000/health >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
curl -sS --max-time 10 http://127.0.0.1:8000/health | head -c 200 || true
echo
curl -sS --max-time 10 -X POST http://127.0.0.1:8000/auth/request-password-reset \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${API_KEY}" \
  -d '{"email":"test@example.com"}' | head -c 200 || true
echo

echo "[deploy] done"
echo "  site: https://tradeforge.art"
echo "  api:  https://api.tradeforge.art/health"
