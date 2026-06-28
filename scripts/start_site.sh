#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/vitamind/my_project/model6"
LOG_DIR="$ROOT"
API_KEY_FILE="$ROOT/.api_key"
if [ ! -f "$API_KEY_FILE" ]; then
  python3 - <<'PY' > "$API_KEY_FILE"
import secrets
print(secrets.token_urlsafe(24))
PY
  chmod 600 "$API_KEY_FILE"
fi
API_KEY=$(cat "$API_KEY_FILE")

# 1) Web (static HTML)
if ! pgrep -f "python -m http.server 8088.*$ROOT/html" >/dev/null 2>&1; then
  nohup /home/vitamind/my_project/model6/.venv/bin/python -m http.server 8088 --directory "$ROOT/html" \
    > "$LOG_DIR/web.log" 2>&1 &
  echo "[start_site] web: started"
else
  echo "[start_site] web: already running"
fi

# 2) API (FastAPI)
if ! pgrep -f "serve_fastapi.py.*--port 8000" >/dev/null 2>&1; then
  nohup env NEWS_PATH=/mnt/data/news/news_raw.parquet ALLOW_ORIGINS="*" API_KEY="$API_KEY" \
    /home/vitamind/my_project/model6/.venv/bin/python "$ROOT/serve_fastapi.py" \
    --model-h20 model_battle_itransformer.keras \
    --stats-h20 norm_stats_battle_itransformer.npz \
    --model-multi model_15m_itransformer_tb_multi.keras \
    --stats-multi norm_stats_15m_itransformer_tb_multi.npz \
    --features data/BTCUSDT_15m_features_h20_v2.parquet \
    --seq-len 256 \
    --host 0.0.0.0 --port 8000 \
    > "$LOG_DIR/serve_fastapi.log" 2>&1 &
  echo "[start_site] api: started"
else
  echo "[start_site] api: already running"
fi

# 3) Cloudflared tunnel for web
if ! pgrep -f "cloudflared tunnel --url http://localhost:8088" >/dev/null 2>&1; then
  nohup /home/vitamind/.local/bin/cloudflared tunnel --url http://localhost:8088 \
    > "$LOG_DIR/cloudflared.log" 2>&1 &
  echo "[start_site] cloudflared web: started"
else
  echo "[start_site] cloudflared web: already running"
fi

# 4) Cloudflared tunnel for api
if ! pgrep -f "cloudflared tunnel --url http://localhost:8000" >/dev/null 2>&1; then
  nohup /home/vitamind/.local/bin/cloudflared tunnel --url http://localhost:8000 \
    > "$LOG_DIR/cloudflared_api.log" 2>&1 &
  echo "[start_site] cloudflared api: started"
else
  echo "[start_site] cloudflared api: already running"
fi

# 5) Live news loop
if ! pgrep -f "live_update_loop.sh" >/dev/null 2>&1; then
  nohup "$ROOT/scripts/live_update_loop.sh" \
    > "$LOG_DIR/live_update.log" 2>&1 &
  echo "[start_site] live_update: started"
else
  echo "[start_site] live_update: already running"
fi

# Output latest public URLs if present in logs
WEB_URL=$(rg -n "https://.*trycloudflare.com" -m 1 "$LOG_DIR/cloudflared.log" | sed -E 's/.*(https:\/\/[^ ]+).*/\1/' || true)
API_URL=$(rg -n "https://.*trycloudflare.com" -m 1 "$LOG_DIR/cloudflared_api.log" | sed -E 's/.*(https:\/\/[^ ]+).*/\1/' || true)
if [ -n "${WEB_URL:-}" ]; then
  echo "[start_site] web url: $WEB_URL"
fi
if [ -n "${API_URL:-}" ]; then
  echo "[start_site] api url: $API_URL"
  cat > "$ROOT/html/config.json" <<JSON
{
  "api_base": "${API_URL}",
  "api_key": "${API_KEY}"
}
JSON
  echo "[start_site] wrote config.json with api_base"
fi
