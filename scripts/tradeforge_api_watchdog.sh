#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/helpd3/my_project/model6}
COMPOSE_FILE=${COMPOSE_FILE:-$ROOT/docker-compose.site.yml}
API_URL=${API_URL:-http://127.0.0.1:8000}
API_KEY_FILE=${API_KEY_FILE:-$ROOT/.api_key}
MAX_FORECAST_AGE_SEC=${MAX_FORECAST_AGE_SEC:-43200}
MAX_NEWS_AGE_SEC=${MAX_NEWS_AGE_SEC:-259200}
LOG_FILE=${LOG_FILE:-/tmp/tradeforge_api_watchdog.log}

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*" | tee -a "$LOG_FILE"; }

API_KEY=""
if [[ -f "$API_KEY_FILE" ]]; then
  API_KEY="$(cat "$API_KEY_FILE" 2>/dev/null || true)"
fi

HDR=()
if [[ -n "${API_KEY:-}" ]]; then
  HDR=(-H "X-API-Key: ${API_KEY}")
fi

restart_api() {
  log "restarting docker compose api service"
  (cd "$ROOT" && docker compose -f "$COMPOSE_FILE" restart api) >>"$LOG_FILE" 2>&1 || true
}

wait_health() {
  local tries=${1:-12}
  local delay=${2:-5}
  local i=0
  while (( i < tries )); do
    if curl -fsS --max-time 8 "${API_URL}/health" >/dev/null; then
      return 0
    fi
    sleep "$delay"
    i=$((i + 1))
  done
  return 1
}

needs_restart=false
stale_warn=false

# 1) health
if ! curl -fsS --max-time 8 "${API_URL}/health" >/dev/null; then
  log "health check failed"
  needs_restart=true
fi

# 2) forecast freshness
if [[ "$needs_restart" != true ]]; then
  if ! python3 - "$API_URL" "$MAX_FORECAST_AGE_SEC" "${HDR[@]}" <<'PY'
import json, sys, time, urllib.request
api=sys.argv[1]; max_age=int(sys.argv[2]); args=sys.argv[3:]
headers={}
if len(args)>=2 and args[0]=="-H" and ":" in args[1]:
    k,v=args[1].split(":",1); headers[k.strip()]=v.strip()
req=urllib.request.Request(f"{api}/forecast?interval=h20", headers=headers)
obj=json.load(urllib.request.urlopen(req, timeout=10))
base=int(obj.get("base_time", 0))
if base <= 0:
    raise SystemExit(2)
age=int(time.time())-base
if age > max_age:
    raise SystemExit(3)
PY
  then
    log "forecast stale/invalid (warning)"
    stale_warn=true
  fi
fi

# 3) news freshness
if [[ "$needs_restart" != true ]]; then
  if ! python3 - "$API_URL" "$MAX_NEWS_AGE_SEC" "${HDR[@]}" <<'PY'
import json, sys, time, datetime as dt, urllib.request
api=sys.argv[1]; max_age=int(sys.argv[2]); args=sys.argv[3:]
headers={}
if len(args)>=2 and args[0]=="-H" and ":" in args[1]:
    k,v=args[1].split(":",1); headers[k.strip()]=v.strip()
req=urllib.request.Request(f"{api}/news?limit=50", headers=headers)
obj=json.load(urllib.request.urlopen(req, timeout=10))
items=obj.get("items", [])
mx=None
for it in items:
    raw=it.get("published_at") or it.get("published")
    if not raw:
        continue
    try:
        d=dt.datetime.fromisoformat(str(raw).replace("Z","+00:00"))
    except Exception:
        continue
    if mx is None or d>mx: mx=d
if mx is None:
    raise SystemExit(4)
age=int(time.time()-mx.timestamp())
if age > max_age:
    raise SystemExit(5)
PY
  then
    log "news stale/invalid (warning)"
    stale_warn=true
  fi
fi

if [[ "$needs_restart" == true ]]; then
  restart_api
  if ! wait_health 12 5; then
    log "health still failing after restart"
    exit 1
  fi
  log "api recovered after restart"
else
  if [[ "$stale_warn" == true ]]; then
    log "api healthy, but freshness warnings present"
  else
    log "api healthy"
  fi
fi
