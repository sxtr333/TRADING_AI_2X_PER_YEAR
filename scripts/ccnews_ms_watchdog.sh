#!/usr/bin/env bash
set -euo pipefail

# Watchdog for ms-7972 news pipeline:
#  - ccnews-download.service
#  - ccnews-sentiment.service
#  - ccnews-sync.service
#
# Also checks file freshness to catch "alive but stuck" loops.

ROOT=${ROOT:-/home/vitamind/my_project/model6}
DOWNLOAD_SVC=${DOWNLOAD_SVC:-ccnews-download.service}
SENTIMENT_SVC=${SENTIMENT_SVC:-ccnews-sentiment.service}
SYNC_SVC=${SYNC_SVC:-ccnews-sync.service}

DOWNLOAD_LOG_GLOB=${DOWNLOAD_LOG_GLOB:-/mnt/data/cc-news-2026/run_*.log}
SENTIMENT_FILE=${SENTIMENT_FILE:-/mnt/data/news/ccnews_2026_sentiment.parquet}

MAX_DOWNLOAD_AGE_SEC=${MAX_DOWNLOAD_AGE_SEC:-3600}
MAX_SENTIMENT_AGE_SEC=${MAX_SENTIMENT_AGE_SEC:-21600}

LOG_FILE=${LOG_FILE:-/tmp/ccnews_ms_watchdog.log}

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*" | tee -a "$LOG_FILE"; }

svc_active() {
  local svc="$1"
  systemctl is-active --quiet "$svc"
}

restart_svc() {
  local svc="$1"
  log "restarting $svc"
  sudo systemctl restart "$svc" >>"$LOG_FILE" 2>&1 || true
}

file_age_sec() {
  local f="$1"
  if [[ ! -f "$f" ]]; then
    echo -1
    return
  fi
  local now mtime
  now="$(date +%s)"
  mtime="$(stat -c %Y "$f" 2>/dev/null || echo 0)"
  echo $((now - mtime))
}

latest_download_log() {
  ls -1t $DOWNLOAD_LOG_GLOB 2>/dev/null | head -n 1 || true
}

needs_attention=false

# 1) service liveness
for svc in "$DOWNLOAD_SVC" "$SENTIMENT_SVC" "$SYNC_SVC"; do
  if ! svc_active "$svc"; then
    log "$svc is not active"
    restart_svc "$svc"
    needs_attention=true
  fi
done

# 2) downloader freshness
dl_log="$(latest_download_log)"
if [[ -z "${dl_log:-}" ]]; then
  log "no downloader log found ($DOWNLOAD_LOG_GLOB)"
  restart_svc "$DOWNLOAD_SVC"
  needs_attention=true
else
  dl_age="$(file_age_sec "$dl_log")"
  if (( dl_age < 0 || dl_age > MAX_DOWNLOAD_AGE_SEC )); then
    log "downloader stale: $dl_log age=${dl_age}s > ${MAX_DOWNLOAD_AGE_SEC}s"
    restart_svc "$DOWNLOAD_SVC"
    needs_attention=true
  fi
fi

# 3) sentiment output freshness
sent_age="$(file_age_sec "$SENTIMENT_FILE")"
if (( sent_age < 0 )); then
  log "sentiment output missing: $SENTIMENT_FILE"
  restart_svc "$SENTIMENT_SVC"
  needs_attention=true
elif (( sent_age > MAX_SENTIMENT_AGE_SEC )); then
  log "sentiment stale: $SENTIMENT_FILE age=${sent_age}s > ${MAX_SENTIMENT_AGE_SEC}s"
  restart_svc "$SENTIMENT_SVC"
  needs_attention=true
fi

# 4) final verification
ok=true
for svc in "$DOWNLOAD_SVC" "$SENTIMENT_SVC" "$SYNC_SVC"; do
  if ! svc_active "$svc"; then
    log "post-check failed: $svc still inactive"
    ok=false
  fi
done

if [[ "$ok" != true ]]; then
  exit 1
fi

if [[ "$needs_attention" == true ]]; then
  log "pipeline recovered"
else
  log "pipeline healthy"
fi

