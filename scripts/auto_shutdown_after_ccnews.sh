#!/usr/bin/env bash
set -euo pipefail

CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-120}"

log() {
  printf "[%s] %s\n" "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" "$*"
}

all_done_logs() {
  local done_count=0
  for q in q1 q2 q3 q4; do
    local log_path="/mnt/data/cc-news-12m-w30-${q}/ccnews_${q}.log"
    if [[ -f "$log_path" ]] && grep -q "Done." "$log_path"; then
      done_count=$((done_count+1))
    fi
  done
  [[ "$done_count" -eq 4 ]]
}

any_running() {
  pgrep -af cc_news_pipeline.py | grep -q "cc-news-12m-w30-" && return 0
  return 1
}

log "Waiting for cc-news-12m-w30 q1-q4 to finish..."
while true; do
  if all_done_logs && ! any_running; then
    log "All processes done. Shutting down."
    shutdown -h now
    exit 0
  fi
  sleep "$CHECK_INTERVAL_SEC"
done
