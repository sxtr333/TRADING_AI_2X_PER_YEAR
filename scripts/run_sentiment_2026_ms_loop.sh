#!/usr/bin/env bash
set -euo pipefail
LOCK=/tmp/ccnews_sentiment_ms.lock
while true; do
  if command -v flock >/dev/null 2>&1; then
    flock -n "$LOCK" /home/vitamind/my_project/model6/scripts/run_sentiment_2026_ms.sh || true
  else
    /home/vitamind/my_project/model6/scripts/run_sentiment_2026_ms.sh || true
  fi
  sleep 300
done
