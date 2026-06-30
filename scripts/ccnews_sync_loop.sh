#!/usr/bin/env bash
set -euo pipefail

DEST_HOST=${DEST_HOST:-helpd3@100.77.192.122}
RSYNC_SSH=${RSYNC_SSH:-"ssh -o ConnectTimeout=10 -o ServerAliveInterval=20 -o ServerAliveCountMax=3"}

SRC_RAW=${SRC_RAW:-/mnt/data/cc-news-2026/}
SRC_NEWS=${SRC_NEWS:-/mnt/data/news/}
DST_RAW=${DST_RAW:-/mnt/data/cc-news-2026/}
DST_NEWS=${DST_NEWS:-/mnt/data/news/}

while true; do
  rsync -az --partial --inplace -e "$RSYNC_SSH" "$SRC_RAW" "$DEST_HOST:$DST_RAW" || true
  rsync -az --partial --inplace -e "$RSYNC_SSH" \
    "$SRC_NEWS/ccnews_2026_sentiment.parquet" \
    "$SRC_NEWS/ccnews_2026_sentiment_cache.parquet" \
    "$DEST_HOST:$DST_NEWS" || true
  sleep 300
done
