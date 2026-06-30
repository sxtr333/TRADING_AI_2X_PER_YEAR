#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/helpd3/my_project/model6
RUN=$ROOT/scripts/run_ccnews_shard_loop.sh
export PYTHON_BIN=/home/helpd3/my_project/model6/.venv_ccnews/bin/python

LOGROOT=/home/helpd3/cc-news/logs
CKPTROOT=/home/helpd3/cc-news/checkpoints

mkdir -p "$LOGROOT" "$CKPTROOT" /home/helpd3/cc-news/2023 /home/helpd3/cc-news/2024 /home/helpd3/cc-news/2025_2026

# Safe stop of old shard loops (avoid pkill -f pitfalls over SSH)
PIDS=$(ps -eo pid,args | awk '$2=="bash" && $3=="/home/helpd3/my_project/model6/scripts/run_ccnews_shard_loop.sh" {print $1}')
if [ -n "$PIDS" ]; then
  kill $PIDS || true
  sleep 1
fi

rm -f /tmp/ccnews_shard_2023.lock /tmp/ccnews_shard_2024.lock /tmp/ccnews_shard_2025_2026.lock

nohup "$RUN" shard_2023 2023-01 2023-12 /home/helpd3/cc-news/2023 "$CKPTROOT/processed_2023.txt" "$LOGROOT/run_2023.log" >/tmp/ccnews_shard_2023.nohup.log 2>&1 &
nohup "$RUN" shard_2024 2024-01 2024-12 /home/helpd3/cc-news/2024 "$CKPTROOT/processed_2024.txt" "$LOGROOT/run_2024.log" >/tmp/ccnews_shard_2024.nohup.log 2>&1 &
nohup "$RUN" shard_2025_2026 2025-01 2026-12 /home/helpd3/cc-news/2025_2026 "$CKPTROOT/processed_2025_2026.txt" "$LOGROOT/run_2025_2026.log" >/tmp/ccnews_shard_2025_2026.nohup.log 2>&1 &

echo "started"
