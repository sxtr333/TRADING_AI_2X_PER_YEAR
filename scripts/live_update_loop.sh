#!/usr/bin/env bash
set -euo pipefail
cd /home/vitamind/my_project/model6

# Use .env if present
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# Keep sentiment on CPU for stability while GPU serves models
export NEWS_DEVICE=cpu
export NEWS_BATCH_SIZE=8
export NEWS_MAX_LENGTH=256
export NEWS_SKIP_SENTIMENT=1

while true; do
  ./scripts/live_update_once.sh
  sleep 300
done
