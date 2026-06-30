#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/vitamind/my_project/model6"
"$ROOT/scripts/start_site.sh"
WEB_URL=$(rg -n "https://.*trycloudflare.com" -m 1 "$ROOT/cloudflared.log" | sed -E 's/.*(https:\/\/[^ ]+).*/\1/' || true)
if [ -n "${WEB_URL:-}" ]; then
  echo "OPEN: ${WEB_URL}/tradeforge_demo_mobile_v9.html"
fi
