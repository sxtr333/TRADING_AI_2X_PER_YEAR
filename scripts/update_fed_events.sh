#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_FILE="${ROOT_DIR}/html/fed_events.json"

python3 "${ROOT_DIR}/backend/update_fed_events.py" --out "${OUT_FILE}"
echo "Saved ${OUT_FILE}"
