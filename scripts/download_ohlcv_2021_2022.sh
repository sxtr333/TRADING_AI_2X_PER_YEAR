#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/vitamind/my_project/model6"
PY="$ROOT/.venv/bin/python"
# Download Bybit public trades and aggregate to 15m candles (2021-2022)
nohup "$PY" "$ROOT/bybit_public_trades.py" \
  --symbol BTCUSDT \
  --start 2021-01-01 \
  --end 2022-12-31 \
  --timeframes 15m \
  --out "$ROOT/data" \
  > "$ROOT/logs/bybit_public_trades_2021_2022.log" 2>&1 &

echo "Started. Log: $ROOT/logs/bybit_public_trades_2021_2022.log"
