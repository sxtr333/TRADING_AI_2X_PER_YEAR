#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PY="${ROOT_DIR}/.venv/bin/python"

NEWS_2023_2024="/mnt/data/news/ccnews_2023_2024_sentiment.parquet"
NEWS_12M="/mnt/data/news/ccnews_12m_all_sentiment.parquet"
NEWS_OUT="/mnt/data/news/ccnews_2023_2026_all_sentiment.parquet"

FEATURES_OUT="${ROOT_DIR}/data/BTCUSDT_15m_features_h20_v2_news_xlmr_full_2023_2026.parquet"
SERVE_OUT="${ROOT_DIR}/data/BTCUSDT_15m_features_h20_v2_news_xlmr_full_2023_2026_serve.parquet"

OHLCV_PATH="${ROOT_DIR}/data/BTCUSDT_15m.parquet"
AUX_PATH="${ROOT_DIR}/data/BTCUSDT_15m_aux.parquet"
MACRO_DAILY_PATH="${ROOT_DIR}/macro and liquidity/data/macro_daily.parquet"
FED_DAILY_PATH="${ROOT_DIR}/macro and liquidity/data/fed_rates_daily.parquet"
INST_DAILY_PATH="${ROOT_DIR}/institutional flows/data/institutional_daily.parquet"

LOG="/mnt/data/news/news_full_pipeline_2023_2026.log"

# Wait for sentiment file
while [ ! -s "$NEWS_2023_2024" ]; do
  echo "[wait] $NEWS_2023_2024 not ready yet" >> "$LOG"
  sleep 30
  # If the sentiment process died, exit so user can restart
  if ! ps aux | rg -q "news_sentiment_hf.py"; then
    echo "[error] news_sentiment_hf.py is not running; aborting" >> "$LOG"
    exit 1
  fi
  if [ -f "$NEWS_2023_2024" ] && [ ! -s "$NEWS_2023_2024" ]; then
    echo "[warn] $NEWS_2023_2024 exists but is empty; waiting" >> "$LOG"
  fi
  if [ -f "$NEWS_2023_2024" ] && [ -s "$NEWS_2023_2024" ]; then
    break
  fi

done

echo "[info] Merging sentiment files" >> "$LOG"

$VENV_PY - <<PY
from pathlib import Path
import pandas as pd

p_2023_2024 = Path("$NEWS_2023_2024")
p_12m = Path("$NEWS_12M")

frames = []
for p in [p_2023_2024, p_12m]:
    if p.exists() and p.stat().st_size > 0:
        frames.append(pd.read_parquet(p))

if not frames:
    raise SystemExit("No sentiment sources found")

df = pd.concat(frames, ignore_index=True)
cols = df.columns
if 'url' in cols:
    df = df.dropna(subset=['url']).drop_duplicates(subset=['url'])
elif 'canonical_url' in cols:
    df = df.dropna(subset=['canonical_url']).drop_duplicates(subset=['canonical_url'])
else:
    keys = [c for c in ['title','published_at'] if c in cols]
    df = df.drop_duplicates(subset=keys) if keys else df.drop_duplicates()

out = Path("$NEWS_OUT")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(out, index=False)
print(f"Saved {out} rows={len(df)}")
PY

echo "[info] Building features with merged news" >> "$LOG"

NEWS_PATH="$NEWS_OUT" \
$VENV_PY "${ROOT_DIR}/build_features.py" \
  --input "$OHLCV_PATH" \
  --aux "$AUX_PATH" \
  --output "$FEATURES_OUT" \
  --serve-output "$SERVE_OUT" \
  --horizon 20 \
  --multi-horizons 20,40,60,80,100,120,140,160 \
  --news "$NEWS_OUT" \
  --news-windows "1h,4h,12h,24h" \
  --news-ewm "4h,12h,24h" \
  --macro-daily-path "$MACRO_DAILY_PATH" \
  --fed-daily-path "$FED_DAILY_PATH" \
  --inst-daily-path "$INST_DAILY_PATH" \
  >> "$LOG" 2>&1

echo "[done] Features saved: $FEATURES_OUT" >> "$LOG"
