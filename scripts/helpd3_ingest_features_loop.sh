#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/helpd3/my_project/model6
if [ -f "$ROOT/.venv-news/bin/activate" ]; then
  # Fallback venv on helpd3
  # shellcheck disable=SC1091
  source "$ROOT/.venv-news/bin/activate"
elif [ -f "$ROOT/.venv/bin/activate" ]; then
  # Preferred project venv
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

NEWS_IN=${NEWS_IN:-/mnt/data/news/ccnews_2026_sentiment.parquet}
NEWS_MERGED=${NEWS_MERGED:-$ROOT/data/news/ccnews_2026_live_merged.parquet}
NEWS_MERGED_LOCAL=${NEWS_MERGED_LOCAL:-$ROOT/data/news/ccnews_2026_live_merged.parquet}
OHLCV_PATH=${OHLCV_PATH:-$ROOT/data/BTCUSDT_15m.parquet}
FEATURES_OUT=${FEATURES_OUT:-$ROOT/data/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes.parquet}
SERVE_OUT=${SERVE_OUT:-$ROOT/data/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes_serve.parquet}

mkdir -p "$ROOT/data/news" /mnt/data/news

PY_BIN="$(command -v python || true)"
if [ -z "$PY_BIN" ]; then
  PY_BIN="$(command -v python3 || true)"
fi
if [ -z "$PY_BIN" ]; then
  echo "[ingest-features] no python/python3 in PATH" >&2
  exit 127
fi

while true; do
  echo "[ingest-features] $(date -u '+%Y-%m-%d %H:%M:%S') cycle start" >> /tmp/helpd3_ingest_features.log

  export NEWS_IN NEWS_MERGED
  if [ -f "$NEWS_IN" ]; then
    "$PY_BIN" - <<'PY'
import os
import pandas as pd
news_in = os.environ.get('NEWS_IN', '/mnt/data/news/ccnews_2026_sentiment.parquet')
news_merged = os.environ.get('NEWS_MERGED', '/mnt/data/news/ccnews_2026_live_merged.parquet')
new_df = pd.read_parquet(news_in)
if os.path.exists(news_merged):
    base = pd.read_parquet(news_merged)
    df = pd.concat([base, new_df], ignore_index=True)
else:
    df = new_df
if 'url' in df.columns:
    df = df.dropna(subset=['url']).drop_duplicates(subset=['url'], keep='last')
if 'published_at' in df.columns:
    try:
        df['published_at'] = pd.to_datetime(df['published_at'], utc=True, errors='coerce')
        df = df.sort_values('published_at')
    except Exception:
        pass
df.to_parquet(news_merged, index=False)
print('merged_rows', len(df))
PY
    # API container sees project bind mount (/app -> $ROOT), not host /mnt paths.
    # Keep a local mirror under $ROOT/data/news to ensure build_features can read it.
    if [ "$NEWS_MERGED" != "$NEWS_MERGED_LOCAL" ]; then
      cp -f "$NEWS_MERGED" "$NEWS_MERGED_LOCAL" || true
    fi
  fi

  read -r START_ISO END_ISO < <("$PY_BIN" - <<'PY'
from pathlib import Path
import datetime as dt
import pandas as pd
path = Path('/home/helpd3/my_project/model6/data/BTCUSDT_15m.parquet')
now = dt.datetime.now(dt.timezone.utc)
end = now.isoformat()
if path.exists():
    df = pd.read_parquet(path, columns=['timestamp'])
    last_ts = pd.to_datetime(df['timestamp'], utc=True).max()
    if pd.notna(last_ts):
        start_dt = last_ts - pd.Timedelta(days=3)
        # If local OHLCV is too old, avoid multi-month backfill in realtime loop.
        min_start = now - pd.Timedelta(days=7)
        if start_dt < min_start:
            start_dt = min_start
        start = start_dt.isoformat()
    else:
        start = (now - pd.Timedelta(days=7)).isoformat()
else:
    start = (now - pd.Timedelta(days=7)).isoformat()
print(start, end)
PY
)

  "$PY_BIN" "$ROOT/bybit_data.py" --symbol BTCUSDT --timeframes 15m --start "$START_ISO" --end "$END_ISO" --out "$ROOT/data" || true

  if [ -f "$NEWS_MERGED_LOCAL" ]; then
    (cd "$ROOT" && docker compose -f docker-compose.site.yml exec -T api python /app/build_features.py \
      --input /app/data/BTCUSDT_15m.parquet \
      --output /app/data/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes.parquet \
      --serve-output /app/data/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes_serve.parquet \
      --horizon 20 \
      --multi-horizons 20,40,60,80,100,120,140,160 \
      --news /app/data/news/ccnews_2026_live_merged.parquet \
      --news-windows "1h,4h,12h,24h" \
      --news-ewm "4h,12h,24h") || true
  fi

  echo "[ingest-features] $(date -u '+%Y-%m-%d %H:%M:%S') cycle done" >> /tmp/helpd3_ingest_features.log
  sleep 900
done
