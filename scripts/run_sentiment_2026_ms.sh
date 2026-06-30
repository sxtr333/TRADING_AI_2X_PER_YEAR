#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/vitamind/my_project/model6
source "$ROOT/.venv-news/bin/activate"

INPUT_DIR=${INPUT_DIR:-/mnt/data/cc-news-2026}
OUT_DIR=${OUT_DIR:-/mnt/data/news}
OUT=${OUT:-$OUT_DIR/ccnews_2026_sentiment.parquet}
CACHE=${CACHE:-$OUT_DIR/ccnews_2026_sentiment_cache.parquet}
TMP=${TMP:-/tmp/ccnews_2026_all.parquet}

NEWS_DEVICE=${NEWS_DEVICE:-auto}
NEWS_BATCH_SIZE=${NEWS_BATCH_SIZE:-8}
NEWS_MAX_LENGTH=${NEWS_MAX_LENGTH:-512}
NEWS_MAX_CHARS=${NEWS_MAX_CHARS:-2000}
NEWS_MAX_ROWS=${NEWS_MAX_ROWS:-2000}

mkdir -p "$OUT_DIR"

if [[ "$NEWS_DEVICE" == "auto" ]]; then
  if "$ROOT/.venv-news/bin/python" - <<'PY' >/dev/null 2>&1
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
  then
    NEWS_DEVICE="cuda"
  else
    NEWS_DEVICE="cpu"
  fi
fi

echo "[sentiment-ms] $(date -u '+%Y-%m-%d %H:%M:%S') merge start" >> /tmp/ccnews_2026_sentiment_ms.log

python - <<'PY'
import glob
import os
import pandas as pd
paths = sorted(glob.glob('/mnt/data/cc-news-2026/*.parquet'))
if not paths:
    raise SystemExit('no parquet files yet')
frames = [pd.read_parquet(p) for p in paths]
df = pd.concat(frames, ignore_index=True)
if 'url' in df.columns:
    df = df.dropna(subset=['url']).drop_duplicates(subset=['url'], keep='last')
max_rows = int(os.environ.get('NEWS_MAX_ROWS', '2000'))
if max_rows > 0 and len(df) > max_rows:
    df = df.tail(max_rows).copy()
df.to_parquet('/tmp/ccnews_2026_all.parquet', index=False)
print('merged_rows', len(df))
PY

HF_HOME=${HF_HOME:-/home/vitamind/.cache_hf} \
HF_HUB_OFFLINE=0 \
TRANSFORMERS_OFFLINE=0 \
"$ROOT/.venv-news/bin/python" "$ROOT/scripts/news_sentiment_hf.py" \
  --input "$TMP" \
  --output "$OUT" \
  --cache "$CACHE" \
  --model-ledger "$(if [[ -d "$ROOT/models/ledgerbert-sentiment" ]]; then echo "$ROOT/models/ledgerbert-sentiment"; else echo "ExponentialScience/LedgerBERT-Market-Sentiment"; fi)" \
  --model-xlm "$(if [[ -d "$ROOT/models/twitter-xlmr-sentiment" ]]; then echo "$ROOT/models/twitter-xlmr-sentiment"; else echo "cardiffnlp/twitter-xlm-roberta-base-sentiment"; fi)" \
  --device "$NEWS_DEVICE" \
  --batch-size "$NEWS_BATCH_SIZE" \
  --max-length "$NEWS_MAX_LENGTH" \
  --max-chars "$NEWS_MAX_CHARS" \
  --save-every 1000 \
  --require-xlmr

echo "[sentiment-ms] $(date -u '+%Y-%m-%d %H:%M:%S') done" >> /tmp/ccnews_2026_sentiment_ms.log
