#!/usr/bin/env bash
set -euo pipefail
export HF_HOME=/home/vitamind/.cache_hf
export TRANSFORMERS_CACHE=/home/vitamind/.cache_hf
export HF_HUB_CACHE=/home/vitamind/.cache_hf
export TOKENIZERS_PARALLELISM=false

PY="/home/vitamind/my_project/model6/.venv-news/bin/python"
$PY /home/vitamind/my_project/model6/scripts/news_sentiment_hf.py \
  --input /mnt/data/cc-news-all/ccnews_all_dedup.parquet \
  --output /mnt/data/news/ccnews_all_sentiment.parquet \
  --cache /mnt/data/news/ccnews_all_sentiment_cache.parquet \
  --model-ledger /home/vitamind/my_project/model6/models/ledgerbert-sentiment \
  --model-xlm /home/vitamind/my_project/model6/models/twitter-xlmr-sentiment \
  --device cuda \
  --batch-size 16 --max-length 512 --max-chars 2000 --save-every 1000 \
  --require-xlmr
