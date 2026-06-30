#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/vitamind/my_project/model6"
PY="$ROOT/.venv/bin/python"
AUX20="$ROOT/data/BTCUSDT_1h_aux_binance_2020Q4.parquet"
AUX21_25="$ROOT/data/BTCUSDT_1h_aux_binance_2021_2025.parquet"
AUX_MERGED="$ROOT/data/BTCUSDT_1h_aux_binance_2020_2025.parquet"
AUX_MERGED_FINAL="$ROOT/data/BTCUSDT_1h_aux_merged_2020_2025.parquet"

# wait for 2020Q4 aux
for i in {1..900}; do
  if [ -s "$AUX20" ]; then
    break
  fi
  sleep 1
done

if [ ! -s "$AUX20" ]; then
  echo "Aux 2020Q4 not ready: $AUX20" >&2
  exit 1
fi

if [ -s "$AUX21_25" ]; then
  $PY - <<'PY'
import pandas as pd
from pathlib import Path
p_old=Path('/home/vitamind/my_project/model6/data/BTCUSDT_1h_aux_binance_2021_2025.parquet')
p_new=Path('/home/vitamind/my_project/model6/data/BTCUSDT_1h_aux_binance_2020Q4.parquet')
out=Path('/home/vitamind/my_project/model6/data/BTCUSDT_1h_aux_binance_2020_2025.parquet')
old=pd.read_parquet(p_old)
new=pd.read_parquet(p_new)
combined=pd.concat([new, old], ignore_index=True).drop_duplicates(subset=['timestamp']).sort_values('timestamp')
combined.to_parquet(out, index=False)
print('Saved', out, 'rows', len(combined))
PY
else
  echo "No $AUX21_25, using 2020Q4 only"
  cp "$AUX20" "$AUX_MERGED"
fi

$PY "$ROOT/merge_aux.py" --binance "$AUX_MERGED" --out "$AUX_MERGED_FINAL"

# Build features without news (news will be added when available)
$PY "$ROOT/build_features.py" \
  --input "$ROOT/data/BTCUSDT_15m.parquet" \
  --output "$ROOT/data/BTCUSDT_15m_features_h20_v2_base_2020_2026.parquet" \
  --aux "$AUX_MERGED_FINAL" \
  --horizon 1 --target-mode log_return --base-tf-min 15

# If news file exists, rebuild with news
NEWS_PATH="/mnt/data/news/ccnews_2020_2026_sentiment.parquet"
if [ -f "$NEWS_PATH" ]; then
  $PY "$ROOT/build_features.py" \
    --input "$ROOT/data/BTCUSDT_15m.parquet" \
    --output "$ROOT/data/BTCUSDT_15m_features_h20_v2_news_xlmr_full_2020_2026.parquet" \
    --aux "$AUX_MERGED_FINAL" \
    --news "$NEWS_PATH" \
    --horizon 1 --target-mode log_return --base-tf-min 15
fi

