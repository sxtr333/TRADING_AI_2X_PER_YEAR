#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/vitamind/my_project/model6"
PY="$ROOT/.venv/bin/python"

# Build Binance aux for 2021-2022
$PY "$ROOT/binance_aux.py" --start 2021-01-01 --end 2022-12-31 --out "$ROOT/data/BTCUSDT_1h_aux_binance_2021_2022.parquet"

# Merge with existing 2023-2025 aux (if present)
if [ -f "$ROOT/data/BTCUSDT_1h_aux_binance.parquet" ]; then
  $PY - <<'PY'
import pandas as pd
from pathlib import Path
p_old=Path('/home/vitamind/my_project/model6/data/BTCUSDT_1h_aux_binance.parquet')
p_new=Path('/home/vitamind/my_project/model6/data/BTCUSDT_1h_aux_binance_2021_2022.parquet')
out=Path('/home/vitamind/my_project/model6/data/BTCUSDT_1h_aux_binance_2021_2025.parquet')
old=pd.read_parquet(p_old)
new=pd.read_parquet(p_new)
combined=pd.concat([new, old], ignore_index=True).drop_duplicates(subset=['timestamp']).sort_values('timestamp')
combined.to_parquet(out, index=False)
print('Saved', out, 'rows', len(combined))
PY
else
  echo "No existing BTCUSDT_1h_aux_binance.parquet, skipping merge."
fi

# Merge aux for features
if [ -f "$ROOT/data/BTCUSDT_1h_aux_binance_2021_2025.parquet" ]; then
  $PY "$ROOT/merge_aux.py" --binance "$ROOT/data/BTCUSDT_1h_aux_binance_2021_2025.parquet" --out "$ROOT/data/BTCUSDT_1h_aux_merged_2021_2025.parquet"
fi

