#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/vitamind/my_project/model6"
PY="$ROOT/.venv/bin/python"
AUX20="$ROOT/data/BTCUSDT_1h_aux_binance_2020Q4.parquet"
AUX21_22="$ROOT/data/BTCUSDT_1h_aux_binance_2021_2022.parquet"
AUX_BIN_OUT="$ROOT/data/BTCUSDT_1h_aux_binance_2020_2022.parquet"
AUX_MERGED="$ROOT/data/BTCUSDT_1h_aux_merged_2020_2022.parquet"
CANDLES="$ROOT/data/BTCUSDT_15m.parquet"
CANDLES_2020_2022="$ROOT/data/BTCUSDT_15m_2020_2022.parquet"
FEAT_2020_2022="$ROOT/data/BTCUSDT_15m_features_h20_v2_base_2020_2022.parquet"
FEAT_EXISTING="$ROOT/data/BTCUSDT_15m_features_h20_v2_news_xlmr_full_2023_2026.parquet"
FEAT_OUT="$ROOT/data/BTCUSDT_15m_features_h20_v2_news_xlmr_full_2020_2026.parquet"

# wait for aux files
for i in {1..7200}; do
  if [ -s "$AUX20" ] && [ -s "$AUX21_22" ]; then
    break
  fi
  sleep 1
done

if [ ! -s "$AUX20" ] || [ ! -s "$AUX21_22" ]; then
  echo "Aux files not ready" >&2
  exit 1
fi

# merge aux binance
$PY - <<'PY'
import pandas as pd
from pathlib import Path
p20=Path('/home/vitamind/my_project/model6/data/BTCUSDT_1h_aux_binance_2020Q4.parquet')
p21=Path('/home/vitamind/my_project/model6/data/BTCUSDT_1h_aux_binance_2021_2022.parquet')
out=Path('/home/vitamind/my_project/model6/data/BTCUSDT_1h_aux_binance_2020_2022.parquet')
old=pd.read_parquet(p21)
new=pd.read_parquet(p20)
combined=pd.concat([new, old], ignore_index=True).drop_duplicates(subset=['timestamp']).sort_values('timestamp')
combined.to_parquet(out, index=False)
print('Saved', out, 'rows', len(combined))
PY

# merge aux (binance only)
$PY "$ROOT/merge_aux.py" --binance "$AUX_BIN_OUT" --out "$AUX_MERGED"

# subset candles 2020-10-23..2022-12-31
$PY - <<'PY'
import pandas as pd
from pathlib import Path
cand=Path('/home/vitamind/my_project/model6/data/BTCUSDT_15m.parquet')
out=Path('/home/vitamind/my_project/model6/data/BTCUSDT_15m_2020_2022.parquet')
start=pd.Timestamp('2020-10-23T00:00:00Z')
end=pd.Timestamp('2022-12-31T23:59:59Z')
df=pd.read_parquet(cand)
df['timestamp']=pd.to_datetime(df['timestamp'], utc=True)
sub=df[(df['timestamp']>=start)&(df['timestamp']<=end)].sort_values('timestamp')
sub.to_parquet(out, index=False)
print('Saved', out, 'rows', len(sub))
PY

# build features for 2020-2022 (no news)
$PY "$ROOT/build_features.py" \
  --input "$CANDLES_2020_2022" \
  --output "$FEAT_2020_2022" \
  --aux "$AUX_MERGED" \
  --horizon 1 --target-mode log_return --base-tf-min 15

# concat with existing 2023-2026 features (align columns, fill missing with 0)
$PY - <<'PY'
import pandas as pd
from pathlib import Path
f20=Path('/home/vitamind/my_project/model6/data/BTCUSDT_15m_features_h20_v2_base_2020_2022.parquet')
fold=Path('/home/vitamind/my_project/model6/data/BTCUSDT_15m_features_h20_v2_news_xlmr_full_2023_2026.parquet')
out=Path('/home/vitamind/my_project/model6/data/BTCUSDT_15m_features_h20_v2_news_xlmr_full_2020_2026.parquet')

new=pd.read_parquet(f20)
old=pd.read_parquet(fold)
# align columns
all_cols=sorted(set(new.columns) | set(old.columns))
for df in (new, old):
    for c in all_cols:
        if c not in df.columns:
            df[c]=0.0
    df=df[all_cols]

combined=pd.concat([new[all_cols], old[all_cols]], ignore_index=True)
combined=combined.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
combined.to_parquet(out, index=False)
print('Saved', out, 'rows', len(combined))
PY

