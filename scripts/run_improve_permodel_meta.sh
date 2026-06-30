#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/vitamind/my_project/model6"
PY="$ROOT/.venv/bin/python"
FEATURES="$ROOT/data/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes.parquet"
META="$ROOT/data/meta/meta_dataset_pruned_newsflag_mix15_2024calib_full.parquet"
META_DIR="$ROOT/new_models/meta_2026-01-24_newsflag_mix15_2024calib_full"
BEST_H20="$ROOT/reports/backtest_v7_long_short_sweep_newsflag_mix15_2024_best.csv"
BEST_V2="$ROOT/reports/backtest_v7_long_short_v2_sweep_newsflag_mix15_2024_best.csv"
PERMODEL_2024="$ROOT/reports/backtest_meta_newsflag_mix15_2024calib_full_2024_permodel_grid.csv"
BEST_JSON="$ROOT/reports/best_permodel_mix15_2024.json"

echo "[wait] per-model 2024 grid..."
while [[ ! -s "$PERMODEL_2024" ]]; do
  sleep 30
done

echo "[1/3] pick best per-model thresholds (2024)"
$PY - <<'PY'
import json, pandas as pd
src="/home/vitamind/my_project/model6/reports/backtest_meta_newsflag_mix15_2024calib_full_2024_permodel_grid.csv"
dst="/home/vitamind/my_project/model6/reports/best_permodel_mix15_2024.json"
df=pd.read_csv(src)
row=df.loc[df['final_equity'].idxmax()]
best={
  "h20_long": float(row["meta_prob_h20_long"]),
  "h20_short": float(row["meta_prob_h20_short"]),
  "h80_short_v2": float(row["meta_prob_h80_short_v2"]),
  "h160_long_v2": float(row["meta_prob_h160_long_v2"]),
}
with open(dst,"w") as f:
  json.dump(best,f,indent=2)
print("Saved:", dst, best)
PY

H20_LONG=$($PY -c "import json; print(json.load(open('/home/vitamind/my_project/model6/reports/best_permodel_mix15_2024.json'))['h20_long'])")
H20_SHORT=$($PY -c "import json; print(json.load(open('/home/vitamind/my_project/model6/reports/best_permodel_mix15_2024.json'))['h20_short'])")
H80_SHORT=$($PY -c "import json; print(json.load(open('/home/vitamind/my_project/model6/reports/best_permodel_mix15_2024.json'))['h80_short_v2'])")
H160_LONG=$($PY -c "import json; print(json.load(open('/home/vitamind/my_project/model6/reports/best_permodel_mix15_2024.json'))['h160_long_v2'])")
PERMODEL_STR="h20_long=$H20_LONG,h20_short=$H20_SHORT,h80_short_v2=$H80_SHORT,h160_long_v2=$H160_LONG"

echo "[1b] backtest 2025 with per-model thresholds"
$PY "$ROOT/scripts/backtest_trade_combo_meta.py" \
  --features "$FEATURES" \
  --meta-features "$META" \
  --meta-model-dir "$META_DIR" \
  --best-h20 "$BEST_H20" \
  --best-v2 "$BEST_V2" \
  --meta-prob-per-model "$PERMODEL_STR" \
  --out-csv "$ROOT/reports/backtest_meta_newsflag_mix15_2024calib_full_2025_permodel.csv"

echo "[2/3] meta-sizing grid on 2024"
mkdir -p "$ROOT/reports/meta_sizing_2024"
for scale in 0.6 0.8 1.0; do
  for power in 1.0 1.2 1.4; do
    out="$ROOT/reports/meta_sizing_2024/permodel_scale_${scale}_power_${power}.csv"
    $PY "$ROOT/scripts/backtest_trade_combo_meta.py" \
      --features "$FEATURES" \
      --meta-features "$META" \
      --meta-model-dir "$META_DIR" \
      --best-h20 "$BEST_H20" \
      --best-v2 "$BEST_V2" \
      --meta-prob-per-model "$PERMODEL_STR" \
      --meta-sizing \
      --meta-size-scale "$scale" \
      --meta-size-power "$power" \
      --start 2024-01-01T00:00:00+00:00 --end 2025-01-01T00:00:00+00:00 \
      --out-csv "$out"
  done
done

echo "[2b] pick best meta-sizing and run 2025"
$PY - <<'PY'
import glob, pandas as pd
import re
files=glob.glob("/home/vitamind/my_project/model6/reports/meta_sizing_2024/permodel_scale_*_power_*.csv")
rows=[]
for f in files:
    df=pd.read_csv(f)
    row=df.iloc[0].to_dict()
    m=re.search(r"scale_([0-9.]+)_power_([0-9.]+)", f)
    row["scale"]=float(m.group(1))
    row["power"]=float(m.group(2))
    rows.append(row)
best=max(rows, key=lambda r: r["final_equity"])
pd.DataFrame(rows).to_csv("/home/vitamind/my_project/model6/reports/meta_sizing_2024_grid.csv", index=False)
print("Best sizing:", best)
with open("/home/vitamind/my_project/model6/reports/meta_sizing_best_2024.txt","w") as f:
    f.write(f"{best['scale']} {best['power']}")
PY

SCALE=$(awk '{print $1}' /home/vitamind/my_project/model6/reports/meta_sizing_best_2024.txt)
POWER=$(awk '{print $2}' /home/vitamind/my_project/model6/reports/meta_sizing_best_2024.txt)

$PY "$ROOT/scripts/backtest_trade_combo_meta.py" \
  --features "$FEATURES" \
  --meta-features "$META" \
  --meta-model-dir "$META_DIR" \
  --best-h20 "$BEST_H20" \
  --best-v2 "$BEST_V2" \
  --meta-prob-per-model "$PERMODEL_STR" \
  --meta-sizing \
  --meta-size-scale "$SCALE" \
  --meta-size-power "$POWER" \
  --out-csv "$ROOT/reports/backtest_meta_newsflag_mix15_2024calib_full_2025_permodel_sized.csv"

echo "[3/3] regime/news filters on 2025"
$PY "$ROOT/scripts/backtest_trade_combo_meta.py" \
  --features "$FEATURES" \
  --meta-features "$META" \
  --meta-model-dir "$META_DIR" \
  --best-h20 "$BEST_H20" \
  --best-v2 "$BEST_V2" \
  --meta-prob-per-model "$PERMODEL_STR" \
  --meta-sizing \
  --meta-size-scale "$SCALE" \
  --meta-size-power "$POWER" \
  --rv-gate-pct 60 \
  --out-csv "$ROOT/reports/backtest_meta_newsflag_mix15_2024calib_full_2025_permodel_sized_rvgate60.csv"

$PY "$ROOT/scripts/backtest_trade_combo_meta.py" \
  --features "$FEATURES" \
  --meta-features "$META" \
  --meta-model-dir "$META_DIR" \
  --best-h20 "$BEST_H20" \
  --best-v2 "$BEST_V2" \
  --meta-prob-per-model "$PERMODEL_STR" \
  --meta-sizing \
  --meta-size-scale "$SCALE" \
  --meta-size-power "$POWER" \
  --rv-gate-pct 60 \
  --require-news-present \
  --out-csv "$ROOT/reports/backtest_meta_newsflag_mix15_2024calib_full_2025_permodel_sized_rvgate60_news.csv"

echo "Done."
