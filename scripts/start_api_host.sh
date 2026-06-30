#!/usr/bin/env bash
set -euo pipefail
/home/vitamind/my_project/model6/.venv/bin/python /home/vitamind/my_project/model6/serve_fastapi.py \
  --model-h20 /home/vitamind/my_project/model6/model_battle_itransformer.keras \
  --stats-h20 /home/vitamind/my_project/model6/norm_stats_battle_itransformer.npz \
  --model-multi /home/vitamind/my_project/model6/model_15m_itransformer_tb_multi.keras \
  --stats-multi /home/vitamind/my_project/model6/norm_stats_15m_itransformer_tb_multi.npz \
  --features /home/vitamind/my_project/model6/data/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes_serve.parquet \
  --seq-len 256 \
  --host 0.0.0.0 \
  --port 8000
