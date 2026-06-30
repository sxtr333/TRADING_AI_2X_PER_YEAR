#!/usr/bin/env bash
set -euo pipefail

# v7 combo: short-horizon pruned + long-horizon daily

BIAS_H20_USD="50.92006187439256" \
BIAS_H80_USD="101.25942521095567" \
BIAS_H160_USD="427.0637288093567" \
python3 /home/vitamind/my_project/model6/serve_fastapi.py \
  --model-h20 /home/vitamind/my_project/model6/new_models/2026-01-18_v7_pruned/model_15m_itransformer_v7_pruned.keras \
  --stats-h20 /home/vitamind/my_project/model6/new_models/2026-01-18_v7_pruned/norm_stats_v7_pruned.npz \
  --model-h80 /home/vitamind/my_project/model6/new_models/2026-01-18_v7_long_daily_v2/model_15m_itransformer_v7_long_daily_v2.keras \
  --stats-h80 /home/vitamind/my_project/model6/new_models/2026-01-18_v7_long_daily_v2/norm_stats_v7_long_daily_v2.npz \
  --model-h160 /home/vitamind/my_project/model6/new_models/2026-01-18_v7_long_daily_v2/model_15m_itransformer_v7_long_daily_v2.keras \
  --stats-h160 /home/vitamind/my_project/model6/new_models/2026-01-18_v7_long_daily_v2/norm_stats_v7_long_daily_v2.npz
