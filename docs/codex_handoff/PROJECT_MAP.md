# Project map for model6 (Codex handoff)

This is a practical map of the repository: what each major piece does, where data comes from, and how to run realtime updates safely.

## 0) Quick orientation

- Repo root: `/home/vitamind/my_project/model6`
- Virtualenv: `.venv` (required for local runs)
- Main time frame: **BTCUSDT 15m**
- Key goals: price direction models + trading backtests + meta‑labeling filter + website visualization

## 1) Core folders

- `data/` — raw & feature datasets (Parquet). Includes 15m OHLCV and derived features.
- `data/meta/` — meta‑label datasets for trade/no‑trade models.
- `new_models/` — trained models and stats (often a symlink to `/mnt/oldssd/...`).
- `logs/` — training logs and runtime logs.
- `reports/` — backtest outputs, sweeps, plots, and result summaries.
- `html/` — UI (TradingView‑style frontend).
- `scripts/` — all pipelines: data ingestion, news, training, backtesting, serving.

## 2) Data pipeline (end‑to‑end)

### 2.1 OHLCV updates
- Source: `bybit_data.py`
- Script: `scripts/live_update_once.sh`
  - updates `data/BTCUSDT_15m.parquet`

### 2.2 News ingestion + sentiment
- Ingest: `scripts/news_ingest.py` → raw news in `/mnt/data/news/news_raw.parquet`
- Dedup: `scripts/news_dedup.py`
- Sentiment: `scripts/news_sentiment_hf.py`
  - two HF models: LedgerBERT + XLM‑R
  - outputs `/mnt/data/news/news_sentiment.parquet`
  - sentiment range roughly **−1..+1** (per‑article)

### 2.3 Feature build (with news + macro)
- Script: `build_features.py`
- Key outputs:
  - `data/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes*.parquet`
  - `data/BTCUSDT_15m_features_h20_v2_news_xlmr_v3*.parquet` (legacy for old v5 model)
- Optional daily inputs:
  - `macro and liquidity/data/macro_daily.parquet`
  - `macro and liquidity/data/fed_rates_daily.parquet`
  - `institutional flows/data/institutional_daily.parquet`

### 2.4 Meta dataset build
- Script: `scripts/build_meta_dataset.py`
- Output: `data/meta/meta_dataset_pruned.parquet`
- Important: `meta_y_*` and `meta_label_*` must **never** be used as features (leakage).

## 3) Realtime update loop (every N minutes)

Use the loop script and set interval via env var:

```bash
LIVE_UPDATE_INTERVAL_SEC=60 bash /home/vitamind/my_project/model6/scripts/live_update_loop.sh
```

What happens inside one cycle (`scripts/live_update_once.sh`):
1) ingest + dedup news
2) run sentiment models (GPU if `NEWS_DEVICE=cuda`)
3) update OHLCV (Bybit)
4) rebuild features with news + macro
5) rebuild v3 features for legacy model

Outputs updated every cycle:
- `data/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes_serve.parquet`
- `data/BTCUSDT_15m_features_h20_v2_news_xlmr_v3_serve.parquet`

## 4) Training (core)

### 4.1 Price / multi‑head models
- Scripts: `train_keras.py`, `train_keras_v7.py`
- Output files:
  - `*.keras` model
  - `*_stats.npz` normalization stats (feature order + mean/std + seq_len)
- **Stats must match features** for inference or backtests.

### 4.2 Meta models (trade / no‑trade)
- Script: `train_keras_v7.py` with `--no-price-head` and `--target-col meta_label_*`
- Output directory (latest):
  - `new_models/meta_2026-01-20/`
    - `meta_h20_long.keras`
    - `meta_h20_short.keras`
    - `meta_h80_short_v2.keras`
    - `meta_h160_long_v2.keras`

## 5) Backtesting

### 5.1 Base combo backtest
- Script: `scripts/backtest_trade_combo_leverage.py`
- Grid search: `scripts/backtest_trade_combo_grid.py` → `reports/backtest_trade_combo_grid.csv`
- Old best combo (no meta): ~`+0.7` PnL

### 5.2 Meta‑filtered combo
- Script: `scripts/backtest_trade_combo_meta.py`
- Sweep output: `reports/backtest_trade_combo_meta_*.csv`
- Best known (older meta models):
  - baseline: `+18.18`
  - boosted risk: `+26.05`
  - some local tweaks: `~+29.9`

### 5.3 Purged CV thresholds
- Script: `scripts/purged_cv_thresholds.py`
- Output: `reports/purged_cv_thresholds.csv`

## 6) Website / Serving

### 6.1 FastAPI
- Script: `serve_fastapi.py`
- Endpoints:
  - `/predict` (model inference)
  - `/candles` (OHLCV from features)
  - `/trades` (trade markers)
  - `/news` + `/news_refresh`

### 6.2 UI
- Main UI: `html/aladin_from_image.html`
- Dual model selector (NEW +29 / V5 old)
- Trades overlay from `/trades`

### 6.3 Run everything
- Script: `scripts/run_servers.sh`
  - Starts FastAPI for new model (port 8000)
  - Starts FastAPI for old v5 (port 8001)
  - Starts arbitration backend + UI (ports 8091/8090)

## 7) Datasets (most used)

- Base OHLCV: `data/BTCUSDT_15m.parquet`
- Features (new): `data/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes_with_news_pruned.parquet`
- Features (serve): `data/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes_serve.parquet`
- Features (old v5): `data/BTCUSDT_15m_features_h20_v2_news_xlmr_v3.parquet`
- Meta dataset: `data/meta/meta_dataset_pruned.parquet`

Full schema list is large — see:
- `reports/data_schema_overview.txt`

## 8) Models (key references)

- Best legacy 3‑head price model:
  - `new_models/2026-01-14_news_xlmr_v5_dr_scale/model_15m_itransformer_price_multi_h20_h80_h160_news_xlmr_v5_dr_scale_e14_b32.keras`

- Latest meta models (trade/no‑trade):
  - `new_models/meta_2026-01-20/` (see Section 4.2)

## 9) Known pitfalls

- **Leakage**: never use `meta_y_*`, `meta_label_*`, or future targets as features.
- **Alignment**: meta dataset must align 1‑to‑1 with features by timestamp.
- **Stats**: never mix `*_stats.npz` from another dataset.
- **GPU**: training must abort if no GPU (see `GPU_TRAINING.md`).

## 10) GPU training

See `GPU_TRAINING.md` (updated) for:
- host run and docker run
- CPU limit (~70%)
- GPU preflight (hard fail if no GPU)
- RTX 5070 notes
