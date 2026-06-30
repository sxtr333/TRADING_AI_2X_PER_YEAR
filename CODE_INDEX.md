# Code index

## Code files (detailed)

- `README.md`
  - Purpose: quick notes (latest model, data locations, GPU command).
  - Inputs/outputs: none; documentation only.
  - Dependencies: none.

- `PIPELINE.md`
  - Purpose: leakage‑safe pipeline and benchmark steps.
  - Inputs/outputs: none.
  - Dependencies: references `build_features.py`, `walk_forward_eval.py`, `scripts/benchmark_compare.sh`.

- `build_features.py`
  - Purpose: core feature engineering + target/label generation.
  - Key functions: `build_features`, `maybe_merge_aux`, `maybe_add_basis`, `triple_barrier_labels`.
  - Inputs: OHLCV parquet (`--input`), optional aux/spot parquet/csv.
  - Outputs: features parquet with target/label columns.
  - Dependencies: `trading_keras_core.default_feature_list`, pandas/numpy.

- `trading_keras_core.py`
  - Purpose: shared feature schema, target helpers, tf.data windowing, model blocks.
  - Key items: `default_feature_list`, `add_time_features`, `build_target`, `make_tf_dataset`, transformer/TCN blocks.
  - Inputs/outputs: in‑memory arrays/df; no files.
  - Dependencies: TensorFlow, numpy, pandas.

- `model_layers.py`
  - Purpose: custom Keras layers (RevIN, TSMixer, ITransformer, LastStep, DropPath).
  - Inputs/outputs: tensors.
  - Dependencies: TensorFlow.

- `train_keras.py`
  - Purpose: main training (multi‑arch, cls+price heads, leakage‑safe split).
  - Key functions: `pick_feature_cols`, `compute_norm_stats`, `build_model`, window dataset builders.
  - Inputs: features parquet + CLI config.
  - Outputs: `.keras` model + `norm_stats_*.npz` (mean/std/feature_names/quantiles/seq_len).
  - Dependencies: `model_layers`, TensorFlow, pandas/numpy.

- `trading_keras.py`
  - Purpose: legacy training script (fixed architecture, sample balancing).
  - Inputs: features parquet.
  - Outputs: model + stats.
  - Dependencies: `trading_keras_core`.

- `evaluate_keras.py`
  - Purpose: evaluation of classification/regression heads with threshold tuning.
  - Inputs: features parquet + model + stats.
  - Outputs: metrics CSV per row.
  - Dependencies: `trading_keras_core`.

- `eval_metrics.py`
  - Purpose: quick regression metrics on test split; supports .ckpt/.keras.
  - Inputs: features parquet + model/weights.
  - Outputs: console metrics.
  - Dependencies: `trading_keras_core`.

- `walk_forward_eval.py`
  - Purpose: walk‑forward backtest with fees/slippage/risk rules.
  - Inputs: features parquet + model + stats + split timestamps.
  - Outputs: optional report/equity CSV/PNG.
  - Dependencies: `train_keras.DropPath`, `model_layers`.

- `inference_keras.py`
  - Purpose: offline inference on last N windows.
  - Inputs: features parquet + model + stats.
  - Outputs: `predictions.csv`.
  - Dependencies: `trading_keras_core.default_feature_list`.

- `realtime_stream.py`
  - Purpose: realtime Bybit predictor (websocket stream, rolling buffer).
  - Inputs: live data + model.
  - Outputs: console/log CSV of predictions.
  - Dependencies: websockets/asyncio, TF, pandas.

- `serve_fastapi.py`
  - Purpose: serve model via FastAPI REST (predict endpoint).
  - Inputs: model path + stats.
  - Outputs: HTTP server.
  - Dependencies: FastAPI, uvicorn, TF.

- `forecast_itransformer_24h.py`
  - Purpose: 24h forecast using iTransformer; generates CSV/HTML/PNG.
  - Inputs: features parquet + model + stats.
  - Outputs: `forecast_24h.csv/html/png`.
  - Dependencies: `model_layers`, `trading_keras_core`.

- `eval_and_forecast.py`
  - Purpose: eval + naive 24h projection (SVG in HTML).
  - Inputs: features parquet + model + stats.
  - Outputs: `forecast_24h.csv/html`.
  - Dependencies: `model_layers`, `trading_keras_core`.

- `ensemble_predict.py`
  - Purpose: ensemble predictions from multiple models.
  - Inputs: features parquet + list of models + stats.
  - Outputs: `ensemble_preds.csv`.

- `feature_importance.py`
  - Purpose: permutation importance on features.
  - Inputs: features parquet + model.
  - Outputs: console/metrics.

- `tune_keras.py`
  - Purpose: KerasTuner hyperparameter search.
  - Inputs: features parquet.
  - Outputs: tuner project dir.

- `retrain.py`
  - Purpose: fine‑tune on recent data; optional download.
  - Inputs: base OHLCV + aux + spot.
  - Outputs: new features + finetuned model.

- `build_features_multi_tf.py`
  - Purpose: multi‑TF features (base + higher TFs, EMA/ATR/BB/RV).
  - Inputs: multiple OHLCV parquets.
  - Outputs: single merged features parquet.

- `train_keras_multitf.py`
  - Purpose: train explicit multi‑TF fusion model.
  - Inputs: base features + ctx features.
  - Outputs: model `.keras`.

- `trading_keras_multitf.py`
  - Purpose: model definition for multi‑TF fusion (cross‑attention).
  - Inputs/outputs: tensors.

- `train_keras_multihorizon.py`
  - Purpose: multi‑horizon model with multiple price/cls heads.
  - Inputs: features parquet (with `label_3cls_h{h}` etc).
  - Outputs: model + stats.

- `eval_tb_multi.py`
  - Purpose: evaluate multi‑horizon TB model on test windows.
  - Inputs: hardcoded features/model paths.
  - Outputs: console metrics.

- `export_model.py`
  - Purpose: export `.keras` → SavedModel / ONNX.
  - Inputs: model path.
  - Outputs: SavedModel dir / ONNX file.

- `leakage_sanity.py`
  - Purpose: quick leakage checks (corrs vs labels and future corr).
  - Inputs: features parquet + label column.
  - Outputs: console diagnostics.

- `bybit_data.py`
  - Purpose: Bybit OHLCV downloader → parquet.
  - Inputs: symbol/time range/category.
  - Outputs: `data/BTCUSDT_*m.parquet`.

- `bybit_public_trades.py`
  - Purpose: public trades → aggregated klines parquet.
  - Inputs: symbol/date range.
  - Outputs: parquet per TF.

- `bybit_aux.py`
  - Purpose: Bybit OI/funding/ratios → aux parquet.
  - Inputs: symbol/time range.
  - Outputs: `data/aux.parquet` (default).

- `binance_aux.py`
  - Purpose: Binance metrics/aux from Binance Vision zips.
  - Inputs: start/end, output path.
  - Outputs: `data/BTCUSDT_1h_aux_binance.parquet`.

- `merge_aux.py`
  - Purpose: merge Bybit/Binance aux.
  - Inputs: bybit/binance aux files.
  - Outputs: merged aux parquet.

- `backend/update_fed_events.py`
  - Purpose: download/update Fed events JSON for UI.
  - Inputs: none/CLI.
  - Outputs: `html/fed_events.json`.

- `scripts/fetch_bybit_liquidations.py`
  - Purpose: pull Bybit liquidations and aggregate to 1h.
  - Outputs: `data/liquidations/bybit_liq_*.parquet` (default path).

- `scripts/fetch_bitmex_liquidations.py`
  - Purpose: pull BitMEX liquidations and aggregate to 1h.
  - Outputs: `data/liquidations/bitmex_liq_*.parquet` (default path).

- `scripts/benchmark_compare.sh`
  - Purpose: run PatchTST/TSMixer/iTransformer benchmark.
  - Inputs/outputs: uses `train_keras.py` + outputs models/stats.

- `scripts/update_fed_events.sh`
  - Purpose: wrapper for `backend/update_fed_events.py`.

- `run_all.sh`
  - Purpose: end‑to‑end aux + feature build (1m/15m).

- `run_train_docker.sh` / `run_eval_docker.sh`
  - Purpose: GPU docker training/eval.

- `Dockerfile`
  - Purpose: container definition.

## Опасные места и области внимания

- `build_features.py` / `build_features_multi_tf.py`: смещение `target_ret` и `tb_label` должно использовать `shift(-horizon)`/`shift(-tb_width)` + `forward-fill` чтобы не вводить look-ahead. Осторожно с timezone и UTC-метками (всегда UTC) и Purge Gap.
- `train_*.py`: нормализация (`norm_stats_*.npz`) и `tf.data` окна должны совпадать с `features`. Перепутанные `seq_len`/feature order вызовут mismatch при inference; слепо менять `feature_cols` может сломать `RevIN`.
- `serve_fastapi.py`: требует `TF_USE_LEGACY_KERAS=1`, импорт кастомных классов (`model_layers.RevIN`, `ITransformerBlock`, `DropPath`) до `load_model`. Stats `mean/std` должны быть из того же файла, иначе `Stats feature length mismatch` (например 54 vs 24) или `MC dropout` выдаёт NaN.
- `forecast[_multi]` endpoints используют 15m spacing и принимают `h20/h80/h160`. Не подставляй `interval=h160` без multi stats — multi-модель должна быть загружена вместе со stats.
- `realtime_stream.py` / `inference_keras.py`: если Bybit/WS теряет данные, window shift может создать NaN; MC-таплы `mc_samples>1` увеличивают вычисление и нуждаются в `DropPath`/`RevIN` ready.
- `scripts/fetch_*liquidations.py`: неконсистентный timezone или пропущенные дни в LIQ могут пролить многократные zero-рядные признаки и искажают feature importance.

- `html/*`
  - Purpose: UI mockups (Lightweight Charts) + CSS + news JSON.
  - Special note from `chat_summary.pdf`: all overlays inside `#mockWrap`, percent‑based coords, active timeframe buttons, etc.

- `chat_summary.pdf`
  - Purpose: UI overlay rules + chart requirements (not ML).

## Artifacts / data (grouped inventory)

> Полный перечислитель zip‑кэшей и отчетов очень большой; ниже — группами. Если нужен полный per‑file список, могу сгенерировать отдельный индекс.

- `data/*.parquet`
  - Purpose: OHLCV (`BTCUSDT_*m.parquet`) and features (`*_features*.parquet`) and aux (`*_aux*.parquet`).
  - I/O: inputs for training/eval/inference.

- `data/binance_cache/**.zip`
  - Purpose: raw Binance Vision data cache (spot/perp/funding).
  - I/O: inputs for `binance_aux.py`.

- `data/metrics/binance_raw/**.zip`
  - Purpose: raw Binance metrics zip archives.
  - I/O: inputs for `binance_aux.py`.

- `model_*.keras`, `*.weights.h5`, `*.ckpt.*`, `model_1m/`
  - Purpose: trained models and checkpoints.
  - I/O: inputs for inference/eval.

- `norm_stats_*.npz`
  - Purpose: normalization stats, feature_names, quantiles, seq_len.

- `reports/*.csv` / `reports/*.png` / `*.log`
  - Purpose: walk‑forward outputs, equity curves, optimization grids.

- `forecast_24h.*`, `forecast_itransformer_24h.*`
  - Purpose: forecast outputs (CSV/HTML/PNG).

- `metrics_*.csv`, `metrics_eval.csv`
  - Purpose: offline evaluation metrics outputs.

## Опасные места (leakage / data snooping / timezone / costs)

- Leakage через таргеты: исключать `target_*`, `label_*`, `tb_*` из features (см. `train_keras.py:pick_feature_cols`).
- Split leakage: учитывать `--target-horizon` и `--purge-gap` при time‑split.
- Нормализация: stats считаются только на train; использовать те же stats на val/test/inference.
- Временная зона: все `timestamp` ожидаются в UTC; не смешивать локальные TZ.
- Feature leakage: `build_features.py` использует future prices для labels — не включать их в features.
- Data snooping: подбор порогов/стратегий на test‑сегменте может завышать метрики.
- Комиссии/слиппедж: в `walk_forward_eval.py` по умолчанию 0 — не забыть задать.
- Realtime: Bybit interim bars (`confirm=False`) игнорируются, иначе могут быть look‑ahead.
