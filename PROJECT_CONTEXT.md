# PROJECT_CONTEXT (model6)

- **Проект** объединяет сбор фич, обучение трансформерных моделей и FastAPI/HTML-интерфейс для прогноза цены BTCUSDT на 15-минутных свечах и вывода пунктирной линии в масштабе графика.
- **Фичи** строятся из паркетов `data/BTCUSDT_*m.parquet` и вспомогательных датасетов (LIQ, funding, OB, basis, Bybit/Binance метрики) в `build_features.py`/`build_features_multi_tf.py`.
- **Модели** — iTransformer/patchtst/tsmixer (цели `target_ret`, `label_3cls`, `tb_label`, VF-мультиградиенты), сохраняются `.keras` + `.npz` stats (напр. `norm_stats_battle_itransformer.npz` и `norm_stats_15m_itransformer_tb_multi.npz`).
- **Боевая ветка**: `best/boevaya` содержит стабильные веса модели на 5h (h20) и соответствующие stats для фронтенда.
- **Сервер**: `serve_fastapi.py` загружает h20 и multi-модели и возвращает `/forecast` + `/forecast_multi` с шагом 15m; используется UI `html/aladin_from_image.html`.
- **UI** теперь строится на Lightweight Charts, слушает кнопки `PREDICT → 5H/20H/40H/80H/160H → ПРОГНОЗ` и рисует прогноз от FastAPI.
- **Реалтайм**: `realtime_stream.py` подключается к Bybit, делает MC-прогон и логирует drift/toxic сигналы чтобы вскрыть фазу исполнения.
- **Отчеты**: `reports/` и `walk_forward_eval.py` выводят equity curve, thresholds, fee/slippage тесты; `eval_metrics.py`/`eval_tb_multi.py` оценивают классификационные метрики.
- **Инструменты**: `ensemble_predict.py`, `feature_importance.py`, `tune_keras.py`, `retrain.py` помогают собрать ансамбли, оценить признаки и переобучить модель.
- **Интеграция**: `html/` содержит layout + `chat_summary.pdf` с hotspots; FastAPI ожидает `norm_stats`/`features` из `data/BTCUSDT_15m_features_h20_v2.parquet`.

## Быстрый старт (команды для ключевых сценариев)

1. Построить 15m фичи (OHLCV + aux + labels):
   ```bash
   python3 build_features.py \
     --input data/BTCUSDT_15m.parquet \
     --aux data/BTCUSDT_15m_aux.parquet \
     --output data/BTCUSDT_15m_features_h20_v2.parquet \
     --horizon 20 \
     --multi-horizons 20,80,160
   ```
2. Обучить iTransformer (VF + multi):
   ```bash
   python3 train_keras.py \
     --features data/BTCUSDT_15m_features_h20_v2.parquet \
     --model-out model_15m_itransformer_vf.keras \
     --stats-out norm_stats_15m_itransformer_vf.npz \
     --seq-len 256 --epochs 12 --arch itransformer --revin --multi-horizons 20,80,160
   ```
3. Оценить walk-forward/fee/slippage:
   ```bash
   python3 walk_forward_eval.py \
     --features data/BTCUSDT_15m_features_h20_v2.parquet \
     --model model_15m_itransformer_vf.keras \
     --stats norm_stats_15m_itransformer_vf.npz \
     --train-end 2024-12-01T00:00:00Z --val-end 2025-06-01T00:00:00Z \
     --threshold 0.58 --fee-bps 2.0 --slip-bps 3.0
   ```
4. Запустить FastAPI (сервер для сайта, требует `.venv` + `tf_keras`):
   ```bash
   source .venv/bin/activate
   export TF_USE_LEGACY_KERAS=1
   python3 serve_fastapi.py \
     --model-h20 model_battle_itransformer.keras \
     --stats-h20 norm_stats_battle_itransformer.npz \
     --model-multi model_15m_itransformer_tb_multi.keras \
     --stats-multi norm_stats_15m_itransformer_tb_multi.npz \
     --features data/BTCUSDT_15m_features_h20_v2.parquet \
     --seq-len 256
   ```
5. Запустить веб-интерфейс (локальный HTTP чтобы подгрузить скрипты):
   ```bash
   cd html
   python3 -m http.server 8082
   # открыть http://localhost:8082/aladin_from_image.html
   ```

## Данные / модели / отчеты (хранилища и форматы)

- `data/`: raw OHLCV/aux/Binance cache (`.parquet`), сгенерённые `.parquet` фичи, `metrics/*.csv`.  
- `best/`, `model_*.keras`, `norm_stats_*.npz`: готовые веса; battle/multi/vf модели для прогноза на 5h/20h/40h/80h/160h.  
- `reports/`: equity-графики, CSV/PNG доклады walk‑forward/feature importance, и файлы `forecast_*.csv` для UI.  
- `html/`: front-end (Lightweight Charts, прогнозная панель) + `chat_summary.pdf` (hotspots, прорисовки, Visual spec).  
