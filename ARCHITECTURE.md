# Architecture (model6)

## Карта модулей (папка/файл → роль)

- `data/` — паркет-файлы raw OHLCV (`BTCUSDT_{1m,5m,15m,30m,1h,4h,1d}.parquet`), aux (`*_aux*.parquet`, Bybit/Binance metrics), кеш Binance (`binance_cache/`) и промежуточные фичи (`*_features*.parquet`).
- `build_features.py` / `build_features_multi_tf.py` — вычисляют 54+ признаков (OHLCV, OI, funding, liq, basis, order-book, volatility, TA, time, Triple-Barrier и VF-таргеты) и сохраняют `.parquet` с обязательными `target_ret`, `label_3cls`, `tb_label`, `vf_label` колонками.
- `model_layers.py` + `trading_keras_core.py` — задаютдоврати, RevIN, DropPath, iTransformer/TSMixer блоки и `tf.data` окна; используется во всех тренировках (`train_keras.py`, `train_keras_multitf.py`, `train_keras_multihorizon.py`).
- `train_*.py` (`train_keras.py`, `train_keras_multitf.py`, `train_keras_multihorizon.py`) — обучают классификаторы/регрессоры (price+cls+multi-horizon), записывают `.keras` и `norm_stats_*.npz`.
- `walk_forward_eval.py`, `eval_metrics.py`, `eval_tb_multi.py`, `eval_and_forecast.py` — делают walk-forward, кластерные пороги, рассчитывают fee/slippage, рисуют equity/metrics CSV и PNG.
- `inference_keras.py`, `realtime_stream.py`, `serve_fastapi.py` — нормализуют окна, выполняют предикт (MC dropout), выдают JSON для UI и, в случае `realtime_stream.py`, подключаются к потокам (Bybit/WS).
- `html/` — frontend (Lightweight Charts, кнопки `PREDICT → 5H/20H/40H/80H/160H → ПРОГНОЗ`, иконки, news). `chat_summary.pdf` описывает hotspots, overlay и взаимодействие с TradingView.
- `best/` — сохранённые модели (battle, multi, profit, etc.) и статистики (`norm_stats_*.npz`) для фронтэнда и анализа.
- `scripts/`, `reports/`, `PIPELINE.md`, `run_*.sh`, `Dockerfile` — вспомогательные скрипты/отчёты/контейнеры для построения данных, обучения и развёртывания.

## Dataflow / Controlflow («данные → фичи → модель → сигналы → риск → исполнение»)

1. **Данные**: raw OHLCV/aux паркет-источники (`data/`), Binance кэш (`binance_cache/`), Bybit LIQ/funding (`scripts/fetch_*liquidations.py`) и новости `html/fed_events.json`.
2. **Фичи**: `build_features*.py` строят окна с time, TA, OB, volatility, zeros-filled liq, target_ret/label_*.pn; сохраняют в `data/BTCUSDT_15m_features_*.parquet`.
3. **Модели**: `train_*.py` создает `model_*.keras` + `norm_stats_*.npz` (mean/std/feature_names/seq_len). RevIN/DropPath обеспечивают leakage-safe training; `--multi-horizons` добавляет 20/80/160 шагов.
4. **Предикт/сигналы**: `serve_fastapi.py` и `realtime_stream.py` загружают stats, нормализуют окна (15m, 5h) и возвращают log-return/price с `forecast`/`forecast_multi`.
5. **Риск**: `walk_forward_eval.py` с `threshold`, `fee-bps`, `slip-bps`, `max-dd` и `cooldown` параметрами превращает probabilistic предики в equity/signal отчёты (CSV/PNG).
6. **UI/исполнение**: `html/aladin_from_image.html` рисует Lightweight Charts + прогнозную линию, кнопки `PREDICT` → “горизонт” → `ПРОГНОЗ` (проверка ready state) и сообщает FastAPI вашим моделям.

## Важные сущности

- **Датасеты**:
  - `data/BTCUSDT_15m_features_h20_v2.parquet` — основной input для FastAPI + training (seq_len 256, 54 фичи).
  - `best/boevaya/model_15m_itransformer_boevaya` (+ `norm_stats_15m_itransformer_boevaya.npz`) — сильная 5h модель для фронта.
  - `best/boevaya/model_15m_itransformer_tb_multi` и `norm_stats_15m_itransformer_tb_multi.npz` — multi-horizon model (20/80/160).
  - `reports/`, `data/binance_cache/`, `metrics_*.csv` — для анализа.
- **Таргеты**: `target_ret`, `target_dir`, `label_3cls`, `tb_label`, `vf_label`, `label_3cls_h{h}`, `tb_label_h{h}`, `target_ret_h{h}` (например, h=20,80,160).
- **Горизонты**: `h20` → 5 hours, `h80` → 20 hours, `h160` → 40+ hours; FastAPI использует `forecast_multi` (15m steps) и ensures `step_min=15`.
- **Stats**: `norm_stats_*.npz` (mean/std/feature_names/seq_len), `stats` must match features and reshape (watch for `54 vs 24` mismatch errors).

## Ключевые точки конфигурации

- `build_features.py`: задаёт input/output parquet, `--horizon`, `--target-mode`, `--multi-horizons`, `--aux` (Base TF 15m), `--vf-k`, `--tb-k`.
- `train_keras.py`: `--features`, `--seq-len`, `--model-out`, `--stats-out`, `--arch` (itransformer/patchtst/tsmixer), `--revin`, `--multi-horizons`, `--feature-dropout`, `--drop-path`, `--price-weight`.
- `walk_forward_eval.py`: `--features`, `--model`, `--stats`, `--seq-len`, `--train-end`, `--val-end`, `--threshold`, `--fee-bps`, `--slip-bps`, `--max-dd`, `--dd-reset`, `--report-csv/png`.
- `serve_fastapi.py`: `--model-h20`, `--stats-h20`, `--model-multi`, `--stats-multi`, `--features`, `--seq-len`, `--host`, `--port`. FastAPI expects `TF_USE_LEGACY_KERAS=1` + `tf_keras` when loading `model_layers.RevIN`.
- `realtime_stream.py`: `--model`, `--seq-len`, `--symbol`, `--log-file`, `--mc-samples` (для MC dropout drift alerts).

