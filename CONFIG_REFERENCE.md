# Config reference (CLI)

> Таблица перечисляет основные параметры для `build_features.py`, `train_keras.py`, `walk_forward_eval.py`, `realtime_stream.py` (runner/бот) и `serve_fastapi.py` (FastAPI-интерфейс сайта). Пояснены дефолты, возможные значения и влияние на результат.

## `build_features.py`

| Arg | Default | Values | Effect |
|---|---|---|---|
| `--input` | required | path | Raw OHLCV Parquet (e.g., `data/BTCUSDT_15m.parquet`). |
| `--output` | required | path | Output features Parquet (needed для training/serve). |
| `--aux` | None | path | Aux data (OI/funding/liq/basis/ratios) aligned by `timestamp`. |
| `--spot` | None | path | Spot CSV/Parquet for basis spreads. |
| `--news` | None | path | News dataset (parquet/jsonl/csv) for `news_*` features. |
| `--horizon` | 1 | int | Forecast horizon (в барах базового TF). |
| `--target-mode` | `log_return` | `log_return` / `price` | Целевой столбец (`target_ret` или прямой `price`). |
| `--base-tf-min` | None | int | Базовый timeframe в минутах (auto-inf если не указан). |
| `--vf-k` | 0.5 | float | K-параметр volatility-filter для `target_dir_vf`. |
| `--vf2-k` | None | float | Второй `k` для `target_dir_vf2`. |
| `--vf-col` | `rv_long` | str | Колонка volatility для VF фильтра. |
| `--tb-horizon` | None | int | Triple-barrier ширина (по умолчанию `horizon*4`). |
| `--tb-k` | 0.8 | float | Множитель `k` в triple-barrier для `tb_label`. |
| `--tb-sigma-col` | `rv_long` | str | Колонка σ в triple-barrier. |
| `--multi-horizons` | None | CSV строка | Добавляет `label_3cls_h{h}` / `tb_label_h{h}` / `target_ret_h{h}` (пример: `20,80,160`). |

## `train_keras.py` (основной тренинг)

| Arg | Default | Values | Effect |
|---|---|---|---|
| `--features` | required | path | Features Parquet (columns include `timestamp`, features, targets). |
| `--seq-len` | 256 | int | Sequence length для окон модели. |
| `--batch-size` | 64 | int | Batch size на этапе тренинга. |
| `--epochs` | 10 | int | Эпохи. |
| `--lr` | 3e-4 | float | Base learning rate. |
| `--model-out` | required | path | Выходной `.keras` файл. |
| `--stats-out` | required | path | `.npz` с normalize stats. |
| `--target-col` | None | str | Имя классификационного target-а (auto по `target_dir`/`label_3cls`). |
| `--price-col` | None | str | Regression target (например `target_ret`). |
| `--purge-gap` | 0 | int | Пробел между train/val агрегатами (rows). |
| `--train-end` | None | ISO ts | Timestamp split (UTC). |
| `--val-end` | None | ISO ts | Timestamp для val. |
| `--target-horizon` | 0 | int | Horizon rows, используемый для предотвращения leakage при split. |
| `--multi-horizons` | None | CSV | Доп. классификационные головы (например `20,80,160`). |
| `--num-classes` | 1 | int | 0=нет cls head, 1=binary, 3=softmax. |
| `--cls-weight` | 1.0 | float | Вес класса в loss (CLS). |
| `--price-weight` | 0.0 | float | Вес regression loss. |
| `--pos-weight` | None | float | Положительный вес (binary). |
| `--auto-pos-weight` | False | flag | Автоподбор `pos_weight`. |
| `--cls-loss` | `bce` | `bce` / `focal` | Тип классификационной функции. |
| `--focal-alpha` | 0.25 | float | Focal loss α. |
| `--focal-gamma` | 2.0 | float | Focal loss γ. |
| `--label-smoothing` | 0.0 | float | Label smoothing для BCE. |
| `--price-loss` | `huber` | `huber`/`mse`/`logcosh`/`mae` | Regression loss. |
| `--huber-delta` | 1.0 | float | Параметр Huber. |
| `--quantiles` | None | CSV | Включает quantile head (e.g. `0.1,0.5,0.9`). |
| `--train-ratio` | 0.70 | float | Train ratio, если `--train-end` не задан. |
| `--val-ratio` | 0.15 | float | Val ratio (если нет `--val-end`). |
| `--feature-cols` | None | JSON list | Принудительный порядок фич (например `["open","close"]`). |
| `--q-low` | 0.001 | float | Clips `target_ret` quantiles before norm (low). |
| `--q-high` | 0.999 | float | Clips quantiles (high). |
| `--d-model` | 128 | int | Hidden size. |
| `--layers` | 2 | int | Количество encoder blocks. |
| `--heads` | 4 | int | Multi-head attention. |
| `--arch` | `transformer` | `transformer` / `patchtst` / `tsmixer` / `itransformer` | Backbone. |
| `--pooling` | `multi` | str | Pooling head (`multi`/`cls`/`reg`). |
| `--patch-size` | 16 | int | PatchTST patch size. |
| `--patch-stride` | 16 | int | Patch stride. |
| `--tsmixer-mlp` | 256 | int | TSMixer MLP dim. |
| `--var-layers` | 2 | int | iTransformer variance layers. |
| `--time-layers` | 2 | int | iTransformer time layers. |
| `--revin` | False | flag | Enable RevIN. |
| `--revin-affine` | False | flag | RevIN affine parameters. |
| `--feature-dropout` | 0.05 | float | SpatialDropout1D rate. |
| `--dropout` | 0.10 | float | Dropout. |
| `--drop-path` | 0.05 | float | Stochastic depth for transformer blocks. |
| `--cosine` | False | flag | Warmup + cosine LR scheduler. |
| `--warmup-steps` | 0 | int | Warmup steps for cosine. |
| `--patience` | 3 | int | Early stopping patience. |
| `--min-delta` | 1e-4 | float | Early stopping min delta. |
| `--start` | None | ISO ts | Filter df `timestamp >= start`. |
| `--end` | None | ISO ts | Filter df `timestamp <= end`. |
| `--sample-weight-col` | None | column | Column for weighting samples. |
| `--sample-weight-k` | 0.0 | float | Weight scaling (`1 + k * zscore`). |
| `--sample-weight-clip` | 3.0 | float | Clip weights to `[1/clip, clip]`. |

## `walk_forward_eval.py`

| Arg | Default | Values | Effect |
|---|---|---|---|
| `--features` | required | path | Features Parquet. |
| `--model` | required | path | Saved `.keras` model. |
| `--stats` | required | path | Normalization stats `.npz`. |
| `--seq-len` | None | int | Sequence length (fallback from stats). |
| `--train-end` | required | ISO ts | Разделение train/val по timestamp. |
| `--val-end` | required | ISO ts | Val end timestamp. |
| `--purge-gap` | 0 | int | Rows gap чтобы исключить leakage. |
| `--target-horizon` | 0 | int | Horizon used in `build_end_indices`. |
| `--threshold` | 0.5 | float | Long signal threshold. |
| `--short-threshold` | None | float | Short threshold (если long_short). |
| `--use-quantile-signal` | False | flag | Use quantile head instead of cls. |
| `--quantiles` | None | CSV | Quantiles defined in training (e.g., `0.1,0.5,0.9`). |
| `--q-min` | 0.0 | float | Minimum return magnitude for quantile signal. |
| `--mode` | `long_short` | `long_only` / `long_short` | Trading mode. |
| `--opt-metric` | `none` | `none`, `sharpe`, `calmar` | Auto optimize threshold. |
| `--thr-min` | 0.50 | float | Threshold scan min. |
| `--thr-max` | 0.70 | float | Threshold scan max. |
| `--thr-step` | 0.01 | float | Threshold step. |
| `--fee-bps` | 0.0 | float | Fee basis points per trade. |
| `--slip-bps` | 0.0 | float | Slippage basis points. |
| `--min-hold` | 1 | int | Minimum bars to hold trade. |
| `--cooldown` | 0 | int | Cooldown bars after trade. |
| `--max-trades-per-day` | 0 | int | Trade cap (0 disables). |
| `--vol-col` | None | column | Volatility column for filters. |
| `--vol-k` | 0.0 | float | Vol threshold scale. |
| `--vol-clip` | 0.25 | float | Clip vol sizing. |
| `--vol-block` | 999.0 | float | Block trades if vol>threshold. |
| `--vol-size-k` | 0.0 | float | Size scaling by vol. |
| `--max-dd` | 0.0 | float | Drawdown stop. |
| `--daily-limit` | 0.0 | float | Daily loss limit. |
| `--dd-reset` | `none` | `none`/`daily`/`monthly`/`quarterly` | Drawdown reset cadence. |
| `--prob-ema` | 0.0 | float | EMA smoothing for probabilities. |
| `--trade-band` | 0.0 | float | Dead-band around 0.5. |
| `--report-csv` | None | path | Monthly report CSV. |
| `--equity-csv` | None | path | Equity curve CSV. |
| `--equity-png` | None | path | Equity PNG. |
| `--model-name` | None | str | Label for reports/logs. |

## `realtime_stream.py` (runner/бот)

| Arg | Default | Values | Effect |
|---|---|---|---|
| `--model` | required | path | `.keras` model for realtime inference. |
| `--seq-len` | 256 | int | Sequence length. |
| `--symbol` | `BTCUSDT` | str | Bybit symbol. |
| `--log-file` | None | path | CSV log for timestamp,pred,close. |
| `--drift-threshold` | 3.0 | float | Z-score for drift alerts. |
| `--ema-alpha` | 0.2 | float | EMA smoothing for predictions. |
| `--mc-samples` | 1 | int | MC dropout samples (>1 enables stochastic forward). |

## `serve_fastapi.py`

| Arg | Default | Values | Effect |
|---|---|---|---|
| `--model-h20` | `model_battle_itransformer.keras` | path | `.keras` model trained for h20 (5h / 20 steps). |
| `--stats-h20` | `norm_stats_battle_itransformer.npz` | path | `.npz` stats matching h20 model (mean/std/feature_names). |
| `--model-multi` | `model_15m_itransformer_tb_multi.keras` | path | Multi-horizon `.keras` (h80/h160). |
| `--stats-multi` | `norm_stats_15m_itransformer_tb_multi.npz` | path | `.npz` stats for multi horizon. |
| `--seq-len` | 256 | int | Sequence length used at training. |
| `--features` | None | path | Features parquet used by `/forecast[_multi]`. |
| `--host` | `0.0.0.0` | str | Bind host. |
| `--port` | 8000 | int | Bind port. |

### Env vars (serve_fastapi)

- `NEWS_PATH` (default: `data/news/news.parquet`) – news store used by `/news` and `/news_agg`.
- `ALLOW_ORIGINS` (default: `http://localhost:8080,http://127.0.0.1:8080`) – CORS allowlist.
  Use `*` for public tunnels.

### News ingest (scripts/news_ingest.py)

Env vars (optional):

- `CRYPTOPANIC_API_KEY`
- `CRYPTOCOMPARE_API_KEY`
- `COINMARKETCAL_API_KEY`

Example:

`python3 scripts/news_ingest.py --out data/news/news.parquet --currency BTC --max-items 200 --start 2023-01-01T00:00:00Z --end 2026-01-12T00:00:00Z`

### CC-NEWS pipeline (scripts/cc_news_pipeline.py)

Stream WARC, filter on-the-fly, save parquet to HDD:

`python3 scripts/cc_news_pipeline.py --start-month 2025-07 --end-month 2025-12 --out-dir /mnt/data/cc-news --checkpoint /mnt/data/cc-news/checkpoints/processed.txt`

### Env vars (serve_fastapi)

- `NEWS_PATH` (default: `data/news/news.parquet`) – news store used by `/news` and `/news_agg`.
- `ALLOW_ORIGINS` (default: `http://localhost:8080,http://127.0.0.1:8080`) – CORS allowlist.
  Use `*` for public tunnels.

### News ingest (scripts/news_ingest.py)

Env vars (optional):

- `CRYPTOPANIC_API_KEY`
- `CRYPTOCOMPARE_API_KEY`
- `COINMARKETCAL_API_KEY`

Example:

`python3 scripts/news_ingest.py --out data/news/news.parquet --currency BTC --max-items 200`
