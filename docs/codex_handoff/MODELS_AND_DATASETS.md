# Актуальные модели и датасеты (с датами)

## 1) Основные датасеты (фактические диапазоны)

| Путь | Строк | Min timestamp (UTC) | Max timestamp (UTC) |
|---|---:|---|---|
| `data/BTCUSDT_15m.parquet` | 105,734 | 2023-01-01 00:00:00+00:00 | 2026-01-19 20:30:00+00:00 |
| `data/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes_with_news_pruned.parquet` | 103,104 | 2023-01-08 00:00:00+00:00 | 2025-12-16 23:45:00+00:00 |
| `data/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes_serve.parquet` | 105,062 | 2023-01-08 00:00:00+00:00 | 2026-01-19 20:30:00+00:00 |
| `data/BTCUSDT_15m_features_h20_v2_news_xlmr_v3.parquet` | 103,104 | 2023-01-08 00:00:00+00:00 | 2025-12-16 23:45:00+00:00 |
| `data/BTCUSDT_15m_features_h20_v2_news_xlmr_v3_serve.parquet` | 105,062 | 2023-01-08 00:00:00+00:00 | 2026-01-19 20:30:00+00:00 |
| `data/meta/meta_dataset_pruned.parquet` | 103,104 | 2023-01-08 00:00:00+00:00 | 2025-12-16 23:45:00+00:00 |

- **OHLCV 15m (сырая свечная база)**
  - `data/BTCUSDT_15m.parquet`
  - Диапазон обновляется realtime (`live_update_loop.sh`)

- **Features (новые, v4_8nodes)**
  - Полный: `data/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes_with_news_pruned.parquet`
  - Serve:  `data/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes_serve.parquet`
  - Диапазон: последние данные зависят от `live_update` и исходного OHLCV

- **Features (старые v3, для v5‑модели)**
  - Полный: `data/BTCUSDT_15m_features_h20_v2_news_xlmr_v3.parquet`
  - Serve:  `data/BTCUSDT_15m_features_h20_v2_news_xlmr_v3_serve.parquet`

- **Meta‑датасет**
  - `data/meta/meta_dataset_pruned.parquet`
  - Должен быть **строго синхронизирован** по timestamp с базовыми features

## 2) Модели (ключевые)

- **Лучший старый 3‑head price model (v5)**
  - `new_models/2026-01-14_news_xlmr_v5_dr_scale/model_15m_itransformer_price_multi_h20_h80_h160_news_xlmr_v5_dr_scale_e14_b32.keras`

- **Meta‑модели (trade/no‑trade), latest**
  - `new_models/meta_2026-01-20/meta_h20_long.keras`
  - `new_models/meta_2026-01-20/meta_h20_short.keras`
  - `new_models/meta_2026-01-20/meta_h80_short_v2.keras`
  - `new_models/meta_2026-01-20/meta_h160_long_v2.keras`

- **Sanity‑train модель (1 эпоха, проверка пайплайна)**
  - `new_models/sanity_v7_2025_h20.keras`

## 3) Лучшие результаты (PnL за 2025 год)

- **Combo без meta**: ~+0.7
  - `reports/backtest_trade_combo_grid.csv`

- **Meta baseline**: +18.18
  - `reports/best_result_meta_2026-01-19.txt`

- **Meta boost**: +26.05
  - `reports/best_result_meta_boost_2026-01-19.txt`

- **Локальный максимум**: ~+29.9
  - `reports/backtest_trade_combo_meta_step1_h160_tradefrac.csv`
  - `reports/backtest_trade_combo_meta_step2_cooldown.csv`

- **Новые meta‑модели (2026‑01‑20)**: максимум +5.585
  - `reports/backtest_trade_combo_meta_sweep_2026-01-20.csv`

## 4) Где смотреть диапазоны дат

Чтобы проверить датасет и датировку:

```bash
.venv/bin/python - <<'PY'
import pandas as pd
path = "data/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes_with_news_pruned.parquet"
df = pd.read_parquet(path, columns=["timestamp"])\nprint(df["timestamp"].min(), df["timestamp"].max())
PY
```

Ту же проверку можно сделать для любого parquet.
