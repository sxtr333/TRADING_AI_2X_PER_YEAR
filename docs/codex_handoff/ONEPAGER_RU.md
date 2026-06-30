# Model6 — краткая русская выжимка (one‑pager)

## 1) Где что лежит

- **Код**: `/home/vitamind/my_project/model6`
- **Модели**: `new_models/` (часто это symlink на `/mnt/oldssd/...`)
- **Датасеты**: `data/`
- **Meta‑датасет**: `data/meta/meta_dataset_pruned.parquet`
- **Отчёты**: `reports/`
- **Логи**: `logs/`
- **Сайт**: `html/aladin_from_image.html`

## 2) Главные датасеты

- Основные фичи: `data/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes_with_news_pruned.parquet`
- Фичи для сервинга (обновляются realtime):
  - `data/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes_serve.parquet`
  - `data/BTCUSDT_15m_features_h20_v2_news_xlmr_v3_serve.parquet` (для старой v5‑модели)

## 3) Realtime обновление фич каждые N минут

```bash
LIVE_UPDATE_INTERVAL_SEC=60 bash /home/vitamind/my_project/model6/scripts/live_update_loop.sh
```

Внутри цикла:
1) news_ingest → news_dedup → news_sentiment_hf (LedgerBERT + XLM‑R)
2) обновление OHLCV (bybit)
3) build_features (с новостями + макро)

## 4) Запуск сайта

```bash
bash /home/vitamind/my_project/model6/scripts/run_servers.sh
```

- UI: http://localhost:8080/aladin_from_image.html
- Арбитраж: http://localhost:8090/arbitration.html

## 5) Трейн на GPU (строго без CPU)

**Всегда использовать GPU‑preflight.**

Host‑вариант:
```bash
scripts/run_train_gpu_cpulimit70.sh -- bash /home/vitamind/my_project/model6/scripts/train_news_xlmr_cosmic_v6_dr_scale_8nodes.sh
```

Docker‑вариант:
```bash
scripts/run_train_gpu_cpulimit70.sh --docker tensorflow/tensorflow:nightly-gpu -- \
  bash /work/scripts/train_news_xlmr_cosmic_v6_dr_scale_8nodes.sh
```

## 6) Meta‑модели (trade/no‑trade)

Последние обученные:
- `new_models/meta_2026-01-20/`
  - meta_h20_long.keras
  - meta_h20_short.keras
  - meta_h80_short_v2.keras
  - meta_h160_long_v2.keras

## 7) Лучшие результаты (важно, чтобы не повторять)

- **Combo без meta**: ~+0.7 PnL (`reports/backtest_trade_combo_grid.csv`)
- **Meta baseline**: +18.18 (`reports/best_result_meta_2026-01-19.txt`)
- **Meta boost**: +26.05 (`reports/best_result_meta_boost_2026-01-19.txt`)
- **Локальный максимум**: ~+29.9 (`reports/backtest_trade_combo_meta_step1_h160_tradefrac.csv`)

**Новые meta‑модели (2026‑01‑20)** дали максимум **+5.585** — хуже старого +29
(`reports/backtest_trade_combo_meta_sweep_2026-01-20.csv`).

## 8) Что дальше сравнивать

Если нужно честное сравнение:
- Прогнать **лучший старый конфиг** (+29) с **новыми meta‑моделями** и сравнить.

## 9) Важные предупреждения

- Нельзя использовать `meta_y_*` и `meta_label_*` как фичи (leakage).
- Датасет для meta и features должны совпадать по timestamp 1‑к‑1.
- `*_stats.npz` всегда должны соответствовать своему датасету.

