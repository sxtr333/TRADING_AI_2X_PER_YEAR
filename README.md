# TradeForge

TradeForge — исследовательский проект по анализу BTCUSDT и проверке торговых моделей на исторических данных. Внутри есть сайт, FastAPI backend, набор ML-скриптов, feature engineering, backtest-логика и финальный архив для сдачи проекта.

Проект не является финансовой рекомендацией. Основная цель — показать полный pipeline: от рыночных данных и признаков до модели, сигналов, истории сделок и веб-интерфейса.

## Что внутри

- `serve_fastapi.py` — основной FastAPI backend: свечи, прогнозы, auth, подписки, платежные endpoints, market radar.
- `html/` — frontend сайта: landing, dashboard, screener, pump scout, страницы оплаты и документы.
- `scripts/` — сборка признаков, обучение, backtest, live-update, Docker/site helpers.
- `docs/` — дополнительные материалы по архитектуре и передаче проекта.
- `submission/trade_model_submission_FINAL.zip` — архив сдачи по регламенту: два ноутбука, sample-данные и готовая модель.
- `TRADEFORGE_CLIENT_README.md` — подробное описание проекта для заказчика.
- `TRADEFORGE_FULL_README.md` — расширенный README для презентации и защиты.

## Архитектура

Общий pipeline:

```text
OHLCV / market data
        ↓
feature engineering
        ↓
training / experiments
        ↓
backtest and trade history
        ↓
FastAPI backend
        ↓
web dashboard
```

Основные компоненты:

- рыночные свечи BTCUSDT на 15m;
- признаки по цене, объему, волатильности, funding/open interest/liquidations;
- news/sentiment ветки как экспериментальный слой;
- sequence-модели: iTransformer, PatchTST, TSMixer, multi-horizon heads;
- отдельные торговые режимы: Conservative, Aggressive, Alpha75, TG Hybrid, MM Supervisor, Chronos2;
- dashboard с графиком TradingView Lightweight Charts и markers входов/выходов.

## Финальная сдача

Актуальная сдача этапа 5 лежит обычной папкой, без ZIP:

```text
stage5_final_nn/
```

Прямая структура папки:

```text
stage5_final_nn/01_final_ready_model.ipynb
stage5_final_nn/02_experiments_model_building.ipynb
stage5_final_nn/btcusdt_15m_sample.csv
stage5_final_nn/trade_signal_model.joblib
stage5_final_nn/stage4_project_report.docx
```

Именно ссылку на `stage5_final_nn/` удобно отправлять на проверку.

Также оставлен архив с теми же материалами:

```text
submission/trade_model_submission_FINAL.zip
```

Внутри ровно два ноутбука:

```text
01_experiments_trade_model.ipynb
02_use_ready_trade_model.ipynb
```

Первый ноутбук показывает эксперименты при создании модели. Второй использует уже готовую модель и выводит понятный сигнал `LONG / SHORT / WAIT`.

## Быстрый запуск сайта

Локальный вариант:

```bash
cp .env.example .env.site
# заполнить значения в .env.site
./scripts/start_site.sh
```

Docker-вариант:

```bash
cp .env.example .env.site
# заполнить значения в .env.site
docker compose -f docker-compose.site.yml up -d --build
```

Проверка API:

```bash
curl http://127.0.0.1:8000/health
```

Frontend обычно отдается из `html/`.

## Данные и модели

Тяжелые датасеты, веса моделей и приватные базы пользователей не хранятся в GitHub. Они вынесены из репозитория намеренно:

- `*.parquet`, `*.csv` с историей рынка;
- `*.keras`, `*.npz` с весами и normalization stats;
- `.env`, `.api_key`, `*.db`;
- live logs и локальные кэши.

Подробная карта датасетов, моделей и отчетов описана в:

```text
TRADEFORGE_CLIENT_README.md
```

## Основные команды research-части

Сборка признаков:

```bash
python build_features.py --help
```

Обучение Keras-моделей:

```bash
python train_keras_v7.py --help
```

Backtest:

```bash
python scripts/backtest_trade_combo_meta.py --help
```

Проверка forecast endpoints:

```bash
python scripts/smoke_forecast.py --help
```

## Структура сайта

- `html/index.html` — главная страница.
- `html/dashboard/index.html` — основной dashboard.
- `html/screener.html` — Market Radar.
- `html/pump-scout/index.html` — Pump Scout.
- `html/how-to-use/index.html` — инструкция.
- `html/settings/index.html` — настройки.
- `html/privacy.html`, `html/offer.html`, `html/refund.html` — юридические страницы.

## Важные замечания

- Исторический backtest не гарантирует будущую доходность.
- Секреты и реальные платежные ключи должны задаваться только через `.env.site`.
- Для полноценного production нужны отдельные backups, monitoring, HTTPS/tunnel и контроль доступа.
- Для повторяемой защиты используйте архив из `submission/`, потому что все нужные файлы лежат внутри него.
