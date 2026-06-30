# TradeForge.art — README для заказчика

Этот документ объясняет проект TradeForge простым языком: что уже сделано, из каких частей состоит система, как она работает, какие модели внутри, как устроен сайт, backend, данные, инфраструктура и что нужно знать для дальнейшей поддержки.

Проект развивался несколько месяцев: от первых экспериментов с прогнозом BTCUSDT до полноценного сайта с регистрацией, подпиской, торговым dashboard, несколькими моделями, историей сделок, Market Radar, Pump Scout, платежным flow, аналитикой и подготовкой к публичной презентации.

---

## 1. Что такое TradeForge

TradeForge.art — это платформа для анализа рынка BTCUSDT и криптовалютных движений.

Главная задача проекта:

- собирать рыночные данные;
- строить признаки;
- обучать модели;
- проверять их на истории;
- превращать прогнозы в понятные сигналы;
- показывать сигналы на сайте;
- давать пользователю доступ через аккаунт и подписку.

Пользователь на сайте видит не сырой ML-вывод, а готовую рабочую картину:

- график BTCUSDT;
- текущий сигнал LONG / SHORT / FLAT;
- confidence;
- риск;
- плечо;
- stop/invalidation;
- историю сделок модели;
- новостной и рыночный контекст;
- Market Radar;
- Pump Scout;
- аккаунт и статус подписки.

---

## 2. Что уже реализовано

На текущем этапе в проекте есть:

- публичный сайт `https://tradeforge.art`;
- backend API `https://api.tradeforge.art`;
- landing page;
- регистрация по email;
- подтверждение email кодом;
- вход по паролю;
- восстановление пароля;
- вход через Yandex OAuth;
- личный кабинет;
- поля ФИО и Telegram/contact;
- trial-доступ;
- подписки и тарифы;
- интеграция платежного flow;
- dashboard с графиком BTC;
- переключение моделей;
- переключение таймфреймов;
- история сделок;
- markers на графике;
- Market Radar;
- Pump Scout;
- live chat через Tawk.to;
- Yandex Metrika;
- GitHub-пакет без секретов;
- пакет для сдачи проекта с ноутбуками и топовой моделью.

---

## 3. Главная архитектура

Проект можно представить так:

```text
Данные рынка
    ↓
Feature engineering
    ↓
Модели
    ↓
Backtest / walk-forward
    ↓
Сигналы
    ↓
FastAPI backend
    ↓
Frontend сайт
    ↓
Пользователь / подписка / поддержка
```

Важная идея: TradeForge — это не один notebook и не одна модель. Это полный продуктовый контур вокруг торговых моделей.

---

## 4. Основные папки проекта

Рабочая папка:

```text
/home/vitamind/my_project/model6
```

Ключевые директории и файлы:

```text
model6/
├── serve_fastapi.py
├── build_features.py
├── build_features_multi_tf.py
├── train_keras.py
├── train_keras_v7.py
├── walk_forward_eval.py
├── backtest_trade_combo_meta_dynamic_exit.py
├── model_layers.py
├── trading_keras_core.py
├── realtime_stream.py
├── data/
├── best/
├── reports/
├── html/
├── scripts/
├── state/
├── project_submission_trade_model/
├── docker-compose.site.yml
├── TRADEFORGE_FULL_README.md
└── TRADEFORGE_CLIENT_README.md
```

### `serve_fastapi.py`

Главный backend. Отвечает за API, auth, подписки, платежи, свечи, сигналы, Market Radar и dashboard bootstrap.

### `html/`

Frontend сайта:

- главная страница;
- dashboard;
- screener;
- Pump Scout;
- settings;
- legal/payment pages.

### `data/`

Данные и признаки. Часть данных вынесена на отдельный диск через symlink.

### `best/`

Лучшие модели и сохраненные веса.

### `reports/`

Отчеты по backtest, walk-forward, equity, trades, monthly stats.

### `project_submission_trade_model/`

Финальная папка для сдачи проекта:

- ноутбуки;
- sample dataset;
- модель;
- zip archive;
- топовая iTransformer-модель для защиты.

---

## 5. Сайт TradeForge.art

Сайт состоит из нескольких основных страниц.

### 5.1. Landing page

Файлы:

```text
html/index.html
html/landing.css
html/landing-core.js
html/landing-ui.js
html/landing-app.js
```

Что делает:

- презентует продукт;
- показывает тарифы;
- запускает trial;
- открывает demo;
- показывает live proof;
- дает вход в аккаунт;
- открывает оплату;
- ведет в dashboard.

На landing есть:

- `Start trial`;
- `Open demo`;
- `Account`;
- тарифы Starter / Pro / Elite;
- блок API uptime;
- блок latency;
- описание dashboard;
- модальное окно авторизации;
- account modal.

### 5.2. Dashboard

Файл:

```text
html/dashboard.html
```

Это основная рабочая страница продукта.

Возможности:

- график BTCUSDT;
- таймфреймы `15m`, `1H`, `4H`, `1D`;
- текущий сигнал;
- выбор модели;
- confidence;
- risk/trade;
- leverage;
- invalidation stop;
- trade markers;
- история сделок;
- новости;
- live refresh;
- mobile layout.

График строится через:

```text
html/lightweight-charts.standalone.production.js
```

Dashboard получает данные через API:

```text
/dashboard/bootstrap
/candles
```

### 5.3. Market Radar

Файл:

```text
html/screener.html
```

Market Radar показывает состояние рынка:

- какие монеты растут;
- какие падают;
- какие дают сильное движение;
- spot mode;
- MEXC perpetual mode;
- pump score;
- volume filter;
- spread alert.

Используемые API:

```text
/market/top
/market/snapshot
/market/icon/{symbol}
```

### 5.4. Pump Scout

Файл:

```text
html/pump-scout/index.html
```

Pump Scout — отдельный модуль для быстрых движений:

- pump candidates;
- dump candidates;
- ignite/fade lanes;
- MEXC perpetual feed;
- live movers.

Используемые API:

```text
/market/top?mode=mexc_perp&direction=pump
/market/top?mode=mexc_perp&direction=dump
```

### 5.5. Legal/payment pages

Файлы:

```text
html/privacy.html
html/refund.html
html/offer.html
html/payment-success.html
html/payment-cancel.html
```

Они нужны для нормального коммерческого продукта: политика, возврат, оферта, страницы после оплаты.

---

## 6. Backend API

Backend написан на FastAPI.

Главный файл:

```text
serve_fastapi.py
```

Основные группы API:

### Health

```text
GET /health
```

Показывает, что API живой.

### Candles

```text
GET /candles
```

Отдает свечи BTCUSDT для графика.

### Dashboard bootstrap

```text
GET /dashboard/bootstrap
```

Одним запросом отдает dashboard данные:

- candles;
- signal;
- news;
- leverage/backtest summary;
- выбранную модель.

### Auth

```text
POST /auth/register
POST /auth/verify-email
POST /auth/resend-code
POST /auth/web-login
POST /auth/request-password-reset
POST /auth/reset-password
GET  /auth/me
POST /auth/profile
GET  /auth/yandex/start
GET  /auth/yandex/callback
GET  /auth/session-bridge
```

Что реализовано:

- регистрация;
- email-код;
- вход;
- восстановление пароля;
- Yandex OAuth;
- session token;
- account profile.

### Subscription

```text
GET /subscription/status
```

Возвращает:

- активна ли подписка;
- какой тариф;
- до какой даты доступ;
- есть ли trial;
- latest payment;
- support code.

### Billing

Подготовлены платежные flow через Prodamus/YooKassa.

Основные endpoints:

```text
/billing/prodamus/create-payment
/billing/prodamus/webhook
```

### Market

```text
/market/top
/market/snapshot
/market/icon/{symbol}
```

Используется Market Radar и Pump Scout.

---

## 7. Аккаунты и подписки

В проекте реализована логика аккаунта:

1. пользователь регистрируется;
2. подтверждает email;
3. получает trial;
4. выбирает тариф;
5. оплачивает;
6. backend обновляет подписку;
7. dashboard открывается только при нужном доступе.

В аккаунте можно видеть:

- email;
- ФИО;
- Telegram/contact;
- plan;
- payment status;
- paid until;
- support code;
- latest payment.

Это важно для поддержки. Если пользователь оплатил, но доступ не открылся, его можно найти по email/support code/txid.

---

## 8. Поддержка и аналитика

### Tawk.to

На сайт добавлен live chat Tawk.to. Он нужен для поддержки пользователей:

- вопросы по продукту;
- помощь с оплатой;
- помощь с доступом;
- trial/support.

### Yandex Metrika

Добавлена Yandex Metrika:

```text
counter id: 108276909
```

Она нужна для:

- анализа посещений;
- анализа кликов;
- понимания поведения пользователей;
- оценки конверсии.

---

## 9. Данные проекта

Проект использует большой набор рыночных и дополнительных данных.

Основные источники:

- BTCUSDT OHLCV;
- volume;
- VWAP;
- open interest;
- funding;
- liquidations;
- basis;
- order book imbalance;
- buy/sell ratio;
- CVD;
- volatility;
- technical indicators;
- news/sentiment;
- macro/risk indicators;
- ETF/stablecoin поля.

Пример важного датасета:

```text
friend_pnl_audit_pack_20260221_110031/data/BTCUSDT_15m_features_h20_v2_news_xlmr_full_2023_2026.parquet
```

Там есть около 170 колонок:

- price features;
- volatility features;
- news features;
- macro features;
- targets на разные горизонты.

---

## 10. Feature engineering

Основные файлы:

```text
build_features.py
build_features_multi_tf.py
merge_aux.py
binance_aux.py
bybit_aux.py
bybit_data.py
bybit_public_trades.py
```

Строятся признаки:

### Price / candle

- log return;
- range norm;
- wick up/down;
- close delta;
- volume delta;
- ATR;
- realized volatility.

### Technical indicators

- RSI;
- MACD;
- ADX;
- Stoch RSI;
- OBV;
- CMF;
- Bollinger bandwidth.

### Market structure

- open interest;
- funding;
- CVD;
- buy/sell ratio;
- liquidation imbalance;
- basis.

### News / sentiment

- news count;
- news sentiment;
- news shock;
- missing news flag.

### Targets

Основные targets:

```text
target_ret_h20
target_dir_h20
target_ret_h80
target_dir_h80
target_ret_h160
target_dir_h160
tb_label
label_3cls
```

Горизонты:

- h20 = примерно 5 часов;
- h80 = примерно 20 часов;
- h160 = примерно 40 часов.

---

## 11. Модели

В проекте есть несколько групп моделей.

### 11.1. Простые baseline-модели

Используются для sanity check и учебной сдачи:

- Logistic Regression;
- Random Forest;
- HistGradientBoosting;
- XGBoost/CatBoost эксперименты.

Они важны, потому что показывают нижнюю планку качества.

### 11.2. Keras sequence models

Основные архитектуры:

- TCN + Transformer;
- PatchTST;
- TSMixer;
- iTransformer;
- multi-task;
- multi-horizon;
- quantile heads;
- triple-barrier heads.

### 11.3. iTransformer

Одна из главных архитектур проекта.

Файлы:

```text
best/model_15m_itransformer_h20.keras
best/BOEVOY_2300pct_model_bench_itransformer.keras
```

Почему важна:

- хорошо работает с большим числом признаков;
- подходит для multivariate time series;
- использовалась в топовых backtest runs.

### 11.4. PatchTST

Использовалась как конкурент iTransformer.

Файлы:

```text
model_bench_patchtst.keras
model_15m_patchtst.keras
```

### 11.5. TSMixer

Более легкая time-series архитектура.

Файлы:

```text
model_bench_tsmixer.keras
```

### 11.6. V7 family

Модели, которые видны на сайте:

- `MODEL V7 (ATR • 3x)`;
- `MODEL V7 (+39% • 3x)`;
- `MODEL V7 (+75% • 3x)`.

Они используются как разные профили риска/доходности.

### 11.7. TG Hybrid

Модель/слой с учетом crowd/news/Telegram сигналов.

Использовались:

- crowd gate;
- soft sizing;
- hybrid filter;
- свежесть сигнала;
- влияние crowd layer на сделки.

### 11.8. MM Supervisor

Риск-менеджмент слой:

- sizing;
- drawdown cap;
- volatility regime;
- risk/trade;
- ограничения сделок.

### 11.9. Chronos2

Экспериментальная ветка с Chronos-style прогнозами.

Использовалась для:

- входных фильтров;
- exit checks;
- дополнительного подтверждения направления.

### 11.10. Meta-combo

Слой, который объединяет разные модели и правила:

- сигнал модели;
- crowd gate;
- Chronos gate;
- ATR exit;
- risk settings;
- meta probability.

---

## 12. Лучшие результаты

В проекте есть несколько сильных backtest-результатов.

### Aggressive iTransformer research run

Файл:

```text
best/BOEVOY_2300pct_report.csv
```

Пример метрик:

```text
total_return ≈ +2376%
Sharpe ≈ 5.79
MaxDD ≈ -65%
trades = 132
```

Это агрессивный исследовательский режим. Он показывает потенциал, но имеет высокий риск.

### Moderate iTransformer walk-forward

Файл:

```text
best/wf_itransformer_moderate_plus.csv
```

Пример метрик:

```text
total_return ≈ +91%
Sharpe ≈ 3.56
MaxDD ≈ -32%
trades = 30
```

Это более осторожный вариант, который лучше подходит для защиты и презентации.

### Meta-combo

Файл:

```text
reports/backtest_trade_combo_meta_best_calib.csv
```

Пример:

```text
final_equity ≈ 113.54
trades = 34
max_dd ≈ -2.36%
```

### Chronos2

Файл:

```text
reports/chronos2_stage12_best_2025_v2.csv
```

Пример:

```text
final_equity ≈ 175.72
trades = 63
max_dd ≈ -9.82%
```

Важно: все эти результаты являются историческим тестом, а не гарантией будущей прибыли.

---

## 13. Как модель превращается в сигнал

Сырая модель дает вероятность или ожидаемый return.

Дальше применяется торговая логика:

1. threshold для long;
2. threshold для short;
3. volatility filter;
4. cooldown;
5. min hold;
6. max trades per day;
7. fee/slippage;
8. drawdown cap;
9. risk sizing;
10. stop/invalidation.

После этого сайт показывает:

```text
LONG / SHORT / FLAT
confidence
risk/trade
leverage
invalidation
why
```

---

## 14. История сделок

Dashboard умеет показывать сделки модели.

Используются CSV:

```text
trades_conservative_canonical.csv
trades_aggressive_canonical.csv
trades_alpha75_canonical.csv
trades_tg_hybrid_canonical.csv
trades_mm_r13_canonical.csv
trades_chronos2_canonical.csv
trades_meta_best_2023.csv
trades_meta_best_2024.csv
trades_meta_best_2025.csv
```

На графике отображаются:

- entry;
- exit;
- LONG/SHORT;
- open/closed;
- model source.

Это делает проект визуально понятным: можно не только читать сигнал, но и видеть, где модель входила раньше.

---

## 15. Как работает график

График построен на TradingView Lightweight Charts.

Frontend:

```text
html/dashboard.html
html/lightweight-charts.standalone.production.js
```

Flow:

1. frontend вызывает `/candles`;
2. получает свечи;
3. нормализует формат;
4. отрисовывает candles;
5. подгружает deep history;
6. поверх рисует markers;
7. обновляет сигнал;
8. кеширует данные.

На мобильной версии отдельно дорабатывались:

- верхняя панель;
- нижняя навигация;
- компактность графика;
- переходы между вкладками;
- размер chart area;
- Market Radar layout.

---

## 16. Инфраструктура

В проекте использовались несколько машин.

### Основной ПК

```text
vitamind-b650m-k
```

Роль:

- основная разработка;
- подготовка пакетов;
- GitHub;
- локальные проверки;
- доступ к другим машинам.

### ms-7972

```text
vitamind-ms-7972
```

Роль:

- физический сервер;
- pipeline/data;
- дополнительные процессы.

Подключение:

```bash
ssh ms-7972
```

Настроен вход по ключу через LAN alias.

### helpd3

```text
helpd3-to-be-filled-by-o-e-m
```

Роль:

- сервер для сайта/экспериментов;
- учебные задания;
- WSGI/ASGI;
- дополнительные backend-процессы.

### Tailscale

Использовался для связи между машинами.

Команды:

```bash
tailscale status
tailscale ping <ip>
```

### Mullvad

На основном ПК включен Mullvad. Его нельзя отключать.

Для работы локальной сети включено:

```bash
mullvad lan set allow
```

---

## 17. Деплой

### Локальный запуск

```bash
cd /home/vitamind/my_project/model6
./scripts/start_site.sh
```

Скрипт запускает:

- web server на `8088`;
- FastAPI на `8000`;
- cloudflared tunnel для сайта;
- cloudflared tunnel для API;
- live update loop.

### Docker Compose

```bash
docker compose -f docker-compose.site.yml up -d --build
```

Сервисы:

- `api`;
- `web`.

### Домены

```text
https://tradeforge.art
https://www.tradeforge.art
https://api.tradeforge.art
```

---

## 18. GitHub

Для GitHub создана отдельная безопасная копия:

```text
/home/vitamind/my_project/tradeforge-public
```

Репозиторий:

```text
https://github.com/sxtr333/TRADING_AI_2X_PER_YEAR
```

Почему не пушится весь `model6`:

- много тяжелых моделей;
- приватные `.env`;
- базы пользователей;
- логи;
- исследовательские файлы;
- секреты;
- 11+ GB данных.

В публичную версию попали:

- сайт;
- backend;
- Docker;
- scripts;
- `.env.example`;
- README.

---

## 19. Пакет для сдачи проекта

Папка:

```text
/home/vitamind/my_project/model6/project_submission_trade_model
```

Финальный архив:

```text
/home/vitamind/my_project/model6/project_submission_trade_model/trade_model_submission_FINAL.zip
```

Внутри:

- `01_final_ready_model.ipynb`;
- `02_experiments_model_building.ipynb`;
- `03_tradeforge_top_model_backtest.ipynb`;
- `btcusdt_15m_sample.csv`;
- `trade_signal_model.joblib`;
- `top_tradeforge_model/`.

Третий ноутбук показывает топовую модель и backtest-графики.

---

## 20. Что было сделано за время проекта

Кратко по этапам:

1. Собрали первые данные BTCUSDT.
2. Сделали первые признаки.
3. Обучили первые Keras-модели.
4. Поняли, что простое предсказание цены слабое.
5. Перешли к direction/trading signal.
6. Добавили multi-horizon targets.
7. Добавили iTransformer, PatchTST, TSMixer.
8. Сделали walk-forward evaluation.
9. Добавили fee/slippage/risk filters.
10. Собрали backtest reports.
11. Сделали первые trade histories.
12. Сделали сайт.
13. Сделали dashboard.
14. Сделали график BTC.
15. Добавили markers сделок.
16. Добавили переключение моделей.
17. Добавили Market Radar.
18. Добавили Pump Scout.
19. Добавили auth.
20. Добавили email verification.
21. Добавили trial.
22. Добавили account modal.
23. Добавили поля ФИО/Telegram.
24. Добавили платежный flow.
25. Добавили support chat.
26. Добавили Yandex Metrika.
27. Доработали мобильную версию.
28. Подготовили GitHub-версию.
29. Подготовили zip для сдачи.
30. Подготовили README для презентации и заказчика.

---

## 21. Что важно помнить заказчику

1. TradeForge — это рабочий прототип продукта, а не просто исследование.
2. В проекте есть настоящие модели и backtest-отчеты.
3. Высокая доходность в отчетах — это исторический результат, не гарантия будущего.
4. Для реального прода нужно:
   - стабильный сервер;
   - мониторинг;
   - backups;
   - аккуратные secrets;
   - регулярный retraining;
   - контроль качества сигналов.
5. Пользовательский сайт уже выглядит как коммерческий продукт.
6. Самая сильная часть проекта — полный pipeline от данных до сайта.

---

## 22. Как презентовать проект

Лучший порядок презентации:

1. Показать landing.
2. Показать регистрацию/account.
3. Показать dashboard.
4. Показать график и сигнал.
5. Показать историю сделок.
6. Показать Market Radar.
7. Показать Pump Scout.
8. Показать backtest notebook.
9. Показать архитектуру.
10. Объяснить, что было сделано от данных до продукта.

Ключевая формулировка:

> TradeForge — это платформа, которая объединяет рыночные данные, time-series модели, backtest, риск-фильтры и сайт с подпиской. Проект прошел путь от экспериментов с BTC до полноценного продукта.

---

## 23. Ограничения

Проект сильный, но важно понимать ограничения:

- рынок BTC шумный;
- backtest может переоценивать результат;
- новости могут менять статистику;
- высокий leverage опасен;
- aggressive режим нельзя выдавать за гарантированную стратегию;
- нужен постоянный мониторинг;
- нужны регулярные проверки на новых данных;
- нужны реальные тесты исполнения.

---

## 24. Что делать дальше

Рекомендуемые следующие шаги:

### Product

- улучшить admin panel;
- сделать страницу сравнения моделей;
- сделать страницу model health;
- добавить экспорт пользователей/оплат;
- улучшить onboarding;
- добавить FAQ.

### ML

- регулярный walk-forward;
- regime classifier;
- calibration;
- meta-labeling;
- better WAIT filter;
- контроль drift;
- регулярный retraining.

### Infrastructure

- перенести на стабильный VPS;
- systemd или Docker Compose prod;
- backups;
- monitoring;
- alerts;
- secrets manager;
- CI/CD.

### Business

- тест трафика;
- TikTok/short videos;
- демо-воронка;
- pricing test;
- поддержка пользователей;
- сбор обратной связи.

---

## 25. Подробная карта данных, моделей и запусков

Этот раздел нужен, чтобы заказчик или проверяющий мог не просто увидеть сайт, а понять, из каких файлов он собран: где лежат датасеты, какие модели к ним относятся, какие отчеты подтверждают результаты и что именно грузится на dashboard.

### 25.1. Главная рабочая папка

Весь основной проект лежит здесь:

```text
/home/vitamind/my_project/model6
```

Папка `data` в проекте является рабочим хранилищем датасетов. На машине она вынесена на отдельный диск:

```text
/home/vitamind/my_project/model6/data
```

Внутри проекта есть четыре важных слоя:

```text
data/       - свечи, признаки, meta-datasets, user db
models      - .keras модели и .npz normalization stats в корне проекта и best/
reports/    - результаты backtest и CSV со сделками
html/       - сайт, dashboard, landing, screener, pump scout
scripts/    - запуск, обучение, сборка признаков, бэктесты
```

### 25.2. Сырые рыночные данные

Базовые свечи BTCUSDT:

```text
data/BTCUSDT_15m.parquet
data/BTCUSDT_15m_2020_2022.parquet
```

Назначение:

- это основа для графика и feature engineering;
- таймфрейм основной модели — 15 минут;
- на этих свечах строятся return, ATR, volatility, wick/range признаки, технические индикаторы и target-колонки.

Дополнительные 1h-данные:

```text
data/BTCUSDT_1h_aux_binance_2020Q4.parquet
data/BTCUSDT_1h_aux_binance_2020_2022.parquet
data/BTCUSDT_1h_aux_binance_2021_2022.parquet
data/BTCUSDT_1h_aux_merged_2020_2022.parquet
```

Назначение:

- расширяют картину рынка;
- используются для auxiliary market context;
- нужны не для самой страницы сайта напрямую, а для построения обучающих признаков.

### 25.3. Основные feature datasets

Самый простой актуальный features-файл без тяжелого news-слоя:

```text
data/BTCUSDT_15m_features_h20_v2.parquet
```

Он используется в простом локальном запуске `scripts/start_site.sh`:

```text
--features data/BTCUSDT_15m_features_h20_v2.parquet
```

Более полный production/research features-файл с news/XLM-R веткой:

```text
data/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes.parquet
```

Его serve-версия для API:

```text
data/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes_serve.parquet
```

Именно serve-версия используется Docker Compose API:

```text
--features /app/data/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes_serve.parquet
```

Полный исторический датасет для research/backtest:

```text
data/BTCUSDT_15m_features_h20_v2_news_xlmr_full_2020_2026.parquet
data/BTCUSDT_15m_features_h20_v2_news_xlmr_full_2020_2026_dailyfill.parquet
data/BTCUSDT_15m_features_h20_v2_news_xlmr_full_2023_2026.parquet
```

Назначение:

- `full_2023_2026` — компактный сильный набор для финальной сдачи и демонстрации модели;
- `full_2020_2026` — широкий исторический набор для проверки устойчивости;
- `dailyfill` — версия, где пропуски новостей/внешних данных заполнены так, чтобы backtest не ломался на старых годах.

Важно: в ходе проекта мы пришли к выводу, что новости могут ухудшать стабильность модели, если поток новостей меняется по годам. Поэтому текущая логика разделяет:

- price/market features — стабильная база;
- news features — экспериментальный слой;
- meta filters — отдельный слой, который можно включать/выключать в backtest.

### 25.4. News / sentiment datasets

Файлы новостного слоя:

```text
warc_by_notebook/ccnews_2023-10_part0001.parquet
warc_by_notebook/ccnews_2024-10_part0001.parquet
warc_by_notebook/ccnews_2024-12_part0001.parquet
```

Назначение:

- это source/sample новостей;
- на их основе строились агрегаты news count, sentiment, missing flag;
- потом эти признаки попадали в `BTCUSDT_15m_features_*news_xlmr*.parquet`.

Скрипты новостной ветки:

```text
scripts/run_news_full_pipeline_2023_2026.sh
scripts/build_features_news_xlmr_v4_8nodes.sh
scripts/build_features_news_xlmr_v3.sh
scripts/build_features_news_hybrid.py
```

Что делают:

- собирают/объединяют news parquet;
- привязывают новости ко времени свечей;
- строят агрегаты;
- сохраняют итоговый features parquet.

### 25.5. Meta datasets

Meta dataset — это не свечи и не сырые признаки. Это слой поверх базовых моделей: вероятности, флаги, режимы, news flags, разметка сделок и дополнительные поля для meta-combo.

Основные файлы:

```text
data/meta/meta_dataset_pruned.parquet
data/meta/meta_dataset_pruned_raw.parquet
data/meta/meta_dataset_pruned_regime.parquet
data/meta/meta_dataset_pruned_newsflag_2024calib.parquet
data/meta/meta_dataset_pruned_newsflag_mix15_2024calib.parquet
data/meta/meta_dataset_pruned_newsflag_mix15_2024calib_full.parquet
data/meta/meta_dataset_pruned_newsflag_mix15_2024calib_full_align_2023_2025.parquet
data/meta/meta_dataset_aligned_to_features_v4_8nodes.parquet
```

Назначение:

- `meta_dataset_pruned.parquet` — базовый meta-набор для combo backtest;
- `meta_dataset_pruned_newsflag_*` — версии с news/missing flags;
- `meta_dataset_aligned_to_features_v4_8nodes.parquet` — версия, выровненная по timestamps с основным features-файлом;
- `mix15` — ветка, где использовались 15m features и несколько горизонтов.

Скрипты:

```text
scripts/build_meta_dataset.py
scripts/build_meta_dataset_cusum_tb.py
scripts/run_mix15_pipeline_2026-01-29.sh
scripts/run_improve_permodel_meta.sh
```

### 25.6. Какие датасеты к каким моделям относятся

| Модель / ветка | Основной датасет | Вес модели | Normalization stats | Отчет / trades |
|---|---|---|---|---|
| Live API h20 / battle | `data/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes_serve.parquet` или локально `data/BTCUSDT_15m_features_h20_v2.parquet` | `model_battle_itransformer.keras` | `norm_stats_battle_itransformer.npz` | API `/forecast`, dashboard signal |
| Live API multi / TB | тот же features-файл API | `model_15m_itransformer_tb_multi.keras` | `norm_stats_15m_itransformer_tb_multi.npz` | API `/forecast-multi`, dashboard status |
| iTransformer h20 | `data/BTCUSDT_15m_features_h20_v2*.parquet` | `model_15m_itransformer_h20.keras` или `best/model_15m_itransformer_h20.keras` | `norm_stats_15m_itransformer_h20.npz` | `best/wf_itransformer_moderate_plus.csv` |
| Aggressive research run | `data/BTCUSDT_15m_features_h20_v2_news_xlmr_full_2023_2026.parquet` | `best/BOEVOY_2300pct_model_bench_itransformer.keras` | `best/BOEVOY_2300pct_norm_stats_bench_itransformer.npz` | `best/BOEVOY_2300pct_report.csv` |
| PatchTST baseline | `data/BTCUSDT_15m_features_h20_v2.parquet` | `model_bench_patchtst.keras` | `norm_stats_bench_patchtst.npz` | benchmark/eval reports |
| TSMixer baseline | `data/BTCUSDT_15m_features_h20_v2.parquet` | `model_bench_tsmixer.keras` | `norm_stats_bench_tsmixer.npz` | benchmark/eval reports |
| Price horizon h20/h80/h160/h320/h640 | `data/BTCUSDT_15m_features_h20_v2*.parquet` | `model_15m_itransformer_price_h*.keras` | `norm_stats_15m_itransformer_price_h*.npz` | price quality reports |
| Meta-combo | `data/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes_with_news_pruned.parquet` + `data/meta/meta_dataset_pruned.parquet` | правила + meta thresholds | не всегда отдельный `.npz` | `reports/backtest_trade_combo_meta_best_calib.csv`, `reports/trades_meta_best_*.csv` |
| Chronos2 | features + Chronos-style forecasts | внешний/экспериментальный слой | нет стандартного `.npz` | `reports/chronos2_stage12_best_2025_v2.csv`, `reports/trades_chronos2_canonical.csv` |
| TG Hybrid | `data/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes_yrnorm_repl.parquet` и related meta/news fields | hybrid rules | нет стандартного `.npz` | `reports/backtest_tg_risk39_lev25_hybrid_chronos2_best_2025_v2.csv`, `reports/trades_tg_hybrid_canonical.csv` |

### 25.7. Модели, которые видит пользователь на сайте

В dashboard пользователь видит не технические имена `.keras`, а продуктовые режимы:

```text
MODEL V7 (ATR • 3x)
MODEL V7 (+39% • 3x)
MODEL V7 (+75% • 3x)
TG Hybrid v3.2_0
MM Supervisor r13
Chronos2 (stage2)
```

Эти режимы связаны с CSV-историей сделок в `html/dashboard/index.html`:

| Режим на сайте | CSV истории сделок |
|---|---|
| Conservative / V7 ATR | `reports/trades_conservative_canonical.csv` |
| Aggressive / V7 +39% | `reports/trades_aggressive_canonical.csv` |
| Alpha75 / V7 +75% | `reports/trades_alpha75_canonical.csv` |
| TG Hybrid | `reports/trades_tg_hybrid_canonical.csv` |
| MM Supervisor | `reports/trades_mm_r13_canonical.csv` |
| Chronos2 | `reports/trades_chronos2_canonical.csv` |
| Meta-combo by year | `reports/trades_meta_best_2023.csv`, `reports/trades_meta_best_2024.csv`, `reports/trades_meta_best_2025.csv` |

На графике эти CSV используются для markers:

- entry marker;
- exit marker;
- LONG/SHORT label;
- open/closed trade;
- source/model name.

Список ниже графика использует те же данные, поэтому график и trade history должны совпадать.

### 25.8. Основные отчеты с результатами

Самые важные отчеты:

```text
best/BOEVOY_2300pct_report.csv
best/wf_itransformer_moderate_plus.csv
reports/backtest_trade_combo_meta_best_calib.csv
reports/chronos2_stage12_best_2025_v2.csv
reports/backtest_tg_risk39_lev25_hybrid_chronos2_best_2025_v2.csv
reports/wf_battle_opt_sharpe_report.csv
reports/wf_battle_price_report.csv
reports/wf_multi_price_report.csv
```

Что они значат:

- `BOEVOY_2300pct_report.csv` — агрессивный research run, высокая доходность, высокий риск;
- `wf_itransformer_moderate_plus.csv` — более спокойный walk-forward;
- `backtest_trade_combo_meta_best_calib.csv` — meta-combo с калибровкой;
- `chronos2_stage12_best_2025_v2.csv` — Chronos2 эксперимент;
- `backtest_tg_*` — hybrid версия с crowd/news/risk logic;
- `wf_*` — walk-forward отчеты для сравнения моделей.

### 25.9. Как собрать признаки заново

Базовая сборка OHLCV/features:

```bash
cd /home/vitamind/my_project/model6
./scripts/update_ohlcv_and_features.sh
```

Сборка news/XLM-R v4 8nodes:

```bash
cd /home/vitamind/my_project/model6
./scripts/build_features_news_xlmr_v4_8nodes.sh
```

Полный pipeline 2023-2026 с news:

```bash
cd /home/vitamind/my_project/model6
./scripts/run_news_full_pipeline_2023_2026.sh
```

Расширение датасета на 2020-2022:

```bash
cd /home/vitamind/my_project/model6
./scripts/extend_dataset_2020_2022.sh
```

Важно: перед пересборкой нужно понимать, что новый датасет может изменить поведение модели. Для защиты и сдачи лучше использовать уже сохраненные parquet/keras/report файлы, чтобы результат был воспроизводимым.

### 25.10. Как запустить сайт локально

Вариант через готовый скрипт:

```bash
cd /home/vitamind/my_project/model6
./scripts/start_site.sh
```

Что делает скрипт:

- запускает API на `8000`;
- запускает web на `8088`;
- поднимает cloudflared tunnel;
- отдает frontend из `html/`;
- подключает модель `model_battle_itransformer.keras`;
- подключает модель `model_15m_itransformer_tb_multi.keras`;
- подключает features parquet.

Вариант через Docker Compose:

```bash
cd /home/vitamind/my_project/model6
docker compose -f docker-compose.site.yml up -d --build
```

Проверка API:

```bash
curl https://api.tradeforge.art/health
```

Проверка локально:

```bash
curl http://127.0.0.1:8000/health
```

### 25.11. Как API запускает модели

Основной backend:

```text
serve_fastapi.py
```

Пример production-команды из Docker Compose:

```bash
python /app/serve_fastapi.py \
  --model-h20 /app/model_battle_itransformer.keras \
  --stats-h20 /app/norm_stats_battle_itransformer.npz \
  --model-multi /app/model_15m_itransformer_tb_multi.keras \
  --stats-multi /app/norm_stats_15m_itransformer_tb_multi.npz \
  --features /app/data/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes_serve.parquet \
  --seq-len 256 \
  --host 0.0.0.0 \
  --port 8000
```

Логика:

1. API загружает parquet с признаками;
2. сортирует по `timestamp`;
3. берет последние `seq_len=256` строк;
4. нормализует признаки через `.npz`;
5. отправляет окно в Keras-модель;
6. получает прогноз;
7. переводит прогноз в LONG/SHORT/FLAT;
8. frontend показывает сигнал и объяснение.

### 25.12. Как сайт получает график и сигналы

Основные frontend-файлы:

```text
html/index.html
html/landing-core.js
html/dashboard/index.html
html/screener.html
html/pump-scout/index.html
```

Dashboard работает так:

```text
html/dashboard/index.html
  -> API_BASE = https://api.tradeforge.art
  -> /candles
  -> /forecast или dashboard bootstrap endpoints
  -> reports/trades_*.csv
  -> TradingView Lightweight Charts
```

График:

- свечи приходят из API `/candles`;
- история сделок приходит из CSV;
- markers рисуются поверх свечей;
- кнопка `TRADES` показывает список тех же сделок;
- переключение модели меняет активный CSV/markers.

Market Radar:

```text
html/screener.html
  -> /market/top
  -> /market/snapshot
  -> /market/icon/{symbol}
```

Pump Scout:

```text
html/pump-scout/index.html
  -> /market/top?mode=mexc_perp&direction=pump
  -> /market/top?mode=mexc_perp&direction=dump
```

Landing/account:

```text
html/landing-core.js
  -> /auth/register
  -> /auth/login
  -> /auth/verify-email
  -> /subscription/status
  -> /billing/prodamus/create-payment
```

### 25.13. Как обучать модели

Обучающие скрипты в проекте:

```text
train_model.py
train_model_advanced.py
train_model_deep.py
train_model_multih.py
train_model_multitask.py
train_model_tb.py
scripts/train_news_xlmr_cosmic_v6_dr_scale_8nodes.sh
scripts/train_long_short_all_mix15_2020_2026.sh
scripts/train_h80_h160_long_short_mix15_2026-01-29.sh
```

Пример запуска ветки news/XLM-R:

```bash
cd /home/vitamind/my_project/model6
./scripts/train_news_xlmr_cosmic_v6_dr_scale_8nodes.sh
```

Пример запуска long/short multi-horizon:

```bash
cd /home/vitamind/my_project/model6
./scripts/train_long_short_all_mix15_2020_2026.sh
```

Правило проекта:

- обучение тяжелых моделей запускалось на GPU;
- CPU использовался только для легкой подготовки файлов, CSV, parquet и проверок;
- для воспроизводимости итоговые веса и отчеты сохранялись рядом с проектом.

### 25.14. Как прогонять backtest

Основные backtest-скрипты:

```text
scripts/backtest_long_short_v7.py
scripts/backtest_long_short_v7_sweep.py
scripts/backtest_trade_combo_meta.py
scripts/backtest_trade_combo_leverage.py
scripts/backtest_combo_v7.py
scripts/run_bt_atr3x_all_years.sh
scripts/run_param_sweep_3x_2023_2025.sh
scripts/run_leverage_grid_thr059.sh
```

Пример meta-combo backtest:

```bash
cd /home/vitamind/my_project/model6
python scripts/backtest_trade_combo_meta.py \
  --features data/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes_with_news_pruned.parquet \
  --meta-features data/meta/meta_dataset_pruned.parquet
```

Пример ATR 3x по годам:

```bash
cd /home/vitamind/my_project/model6
./scripts/run_bt_atr3x_all_years.sh
```

Результаты обычно сохраняются в:

```text
reports/
best/
```

### 25.15. Какие файлы нужны для полной демонстрации

Минимальный набор для сайта:

```text
serve_fastapi.py
docker-compose.site.yml
docker.site.Dockerfile
html/
data/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes_serve.parquet
model_battle_itransformer.keras
norm_stats_battle_itransformer.npz
model_15m_itransformer_tb_multi.keras
norm_stats_15m_itransformer_tb_multi.npz
reports/trades_*.csv
```

Минимальный набор для защиты ML-части:

```text
project_submission_trade_model/trade_model_submission_FINAL.zip
best/BOEVOY_2300pct_model_bench_itransformer.keras
best/BOEVOY_2300pct_norm_stats_bench_itransformer.npz
best/BOEVOY_2300pct_report.csv
best/wf_itransformer_moderate_plus.csv
data/BTCUSDT_15m_features_h20_v2_news_xlmr_full_2023_2026.parquet
```

Минимальный набор для research-воспроизведения:

```text
data/BTCUSDT_15m_features_h20_v2_news_xlmr_full_2020_2026_dailyfill.parquet
data/meta/meta_dataset_pruned_newsflag_mix15_2024calib_full.parquet
scripts/run_mix15_pipeline_2026-01-29.sh
scripts/backtest_trade_combo_meta.py
reports/
best/
```

### 25.16. Как восстановить проект после перезапуска машин

Порядок проверки:

```bash
cd /home/vitamind/my_project/model6
git status
ls data | head
ls reports | head
ls *.keras | head
```

Проверить API:

```bash
curl http://127.0.0.1:8000/health
curl https://api.tradeforge.art/health
```

Запустить сайт:

```bash
./scripts/start_site.sh
```

Или Docker:

```bash
docker compose -f docker-compose.site.yml up -d --build
docker compose -f docker-compose.site.yml ps
```

Если график пустой:

- проверить `/candles`;
- проверить features path;
- проверить, что parquet содержит `timestamp`, `open`, `high`, `low`, `close`;
- проверить browser console;
- проверить `API_BASE` в frontend.

Если сделки не видны на графике:

- проверить файлы `reports/trades_*_canonical.csv`;
- проверить, что timestamp сделок попадает в диапазон свечей;
- проверить выбранную модель на dashboard;
- проверить mapping trade files в `html/dashboard/index.html`.

Если сигнал не обновляется:

- проверить `/forecast`;
- проверить `.keras` и `.npz`;
- проверить, что размер окна совпадает с `seq_len=256`;
- проверить, что порядок feature columns совпадает с тем, на чем обучалась модель.

### 25.17. Что важно не потерять

Критически важные файлы:

```text
model_battle_itransformer.keras
norm_stats_battle_itransformer.npz
model_15m_itransformer_tb_multi.keras
norm_stats_15m_itransformer_tb_multi.npz
best/
reports/
data/meta/
data/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes_serve.parquet
data/BTCUSDT_15m_features_h20_v2_news_xlmr_full_2023_2026.parquet
project_submission_trade_model/trade_model_submission_FINAL.zip
```

Их нужно бэкапить в первую очередь. Без них сайт можно открыть как frontend, но нельзя честно восстановить ML-логику, историю сделок и защитные материалы.

---

## 26. Быстрые ссылки и пути

### Рабочий проект

```text
/home/vitamind/my_project/model6
```

### README для полной презентации

```text
/home/vitamind/my_project/model6/TRADEFORGE_FULL_README.md
```

### README для заказчика

```text
/home/vitamind/my_project/model6/TRADEFORGE_CLIENT_README.md
```

### Сдача проекта

```text
/home/vitamind/my_project/model6/project_submission_trade_model/trade_model_submission_FINAL.zip
```

### GitHub public copy

```text
/home/vitamind/my_project/tradeforge-public
```

### GitHub repo

```text
https://github.com/sxtr333/TRADING_AI_2X_PER_YEAR
```

### Сайт

```text
https://tradeforge.art
```

### API

```text
https://api.tradeforge.art
```

---

## 27. Финальное резюме

TradeForge — это проект, в котором за несколько месяцев была собрана полноценная система:

```text
данные -> признаки -> модели -> бэктест -> сигналы -> backend -> сайт -> аккаунты -> подписка
```

Проект можно показывать как:

- ML-проект;
- торговый research;
- backend/frontend приложение;
- коммерческий SaaS-прототип;
- итоговую работу для защиты.

Самое главное достижение: это не отдельная модель и не отдельный сайт. Это связанный продуктовый pipeline, где каждая часть работает вместе.
