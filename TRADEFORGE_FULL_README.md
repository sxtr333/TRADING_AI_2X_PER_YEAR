# TradeForge.art — полный README проекта

Этот файл нужен как большая карта проекта: для презентации, защиты, восстановления сервиса и объяснения, как устроены сайт, модели, данные и серверы.

TradeForge.art — это веб-платформа для анализа BTCUSDT и крипторынка. В проекте есть публичная витрина, личный кабинет, подписка, демо-доступ, dashboard с графиком, несколько торговых моделей, история сделок, Market Radar, Pump Scout, новости, платежи, email-верификация и backend API.

Главная идея проекта: не просто нарисовать график BTC, а собрать полный исследовательский и продуктовый контур вокруг торговых сигналов:

- собираем рыночные данные;
- строим признаки;
- обучаем разные модели;
- проверяем их через walk-forward/backtest;
- превращаем предсказания в сигналы LONG / SHORT / FLAT;
- показываем это на сайте в понятном интерфейсе;
- закрываем доступ через регистрацию, email, trial и подписку.

Важно: результаты бэктестов не являются гарантией будущей доходности. Проект показывает исследовательский pipeline и рабочий продуктовый прототип, а не обещание прибыли.

---

## 1. С чего начинали

Изначально проект был просто папкой `model6`, где проверялись идеи прогнозирования BTCUSDT:

- простые модели на свечах;
- регрессия будущей цены;
- классификация направления;
- проверка sign accuracy;
- первые Keras-модели;
- первые признаки из OHLCV;
- первые отчеты по PnL.

Первые результаты были слабые. Модель часто угадывала направление почти как случайная, а простая стратегия давала плохой PnL. Это было нормальным этапом: рынок BTC на коротком горизонте шумный, и простое "предскажи следующую цену" почти не работает.

После этого проект стал развиваться в сторону полноценного pipeline:

- больше источников данных;
- больше признаков;
- несколько горизонтов;
- не только price forecast, а trading decision;
- не только точность модели, а результат стратегии с комиссиями и риском;
- сайт как конечный продукт.

---

## 2. К чему пришли

Сейчас TradeForge состоит из нескольких больших частей:

1. **Research / обучение моделей**
   - датасеты BTCUSDT;
   - признаки;
   - Keras-модели;
   - iTransformer / PatchTST / TSMixer;
   - walk-forward;
   - отчеты в `reports/`;
   - лучшие веса в `best/`.

2. **Backend**
   - `serve_fastapi.py`;
   - FastAPI;
   - свечи;
   - сигналы;
   - новости;
   - Market Radar;
   - Pump Scout;
   - auth;
   - email-коды;
   - подписки;
   - платежи;
   - API для dashboard.

3. **Frontend**
   - `html/index.html` — главная страница;
   - `html/dashboard.html` — торговый dashboard;
   - `html/screener.html` — Market Radar;
   - `html/pump-scout/index.html` — Pump Scout;
   - `html/how-to-use/index.html` — пояснения;
   - `html/settings/index.html` — настройки;
   - legal/payment pages.

4. **Инфраструктура**
   - основной ПК;
   - физический сервер `ms-7972`;
   - сервер `helpd3`;
   - Tailscale;
   - Cloudflare tunnel;
   - Docker Compose;
   - статический web server;
   - API server;
   - логи и watchdog.

5. **Публичная упаковка**
   - отдельная GitHub-версия без секретов;
   - учебный zip для сдачи;
   - презентационный README;
   - ноутбуки для защиты.

---

## 3. Основная структура папок

```text
model6/
├── serve_fastapi.py                 # основной backend API
├── build_features.py                # сбор признаков
├── build_features_multi_tf.py       # признаки с несколькими таймфреймами
├── train_keras.py                   # обучение Keras-моделей
├── train_keras_v7.py                # v7-ветка моделей
├── walk_forward_eval.py             # walk-forward и PnL evaluation
├── backtest_trade_combo_meta_dynamic_exit.py
├── model_layers.py                  # кастомные слои: RevIN, iTransformer, TSMixer и др.
├── trading_keras_core.py            # архитектуры sequence-моделей
├── realtime_stream.py               # realtime-поток/предикт
├── data/                            # raw и feature datasets, symlink на oldssd
├── best/                            # лучшие модели и отчеты
├── reports/                         # бэктесты, equity, monthly, trades
├── html/                            # frontend сайта
├── scripts/                         # запуск, watchdog, экспорт, сбор данных
├── state/                           # user/subscription db, локальное состояние
├── project_submission_trade_model/  # пакет для сдачи проекта
└── docker-compose.site.yml          # docker-deploy API + web
```

---

## 4. Главная продуктовая идея сайта

TradeForge.art показывает пользователю не "сырые предсказания модели", а собранный торговый контекст:

- текущий статус модели;
- вероятность LONG / SHORT / FLAT;
- таймфрейм;
- горизонт;
- stop / invalidation;
- confidence;
- risk/trade;
- leverage;
- новости;
- рыночный режим;
- историю сделок модели;
- график BTC;
- точки входа/выхода;
- Market Radar;
- Pump Scout.

То есть сайт не просто говорит "цена будет выше". Он показывает, какая модель выбрана, на каком горизонте, почему она дала сигнал, какие были сделки и как это выглядит на графике.

---

## 5. Основные страницы сайта

### 5.1. Landing page

Файлы:

```text
html/index.html
html/landing.css
html/landing-core.js
html/landing-ui.js
html/landing-app.js
```

Landing page отвечает за:

- первое впечатление;
- описание продукта;
- trial 3 days;
- тарифы;
- вход/регистрацию;
- аккаунт;
- оплату;
- переход в dashboard;
- Yandex OAuth;
- email verification;
- поддержку;
- Yandex Metrika;
- Tawk.to chat.

На landing есть:

- кнопка `Start trial`;
- кнопка `Open demo`;
- кнопка `Account`;
- тарифы Starter / Pro / Elite;
- описание, что внутри dashboard;
- live proof-блок;
- backend health/latency;
- блоки технологии.

### 5.2. Dashboard

Файл:

```text
html/dashboard.html
```

Это главная рабочая страница.

Возможности:

- график BTCUSDT;
- таймфреймы `15m`, `1H`, `4H`, `1D`;
- загрузка свечей через API `/candles`;
- deep history для 1D/4H/1H;
- переключение моделей;
- боковая панель сигнала;
- probability LONG / FLAT / SHORT;
- markers входов/выходов;
- trade history;
- новости;
- liquidation / volatility context;
- кнопки навигации по графику;
- мобильная нижняя навигация.

График строится на:

```text
html/lightweight-charts.standalone.production.js
```

Логика:

- frontend вызывает `/dashboard/bootstrap`;
- получает свечи, сигнал, новости, leverage/backtest summary;
- строит candles;
- накладывает markers;
- обновляет signal panel;
- кеширует данные в memory/localStorage;
- подгружает более глубокую историю отдельным запросом.

Ключевые элементы dashboard:

```text
loadCandles()
scheduleCandles()
loadDashboardBootstrap()
applyDashboardBootstrap()
setDecision()
updateDirectionMarkers()
renderTradeHistory()
```

### 5.3. Market Radar

Файл:

```text
html/screener.html
```

Market Radar нужен, чтобы видеть рынок шире BTC:

- classic spot radar;
- MEXC perpetual mode;
- поиск монеты;
- сортировка по росту/силе;
- min volume;
- spread alert;
- pump score;
- dump mode;
- live refresh.

Frontend ходит в:

```text
/market/top
/market/snapshot
/market/icon/{symbol}
```

Market Radar — это не trading model, а market scanning layer. Он помогает понять, где сейчас движение и какие альты резко растут/падают.

### 5.4. Pump Scout

Файл:

```text
html/pump-scout/index.html
```

Pump Scout — отдельный модуль для быстрых movers:

- Ignite lane;
- Fade lane;
- Dump lane;
- MEXC perpetual feed;
- live market movers;
- pump score;
- board для быстрых решений.

API:

```text
/market/top?mode=mexc_perp&direction=pump
/market/top?mode=mexc_perp&direction=dump
```

### 5.5. How to use / Settings / Legal

Файлы:

```text
html/how-to-use/index.html
html/settings/index.html
html/privacy.html
html/refund.html
html/offer.html
html/payment-success.html
html/payment-cancel.html
```

Эти страницы нужны, чтобы сайт выглядел как продукт, а не как демка:

- условия;
- возвраты;
- оферта;
- инструкции;
- настройки;
- payment result pages.

---

## 6. Backend API

Основной файл:

```text
serve_fastapi.py
```

Backend делает несколько задач сразу:

1. грузит модели;
2. грузит dataset/features;
3. отдает свечи;
4. считает/отдает сигналы;
5. отдает bootstrap для dashboard;
6. отдает новости;
7. отдает market radar;
8. обслуживает auth;
9. обслуживает подписки;
10. создает платежи;
11. принимает webhooks;
12. хранит user state.

### 6.1. Health

```text
GET /health
```

Используется для:

- проверки API;
- live proof на landing;
- мониторинга;
- диагностики.

### 6.2. Candles

```text
GET /candles?symbol=BTCUSDT&interval=15m&limit=6000
```

Отдает свечи в формате, который frontend превращает в Lightweight Charts candles:

```text
time, open, high, low, close, volume
```

Особенность:

- для мобильной версии и dashboard были проблемы с пустым графиком;
- потом добавили fallback, caching и deep loading;
- для 1D/4H/1H используется более глубокая история.

### 6.3. Dashboard bootstrap

```text
GET /dashboard/bootstrap
```

Это удобный endpoint, который одним запросом отдает:

- candles;
- current signal;
- news;
- leverage/backtest snippets;
- текущий model profile.

Так dashboard быстрее открывается и меньше дергает API.

### 6.4. Signals

Сигнал на сайте — это не просто raw output модели. Он приводится к понятному виду:

```text
LONG / SHORT / FLAT
confidence
horizon
leverage
risk_pct
invalidation
why
trap
```

В dashboard это отображается как:

- status;
- confidence;
- leverage;
- risk/trade;
- momentum;
- volatility regime;
- invalidation stop;
- probability LONG/FLAT/SHORT.

### 6.5. Auth

Backend содержит auth flow:

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

- регистрация по email;
- код подтверждения;
- resend code;
- вход по паролю;
- password reset;
- session token;
- cookie bridge для OAuth;
- профиль пользователя;
- поля ФИО и Telegram/contact;
- support code для помощи клиентам.

### 6.6. Subscription

```text
GET /subscription/status
```

Хранит:

- active/inactive;
- plan;
- billing period;
- paid until;
- trial until;
- latest payment;
- txid/support info.

На landing кнопка `Account` показывает данные подписки и оплаты.

### 6.7. Billing

В проекте есть подготовка под YooKassa и Prodamus.

Основные переменные:

```text
YOOKASSA_SHOP_ID
YOOKASSA_SECRET_KEY
PRODAMUS_SECRET_KEY
PRODAMUS_PAYMENT_URL
PRODAMUS_WEBHOOK_URL
```

API flow:

```text
/billing/prodamus/create-payment
/billing/prodamus/webhook
```

Идея:

- пользователь выбирает тариф;
- backend создает payment link;
- после оплаты webhook обновляет подписку;
- аккаунт показывает оплату и срок доступа.

### 6.8. Market endpoints

```text
/market/top
/market/snapshot
/market/icon/{symbol}
```

Используются для:

- Market Radar;
- Pump Scout;
- pump/dump scanner;
- live market board.

---

## 7. Данные

Главный актив проекта — не только модели, а датасеты и признаки.

Источники:

- BTCUSDT OHLCV;
- Bybit/Binance metrics;
- open interest;
- funding;
- liquidations;
- basis;
- order book imbalance;
- tick buy/sell volume;
- volatility features;
- technical indicators;
- macro/risk indicators;
- news/sentiment;
- ETF/stablecoin fields;
- calendar/time features.

Пример основного feature dataset:

```text
friend_pnl_audit_pack_20260221_110031/data/BTCUSDT_15m_features_h20_v2_news_xlmr_full_2023_2026.parquet
```

В нем есть:

- `open`, `high`, `low`, `close`, `volume`;
- `vwap`;
- `open_interest`;
- `funding_rate`;
- `cvd`;
- `liq_long`, `liq_short`, `liq_imbalance`;
- `buy_sell_ratio`;
- `atr`, `rv`, `rv_ratio`;
- `rsi14`, `macd_line`, `macd_signal`, `adx14`;
- `news_count`, `news_sentiment`, `news_shock`;
- `target_ret_h20`, `target_dir_h20`;
- targets для h40/h60/h80/h100/h120/h140/h160.

---

## 8. Feature engineering

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

Что строим:

### 8.1. Price features

- close delta;
- log return;
- range norm;
- wick up/down;
- bollinger bandwidth;
- ATR;
- realized volatility.

### 8.2. Market microstructure

- buy/sell ratio;
- delta;
- CVD;
- open interest delta;
- volume delta;
- liquidity imbalance.

### 8.3. TA indicators

- RSI;
- MACD;
- ADX;
- Stoch RSI;
- OBV;
- CMF.

### 8.4. News/sentiment

Был отдельный поток новостей:

- raw news;
- dedup;
- sentiment cache;
- merged news dataset.

Позже было принято решение осторожнее относиться к новостям, потому что разный баланс новостного потока может портить стабильность моделей. Поэтому новости используются как feature/filter, но не должны слепо ломать основную ценовую модель.

### 8.5. Targets

Основные targets:

```text
target_ret
target_dir
target_dir_vf
tb_label
label_3cls
target_ret_h20
target_dir_h20
target_ret_h80
target_dir_h80
target_ret_h160
target_dir_h160
```

Горизонты:

- h20 = 20 свечей по 15 минут = примерно 5 часов;
- h80 = примерно 20 часов;
- h160 = примерно 40 часов.

---

## 9. Модели проекта

В проекте есть несколько семейств моделей.

### 9.1. Simple / baseline models

Использовались для проверки идеи:

- Logistic Regression;
- Random Forest;
- HistGradientBoosting;
- XGBoost/CatBoost эксперименты;
- simple Keras regressors.

Они нужны для:

- sanity check;
- учебных ноутбуков;
- быстрого reproducible pipeline;
- проверки, есть ли сигнал вообще.

Для сдачи проекта сделан отдельный пакет:

```text
project_submission_trade_model/
```

Там есть простая финальная модель:

```text
trade_signal_model.joblib
```

Она прогнозирует направление BTCUSDT на h20. Это не топовая модель сайта, а удобный reproducible model для проверки преподавателем.

### 9.2. Keras sequence models

Основные файлы:

```text
train_keras.py
train_keras_v7.py
trading_keras_core.py
model_layers.py
```

Архитектуры:

- TCN + Transformer;
- PatchTST-style;
- TSMixer;
- iTransformer;
- multi-task;
- multi-horizon;
- quantile/regression heads;
- triple-barrier heads;
- volatility-filter heads.

### 9.3. iTransformer

iTransformer стал одной из главных сильных архитектур проекта.

Файлы:

```text
model_bench_itransformer.keras
best/model_15m_itransformer_h20.keras
best/BOEVOY_2300pct_model_bench_itransformer.keras
```

Почему iTransformer важен:

- хорошо работает с большим числом признаков;
- смотрит на признаки как на tokens;
- подходит для multivariate time series;
- лучше переносит сложные рыночные признаки, чем простая MLP/linear model;
- использовался в топовых backtest runs.

### 9.4. PatchTST

Файлы:

```text
model_bench_patchtst.keras
model_15m_patchtst.keras
model_15m_patchtst_h20.keras
```

PatchTST использовался как конкурент iTransformer:

- делит ряд на patches;
- работает с sequence windows;
- полезен для сравнения архитектур на одном датасете.

### 9.5. TSMixer

Файлы:

```text
model_bench_tsmixer.keras
norm_stats_bench_tsmixer.npz
```

TSMixer — более легкая архитектура для time series. Нужна была для benchmark:

- быстрее;
- меньше сложность;
- иногда лучше как baseline среди deep models.

### 9.6. V7 model family

На сайте видны модели:

- `MODEL V7 (ATR • 3x)`;
- `MODEL V7 (+39% • 3x)`;
- `MODEL V7 (+75% • 3x)`.

Это семейство сигналов и backtest-конфигов, которые используются в dashboard как разные профили риска/доходности.

В dashboard они соответствуют trade files:

```text
trades_conservative_canonical.csv
trades_aggressive_canonical.csv
trades_alpha75_canonical.csv
```

Смысл:

- conservative — более осторожный профиль;
- aggressive — больше сделок/риск;
- alpha75 — отдельный профиль с более жестким сигналом.

### 9.7. TG Hybrid / crowd layer

На сайте есть:

```text
TG Hybrid v3_2_0
```

Это ветка, где в backtest добавлялся crowd/news/sentiment слой:

- Telegram/crowd signals;
- confidence gate;
- soft/hybrid size multiplier;
- свежесть сигнала;
- влияние crowd-gate на входы.

Файлы отчетов:

```text
reports/telegram_crowd_impact_on_trades_meta_best_2025_sweep.csv
reports/backtest_tg_allin_lev3_hybrid_2025.csv
reports/backtest_tg_risk39_lev25_hybrid_chronos2_best_2025_v2.csv
```

### 9.8. MM Supervisor

На сайте:

```text
MM Supervisor r13
```

Это слой money/risk management:

- ограничение сделок;
- risk sizing;
- volatility regime;
- drawdown cap;
- фильтр риск/доходность;
- soft gating.

### 9.9. Chronos2 stage2

На сайте:

```text
Chronos2 (stage2)
```

Это экспериментальная ветка с Chronos-style прогнозом:

- длинный context;
- pred_len;
- multiple samples;
- gate entries;
- exit checks.

Отчеты:

```text
reports/chronos2_stage12_best_2025_v2.csv
reports/backtest_transfer_2021_2026ytd_chronos_lev5.csv
reports/trades_chronos2_2021_2026ytd.csv
```

Пример сильного результата из отчета:

```text
chronos2_stage12_best_2025_v2.csv
final_equity ≈ 175.72
trades = 63
max_dd ≈ -9.82%
```

Важно: такие результаты требуют осторожной проверки, потому что высокая доходность на backtest может быть следствием режима рынка/порогов/плеча.

### 9.10. Meta-combo model

Файлы:

```text
backtest_trade_combo_meta_dynamic_exit.py
reports/backtest_trade_combo_meta_best_calib.csv
reports/trades_meta_best_2023.csv
reports/trades_meta_best_2024.csv
reports/trades_meta_best_2025.csv
```

Meta-combo — это не одна модель, а слой, который объединяет сигналы нескольких моделей и решает:

- какой сигнал брать;
- когда входить;
- когда пропускать;
- какой risk fraction;
- какой exit;
- как учитывать ATR;
- как учитывать crowd/news/chronos gate.

Пример из отчета:

```text
reports/backtest_trade_combo_meta_best_calib.csv
final_equity ≈ 113.54
trades = 34
max_dd ≈ -2.36%
```

---

## 10. Лучшие результаты и как их правильно объяснять

В проекте есть несколько красивых backtest-результатов.

### 10.1. Aggressive iTransformer research run

Файл:

```text
best/BOEVOY_2300pct_report.csv
```

Метрики:

```text
model = itransformer_moretrades_ddreset
mode = long_short
threshold = 0.64
short_threshold = 0.36
total_return ≈ 23.77
CAGR ≈ 27.32
Sharpe ≈ 5.79
MaxDD ≈ -65.14%
trades = 132
```

Как объяснять:

- это aggressive research-run;
- доходность очень высокая;
- риск тоже высокий;
- нужен как демонстрация потенциала модели;
- не стоит продавать это как стабильную гарантию.

### 10.2. Moderate iTransformer walk-forward

Файл:

```text
best/wf_itransformer_moderate_plus.csv
```

Метрики:

```text
model = itransformer_moderate_plus
total_return ≈ 0.91
CAGR ≈ 0.97
Sharpe ≈ 3.56
MaxDD ≈ -32.12%
trades = 30
```

Как объяснять:

- это более спокойный вариант;
- меньше сделок;
- ниже доходность;
- ниже агрессивность;
- лучше подходит для защиты как "осторожный режим".

### 10.3. Meta-combo / Chronos / TG hybrid

Проект также проверял:

- meta-combo;
- Chronos2;
- Telegram/crowd hybrid;
- ATR exits;
- leverage sweeps.

Эти результаты важны не как финальный ответ, а как доказательство, что была настоящая исследовательская работа: мы не взяли одну модель и не остановились, а проверяли разные гипотезы.

---

## 11. Как сигнал превращается в сделку

Raw model output обычно выглядит как вероятность или предсказанный return. Но на сайте пользователю показывается не это, а понятный сигнал:

```text
LONG
SHORT
FLAT
```

Переход от модели к сделке:

1. Модель считает вероятность/return.
2. Применяется threshold.
3. Проверяется side:
   - long threshold;
   - short threshold.
4. Проверяется volatility/risk gate.
5. Проверяется cooldown/min hold.
6. Проверяется max trades per day.
7. Учитываются fee/slippage.
8. Рассчитывается stop/invalidation.
9. Сигнал попадает в dashboard.

Для пользователя это отображается как:

```text
Status: LL / SS / FLAT
Confidence: 67%
Leverage: 1x / 3x
Risk/trade: 0.5%
Invalidation: price stop
Why: agreement, volatility, news
```

---

## 12. История сделок на графике

Dashboard показывает не только текущий сигнал, но и где модель входила/выходила.

Файлы trades:

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

В dashboard есть mapping:

```text
TRADE_FILES_PREFERRED
```

Логика:

- выбираем модель;
- dashboard выбирает нужный CSV;
- грузит сделки;
- фильтрует по visible range;
- строит markers на свечах;
- список сделок показывает entry/exit/status/model.

На графике markers помогают объяснить, что модель не просто "говорит сейчас", а имеет историю решений.

---

## 13. Как строится график

Frontend:

```text
html/dashboard.html
html/lightweight-charts.standalone.production.js
```

Шаги:

1. Загружается Lightweight Charts.
2. Создается candlestick series.
3. Dashboard вызывает API `/candles`.
4. Свечи нормализуются:

```text
time
open
high
low
close
```

5. Данные кладутся в chart.
6. Сверху накладываются:

- price line;
- signal markers;
- trade markers;
- current signal label;
- probability block.

7. При смене таймфрейма:

- меняется interval;
- старый запрос отменяется/игнорируется;
- загружается quick history;
- потом deep history;
- markers пересчитываются.

8. На mobile:

- верхняя панель сжимается;
- нижнее app-nav меню;
- график занимает больше экрана;
- карточки и side panel становятся компактнее.

---

## 14. Аккаунты, подписка и поддержка

В проекте есть полноценный account flow.

### 14.1. Регистрация

Пользователь:

1. вводит email/password;
2. получает код;
3. подтверждает email;
4. получает trial;
5. может открыть dashboard.

### 14.2. Вход

Поддерживается:

- email/password;
- Yandex OAuth;
- session token;
- cookie bridge.

### 14.3. Account modal

В `Account` показывается:

- email;
- support code;
- plan;
- billing period;
- paid until;
- latest payment;
- txid;
- ФИО;
- Telegram/contact.

Это нужно для поддержки. Например, если клиент оплатил, но доступ не открылся, можно найти его по email/support code/txid.

### 14.4. Support chat

На сайт был добавлен Tawk.to chat.

Он нужен для:

- быстрых вопросов;
- поддержки trial;
- вопросов по оплате;
- ощущения живого продукта.

### 14.5. Analytics

Добавлена Yandex Metrika:

```text
counter id: 108276909
```

Используется для:

- отслеживания посещений;
- просмотра поведения;
- анализа конверсии;
- проверки, где пользователи кликают.

---

## 15. Инфраструктура и компьютеры

В проекте участвовало несколько машин.

### 15.1. Основной ПК

Имя:

```text
vitamind-b650m-k
```

Роль:

- основная разработка;
- Codex workspace;
- запуск локальных проверок;
- работа с GitHub;
- подготовка notebook/zip;
- иногда обучение/эксперименты на GPU;
- доступ к другим машинам по Tailscale/LAN.

### 15.2. ms-7972

Имя:

```text
vitamind-ms-7972
```

Роль:

- физический сервер;
- использовался в pipeline;
- хранение/перенос данных;
- дополнительные процессы;
- сейчас доступ настроен через LAN alias.

Подключение:

```bash
ssh ms-7972
```

Alias в SSH config ведет на LAN IP:

```text
192.168.0.189
```

Почему так:

- Tailscale node был виден;
- Tailscale ping работал;
- но TCP на `100.75.50.55:22` давал `Connection refused`;
- по LAN `192.168.0.189:22` SSH работал;
- Mullvad на основном ПК выключать нельзя, поэтому оставили LAN alias.

### 15.3. helpd3

Имя:

```text
helpd3-to-be-filled-by-o-e-m
```

Роль:

- сервер для сайта/экспериментов;
- запуск учебных веб-заданий;
- WSGI/ASGI homework;
- иногда backend/site processes.

### 15.4. Tailscale

Использовался для:

- доступа к физическим машинам;
- проверки online/offline;
- SSH;
- диагностики сети.

Проверки:

```bash
tailscale status
tailscale ping <ip>
```

### 15.5. Mullvad

На основном ПК включен Mullvad, его нельзя выключать.

Было важно включить:

```bash
mullvad lan set allow
```

Чтобы VPN не ломал локальную сеть/Tailscale LAN path.

---

## 16. Деплой сайта

Есть два сценария.

### 16.1. Локальный запуск без Docker

Скрипт:

```text
scripts/start_site.sh
```

Он запускает:

1. static web server:

```text
python -m http.server 8088 --directory html
```

2. FastAPI:

```text
serve_fastapi.py --port 8000
```

3. Cloudflared tunnel для web;
4. Cloudflared tunnel для API;
5. live update loop для новостей.

### 16.2. Docker Compose

Файл:

```text
docker-compose.site.yml
```

Сервисы:

```text
api
web
```

`api`:

- FastAPI;
- TensorFlow;
- tf_keras;
- pandas/pyarrow;
- transformers;
- torch CPU;
- модель и features volume.

`web`:

- python static server;
- отдает `html/`.

### 16.3. Домены

Frontend:

```text
https://tradeforge.art
https://www.tradeforge.art
```

API:

```text
https://api.tradeforge.art
```

CORS:

```text
ALLOW_ORIGINS=https://tradeforge.art,https://www.tradeforge.art,...
```

---

## 17. GitHub и публичная версия

Для GitHub была создана отдельная безопасная копия:

```text
/home/vitamind/my_project/tradeforge-public
```

Почему не пушили весь `model6`:

- там 11+ GB;
- есть `.env`;
- есть `.api_key`;
- есть базы пользователей;
- есть тяжелые модели;
- есть логи;
- есть research мусор;
- есть приватные файлы.

В публичную копию вошло:

- FastAPI;
- frontend;
- Docker files;
- scripts;
- README;
- `.env.example`;
- legal/payment pages;
- без секретов.

GitHub repo:

```text
https://github.com/sxtr333/TRADING_AI_2X_PER_YEAR
```

---

## 18. Пакет для сдачи проекта

Папка:

```text
project_submission_trade_model/
```

Финальный архив:

```text
/home/vitamind/my_project/model6/project_submission_trade_model/trade_model_submission_FINAL.zip
```

Внутри:

```text
01_final_ready_model.ipynb
02_experiments_model_building.ipynb
03_tradeforge_top_model_backtest.ipynb
btcusdt_15m_sample.csv
trade_signal_model.joblib
top_tradeforge_model/
```

Смысл:

- `01` — простой финальный notebook для проверки готовой модели;
- `02` — эксперименты и сравнение моделей;
- `03` — сильная часть для защиты: iTransformer, backtest, equity, drawdown;
- `top_tradeforge_model/` — реальные `.keras` модели и отчеты.

---

## 19. Что показывать на презентации

Рекомендуемый порядок:

### Слайд 1. Что такое TradeForge

TradeForge — платформа для анализа BTCUSDT и крипторынка: модели, dashboard, сигналы, история сделок, market radar, pump scout, подписка.

### Слайд 2. Проблема

Рынок шумный. Обычный прогноз цены плохо работает. Нужна система, которая объединяет данные, модели, риск-фильтры и backtest.

### Слайд 3. Данные

Показать:

- OHLCV;
- OI/funding;
- liquidations;
- volatility;
- TA;
- news/sentiment;
- macro/risk.

### Слайд 4. Pipeline

```text
raw data -> features -> model -> signal -> risk filter -> backtest -> API -> website
```

### Слайд 5. Модели

Показать семейства:

- Logistic/RF/Boosting baseline;
- PatchTST;
- TSMixer;
- iTransformer;
- V7;
- TG Hybrid;
- MM Supervisor;
- Chronos2;
- Meta-combo.

### Слайд 6. Лучшие результаты

Показать:

- aggressive iTransformer;
- moderate walk-forward;
- drawdown;
- monthly returns.

Говорить аккуратно:

> "Это не гарантия будущей прибыли. Это результат исторического теста и демонстрация потенциала pipeline."

### Слайд 7. Сайт

Показать:

- landing;
- account;
- dashboard;
- chart;
- signal panel;
- trade markers;
- Market Radar;
- Pump Scout.

### Слайд 8. Backend

Показать:

- FastAPI;
- `/candles`;
- `/dashboard/bootstrap`;
- `/auth`;
- `/subscription`;
- `/billing`;
- `/market/top`.

### Слайд 9. Инфраструктура

Показать:

- основной ПК;
- ms-7972;
- helpd3;
- Tailscale;
- Docker;
- Cloudflare tunnel;
- GitHub/public package.

### Слайд 10. Итог

Мы пришли от простой модели BTC к полноценной платформе:

- данные;
- модели;
- backtest;
- сайт;
- аккаунты;
- подписка;
- мониторинг;
- поддержка;
- презентационный пакет.

---

## 20. Что говорить, если спросят про точность

Короткий честный ответ:

> На коротком горизонте BTC почти не дает стабильной высокой accuracy. Поэтому я смотрел не только accuracy, а результат торговой логики: пороги, комиссии, просадка, количество сделок и walk-forward. Простая модель нужна для воспроизводимого примера, а боевой результат показывает iTransformer + risk layer.

---

## 21. Что говорить, если спросят про доходность

Ответ:

> В aggressive research-run была высокая доходность, но там выше риск и просадка. Поэтому я отдельно показываю moderate walk-forward: он спокойнее и честнее для оценки. В реальном продукте я бы давал пользователю не обещание прибыли, а сигнал, риск, историю сделок и предупреждение, что прошлый результат не гарантирует будущий.

---

## 22. Что говорить, если спросят, почему столько моделей

Ответ:

> Потому что на рынке нет одной вечной модели. Разные режимы рынка требуют разных фильтров. Поэтому в TradeForge есть несколько профилей: conservative, aggressive, alpha, TG hybrid, Chronos, meta-combo. Сайт позволяет сравнить их и видеть, как каждая вела себя на истории.

---

## 23. Текущие сильные стороны проекта

- Есть реальный сайт, а не только notebook.
- Есть backend API.
- Есть auth/subscription/payment flow.
- Есть график и визуализация сделок.
- Есть несколько моделей.
- Есть backtest reports.
- Есть market radar и pump scout.
- Есть support/analytics.
- Есть публичный GitHub package.
- Есть zip для сдачи проекта.
- Есть понятная история развития.

---

## 24. Ограничения проекта

Важно знать и честно говорить:

- рынок BTC шумный;
- backtest может переоценивать результат;
- новости могут ломать стабильность, если поток меняется;
- высокий leverage сильно увеличивает риск;
- aggressive-модели нельзя выдавать за гарантированную стратегию;
- нужна постоянная walk-forward проверка;
- реальные комиссии/проскальзывание могут отличаться;
- prod-сервис требует мониторинга и стабильных серверов.

---

## 25. Что можно улучшить дальше

### 25.1. Модели

- walk-forward по месяцам;
- purged CV;
- regime classifier;
- отдельный WAIT-class model;
- calibration;
- ensemble voting;
- meta-labeling;
- better execution model.

### 25.2. Данные

- более стабильный news pipeline;
- order book snapshots;
- liquidation clusters;
- funding anomalies;
- on-chain data;
- ETF/stablecoin flows;
- macro calendar.

### 25.3. Сайт

- admin panel;
- user payments export;
- better support tools;
- strategy comparison page;
- model health page;
- live incidents;
- mobile polish.

### 25.4. Инфраструктура

- production VPS;
- systemd services;
- Docker registry;
- backups;
- DB migrations;
- monitoring;
- alerting;
- secrets manager.

---

## 26. Быстрые команды

### Запуск сайта локально

```bash
cd /home/vitamind/my_project/model6
./scripts/start_site.sh
```

### Docker

```bash
cd /home/vitamind/my_project/model6
docker compose -f docker-compose.site.yml up -d --build
```

### Проверка API

```bash
curl https://api.tradeforge.art/health
```

### Подключение к ms-7972

```bash
ssh ms-7972
```

### Пакет сдачи проекта

```text
/home/vitamind/my_project/model6/project_submission_trade_model/trade_model_submission_FINAL.zip
```

### Публичная GitHub-копия

```text
/home/vitamind/my_project/tradeforge-public
```

---

## 27. Главная формулировка проекта

TradeForge — это не просто модель для BTC. Это полный путь от исследовательских экспериментов до работающего сайта:

```text
данные -> признаки -> модели -> бэктест -> сигналы -> API -> dashboard -> подписка -> пользователь
```

Именно это делает проект сильным для защиты: здесь есть и ML-часть, и продукт, и backend, и frontend, и инфраструктура, и понимание рисков.

