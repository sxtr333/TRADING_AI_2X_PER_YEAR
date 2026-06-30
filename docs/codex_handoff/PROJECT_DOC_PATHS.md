# Пути к файлам с описанием проекта

Ниже список ключевых документов и где они лежат.

## Основные документы

- `PROJECT_REPORT.txt`
  - Большой отчёт по проекту (устаревший, но полезный обзор).

- `README.md`
  - Короткое описание, базовые команды.

- `ARCHITECTURE.md`
  - Архитектура модели/обучения (train_*.py, слои, head’ы, RevIN/DropPath).

- `PIPELINE.md`
  - Описание пайплайна (train/val/test split, purge‑gap и т.д.).

- `CONFIG_REFERENCE.md`
  - Таблица CLI‑параметров для `build_features.py`, `train_keras.py`, `serve_fastapi.py`.

- `CODE_INDEX.md`
  - Индекс ключевых файлов и их назначение.

- `PROJECT_CONTEXT.md`
  - Короткий контекст проекта (обзор модулей).

## Новые материалы для Codex (handoff)

- `docs/codex_handoff/PROJECT_MAP.md`
  - Подробная карта проекта: что где лежит, откуда данные, как обновлять realtime.

- `docs/codex_handoff/SESSION_REPORT_2026-01-20.md`
  - Отчёт последней сессии: что сделано, какие результаты, что не повторять.

- `docs/codex_handoff/CODEX_CHEATSHEET.md`
  - Быстрые команды (realtime, запуск сайта, GPU‑train, backtests).

- `docs/codex_handoff/ONEPAGER_RU.md`
  - Короткая русская выжимка на 1 страницу.

- `docs/codex_handoff/MODELS_AND_DATASETS.md`
  - Список моделей и датасетов + фактические диапазоны дат.

## GPU и обучение

- `GPU_TRAINING.md`
  - Канонический runbook GPU‑обучения (host+docker, preflight, CPU limit).

- `scripts/run_train_gpu_cpulimit70.sh`
  - Универсальный launcher: GPU‑preflight + лимит CPU 70%.

## Отчёты и результаты

- `reports/`
  - Все CSV/PNG/лог‑отчёты (backtests, sweeps, results). Ключевые summary:
    - `reports/best_result_meta_2026-01-19.txt`
    - `reports/best_result_meta_boost_2026-01-19.txt`

## Где смотреть датасеты

- `data/` — основные фичи
- `data/meta/` — meta‑датасеты
- `/mnt/data/news/` — новости и sentiment

