# ТЗ для нового Codex (model6)

Дата: 2026-02-15
Корневая директория проекта: `/home/vitamind/my_project/model6`

## 1) Где работать
Рабочая директория всегда:
- `/home/vitamind/my_project/model6`

Перед началом:
```bash
cd /home/vitamind/my_project/model6
pwd
```

## 2) Главный файл-описание проекта (прочитать первым)
- `docs/codex_handoff/MASTER_HANDOFF_2026-02-14.md`

Это главный источник контекста: инфраструктура, роли ПК, датасеты, модели, обучение, деплой, аварийные сценарии.

## 3) Дополнительные обязательные файлы (после MASTER)
Читать в таком порядке:
1. `GPU_TRAINING.md` — каноничный запуск обучения на GPU (без silent CPU fallback)
2. `docs/codex_handoff/PROJECT_MAP.md` — карта проекта и пайплайнов
3. `docs/codex_handoff/MODELS_AND_DATASETS.md` — актуальные модели/датасеты
4. `docs/codex_handoff/CODEX_CHEATSHEET.md` — быстрые команды
5. `ARCHITECTURE.md` и `CONFIG_REFERENCE.md` — архитектура и параметры

## 4) Критичные директории и зависимости
В проекте есть symlink'и, это нормально:
- `data -> /mnt/oldssd/model6/data`
- `models -> /mnt/oldssd/model6/models`
- `new_models -> /mnt/oldssd/model6/new_models`
- `logs -> /mnt/oldssd/model6/logs`
- `.venv -> /mnt/oldssd/model6/.venv`

Если `/mnt/oldssd` не смонтирован, будет выглядеть как "пропали файлы".

Проверка:
```bash
mount | grep /mnt/oldssd
ls -ld data models new_models logs .venv
readlink -f data models new_models logs .venv
```

## 5) Где лежит что
- Код обучения: `train_keras.py`, `train_keras_v7.py`, `scripts/`
- API: `serve_fastapi.py`
- Фичи/сборка данных: `build_features.py`, `scripts/live_update_once.sh`, `scripts/live_update_loop.sh`
- Веб: `html/`
- Docker site stack: `docker-compose.site.yml`
- Отчеты/бэктесты: `reports/`

## 6) Базовый рабочий регламент (A-Z)
1. Открыть проект:
```bash
cd /home/vitamind/my_project/model6
```
2. Прочитать `MASTER_HANDOFF_2026-02-14.md`.
3. Проверить mount/symlink и GPU:
```bash
mount | grep /mnt/oldssd
nvidia-smi
```
4. Проверить локальные сервисы (если нужна отладка сайта/API):
```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8088/ | head -n 1
```
5. Если нужно обучение — запускать только через GPU-safe launcher:
```bash
scripts/run_train_gpu_cpulimit70.sh -- bash /home/vitamind/my_project/model6/scripts/train_news_xlmr_cosmic_v6_dr_scale_8nodes.sh
```
6. Если нужен сайт через docker:
```bash
docker compose -f docker-compose.site.yml up -d api web
```
7. Если нужен realtime update:
```bash
LIVE_UPDATE_INTERVAL_SEC=60 bash /home/vitamind/my_project/model6/scripts/live_update_loop.sh
```

## 7) Нельзя нарушать
- Нельзя обучать без GPU preflight.
- Нельзя удалять данные/модели в `/mnt/oldssd/model6/*` без явного запроса.
- Нельзя смешивать несоответствующие `model + norm_stats + features`.
- Нельзя делать выводы о "пропаже" файлов без проверки mount.

## 8) Где смотреть деплой домена
- Cloudflared config: `/etc/cloudflared/config.yml`
- Systemd unit status/logs:
```bash
sudo systemctl status cloudflared --no-pager
sudo journalctl -u cloudflared -n 100 --no-pager
```
- Публичные health-check:
```bash
curl -I https://tradeforge.art
curl -s https://api.tradeforge.art/health
```

## 9) Цель нового Codex в этом проекте
- Стабильно поддерживать цикл: данные -> фичи -> модели -> бэктест -> API -> сайт
- Не терять воспроизводимость и совместимость артефактов
- Фиксировать важные изменения в `docs/codex_handoff/MASTER_HANDOFF_2026-02-14.md`

