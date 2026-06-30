# MASTER HANDOFF (model6) — FULL OPS CONTEXT

Last updated: 2026-02-15 (MSK)
Repo: `/home/vitamind/my_project/model6`

## 1) Purpose
This file is the single operational handoff so a new Codex can continue work immediately without losing progress.
Scope:
- project architecture and key files,
- machine roles,
- storage and symlink dependencies,
- canonical datasets/models,
- strict training policy (GPU-only + CPU cap),
- serve/deploy procedures,
- known incidents and proven fixes,
- do/don't safety rules.

If anything conflicts with assumptions, trust this file + live checks from Section 18.

---

## 2) Project mission
`model6` is a BTCUSDT 15m trading/research stack:
- OHLCV + news ingestion,
- feature engineering,
- multi-horizon forecasting models,
- meta-label models (trade/no-trade filters),
- backtests and threshold sweeps,
- live API + website visualization.

Operational goal right now:
1. keep site/API stable,
2. keep model training reproducible and GPU-only,
3. iterate model quality without leakage/regressions.

---

## 3) Machine matrix and roles
Tailscale matrix (historical active nodes):
- `vitamind-b650m-k` (`100.64.204.81`) — main workstation, primary training and development.
- `bookvitamind-modern-14-b11mou` (`100.84.25.48`) — notebook, frequently used for public domain/tunnel runs.
- `helpd3-to-be-filled-by-o-e-m` (`100.77.192.122`) — auxiliary node for migration/ops/possible serving.
- `vitamind-ms-7972` (`100.75.50.55`) — auxiliary desktop (checks, file transfer, support tasks).

Recommended division:
- Main R&D and model training: `vitamind-b650m-k`.
- Public tunnel/domain experiments: notebook.
- Heavy copy/backup and side diagnostics: `helpd3`/`ms-7972`.

---

## 4) Hardware constraints that matter
### Main workstation (`vitamind-b650m-k`)
- CPU: Ryzen 7 7800X3D
- GPU: RTX 5070
- RAM target now tested at 64GB (4x16) on reduced DDR5 clocks (stability tuning done manually in BIOS)
- Use this host for training.

### `helpd3`
- CPU: i7-4790
- RAM: 16GB
- Disk: ST1000LM014 SSHD
- GPU path discussed: GTX 1050 Ti / GTX 1060 (for inference/light workloads)
- Good for serving/support tasks, not preferred for major training.

Operational conclusion:
- Production-quality training belongs on RTX 5070 host.
- `helpd3` can run API/site/light inference if configured.

---

## 5) Storage map (critical)
Main path: `/home/vitamind/my_project/model6`

Critical symlinks in repo root:
- `data -> /mnt/oldssd/model6/data`
- `models -> /mnt/oldssd/model6/models`
- `new_models -> /mnt/oldssd/model6/new_models`
- `logs -> /mnt/oldssd/model6/logs`
- `.venv -> /mnt/oldssd/model6/.venv`

Observed sizes:
- `/mnt/oldssd/model6/data` ~ `1.7G`
- `/mnt/oldssd/model6/models` ~ `3.4G`
- `/mnt/oldssd/model6/new_models` ~ `1.8G`

Non-obvious failure mode:
- If `/mnt/oldssd` is not mounted, repo appears "missing files/models" due to broken symlink targets.

Mandatory check before any training/serving:
```bash
mount | grep /mnt/oldssd
cd /home/vitamind/my_project/model6
ls -ld data models new_models logs .venv
readlink -f data models new_models
```

---

## 6) Repo structure (what matters)
Core docs:
- `GPU_TRAINING.md` (canonical GPU training runbook)
- `ARCHITECTURE.md`
- `CONFIG_REFERENCE.md`
- `PROJECT_CONTEXT.md`
- `docs/codex_handoff/PROJECT_MAP.md`
- `docs/codex_handoff/MODELS_AND_DATASETS.md`
- `docs/codex_handoff/CODEX_CHEATSHEET.md`

Core code and scripts:
- Training: `train_keras.py`, `train_keras_v7.py`
- Serving: `serve_fastapi.py`
- Feature pipeline: `build_features.py`
- Live updates: `scripts/live_update_once.sh`, `scripts/live_update_loop.sh`
- Start local stack: `scripts/run_servers.sh`
- Docker site stack: `docker-compose.site.yml`
- Frontend: `html/` (main working page historically `html/aladin_from_image.html`)

---

## 7) Datasets (canonical)
### Key market/features data
- `data/BTCUSDT_15m.parquet` (base OHLCV 15m)
- `data/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes_with_news_pruned.parquet` (main train/backtest features)
- `data/BTCUSDT_15m_features_h20_v2_news_xlmr_v4_8nodes_serve.parquet` (main serve features)
- `data/BTCUSDT_15m_features_h20_v2_news_xlmr_v3.parquet` and `..._v3_serve.parquet` (legacy line)
- `data/meta/meta_dataset_pruned.parquet` (meta labels dataset)

### Additional variants present
- full/yrnorm/metaalign variants,
- `long_short` and `long_short_mix15` subsets,
- `BTCUSDT_1h_aux_*` datasets.

### News datasets (currently under `/mnt/data/news`)
Examples present:
- `news_raw.parquet`
- `news.parquet`
- `ccnews_all_sentiment.parquet`
- `ccnews_2023_2026_all_sentiment.parquet`
- cache/log/pid files for news sentiment pipelines.

Rule:
- For inference/backtests, always align `features + model + norm_stats` from same feature schema/run family.

---

## 8) Model inventory and pairing rule
Root snapshot contains many models (`*.keras`) and normalization files (`norm_stats*.npz`).
Also many run-specific outputs in `new_models/`.

Frequently referenced models:
- `model_battle_itransformer.keras`
- `model_15m_itransformer_tb_multi.keras`
- multiple `model_15m_itransformer_price_h*` variants
- legacy and meta families under `new_models/*`

Hard rule (non-negotiable):
- Never mix model and stats from unrelated runs.
- If model path changes, matching `norm_stats_*.npz` must change with it.

---

## 9) Strict training policy (mandatory)
### 9.1 Never allow silent CPU training
Use GPU preflight and fail hard if no GPU visible.

### 9.2 CPU cap policy
Run training with ~70% CPU threads for system stability.

### 9.3 Canonical host launcher
```bash
cd /home/vitamind/my_project/model6
scripts/run_train_gpu_cpulimit70.sh -- bash /home/vitamind/my_project/model6/scripts/train_news_xlmr_cosmic_v6_dr_scale_8nodes.sh
```

### 9.4 Canonical Docker training
Documented in `GPU_TRAINING.md` with known-good image:
- `tensorflow/tensorflow:nightly-gpu`
- `tf_keras` + `nvidia-cudnn-cu12`
- explicit GPU preflight
- mounts `/mnt/data` and `/mnt/oldssd`

### 9.5 Required GPU preflight snippet
```python
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", gpus)
raise SystemExit(0 if gpus else 2)
```

---

## 10) Leakage and data integrity guardrails
Never include target-like columns in price-model features:
- `meta_y_*`
- `meta_label_*`
- any future/shifted labels not available at inference time.

Before training:
- validate timestamps monotonicity,
- validate no NaN explosions,
- verify features used by model exactly match stats file schema.

---

## 11) Serving architecture (two common modes)
### Mode A: Docker site stack (`docker-compose.site.yml`)
Current file maps:
- API on `8000`
- web static server on `8088`

Run:
```bash
cd /home/vitamind/my_project/model6
docker compose -f docker-compose.site.yml up -d api web
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8088/ | head -n 1
```

### Mode B: Script-based dual API stack (`scripts/run_servers.sh`)
Starts:
- v6 API on `8000`
- v5 API on `8001`
- web on `8080`
- optional arbitration services (8090/8091)

Use when testing both model lines concurrently.

---

## 12) API endpoints and CORS
`serve_fastapi.py` endpoints include:
- `GET /health`
- `POST /predict`, `POST /predict_batch`
- `GET /forecast`, `GET /forecast_multi`, `GET /forecast.csv`
- `GET /candles`
- `GET /trades`
- `GET /news`, `POST /news_refresh`, `GET /news_agg`
- auth endpoints (`/auth/register`, `/auth/login`)

CORS is enabled via `CORSMiddleware`.
Default allowed origins include:
- `http://localhost:8080`
- `http://127.0.0.1:8080`
- `http://localhost:8090`
- `http://127.0.0.1:8090`
- `http://localhost:8091`
- `http://127.0.0.1:8091`
- `https://tradeforge.art`
- `https://www.tradeforge.art`

If browser shows CORS errors, verify `ALLOW_ORIGINS` environment and API process actually restarted with desired config.

---

## 13) Domain and tunnel (`tradeforge.art`)
Public endpoints:
- `https://tradeforge.art`
- `https://api.tradeforge.art`

Tunnel details:
- managed via `cloudflared` systemd on host used for public serving.
- config file: `/etc/cloudflared/config.yml`

Common working ingress pattern:
```yaml
ingress:
  - hostname: tradeforge.art
    service: http://127.0.0.1:8088
  - hostname: api.tradeforge.art
    service: http://127.0.0.1:<API_PORT>
  - service: http_status:404
```

Important:
- `<API_PORT>` must match actual running API port on that host (8000 or 8001 depending on chosen mode).
- Historical intermittent 502/520/1033 happened when ingress port mismatched or origin was unstable.

Useful checks:
```bash
sudo systemctl status cloudflared --no-pager
sudo journalctl -u cloudflared -n 100 --no-pager
curl -I https://tradeforge.art
curl -s https://api.tradeforge.art/health
```

---

## 14) Known incidents and proven fixes
### 14.1 502/520/1033 from Cloudflare
Typical root causes:
- wrong ingress port,
- origin process dead/hanging,
- unstable network/VPN path.

Fix sequence:
1. verify local origin (`curl 127.0.0.1:<port>`),
2. verify `/etc/cloudflared/config.yml` port,
3. restart cloudflared,
4. re-check logs and external endpoint.

### 14.2 CORS blocking in browser
Symptom: preflight fails, no `Access-Control-Allow-Origin`.
Fix:
- ensure API is running code with CORS middleware,
- ensure origin in `ALLOW_ORIGINS`,
- restart API process/container.

### 14.3 Data/models suddenly "missing"
Usually not deleted: `/mnt/oldssd` not mounted.

### 14.4 NVIDIA driver mismatch / lagging desktop
Observed case: `nvidia-smi` failed with `Driver/library version mismatch` after package drift.
Recovery:
- `sudo dpkg --configure -a`
- `sudo apt --fix-broken install`
- reinstall coherent NVIDIA package set
- rebuild/sign module if needed (MOK enrollment)
- reboot and recheck `nvidia-smi`.

### 14.5 6TB Seagate (`ST6000NM0024`) disk issue
Repeated symptoms on hosts:
- 0B/malformed capacity,
- read capacity/inquiry failures.
Conclusion from prior attempts: likely hardware-level fault path; software formatting did not reliably recover.

---

## 15) Safe cleanup policy
Safe commands used without breaking active project data:
```bash
docker system prune -a -f
sudo journalctl --vacuum-size=200M
sudo apt clean
```

Can also remove old disabled Snap revisions carefully.

Do not:
- delete whole `/var`,
- remove `/mnt/oldssd/model6/*` blindly,
- purge datasets/models without verifying references.

---

## 16) Migration/copy policy between machines
When copying project across hosts (e.g., to `helpd3`):
- preserve symlinks vs real data intentionally,
- verify copy completeness by file count + checksum sample.

Recommended verification:
```bash
# counts
find /src/model6 -type f | wc -l
find /dst/model6 -type f | wc -l

# checksums (sample or full list)
cd /src/model6 && find . -type f -print0 | sort -z | xargs -0 sha256sum > /tmp/src.sha
cd /dst/model6 && find . -type f -print0 | sort -z | xargs -0 sha256sum > /tmp/dst.sha
diff -u /tmp/src.sha /tmp/dst.sha
```

If full checksum is too heavy, do targeted checks for:
- datasets (`data/*.parquet`, `data/meta/*.parquet`),
- active models + stats,
- scripts and serve files.

---

## 17) Command runbook (day-to-day)
### 17.1 Health preflight
```bash
nvidia-smi
mount | grep /mnt/oldssd
cd /home/vitamind/my_project/model6
```

### 17.2 Local API/web checks
```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8088/ | head -n 1
```

### 17.3 Start docker serving
```bash
docker compose -f docker-compose.site.yml up -d api web
docker compose -f docker-compose.site.yml ps
docker compose -f docker-compose.site.yml logs --no-color --tail=100 api
```

### 17.4 Start dual script serving
```bash
bash /home/vitamind/my_project/model6/scripts/run_servers.sh
```

### 17.5 Start realtime update loop
```bash
LIVE_UPDATE_INTERVAL_SEC=60 bash /home/vitamind/my_project/model6/scripts/live_update_loop.sh
```

### 17.6 Start GPU-safe training
```bash
scripts/run_train_gpu_cpulimit70.sh -- bash /home/vitamind/my_project/model6/scripts/train_news_xlmr_cosmic_v6_dr_scale_8nodes.sh
```

### 17.7 Public tunnel checks
```bash
sudo systemctl status cloudflared --no-pager
sudo journalctl -u cloudflared -n 100 --no-pager
curl -I https://tradeforge.art
curl -s https://api.tradeforge.art/health
```

---

## 18) Fast fact regeneration (no guessing)
Use these to refresh reality in 1-2 minutes:

```bash
# machine + time + tailscale
hostname
date
tailscale status

# critical symlinks
cd /home/vitamind/my_project/model6
ls -ld data models new_models logs .venv
readlink -f data models new_models logs .venv

# sizes
du -sh /mnt/oldssd/model6/data /mnt/oldssd/model6/models /mnt/oldssd/model6/new_models

# key data inventory
find /mnt/oldssd/model6/data -maxdepth 2 -type f \( -name '*.parquet' -o -name '*.csv' -o -name '*.json' \) | sort

# key model inventory
find /home/vitamind/my_project/model6 -maxdepth 1 -type f -name '*.keras' | sort
find /home/vitamind/my_project/model6 -maxdepth 1 -type f -name 'norm_stats*.npz' | sort
find /mnt/oldssd/model6/new_models -maxdepth 3 -type f \( -name '*.keras' -o -name '*.npz' -o -name '*.h5' \) | sort
```

---

## 19) Non-negotiable rules for the next Codex
1. Check mount/symlink integrity before declaring files missing.
2. Never run training without GPU preflight hard-fail.
3. Keep CPU cap (~70%) for long runs.
4. Treat model+stats+features as one immutable bundle.
5. Validate local service health before debugging Cloudflare.
6. Do not run destructive cleanup on dataset/model directories.
7. When changing ports, align compose/scripts/cloudflared/config/json simultaneously.

---

## 20) Priority next actions
1. Add persistent `/mnt/oldssd` mount (`/etc/fstab` by UUID) if not already persistent.
2. Add one unified preflight script (`gpu + mounts + required files + ports`) and call it from training/serving wrappers.
3. Add one smoke-test script for API (`/health`, `/candles`, `/forecast`, `/news`) + CORS preflight.
4. Keep this handoff updated after every major infra/training change.

