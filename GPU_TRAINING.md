# GPU Training (model6) — Canonical Runbook

This repo must **never** train on CPU silently. Always use a GPU preflight and stop if TF cannot see a GPU.

## Host run (recommended for local training)

Use the launcher that enforces the CPU limit (~70%) and GPU preflight:

```bash
scripts/run_train_gpu_cpulimit70.sh -- bash /home/vitamind/my_project/model6/scripts/train_news_xlmr_cosmic_v6_dr_scale_8nodes.sh
```

What it does:
- Sets CPU thread limits (OMP / TF) to ~70% of `nproc`.
- Forces `CUDA_VISIBLE_DEVICES=0` unless you override it.
- Runs a **GPU preflight** (TensorFlow must see a GPU, otherwise exit != 0).
- If GPU is missing, it **aborts** (no silent CPU training).

If you need a custom command:

```bash
scripts/run_train_gpu_cpulimit70.sh -- .venv/bin/python train_keras_v7.py --help
```

## Canonical Docker command (works with RTX 5070)

Use `tensorflow/tensorflow:nightly-gpu` and install `tf_keras` + `nvidia-cudnn-cu12` inside the container. This is the **known‑good** recipe that successfully detects the GPU and starts training.

```bash
CPU_TOTAL=$(nproc)
CPU_LIMIT=$(( (CPU_TOTAL*70+99)/100 ))
CPUSET="0-$((CPU_LIMIT-1))"

docker run --rm --runtime=nvidia --gpus all \
  --cpus=${CPU_LIMIT} --cpuset-cpus=${CPUSET} \
  --network=host --shm-size=2g \
  -v /home/vitamind/my_project/model6:/work \
  -v /mnt/data:/mnt/data \
  -v /mnt/oldssd:/mnt/oldssd \
  -v /home/vitamind/.cache/huggingface:/root/.cache/huggingface \
  -e TF_USE_LEGACY_KERAS=1 \
  tensorflow/tensorflow:nightly-gpu bash -lc '
    pip install -q tf_keras pandas pyarrow numpy nvidia-cudnn-cu12

    CUDNN_PATH=$(python3 - <<'"'"'PY'"'"'
import site, glob, os
paths=[]
for sp in site.getsitepackages():
    paths += glob.glob(os.path.join(sp,"nvidia","cudnn","lib*"))
print(paths[0] if paths else "")
PY
)
    export LD_LIBRARY_PATH="$CUDNN_PATH:$LD_LIBRARY_PATH"

    # GPU preflight: abort if GPU not visible
    python3 - <<'"'"'PY'"'"'
import tensorflow as tf
print("TF:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
raise SystemExit(0 if tf.config.list_physical_devices("GPU") else 2)
PY

    # Train script (host path: /home/vitamind/my_project/model6/...)
    bash /work/scripts/train_news_xlmr_cosmic_v6_dr_scale_8nodes.sh
  '
```

## Docker via launcher (same CPU/GPU safety)

```bash
scripts/run_train_gpu_cpulimit70.sh --docker tensorflow/tensorflow:nightly-gpu -- \
  bash /work/scripts/train_news_xlmr_cosmic_v6_dr_scale_8nodes.sh
```

## Train script paths

- Host path: `/home/vitamind/my_project/model6/scripts/train_news_xlmr_cosmic_v6_dr_scale_8nodes.sh`
- Container path (mounted at `/work`): `/work/scripts/train_news_xlmr_cosmic_v6_dr_scale_8nodes.sh`

## Requirements / notes

- `new_models` is a symlink to `/mnt/oldssd/...`, so **must** mount `/mnt/oldssd` into the container.
- The GPU preflight must **exit non‑zero** if TF does not see a GPU.
- Warning about PTX JIT for CC 12.0 is expected on nightly TF builds; first kernel compile can be slow.
- If GPU is not detected inside the container, do **not** proceed with training.
- RTX 5070 (CC 12.0a) needs a modern CUDA toolchain; older CUDA can cause long JIT or failures.

## CPU limit template (70%)

This is the standard formula used everywhere:

```bash
CPU_TOTAL=$(nproc)
CPU_LIMIT=$(( (CPU_TOTAL*70+99)/100 ))   # ceil(0.7 * nproc)
export OMP_NUM_THREADS=${CPU_LIMIT}
export TF_NUM_INTRAOP_THREADS=${CPU_LIMIT}
export TF_NUM_INTEROP_THREADS=$((CPU_LIMIT/4))
if [ ${TF_NUM_INTEROP_THREADS} -lt 1 ]; then export TF_NUM_INTEROP_THREADS=1; fi
```

## GPU preflight (required)

```bash
python3 - <<'PY'
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", gpus)
raise SystemExit(0 if gpus else 2)
PY
```
