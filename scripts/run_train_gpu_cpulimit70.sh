#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_train_gpu_cpulimit70.sh [--docker IMAGE] -- <command...>

Examples:
  scripts/run_train_gpu_cpulimit70.sh -- bash scripts/train_news_xlmr_cosmic_v6_dr_scale_8nodes.sh
  scripts/run_train_gpu_cpulimit70.sh --docker tensorflow/tensorflow:nightly-gpu -- \
    bash /work/scripts/train_news_xlmr_cosmic_v6_dr_scale_8nodes.sh
USAGE
}

MODE="host"
IMAGE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --docker)
      MODE="docker"
      IMAGE="${2:-}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

CPU_TOTAL="$(nproc)"
CPU_LIMIT="$(( (CPU_TOTAL * 70 + 99) / 100 ))"
if [[ "${CPU_LIMIT}" -lt 1 ]]; then
  CPU_LIMIT=1
fi

export OMP_NUM_THREADS="${CPU_LIMIT}"
export TF_NUM_INTRAOP_THREADS="${CPU_LIMIT}"
export TF_NUM_INTEROP_THREADS="$((CPU_LIMIT/4))"
if [[ "${TF_NUM_INTEROP_THREADS}" -lt 1 ]]; then
  export TF_NUM_INTEROP_THREADS=1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="python3"
if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
fi

gpu_preflight() {
  "${PYTHON_BIN}" - <<'PY'
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", gpus)
raise SystemExit(0 if gpus else 2)
PY
}

if [[ "${MODE}" == "host" ]]; then
  echo "[preflight] GPU check (host)"
  if ! gpu_preflight; then
    echo "ERROR: TensorFlow does not see a GPU. Aborting to avoid CPU training."
    exit 2
  fi
  echo "[train] CPU limit ~70%: OMP_NUM_THREADS=${OMP_NUM_THREADS}, TF_NUM_INTRAOP_THREADS=${TF_NUM_INTRAOP_THREADS}, TF_NUM_INTEROP_THREADS=${TF_NUM_INTEROP_THREADS}"
  exec "$@"
fi

if [[ -z "${IMAGE}" ]]; then
  echo "ERROR: --docker requires an IMAGE argument."
  exit 2
fi

CPUSET="0-$((CPU_LIMIT-1))"

DOCKER_MOUNTS=(
  -v "${ROOT_DIR}:/work"
)
if [[ -d /mnt/data ]]; then
  DOCKER_MOUNTS+=(-v /mnt/data:/mnt/data)
fi
if [[ -d "${HOME}/.cache/huggingface" ]]; then
  DOCKER_MOUNTS+=(-v "${HOME}/.cache/huggingface:/root/.cache/huggingface")
fi

echo "[preflight] GPU check (docker)"
docker run --rm --runtime=nvidia --gpus all \
  --cpus="${CPU_LIMIT}" --cpuset-cpus="${CPUSET}" \
  --network=host --shm-size=2g \
  "${DOCKER_MOUNTS[@]}" \
  -e OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
  -e TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS}" \
  -e TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS}" \
  -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  "${IMAGE}" bash -lc '
    python3 - <<'"'"'PY'"'"'
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", gpus)
raise SystemExit(0 if gpus else 2)
PY
  ' || {
    echo "ERROR: TensorFlow does not see a GPU in the container. Aborting to avoid CPU training."
    exit 2
  }

echo "[train] CPU limit ~70%: OMP_NUM_THREADS=${OMP_NUM_THREADS}, TF_NUM_INTRAOP_THREADS=${TF_NUM_INTRAOP_THREADS}, TF_NUM_INTEROP_THREADS=${TF_NUM_INTEROP_THREADS}"
exec docker run --rm --runtime=nvidia --gpus all \
  --cpus="${CPU_LIMIT}" --cpuset-cpus="${CPUSET}" \
  --network=host --shm-size=2g \
  "${DOCKER_MOUNTS[@]}" \
  -e OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
  -e TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS}" \
  -e TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS}" \
  -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  "${IMAGE}" bash -lc "$(printf '%q ' "$@")"
