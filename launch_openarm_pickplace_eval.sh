#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_LOG_ROOT="${REPO_ROOT}/logs/rsl_rl/openarm_bi_pick_place_cube"

TASK="${TASK:-Isaac-PickPlace-Cube-OpenArm-Bi-Play-v0}"
DEVICE="${DEVICE:-cuda:0}"
NUM_ENVS="${NUM_ENVS:-8}"
VIDEO_LENGTH="${VIDEO_LENGTH:-200}"
VISUALIZE_BODY="${VISUALIZE_BODY:-}"
VISUALIZE_BODY_RADIUS="${VISUALIZE_BODY_RADIUS:-0.025}"

RUN_DIR="${1:-${RUN_DIR:-}}"
CHECKPOINT_PATH="${2:-${CHECKPOINT:-}}"

if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR="$(find "${DEFAULT_LOG_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
fi

if [[ -z "${RUN_DIR}" || ! -d "${RUN_DIR}" ]]; then
  echo "[ERROR] Could not resolve a valid run directory." >&2
  exit 1
fi

RUN_DIR="$(realpath "${RUN_DIR}")"

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  CHECKPOINT_PATH="$(find "${RUN_DIR}" -maxdepth 1 -name 'model_*.pt' | sort -V | tail -n 1)"
fi

if [[ -z "${CHECKPOINT_PATH}" || ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "[ERROR] Could not resolve a valid checkpoint file." >&2
  exit 1
fi

CHECKPOINT_PATH="$(realpath "${CHECKPOINT_PATH}")"
CHECKPOINT_STEM="$(basename "${CHECKPOINT_PATH}" .pt)"
VIDEO_DIR="${RUN_DIR}/videos/play"
RAW_VIDEO_PATH="${VIDEO_DIR}/rl-video-step-0.mp4"
FINAL_VIDEO_PATH="${VIDEO_DIR}/latest_eval_${CHECKPOINT_STEM}.mp4"

mkdir -p "${VIDEO_DIR}"
rm -f "${RAW_VIDEO_PATH}"

RUN_DIR_REL="${RUN_DIR#${REPO_ROOT}/}"
CHECKPOINT_REL="${CHECKPOINT_PATH#${REPO_ROOT}/}"
CHECKPOINT_IN_CONTAINER="/workspace/isaaclab/${CHECKPOINT_REL}"

VIS_ARGS=()
if [[ -n "${VISUALIZE_BODY}" ]]; then
  VIS_ARGS+=(--visualize_body "${VISUALIZE_BODY}" --visualize_body_radius "${VISUALIZE_BODY_RADIUS}")
fi

cd "${REPO_ROOT}"

apptainer exec --nv \
  --bind "${REPO_ROOT}:/workspace/isaaclab" \
  --bind "${REPO_ROOT}/docker/apptainer/build/tmp/kit-data:/isaac-sim/kit/data" \
  --bind "${REPO_ROOT}/docker/apptainer/build/tmp/kit-cache:/isaac-sim/kit/cache" \
  --bind "${REPO_ROOT}/docker/apptainer/build/tmp/kit-logs:/isaac-sim/kit/logs" \
  --bind "${REPO_ROOT}/docker/apptainer/build/tmp/ov-cache:/root/.cache/ov" \
  --bind "${REPO_ROOT}/docker/apptainer/build/tmp/documents:/root/Documents" \
  --bind "${REPO_ROOT}/docker/apptainer/build/tmp/kit-logs:/root/.nvidia-omniverse/logs" \
  docker/apptainer/build/isaac-lab-v2.3.2.sif \
  bash -lc "
    source /workspace/isaaclab/env_isaaclab/bin/activate
    source /isaac-sim/setup_conda_env.sh
    export ISAACLAB_PATH=/workspace/isaaclab
    export OMNI_KIT_ALLOW_ROOT=1
    export CUDA_VISIBLE_DEVICES=0
    export PYTHONPATH=/workspace/isaaclab/source/isaaclab:/workspace/isaaclab/source/isaaclab_assets:/workspace/isaaclab/source/isaaclab_tasks:/workspace/isaaclab/source/isaaclab_rl:/workspace/isaaclab/source/isaaclab_mimic:\${PYTHONPATH:-}
    python -u scripts/reinforcement_learning/rsl_rl/play.py \
      --task ${TASK} \
      --headless \
      --enable_cameras \
      --device ${DEVICE} \
      --num_envs ${NUM_ENVS} \
      --video \
      --video_length ${VIDEO_LENGTH} \
      --checkpoint ${CHECKPOINT_IN_CONTAINER} \
      ${VIS_ARGS[*]:-}
  "

if [[ ! -f "${RAW_VIDEO_PATH}" ]]; then
  echo "[ERROR] Expected video was not generated at ${RAW_VIDEO_PATH}" >&2
  exit 1
fi

cp -f "${RAW_VIDEO_PATH}" "${FINAL_VIDEO_PATH}"
echo "[INFO] Run directory: ${RUN_DIR}"
echo "[INFO] Checkpoint: ${CHECKPOINT_PATH}"
echo "[INFO] Video: ${FINAL_VIDEO_PATH}"
