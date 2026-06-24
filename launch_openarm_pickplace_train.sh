#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_NAME="${1:-byobu_openarm_pickplace}"
NUM_ENVS="${NUM_ENVS:-2048}"
MAX_ITERS="${MAX_ITERS:-1000000}"

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
    python -u scripts/reinforcement_learning/rsl_rl/train.py \
      --task Isaac-PickPlace-Cube-OpenArm-Bi-v0 \
      --headless \
      --device cuda:0 \
      --num_envs ${NUM_ENVS} \
      --max_iterations ${MAX_ITERS} \
      --run_name ${RUN_NAME}
  "
