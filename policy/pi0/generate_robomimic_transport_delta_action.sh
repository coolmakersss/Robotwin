#!/usr/bin/env bash
set -euo pipefail

input_hdf5=${1:-/mnt/afs/huangdi/xiangenda/robomimic/datasets/transport/ph/image_v15.hdf5}
raw_dir=${2:-training_data_robomimic_transport_delta_action}
repo_id=${3:-multi-robomimic-transport-delta_action}
robotwin_python=${ROBOTWIN_PYTHON:-/mnt/afs/huangdi/xiangenda/miniconda3/envs/robotwin/bin/python}
robotwin_bin=$(dirname "${robotwin_python}")
overwrite_args=()
max_episode_args=()

if [[ "${OVERWRITE:-0}" == "1" ]]; then
  overwrite_args+=(--overwrite)
fi
if [[ -n "${MAX_EPISODES:-}" ]]; then
  max_episode_args+=(--max-episodes "${MAX_EPISODES}")
fi

"${robotwin_python}" scripts/process_robomimic_transport_delta_action.py \
  --input-hdf5 "${input_hdf5}" \
  --save-dir "${raw_dir}" \
  "${overwrite_args[@]}" \
  "${max_episode_args[@]}"

PATH="${robotwin_bin}:${PATH}" \
  HF_LEROBOT_HOME=/mnt/afs/huangdi/xiangenda/.cache/huggingface/lerobot \
  UV_CACHE_DIR=/tmp/uv-cache-${USER:-root} \
  TMPDIR=/tmp \
  UV_LINK_MODE=copy \
  UV_PROJECT_ENVIRONMENT=/mnt/afs/huangdi/xiangenda/.venv \
  uv run examples/aloha_real/convert_aloha_data_to_lerobot_robomimic_delta.py \
    --raw_dir "${raw_dir}" \
    --repo_id "${repo_id}"
