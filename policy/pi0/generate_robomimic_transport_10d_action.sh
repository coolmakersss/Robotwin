#!/usr/bin/env bash
set -euo pipefail

input_hdf5=${1:-/mnt/afs/huangdi/xiangenda/robomimic/datasets/transport/ph/image_v15.hdf5}
raw_dir=${2:-training_data_robomimic_transport_10d_action}
repo_id=${3:-multi-robomimic-transport-10d_action}
robotwin_python=${ROBOTWIN_PYTHON:-/mnt/afs/huangdi/xiangenda/miniconda3/envs/robotwin/bin/python}
robotwin_bin=$(dirname "${robotwin_python}")
overwrite_args=()

if [[ "${OVERWRITE:-0}" == "1" ]]; then
  overwrite_args+=(--overwrite)
fi

"${robotwin_python}" scripts/process_robomimic_transport_10d_action.py \
  --input-hdf5 "${input_hdf5}" \
  --save-dir "${raw_dir}" \
  "${overwrite_args[@]}"

PATH="${robotwin_bin}:${PATH}" \
  HF_LEROBOT_HOME=/mnt/afs/huangdi/xiangenda/.cache/huggingface/lerobot \
  bash generate.sh "${raw_dir}" "${repo_id}"
