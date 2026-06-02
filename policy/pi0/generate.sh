data_dir=${1}
repo_id=${2}
mode=${3:-default}

converter=examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py
if [ "${mode}" = "mode-ratio" ] || [ "${mode}" = "mode_ratio" ] || [[ "${data_dir}" == *mode-ratio* ]] || [[ "${data_dir}" == *mode_ratio* ]]; then
converter=examples/aloha_real/convert_aloha_data_to_lerobot_robotwin_mode_ratio.py
fi

export UV_CACHE_DIR=/tmp/uv-cache-${USER:-root}
export TMPDIR=/tmp
export UV_LINK_MODE=copy
export UV_PROJECT_ENVIRONMENT=/mnt/afs/huangdi/xiangenda/.venv
uv run $converter --raw_dir $data_dir --repo_id $repo_id


# bash generate.sh ./training_data/grab_roller-aloha-agilex_clean_50-50 grab_roller-aloha-agilex_clean_50-50


# bash generate.sh ./training_data_cts/grab_roller-aloha-agilex_clean_50-50 grab_roller-aloha-agilex_clean_50-50-cts
# bash generate.sh ./training_data_cts/grab_roller-arx-x5_clean_50-50 grab_roller-arx-x5_clean_50-50-cts
# bash generate.sh ./training_data_cts/lift_pot-aloha-agilex_clean_50-50 lift_pot-aloha-agilex_clean_50-50-cts
# bash generate.sh ./training_data_cts/lift_pot-arx-x5_clean_50-50 lift_pot-arx-x5_clean_50-50-cts

# bash generate.sh ./training_data_delta/lift_pot-aloha-agilex_clean_50-50 lift_pot-aloha-agilex_clean_50-50-delta

# bash generate.sh ./training_data_cts_10d_action/lift_pot-aloha-agilex_clean_50-50 lift_pot-aloha-agilex_clean_50-50-cts-10d_action
# bash generate.sh ./training_data_cts_10d_action/lift_pot-arx-x5_clean_50-50 lift_pot-arx-x5_clean_50-50-cts-10d_action
# bash generate.sh ./training_data_cts_10d_action/grab_roller-aloha-agilex_clean_50-50 grab_roller-aloha-agilex_clean_50-50-cts-10d_action
# bash generate.sh ./training_data_cts_10d_action/grab_roller-arx-x5_clean_50-50 grab_roller-arx-x5_clean_50-50-cts-10d_action

# bash generate.sh ./training_data_50_tasks_cts_10d_action_mode_ratio/swep_table-realworld-50 swep_table-realworld-50-cts-10d_action-mode-ratio mode-ratio
