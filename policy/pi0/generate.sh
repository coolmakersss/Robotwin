data_dir=${1}
repo_id=${2}
uv run examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py --raw_dir $data_dir --repo_id $repo_id


# bash generate.sh ./training_data/grab_roller-aloha-agilex_clean_50-50 grab_roller-aloha-agilex_clean_50-50


# bash generate.sh ./training_data_cts/grab_roller-aloha-agilex_clean_50-50 grab_roller-aloha-agilex_clean_50-50-cts
# bash generate.sh ./training_data_cts/grab_roller-arx-x5_clean_50-50 grab_roller-arx-x5_clean_50-50-cts
# bash generate.sh ./training_data_cts/lift_pot-aloha-agilex_clean_50-50 lift_pot-aloha-agilex_clean_50-50-cts
# bash generate.sh ./training_data_cts/lift_pot-arx-x5_clean_50-50 lift_pot-arx-x5_clean_50-50-cts

# bash generate.sh ./training_data_delta/lift_pot-aloha-agilex_clean_50-50 lift_pot-aloha-agilex_clean_50-50-delta

# bash generate.sh ./training_data_10d_action/lift_pot-aloha-agilex_clean_50-50 lift_pot-aloha-agilex_clean_50-50-cts-10d_action
# bash generate.sh ./training_data_10d_action/lift_pot-arx-x5_clean_50-50 lift_pot-arx-x5_clean_50-50-cts-10d_action
# bash generate.sh ./training_data_10d_action/grab_roller-aloha-agilex_clean_50-50 grab_roller-aloha-agilex_clean_50-50-cts-10d_action
# bash generate.sh ./training_data_10d_action/grab_roller-arx-x5_clean_50-50 grab_roller-arx-x5_clean_50-50-cts-10d_action