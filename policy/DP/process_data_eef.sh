#!/bin/bash

task_name=${1}
task_config=${2}
expert_data_num=${3}

python process_data_eef.py $task_name $task_config $expert_data_num

# bash process_data_eef.sh grab_roller aloha-agilex_clean_50 50
# bash process_data_eef.sh grab_roller arx-x5_clean_50 50
# bash process_data_eef.sh lift_pot aloha-agilex_clean_50 50
# bash process_data_eef.sh lift_pot arx-x5_clean_50 50
# bash process_data_eef.sh handover_mic aloha-agilex_clean_50 50
# bash process_data_eef.sh handover_mic arx-x5_clean_50 50
# bash process_data_eef.sh place_bread_skillet aloha-agilex_clean_50 50
# bash process_data_eef.sh place_bread_skillet arx-x5_clean_50 50
# bash process_data_eef.sh place_cans_plasticbox aloha-agilex_clean_50 50
# bash process_data_eef.sh place_cans_plasticbox arx-x5_clean_50 50