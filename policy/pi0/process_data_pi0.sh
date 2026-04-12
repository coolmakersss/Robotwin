task_name=${1}
setting=${2}
expert_data_num=${3}
mode=${4}

if [ ${mode} = "eef" ]; then
python scripts/process_data.py $task_name $setting $expert_data_num
fi

if [ ${mode} = "cts" ]; then
python scripts/process_data_cts.py $task_name $setting $expert_data_num
fi

if [ ${mode} = "delta" ]; then
python scripts/process_data_delta.py $task_name $setting $expert_data_num
fi

if [ ${mode} = "eef-10d" ]; then
python scripts/process_data_10d_action.py $task_name $setting $expert_data_num
fi

if [ ${mode} = "cts-10d" ]; then
python scripts/process_data_cts_10d_action.py $task_name $setting $expert_data_num
fi

# bash process_data_pi0.sh grab_roller aloha-agilex_clean_50 50 eef
# bash process_data_pi0.sh grab_roller aloha-agilex_clean_50 50 cts
# bash process_data_pi0.sh lift_pot aloha-agilex_clean_50 50 delta
# bash process_data_pi0.sh lift_pot aloha-agilex_clean_50 50 eef-10d
# bash process_data_pi0.sh lift_pot aloha-agilex_clean_50 50 cts-10d
