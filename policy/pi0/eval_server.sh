#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 # ensure GPU < 24G

policy_name=pi0
task_name=${1}
task_config=${2}
train_config_name=${3}
model_name=${4}
seed=${5}
gpu_id=${6}
server_port=${7:-1234}
fast_tokenizer_path=${8:-${OPENPI_FAST_TOKENIZER_PATH:-/mnt/afs/huangdi/xiangenda/RoboTwin/policy/pi0/fast/fast_tokenizer_10d_action}}

export CUDA_VISIBLE_DEVICES=${gpu_id}
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-$HOME/.cache/openpi}
export OPENPI_FAST_TOKENIZER_PATH=${fast_tokenizer_path}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[33mserver port: ${server_port}\033[0m"
echo -e "\033[33mopenpi cache dir: ${OPENPI_DATA_HOME}\033[0m"
echo -e "\033[33mFAST tokenizer: ${OPENPI_FAST_TOKENIZER_PATH}\033[0m"

OPENPI_PYTHON=/mnt/afs/huangdi/xiangenda/.venv/bin/python
cd ../.. # move to root


PYTHONWARNINGS=ignore::UserWarning \
"${OPENPI_PYTHON}" script/policy_model_server.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --ckpt_setting ${model_name} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --port ${server_port}

#PYTHONWARNINGS=ignore::UserWarning \
#python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
#    --overrides \
#    --task_name ${task_name} \
#    --task_config ${task_config} \
#    --train_config_name ${train_config_name} \
#    --model_name ${model_name} \
#    --ckpt_setting ${model_name} \
#    --seed ${seed} \
#    --policy_name ${policy_name} 


# bash eval.sh grab_roller demo_clean pi0_base_aloha_robotwin_full grab_roller-aloha-agilex_clean_50 0 0
# bash eval.sh lift_pot demo_clean pi0_base_aloha_robotwin_full lift_pot-aloha-agilex_clean_50 0 0
# bash eval.sh lift_pot demo_clean pi0_base_aloha_robotwin_full lift_pot-aloha-agilex_clean_50-delta 0 0

# nohup bash eval.sh grab_roller demo_clean pi0_base_aloha_robotwin_full_chunk_delta_position grab_roller-aloha-agilex_clean_50-chunk_delta_position 0 0 &

# nohup bash eval_server.sh grab_roller demo_clean pi0_base_aloha_robotwin_full_chunk_delta_cts_position grab_roller-aloha-agilex_clean_50-chunk_delta_cts_position 0 0 &

# nohup bash eval_server.sh grab_roller demo_clean pi0_base_aloha_robotwin_full_chunk_delta_position_10d_action_grab_roller-aloha-agilex_clean_50 grab_roller-aloha-agilex_clean_50-chunk_delta_position_10d_action 0 0 &

# nohup bash eval_server.sh grab_roller demo_clean pi0_base_aloha_robotwin_full_chunk_delta_position_10d_action_grab_roller-arx-x5_clean_50 grab_roller-arx-x5_clean_50-chunk_delta_position_10d_action 0 0 &

# nohup bash eval_server.sh lift_pot demo_clean pi0_base_aloha_robotwin_full_chunk_delta_position_10d_action_lift_pot-arx-x5_clean_50 lift_pot-arx-x5_clean_50-chunk_delta_position_10d_action 0 0 &

# nohup bash eval_server.sh grab_roller demo_clean pi0_base_aloha_robotwin_full_chunk_delta_cts_position_10d_action_grab_roller-aloha-agilex_clean_50 grab_roller-aloha-agilex_clean_50-chunk_delta_cts_position_10d_action 0 0 &
# bash eval_server.sh grab_roller demo_clean pi0_base_aloha_robotwin_full_chunk_delta_position_10d_action_grab_roller-aloha-agilex_clean_50 grab_roller-aloha-agilex_clean_50-50-10d_action ee_10d 0 0 1234
