#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 # ensure GPU < 24G

policy_name=pi0
task_name=${1}
task_config=${2}
train_config_name=${3}
model_name=${4}
seed=${5}
gpu_id=${6}

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

source .venv/bin/activate
cd ../.. # move to root


PYTHONWARNINGS=ignore::UserWarning \
python script/policy_model_server.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --ckpt_setting ${model_name} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --port 1234

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

# nohup bash eval.sh grab_roller demo_clean pi0_base_aloha_robotwin_full_chunk_delta_cts_position grab_roller-aloha-agilex_clean_50-chunk_delta_cts_position 0 0 &
