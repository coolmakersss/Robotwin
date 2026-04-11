#!/bin/bash

# == keep unchanged ==
policy_name=DP
task_name=${1}
task_config=${2}
ckpt_setting=${3}
expert_data_num=${4}
action_dim=${5}
mode=${6}
seed=${7}
gpu_id=${8}
DEBUG=False

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../..

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --action_dim ${action_dim} \
    --expert_data_num ${expert_data_num} \
    --mode ${mode} \
    --seed ${seed}

# c
# bash eval.sh grab_roller demo_clean arx-x5_clean_50 50 16 eef 0 0
# bash eval.sh lift_pot demo_clean aloha-agilex_clean_50 50 16 eef 0 0
# bash eval.sh lift_pot demo_clean arx-x5_clean_50 50 16 eef 0 0
