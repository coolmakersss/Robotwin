#!/bin/bash

# == keep unchanged ==
policy_name=ACT
task_name=${1}
task_config=${2}
ckpt_setting=${3}
expert_data_num=${4}
seed=${5}
gpu_id=${6}
mode=${7:-qpos}
test_num=${8:-100}
expert_check=${9:-true}
ckpt_name=${10:-policy_last.ckpt}
# temporal_agg=${5} # use temporal_agg
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
    --ckpt_dir policy/ACT/act_ckpt/act-${task_name}/${ckpt_setting}-${expert_data_num} \
    --seed ${seed} \
    --mode ${mode} \
    --test_num ${test_num} \
    --expert_check ${expert_check} \
    --ckpt_name ${ckpt_name} \
    --temporal_agg true
