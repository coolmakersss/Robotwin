#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 # ensure GPU < 24G

policy_name=pi0
task_name=${1}
task_config=${2}
train_config_name=${3}
model_name=${4}
seed=${5}
gpu_id=${6}
server_host=${7:-180.184.148.133}
server_port=${8:-60913}
run_all_tasks=${9:-false}

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[33mserver host: ${server_host}\033[0m"
echo -e "\033[33mserver port: ${server_port}\033[0m"
echo -e "\033[33mrun all 50 tasks: ${run_all_tasks}\033[0m"

source .venv/bin/activate
cd ../.. # move to root

run_eval() {
    local current_task_name=$1

    echo -e "\033[36mstarting evaluation for task: ${current_task_name}\033[0m"

    PYTHONWARNINGS=ignore::UserWarning \
    python script/eval_policy_client.py --config policy/$policy_name/deploy_policy.yml \
        --host ${server_host} \
        --overrides \
        --task_name ${current_task_name} \
        --task_config ${task_config} \
        --train_config_name ${train_config_name} \
        --model_name ${model_name} \
        --ckpt_setting ${model_name} \
        --seed ${seed} \
        --policy_name ${policy_name} \
        --port ${server_port}
}

if [[ "${run_all_tasks}" == "true" || "${run_all_tasks}" == "--all-tasks" ]]; then
    mapfile -t all_task_names < <(
        find envs -maxdepth 1 -type f -name "*.py" \
            ! -name "__init__.py" \
            ! -name "_*.py" \
            -printf "%f\n" | sed 's/\.py$//' | sort
    )

    echo -e "\033[36mfound ${#all_task_names[@]} tasks for full evaluation\033[0m"

    for current_task_name in "${all_task_names[@]}"; do
        run_eval "${current_task_name}" || exit 1
    done
else
    run_eval "${task_name}"
fi

# bash eval.sh grab_roller demo_clean pi0_base_aloha_robotwin_full grab_roller-aloha-agilex_clean_50 0 0
# bash eval_client.sh grab_roller demo_clean pi0_base_aloha_robotwin_full_chunk_delta_position_10d_action_grab_roller-aloha-agilex_clean_50 grab_roller-aloha-agilex_clean_50-50-10d_action ee_10d 0 0
# bash eval_client.sh grab_roller demo_clean pi0_base_aloha_robotwin_full_chunk_delta_position_10d_action_grab_roller-aloha-agilex_clean_50 grab_roller-aloha-agilex_clean_50-50-10d_action ee_10d 0 0 127.0.0.1 1234
# bash eval_client.sh placeholder demo_clean pi0_base_aloha_robotwin_full_50_tasks_clean_cts_10d_action multi_clean_50-cts-10d_action ee_10d 0 127.0.0.1 1234 --all-tasks
