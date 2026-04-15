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

append_task_success_rate() {
    local current_task_name=$1
    local summary_file=$2
    local batch_eval_id=$3

    local result_dir="eval_result/${current_task_name}/${policy_name}/${task_config}/${model_name}/${batch_eval_id}"
    local result_file="${result_dir}/_result.txt"

    if [[ ! -f "${result_file}" ]]; then
        echo -e "\033[31mfailed to find result file for task: ${current_task_name}\033[0m"
        echo "| ${current_task_name} | N/A | ${result_dir} |" >> "${summary_file}"
        return 1
    fi

    local success_rate
    success_rate=$(tail -n 1 "${result_file}" | tr -d '[:space:]')

    if [[ -z "${success_rate}" ]]; then
        success_rate="N/A"
    fi

    echo "| ${current_task_name} | ${success_rate} | ${result_dir} |" >> "${summary_file}"
}

run_task_group() {
    local summary_file=$1
    local batch_eval_id=$2
    shift 2

    local current_task_name
    for current_task_name in "$@"; do
        run_eval "${current_task_name}" "${batch_eval_id}" || return 1
        append_task_success_rate "${current_task_name}" "${summary_file}" "${batch_eval_id}" || return 1
    done
}

append_overall_average() {
    local summary_file=$1

    local overall_average
    overall_average=$(awk -F'|' '
        /^\| / {
            rate = $3
            gsub(/^[ \t]+|[ \t]+$/, "", rate)
            if (rate != "Success Rate" && rate != "N/A" && rate != "") {
                sum += rate
                count += 1
            }
        }
        END {
            if (count > 0) {
                printf "%.4f", sum / count
            } else {
                print "N/A"
            }
        }
    ' "${summary_file}")

    {
        echo
        echo "- Overall average: ${overall_average}"
    } >> "${summary_file}"
}

run_eval() {
    local current_task_name=$1
    local batch_eval_id=${2:-}

    echo -e "\033[36mstarting evaluation for task: ${current_task_name}\033[0m"

    local cmd=(
        python script/eval_policy_client.py --config policy/$policy_name/deploy_policy.yml
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
    )

    if [[ -n "${batch_eval_id}" ]]; then
        cmd+=(--save_timestamp "${batch_eval_id}")
    fi

    PYTHONWARNINGS=ignore::UserWarning "${cmd[@]}"
}

if [[ "${run_all_tasks}" == "true" || "${run_all_tasks}" == "--all-tasks" ]]; then
    mapfile -t all_task_names < <(
        find envs -maxdepth 1 -type f -name "*.py" \
            ! -name "__init__.py" \
            ! -name "_*.py" \
            -printf "%f\n" | sed 's/\.py$//' | sort
    )

    echo -e "\033[36mfound ${#all_task_names[@]} tasks for full evaluation\033[0m"

    batch_eval_id=$(date +"%Y-%m-%d %H:%M:%S")
    summary_dir="eval_result/_all_tasks/${policy_name}/${task_config}/${model_name}/${batch_eval_id}"
    summary_file="${summary_dir}/task_success_rates.md"
    summary_file_1="${summary_dir}/task_success_rates_part1.md"
    summary_file_2="${summary_dir}/task_success_rates_part2.md"
    mkdir -p "${summary_dir}"

    {
        echo "# Task Success Rates"
        echo
        echo "- Batch timestamp: ${batch_eval_id}"
        echo "- Policy: ${policy_name}"
        echo "- Task config: ${task_config}"
        echo "- Checkpoint: ${model_name}"
        echo "- Seed: ${seed}"
        echo "- Server: ${server_host}:${server_port}"
        echo
        echo "| Task | Success Rate | Result Dir |"
        echo "| --- | --- | --- |"
    } > "${summary_file}"

    echo -e "\033[36mall-task summary will be saved to: ${summary_file}\033[0m"

    mid_index=$(( (${#all_task_names[@]} + 1) / 2 ))
    task_group_1=( "${all_task_names[@]:0:mid_index}" )
    task_group_2=( "${all_task_names[@]:mid_index}" )

    : > "${summary_file_1}"
    : > "${summary_file_2}"

    run_task_group "${summary_file_1}" "${batch_eval_id}" "${task_group_1[@]}" &
    pid_1=$!
    run_task_group "${summary_file_2}" "${batch_eval_id}" "${task_group_2[@]}" &
    pid_2=$!

    wait "${pid_1}" || exit 1
    wait "${pid_2}" || exit 1

    cat "${summary_file_1}" "${summary_file_2}" >> "${summary_file}"
    append_overall_average "${summary_file}"

    rm -f "${summary_file_1}" "${summary_file_2}"
else
    run_eval "${task_name}"
fi

# bash eval.sh grab_roller demo_clean pi0_base_aloha_robotwin_full grab_roller-aloha-agilex_clean_50 0 0
# bash eval_client.sh grab_roller demo_clean pi0_base_aloha_robotwin_full_chunk_delta_position_10d_action_grab_roller-aloha-agilex_clean_50 grab_roller-aloha-agilex_clean_50-50-10d_action ee_10d 0 0
# bash eval_client.sh grab_roller demo_clean pi0_base_aloha_robotwin_full_chunk_delta_position_10d_action_grab_roller-aloha-agilex_clean_50 grab_roller-aloha-agilex_clean_50-50-10d_action ee_10d 0 0 127.0.0.1 1234
# bash eval_client.sh placeholder demo_clean pi0_base_aloha_robotwin_full_50_tasks_clean_cts_10d_action multi_clean_50-cts-10d_action ee_10d 0 127.0.0.1 1234 --all-tasks
