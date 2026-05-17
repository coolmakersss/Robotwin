#!/bin/bash

#加速评估 bash eval_client.sh ...
#恢复旧行为和视频 ROBOTWIN_EVAL_VIDEO_LOG=true PI0_UPDATE_OBS_EVERY_ACTION=true bash eval_client.sh ...

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
test_num=${ROBOTWIN_TEST_NUM:-100}
eval_video_log=${ROBOTWIN_EVAL_VIDEO_LOG:-false}
render_freq=${ROBOTWIN_RENDER_FREQ:-0}
expert_check=${ROBOTWIN_EXPERT_CHECK:-true}
export PI0_ACTION_CHUNK_STEPS=${PI0_ACTION_CHUNK_STEPS:-20}
export PI0_UPDATE_OBS_EVERY_ACTION=${PI0_UPDATE_OBS_EVERY_ACTION:-false}

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[33mserver host: ${server_host}\033[0m"
echo -e "\033[33mserver port: ${server_port}\033[0m"
echo -e "\033[33mbatch eval mode: ${run_all_tasks}\033[0m"
echo -e "\033[33mtest episodes: ${test_num}\033[0m"
echo -e "\033[33meval video log: ${eval_video_log}\033[0m"
echo -e "\033[33mrender freq: ${render_freq}\033[0m"
echo -e "\033[33mexpert check: ${expert_check}\033[0m"
echo -e "\033[33maction chunk steps: ${PI0_ACTION_CHUNK_STEPS}\033[0m"
echo -e "\033[33mupdate obs every action: ${PI0_UPDATE_OBS_EVERY_ACTION}\033[0m"

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

has_task_success_rate() {
    local current_task_name=$1
    local summary_file=$2

    [[ -f "${summary_file}" ]] && grep -Fq "| ${current_task_name} |" "${summary_file}"
}

ensure_summary_header() {
    local summary_file=$1
    local batch_eval_id=$2
    local task_group=${3:-}

    if [[ -f "${summary_file}" ]]; then
        return 0
    fi

    mkdir -p "$(dirname "${summary_file}")"
    {
        echo "# Task Success Rates"
        echo
        echo "- Batch timestamp: ${batch_eval_id}"
        echo "- Policy: ${policy_name}"
        echo "- Task config: ${task_config}"
        echo "- Checkpoint: ${model_name}"
        echo "- Seed: ${seed}"
        echo "- Server: ${server_host}:${server_port}"
        if [[ -n "${task_group}" ]]; then
            echo "- Task group: ${task_group}"
        fi
        echo
        echo "| Task | Success Rate | Result Dir |"
        echo "| --- | --- | --- |"
    } > "${summary_file}"
}

append_overall_average() {
    local summary_file=$1

    sed -i '/^- Overall average:/d' "${summary_file}"

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
        --port ${server_port} \
        --test_num ${test_num} \
        --eval_video_log ${eval_video_log} \
        --render_freq ${render_freq} \
        --expert_check ${expert_check}
    )

    if [[ -n "${batch_eval_id}" ]]; then
        cmd+=(--save_timestamp "${batch_eval_id}")
    fi

    PYTHONWARNINGS=ignore::UserWarning "${cmd[@]}"
}

if [[ "${run_all_tasks}" == "true" || "${run_all_tasks}" == "--all-tasks" || "${run_all_tasks}" == "--coupled-tasks" || "${run_all_tasks}" == "--coupled" ]]; then
    task_group="all_tasks"
    summary_group="_all_tasks"

    mapfile -t all_task_names < <(
        find envs -maxdepth 1 -type f -name "*.py" \
            ! -name "__init__.py" \
            ! -name "_*.py" \
            -printf "%f\n" | sed 's/\.py$//' | sort
    )

    if [[ "${run_all_tasks}" == "--coupled-tasks" || "${run_all_tasks}" == "--coupled" ]]; then
        task_group="coupled_tasks"
        summary_group="_coupled_tasks"
        all_task_names=(
            dump_bin_bigbin
            handover_block
            handover_mic
            hanging_mug
            pick_diverse_bottles
            pick_dual_bottles
            place_bread_basket
            place_burger_fries
            place_can_basket
            place_cans_plasticbox
            place_dual_shoes
            place_object_basket
            put_bottles_dustbin
            put_object_cabinet
            scan_object
            grab_roller
            lift_pot
        )
    fi

    echo -e "\033[36mfound ${#all_task_names[@]} tasks for ${task_group} evaluation\033[0m"

    summary_root="eval_result/${summary_group}/${policy_name}/${task_config}/${model_name}"
    latest_summary_file=$(find "${summary_root}" -mindepth 2 -maxdepth 2 -type f -name 'task_success_rates.md' 2>/dev/null | sort | tail -n 1)

    if [[ -n "${latest_summary_file}" ]]; then
        summary_file="${latest_summary_file}"
        summary_dir=$(dirname "${summary_file}")
        batch_eval_id=$(basename "${summary_dir}")
        echo -e "\033[36mresuming ${task_group} evaluation from: ${summary_file}\033[0m"
    else
        batch_eval_id=$(date +"%Y-%m-%d %H:%M:%S")
        summary_dir="${summary_root}/${batch_eval_id}"
        summary_file="${summary_dir}/task_success_rates.md"
        echo -e "\033[36m${task_group} summary will be saved to: ${summary_file}\033[0m"
    fi

    ensure_summary_header "${summary_file}" "${batch_eval_id}" "${task_group}"

    for current_task_name in "${all_task_names[@]}"; do
        if has_task_success_rate "${current_task_name}" "${summary_file}"; then
            echo -e "\033[33mskipping completed task: ${current_task_name}\033[0m"
            continue
        fi

        result_file="eval_result/${current_task_name}/${policy_name}/${task_config}/${model_name}/${batch_eval_id}/_result.txt"
        if [[ -f "${result_file}" ]]; then
            echo -e "\033[33mfound existing result for task: ${current_task_name}, appending to summary\033[0m"
            append_task_success_rate "${current_task_name}" "${summary_file}" "${batch_eval_id}" || exit 1
            continue
        fi

        run_eval "${current_task_name}" "${batch_eval_id}" || exit 1
        append_task_success_rate "${current_task_name}" "${summary_file}" "${batch_eval_id}" || exit 1
    done

    append_overall_average "${summary_file}"
else
    run_eval "${task_name}"
fi

# bash eval.sh grab_roller demo_clean pi0_base_aloha_robotwin_full grab_roller-aloha-agilex_clean_50 0 0
# bash eval_client.sh grab_roller demo_clean pi0_base_aloha_robotwin_full_chunk_delta_position_10d_action_grab_roller-aloha-agilex_clean_50 grab_roller-aloha-agilex_clean_50-50-10d_action ee_10d 0 0
# bash eval_client.sh grab_roller demo_clean pi0_base_aloha_robotwin_full_chunk_delta_position_10d_action_grab_roller-aloha-agilex_clean_50 grab_roller-aloha-agilex_clean_50-50-10d_action ee_10d 0 0 127.0.0.1 1234
# bash eval_client.sh placeholder demo_clean pi0_base_aloha_robotwin_full_50_tasks_clean_cts_10d_action multi_clean_50-cts-10d_action ee_10d 0 127.0.0.1 1234 --all-tasks
# bash eval_client.sh placeholder demo_clean pi0_base_aloha_robotwin_full_50_tasks_clean_cts_10d_action multi_clean_50-cts-10d_action ee_10d 0 127.0.0.1 1234 --coupled-tasks
