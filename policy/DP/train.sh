#!/bin/bash

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
action_dim=${5}
gpu_id=${6}
mode=${7}

head_camera_type=D435

DEBUG=False
save_ckpt=True

alg_name=robot_dp_${action_dim}
if [ ${mode} = "cts" ]; then
    alg_name=robot_dp_${action_dim}_cts
fi
if [ ${mode} = "joint" ]; then
    alg_name=robot_dp_${action_dim}_joint
fi
if [ ${mode} = "delta" ]; then
    alg_name=robot_dp_${action_dim}_delta_horizon_32
fi
if [ ${mode} = "chunk_delta" ]; then
    alg_name=robot_dp_${action_dim}_chunk_delta
fi
if [ ${mode} = "chunk_delta_position" ]; then
    alg_name=robot_dp_${action_dim}_chunk_delta_position
fi
if [ ${mode} = "chunk_delta_no_norm" ]; then
    alg_name=robot_dp_${action_dim}_chunk_delta_no_norm
fi
if [ ${mode} = "delta_cts" ]; then
    alg_name=robot_dp_${action_dim}_delta_cts
fi
if [ ${mode} = "chunk_delta_cts_position" ]; then
    alg_name=robot_dp_${action_dim}_chunk_delta_cts_position
fi
config_name=${alg_name}
addition_info=train
exp_name=${task_name}-robot_dp-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"


if [ $DEBUG = True ]; then
    wandb_mode=offline
    # wandb_mode=online
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
fi

export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}

if [ ! -d "./data/${task_name}-${task_config}-${expert_data_num}.zarr" ]; then
    bash process_data.sh ${task_name} ${task_config} ${expert_data_num}
fi

python train.py --config-name=${config_name}.yaml \
                            task.name=${task_name} \
                            task.dataset.zarr_path="data/${task_name}-${task_config}-${expert_data_num}.zarr" \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            setting=${task_config} \
                            expert_data_num=${expert_data_num} \
                            head_camera_type=$head_camera_type
                            # checkpoint.save_ckpt=${save_ckpt}
                            # hydra.run.dir=${run_dir} \


### 
# nohup bash train.sh grab_roller aloha-agilex_clean_50 50 0 16 0 &
# nohup bash train.sh grab_roller arx-x5_clean_50 50 0 16 1 &
# nohup bash train.sh lift_pot aloha-agilex_clean_50 50 0 16 0 &
# nohup bash train.sh lift_pot arx-x5_clean_50 50 0 16 1 &

### joint
# nohup bash train.sh grab_roller aloha-agilex_clean_50 50 0 14 0 joint &
# nohup bash train.sh grab_roller arx-x5_clean_50 50 0 14 0 joint &
# nohup bash train.sh lift_pot aloha-agilex_clean_50 50 0 14 0 joint &
# nohup bash train.sh lift_pot arx-x5_clean_50 50 0 14 0 joint &


### cts
# nohup bash train.sh handover_mic aloha-agilex_clean_50 50 0 16 0 cts &
# nohup bash train.sh handover_mic arx-x5_clean_50 50 0 16 0 cts &
# nohup bash train.sh place_bread_skillet aloha-agilex_clean_50 50 0 16 0 cts &
# nohup bash train.sh place_bread_skillet arx-x5_clean_50 50 0 16 0 cts &
# nohup bash train.sh place_cans_plasticbox aloha-agilex_clean_50 50 0 16 0 cts &
# nohup bash train.sh place_cans_plasticbox arx-x5_clean_50 50 0 16 0 cts &

### delta cts
# nohup bash train.sh grab_roller aloha-agilex_clean_50 50 0 16 0 delta_cts &
# nohup bash train.sh grab_roller arx-x5_clean_50 50 0 16 1 delta_cts &
# nohup bash train.sh lift_pot aloha-agilex_clean_50 50 0 16 0 delta_cts &
# nohup bash train.sh lift_pot arx-x5_clean_50 50 0 16 1 delta_cts &

### chunk delta
# nohup bash train.sh lift_pot aloha-agilex_clean_50 50 0 16 0 chunk_delta &
# nohup bash train.sh grab_roller aloha-agilex_clean_50 50 0 16 0 chunk_delta &

### chunk delta no norm
# nohup bash train.sh lift_pot aloha-agilex_clean_50 50 0 16 0 chunk_delta_no_norm &
# nohup bash train.sh grab_roller aloha-agilex_clean_50 50 0 16 0 chunk_delta_no_norm &

### chunk delta position
# nohup bash train.sh lift_pot aloha-agilex_clean_50 50 0 16 0 chunk_delta_position &
# nohup bash train.sh grab_roller aloha-agilex_clean_50 50 0 16 0 chunk_delta_position &

### chunk delta cts position
# nohup bash train.sh lift_pot aloha-agilex_clean_50 50 0 16 0 chunk_delta_cts_position &
# nohup bash train.sh grab_roller aloha-agilex_clean_50 50 0 16 0 chunk_delta_cts_position &

# bash train.sh handover_mic aloha-agilex_clean_50 50 0 16 0 chunk_delta_cts_position
# bash train.sh handover_mic arx-x5_clean_50 50 0 16 0 chunk_delta_cts_position
# bash train.sh place_bread_skillet aloha-agilex_clean_50 50 0 16 0 chunk_delta_cts_position
# bash train.sh place_bread_skillet  arx-x5_clean_50 50 0 16 0 chunk_delta_cts_position
# bash train.sh place_cans_plasticbox aloha-agilex_clean_50 50 0 16 0 chunk_delta_cts_position
# bash train.sh place_cans_plasticbox arx-x5_clean_50 50 0 16 0 chunk_delta_cts_position

# cp --parents checkpoints/*/delta/600.ckpt /mnt/aoss/xiangenda/tmp/
# cp --parents checkpoints/*/delta_cts/600.ckpt /mnt/aoss/xiangenda/tmp/



# https_proxy="http://127.0.0.1:1081" HF_ENDPOINT=https://huggingface.co HF_HUB_ENABLE_HF_TRANSFER=1 hf upload Arosy24/robotwin_dp ./policy/DP/checkpoints/handover_mic-aloha-agilex_clean_50-50-0/chunk_delta_cts_position/600.ckpt handover_mic-aloha-agilex_clean_50-50-0/chunk_delta_cts_position/600.ckpt
# https_proxy="http://127.0.0.1:1081" HF_ENDPOINT=https://huggingface.co HF_HUB_ENABLE_HF_TRANSFER=1 hf upload Arosy24/robotwin_dp ./policy/DP/checkpoints/handover_mic-arx-x5_clean_50-50-0/chunk_delta_cts_position/600.ckpt handover_mic-arx-x5_clean_50-50-0/chunk_delta_cts_position/600.ckpt

# https_proxy="http://127.0.0.1:1081" HF_ENDPOINT=https://huggingface.co HF_HUB_ENABLE_HF_TRANSFER=1 hf upload Arosy24/robotwin_dp ./policy/DP/checkpoints/place_bread_skillet-aloha-agilex_clean_50-50-0/chunk_delta_cts_position/600.ckpt place_bread_skillet-aloha-agilex_clean_50-50-0/chunk_delta_cts_position/600.ckpt
# https_proxy="http://127.0.0.1:1081" HF_ENDPOINT=https://huggingface.co HF_HUB_ENABLE_HF_TRANSFER=1 hf upload Arosy24/robotwin_dp ./policy/DP/checkpoints/place_bread_skillet-arx-x5_clean_50-50-0/chunk_delta_cts_position/600.ckpt place_bread_skillet-arx-x5_clean_50-50-0/chunk_delta_cts_position/600.ckpt

# https_proxy="http://127.0.0.1:1081" HF_ENDPOINT=https://huggingface.co HF_HUB_ENABLE_HF_TRANSFER=1 hf upload Arosy24/robotwin_dp ./policy/DP/checkpoints/place_cans_plasticbox-aloha-agilex_clean_50-50-0/chunk_delta_cts_position/600.ckpt place_cans_plasticbox-aloha-agilex_clean_50-50-0/chunk_delta_cts_position/600.ckpt
# https_proxy="http://127.0.0.1:1081" HF_ENDPOINT=https://huggingface.co HF_HUB_ENABLE_HF_TRANSFER=1 hf upload Arosy24/robotwin_dp ./policy/DP/checkpoints/place_cans_plasticbox-arx-x5_clean_50-50-0/chunk_delta_cts_position/600.ckpt place_cans_plasticbox-arx-x5_clean_50-50-0/chunk_delta_cts_position/600.ckpt