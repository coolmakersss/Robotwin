import argparse
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import h5py
import matplotlib
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ARM_LABELS = ("left", "right")
ARM_DIM = 10
ROT6D_DIM = 6


def quaternion_to_rotation_6d(quaternion):
    rotation_matrix = R.from_quat(quaternion).as_matrix()
    return rotation_matrix[:, :2].T.reshape(-1)


def pose_to_10d(arm_pose, gripper):
    position = arm_pose[:3]
    rotation_6d = quaternion_to_rotation_6d(arm_pose[3:7])
    return np.concatenate([position, rotation_6d, np.array([gripper])], axis=0)


def pose_pair_to_20d(left_arm, left_gripper, right_arm, right_gripper):
    left_pose = pose_to_10d(left_arm, left_gripper)
    right_pose = pose_to_10d(right_arm, right_gripper)
    return np.concatenate([left_pose, right_pose], axis=0)


def visualize_action_dimensions(actions, save_path):
    actions = np.asarray(actions, dtype=np.float32)
    if actions.size == 0:
        return

    fig, axes = plt.subplots(5, 4, figsize=(20, 16), sharex=True)
    axes = axes.flatten()
    time_index = np.arange(actions.shape[0])

    for dim_idx, ax in enumerate(axes):
        arm_name = ARM_LABELS[dim_idx // ARM_DIM]
        local_dim = dim_idx % ARM_DIM
        if local_dim < 3:
            dim_name = f"pos_{local_dim}"
        elif local_dim < 3 + ROT6D_DIM:
            dim_name = f"rot6d_{local_dim - 3}"
        else:
            dim_name = "gripper"

        ax.plot(time_index, actions[:, dim_idx], linewidth=1.0)
        ax.set_title(f"{arm_name}_{dim_name}")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    for ax in axes[-4:]:
        ax.set_xlabel("step")

    fig.suptitle("Action Dimension Trajectories", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        left_gripper, left_arm = (
            root["/endpose/left_gripper"][()],
            root["/endpose/left_endpose"][()],
        )
        right_gripper, right_arm = (
            root["/endpose/right_gripper"][()],
            root["/endpose/right_endpose"][()],
        )
        joint_vector = root["/joint_action/vector"][()]

        image_dict = {}
        for cam_name in root["/observation/"].keys():
            image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]

    return left_gripper, left_arm, right_gripper, right_arm, joint_vector, image_dict


def images_encoding(imgs):
    encode_data = []
    max_len = 0
    for img in imgs:
        success, encoded_image = cv2.imencode(".jpg", img)
        if not success:
            raise RuntimeError("Failed to encode image as JPEG.")
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    return encode_data, max_len


def get_task_config(task_name):
    with open(f"./task_config/{task_name}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return args


def data_transform(path, episode_num, save_path, visualize_actions=True):
    begin = 0

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(episode_num):
        desc_type = "seen"
        instruction_data_path = os.path.join(path, "instructions", f"episode{i}.json")
        with open(instruction_data_path, "r") as f_instr:
            instruction_dict = json.load(f_instr)
        instructions = instruction_dict[desc_type]
        save_instructions_json = {"instructions": instructions}

        episode_save_dir = os.path.join(save_path, f"episode_{i}")
        os.makedirs(episode_save_dir, exist_ok=True)

        with open(os.path.join(episode_save_dir, "instructions.json"), "w") as f:
            json.dump(save_instructions_json, f, indent=2)

        left_gripper_all, left_arm_all, right_gripper_all, right_arm_all, joint_vector_all, image_dict = load_hdf5(
            os.path.join(path, "data", f"episode{i}.hdf5")
        )

        qpos = []
        actions = []
        cam_high = []
        cam_right_wrist = []
        cam_left_wrist = []
        left_arm_dim = []
        right_arm_dim = []

        for j in range(left_gripper_all.shape[0]):
            left_gripper, left_arm, right_gripper, right_arm = (
                left_gripper_all[j],
                left_arm_all[j],
                right_gripper_all[j],
                right_arm_all[j],
            )

            state_10d = pose_pair_to_20d(left_arm, left_gripper, right_arm, right_gripper).astype(np.float32)

            if j != left_gripper_all.shape[0] - 1:
                qpos.append(state_10d)

                camera_high_bits = image_dict["head_camera"][j]
                camera_high = cv2.imdecode(np.frombuffer(camera_high_bits, np.uint8), cv2.IMREAD_COLOR)
                cam_high.append(cv2.resize(camera_high, (640, 480)))

                camera_right_wrist_bits = image_dict["right_camera"][j]
                camera_right_wrist = cv2.imdecode(
                    np.frombuffer(camera_right_wrist_bits, np.uint8), cv2.IMREAD_COLOR
                )
                cam_right_wrist.append(cv2.resize(camera_right_wrist, (640, 480)))

                camera_left_wrist_bits = image_dict["left_camera"][j]
                camera_left_wrist = cv2.imdecode(
                    np.frombuffer(camera_left_wrist_bits, np.uint8), cv2.IMREAD_COLOR
                )
                cam_left_wrist.append(cv2.resize(camera_left_wrist, (640, 480)))

            if j != 0:
                actions.append(state_10d)
                left_arm_dim.append(ARM_DIM)
                right_arm_dim.append(ARM_DIM)

        hdf5path = os.path.join(episode_save_dir, f"episode_{i}.hdf5")

        with h5py.File(hdf5path, "w") as f:
            f.create_dataset("action", data=np.asarray(actions, dtype=np.float32))
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=np.asarray(qpos, dtype=np.float32))
            obs.create_dataset("left_arm_dim", data=np.asarray(left_arm_dim, dtype=np.int32))
            obs.create_dataset("right_arm_dim", data=np.asarray(right_arm_dim, dtype=np.int32))
            image = obs.create_group("images")
            cam_high_enc, len_high = images_encoding(cam_high)
            cam_right_wrist_enc, len_right = images_encoding(cam_right_wrist)
            cam_left_wrist_enc, len_left = images_encoding(cam_left_wrist)
            image.create_dataset("cam_high", data=cam_high_enc, dtype=f"S{len_high}")
            image.create_dataset("cam_right_wrist", data=cam_right_wrist_enc, dtype=f"S{len_right}")
            image.create_dataset("cam_left_wrist", data=cam_left_wrist_enc, dtype=f"S{len_left}")

        if visualize_actions:
            visualize_action_dimensions(actions, os.path.join(episode_save_dir, "action_dims.png"))

        begin += 1
        print(f"proccess {i} success!")

    return begin


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process episodes into 10d-per-arm action/state data.")
    parser.add_argument(
        "task_name",
        type=str,
        default="beat_block_hammer",
        help="The name of the task (e.g., beat_block_hammer)",
    )
    parser.add_argument("setting", type=str)
    parser.add_argument(
        "expert_data_num",
        type=int,
        default=50,
        help="Number of episodes to process (e.g., 50)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Optional override for the output directory.",
    )
    parser.add_argument(
        "--skip-action-plot",
        action="store_true",
        help="Skip saving per-episode action dimension plots.",
    )
    args = parser.parse_args()

    task_name = args.task_name
    setting = args.setting
    expert_data_num = args.expert_data_num

    load_dir = os.path.join("../../data", str(task_name), str(setting))

    print(f'read data from path:{os.path.join("data", load_dir)}')

    target_dir = args.save_dir or f"training_data_10d_action/{task_name}-{setting}-{expert_data_num}"
    data_transform(
        load_dir,
        expert_data_num,
        target_dir,
        visualize_actions=not args.skip_action_plot,
    )
