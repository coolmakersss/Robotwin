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


ROT6D_DIM = 6
DEFAULT_ROTATION_SCALE = 0.05
DEFAULT_MOTION_EPS = 1e-5
DEFAULT_MODE_THRESHOLD = 0.6
DEFAULT_CHUNK_HORIZON = 50
DEFAULT_TAIL_HOLD_STEPS = 20
MODE_NAMES = (
    "stabilize",
    "absolute",
    "relative_translation_close",
    "relative_translation_apart",
    "relative_rotation",
    "mixed",
)
RATIO_LABELS = ("absolute", "relative_translation", "relative_rotation")
# Auxiliary labels are aligned with qpos[k] and summarize the future action chunk action[k:k+H].
CTS_DIM_LABELS = (
    "Pa_x",
    "Pa_y",
    "Pa_z",
    "Qa_6d_0",
    "Qa_6d_1",
    "Qa_6d_2",
    "Qa_6d_3",
    "Qa_6d_4",
    "Qa_6d_5",
    "left_gripper",
    "Pr_x",
    "Pr_y",
    "Pr_z",
    "Qr_6d_0",
    "Qr_6d_1",
    "Qr_6d_2",
    "Qr_6d_3",
    "Qr_6d_4",
    "Qr_6d_5",
    "right_gripper",
)


def quaternion_to_rotation_6d(quaternion):
    rotation_matrix = R.from_quat(quaternion).as_matrix()
    return rotation_matrix[:, :2].T.reshape(-1)


def cal_cts(end_pose_vector, qa_last):
    pa = (end_pose_vector[8:11] + end_pose_vector[:3]) / 2.0
    pr = end_pose_vector[8:11] - end_pose_vector[:3]
    r1_true = R.from_quat(end_pose_vector[3:7]).as_matrix()
    r2_true = R.from_quat(end_pose_vector[11:15]).as_matrix()
    rr = r1_true.T @ r2_true
    qr = R.from_matrix(rr).as_quat()

    def mid_rotation_scipy(r1, r2, t=0.5):
        q1 = R.from_matrix(r1).as_quat()
        q2 = R.from_matrix(r2).as_quat()
        if np.dot(q1, q2) < 0.0:
            q2 = -q2
        q_mid = q1
        return q_mid

    qa = mid_rotation_scipy(r1_true, r2_true, t=0.5)
    if np.dot(qa, qa_last) < -1e-4:
        qa = -qa

    return np.concatenate([pa, qa, end_pose_vector[7:8], pr, qr, end_pose_vector[15:16]])


def cts_quat_to_cts_10d(cts_pose_state):
    pa = cts_pose_state[:3]
    qa_6d = quaternion_to_rotation_6d(cts_pose_state[3:7])
    left_gripper = cts_pose_state[7:8]
    pr = cts_pose_state[8:11]
    qr_6d = quaternion_to_rotation_6d(cts_pose_state[11:15])
    right_gripper = cts_pose_state[15:16]
    return np.concatenate([pa, qa_6d, left_gripper, pr, qr_6d, right_gripper], axis=0)


def quaternion_distance(q1, q2):
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    q1 = q1 / max(np.linalg.norm(q1), 1e-12)
    q2 = q2 / max(np.linalg.norm(q2), 1e-12)
    if np.dot(q1, q2) < 0.0:
        q2 = -q2
    return (R.from_quat(q1).inv() * R.from_quat(q2)).magnitude()


def compute_motion_energy(prev_cts_state, next_cts_state, rotation_scale):
    abs_trans = np.linalg.norm(next_cts_state[:3] - prev_cts_state[:3])
    abs_rot = rotation_scale * quaternion_distance(prev_cts_state[3:7], next_cts_state[3:7])
    rel_trans = np.linalg.norm(next_cts_state[8:11] - prev_cts_state[8:11])
    rel_rot = rotation_scale * quaternion_distance(prev_cts_state[11:15], next_cts_state[11:15])
    return np.asarray([abs_trans + abs_rot, rel_trans, rel_rot], dtype=np.float32)


def classify_motion_energy(energy, relative_distance_delta, motion_eps, mode_threshold):
    total_energy = float(np.sum(energy))
    if total_energy < motion_eps:
        return 0, np.zeros(len(RATIO_LABELS), dtype=np.float32), 0.0, energy

    ratio = (energy / total_energy).astype(np.float32)
    dominant_idx = int(np.argmax(ratio))
    if ratio[dominant_idx] >= mode_threshold:
        if dominant_idx == 0:
            mode = 1
        elif dominant_idx == 1:
            mode = 2 if relative_distance_delta < 0.0 else 3
        else:
            mode = 4
    else:
        mode = 5
    return mode, ratio, 1.0, energy


def compute_chunk_mode_ratio(cts_states, start_index, chunk_horizon, rotation_scale, motion_eps, mode_threshold):
    chunk_end = min(start_index + chunk_horizon, len(cts_states) - 1)
    energy = np.zeros(len(RATIO_LABELS), dtype=np.float32)
    for step_idx in range(start_index + 1, chunk_end + 1):
        energy += compute_motion_energy(cts_states[step_idx - 1], cts_states[step_idx], rotation_scale)
    start_relative_distance = np.linalg.norm(cts_states[start_index][8:11])
    end_relative_distance = np.linalg.norm(cts_states[chunk_end][8:11])
    relative_distance_delta = float(end_relative_distance - start_relative_distance)
    return classify_motion_energy(energy, relative_distance_delta, motion_eps, mode_threshold)


def visualize_action_dimensions(actions, save_path):
    actions = np.asarray(actions, dtype=np.float32)
    if actions.size == 0:
        return

    fig, axes = plt.subplots(5, 4, figsize=(20, 16), sharex=True)
    axes = axes.flatten()
    time_index = np.arange(actions.shape[0])

    for dim_idx, ax in enumerate(axes):
        ax.plot(time_index, actions[:, dim_idx], linewidth=1.0)
        ax.set_title(CTS_DIM_LABELS[dim_idx])
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
        end_pose_vector = np.concatenate(
            [
                left_arm,
                left_gripper.reshape(-1, 1),
                right_arm,
                right_gripper.reshape(-1, 1),
            ],
            axis=1,
        )

        image_dict = {}
        for cam_name in root["/observation/"].keys():
            image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]

    return left_gripper, left_arm, right_gripper, right_arm, joint_vector, end_pose_vector, image_dict


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


def data_transform(
    path,
    episode_num,
    save_path,
    visualize_actions=True,
    rotation_scale=DEFAULT_ROTATION_SCALE,
    motion_eps=DEFAULT_MOTION_EPS,
    mode_threshold=DEFAULT_MODE_THRESHOLD,
    chunk_horizon=DEFAULT_CHUNK_HORIZON,
    tail_hold_steps=DEFAULT_TAIL_HOLD_STEPS,
):
    if rotation_scale < 0.0:
        raise ValueError("--rotation-scale must be non-negative.")
    if motion_eps < 0.0:
        raise ValueError("--motion-eps must be non-negative.")
    if not 0.0 < mode_threshold <= 1.0:
        raise ValueError("--mode-threshold must be in (0, 1].")
    if chunk_horizon <= 0:
        raise ValueError("--chunk-horizon must be positive.")
    if tail_hold_steps < 0:
        raise ValueError("--tail-hold-steps must be non-negative.")

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

        (
            left_gripper_all,
            left_arm_all,
            right_gripper_all,
            right_arm_all,
            joint_vector_all,
            end_pose_vector_all,
            image_dict,
        ) = load_hdf5(os.path.join(path, "data", f"episode{i}.hdf5"))

        qpos = []
        actions = []
        cam_high = []
        cam_right_wrist = []
        cam_left_wrist = []
        left_arm_dim = []
        right_arm_dim = []
        mode_labels = []
        ratios = []
        ratio_masks = []
        motion_energies = []
        cts_states = []
        states_10d = []
        qa_last = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        for j in range(left_gripper_all.shape[0]):
            cts_state = cal_cts(end_pose_vector_all[j], qa_last)
            qa_last = cts_state[3:7]
            cts_states.append(cts_state)
            states_10d.append(cts_quat_to_cts_10d(cts_state).astype(np.float32))

        for j, state_10d in enumerate(states_10d):
            if j != left_gripper_all.shape[0] - 1:
                qpos.append(state_10d)
                remaining_steps = len(cts_states) - 1 - j
                if tail_hold_steps > 0 and remaining_steps < tail_hold_steps and mode_labels:
                    aux_mode = mode_labels[-1]
                    aux_ratio = ratios[-1].copy()
                    aux_ratio_mask = ratio_masks[-1]
                    aux_motion_energy = motion_energies[-1].copy()
                else:
                    aux_mode, aux_ratio, aux_ratio_mask, aux_motion_energy = compute_chunk_mode_ratio(
                        cts_states,
                        j,
                        chunk_horizon,
                        rotation_scale,
                        motion_eps,
                        mode_threshold,
                    )
                mode_labels.append(aux_mode)
                ratios.append(aux_ratio)
                ratio_masks.append(aux_ratio_mask)
                motion_energies.append(aux_motion_energy)

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
                left_arm_dim.append(10)
                right_arm_dim.append(10)

        hdf5path = os.path.join(episode_save_dir, f"episode_{i}.hdf5")
        ratio_array = np.asarray(ratios, dtype=np.float32).reshape(-1, len(RATIO_LABELS))
        motion_energy_array = np.asarray(motion_energies, dtype=np.float32).reshape(-1, len(RATIO_LABELS))

        with h5py.File(hdf5path, "w") as f:
            f.create_dataset("action", data=np.asarray(actions, dtype=np.float32))
            f.create_dataset("mode", data=np.asarray(mode_labels, dtype=np.int32))
            f.create_dataset("ratio", data=ratio_array)
            f.create_dataset("ratio_mask", data=np.asarray(ratio_masks, dtype=np.float32))
            f.create_dataset("motion_energy", data=motion_energy_array)
            f.attrs["mode_names"] = json.dumps(MODE_NAMES)
            f.attrs["ratio_labels"] = json.dumps(RATIO_LABELS)
            f.attrs["rotation_scale"] = float(rotation_scale)
            f.attrs["motion_eps"] = float(motion_eps)
            f.attrs["mode_threshold"] = float(mode_threshold)
            f.attrs["chunk_horizon"] = int(chunk_horizon)
            f.attrs["tail_hold_steps"] = int(tail_hold_steps)
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
    parser = argparse.ArgumentParser(
        description="Process episodes into CTS 10d-per-arm data with mode and ratio auxiliary labels."
    )
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
    parser.add_argument(
        "--rotation-scale",
        type=float,
        default=DEFAULT_ROTATION_SCALE,
        help="Meters-per-radian scale used to compare rotation and translation energy.",
    )
    parser.add_argument(
        "--motion-eps",
        type=float,
        default=DEFAULT_MOTION_EPS,
        help="Total motion energy below this value is labeled as stabilize and has ratio_mask=0.",
    )
    parser.add_argument(
        "--mode-threshold",
        type=float,
        default=DEFAULT_MODE_THRESHOLD,
        help=(
            "Minimum ratio for absolute/relative_translation/relative_rotation to be a dominant mode. "
            "Dominant relative_translation is split into close/apart by the chunk distance change."
        ),
    )
    parser.add_argument(
        "--chunk-horizon",
        type=int,
        default=DEFAULT_CHUNK_HORIZON,
        help="Number of future action steps summarized by each mode/ratio label.",
    )
    parser.add_argument(
        "--tail-hold-steps",
        type=int,
        default=DEFAULT_TAIL_HOLD_STEPS,
        help="Reuse the previous label when fewer than this many future action steps remain.",
    )
    args = parser.parse_args()

    task_name = args.task_name
    setting = args.setting
    expert_data_num = args.expert_data_num

    load_dir = os.path.join("../../data", str(task_name), str(setting))

    print(f'read data from path:{os.path.join("data", load_dir)}')

    target_dir = args.save_dir or f"training_data_50_tasks_cts_10d_action_mode_ratio/{task_name}-{setting}-{expert_data_num}"
    data_transform(
        load_dir,
        expert_data_num,
        target_dir,
        visualize_actions=not args.skip_action_plot,
        rotation_scale=args.rotation_scale,
        motion_eps=args.motion_eps,
        mode_threshold=args.mode_threshold,
        chunk_horizon=args.chunk_horizon,
        tail_hold_steps=args.tail_hold_steps,
    )
