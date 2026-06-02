# /mnt/afs/huangdi/xiangenda/.venv/bin/python scripts/convert_realworld_to_pi0_hdf5.py   --source-dir training_data_realworld_sweep_table   --setting realworld   --instruction "Use the two arms to pick up the dustpan and brush respectively, and clear the block from the table"
# /mnt/afs/huangdi/xiangenda/.venv/bin/python scripts/convert_realworld_to_pi0_hdf5.py   --source-dir training_data_realworld_lift_plate   --setting realworld   --instruction "Use both arms to jointly grasp the plate holding a cup, move it steadily to the front of the table, and place it down"
# /mnt/afs/huangdi/xiangenda/.venv/bin/python scripts/convert_realworld_to_pi0_hdf5.py   --source-dir training_data_realworld_carry_basket   --setting realworld   --instruction "Use both arms to place the bottle into the basket, then lift and carry the basket steadily across the table"
# /mnt/afs/huangdi/xiangenda/.venv/bin/python scripts/convert_realworld_to_pi0_hdf5.py   --source-dir training_data_realworld_dual_pour_water   --setting realworld   --instruction "Use one arm to hold the cup and the other arm to pick up the bottle, then pour water from the bottle into the cup"
# /mnt/afs/huangdi/xiangenda/.venv/bin/python scripts/convert_realworld_to_pi0_hdf5.py   --source-dir training_data_realworld_dual_sample_loading   --setting realworld   --instruction "Use one arm to pick up the test tube and hand it over to the other arm, then load it into the test tube rack"

import argparse
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import h5py
import matplotlib
import numpy as np
from scipy.spatial.transform import Rotation as R

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ARM_LABELS = ("left", "right")
ARM_DIM = 10
ROT6D_DIM = 6
TARGET_IMAGE_SIZE = (640, 480)
GRIPPER_SCALE = 70.0 / 100.0

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

TARGET_ROOTS = {
    "eef-10d": "training_data_realworld_10d_action",
    "cts-10d": "training_data_realworld_cts_10d_action",
    "cts-10d-mode-ratio": "training_data_realworld_cts_10d_action_mode_ratio",
}

CAMERA_MAP = {
    "cam_high": "/slave_cam_high/color",
    "cam_left_wrist": "/slave_cam_left/color",
    "cam_right_wrist": "/slave_cam_right/color",
}

REQUIRED_DATASETS = (
    "/slave_left_arm/ee_pose",
    "/slave_right_arm/ee_pose",
    "/slave_left_arm/gripper",
    "/slave_right_arm/gripper",
    *CAMERA_MAP.values(),
)


def numeric_sort_key(path: Path):
    return (0, int(path.stem)) if path.stem.isdigit() else (1, path.name)


def quaternion_to_rotation_6d(quaternion):
    rotation_matrix = R.from_quat(quaternion).as_matrix()
    return rotation_matrix[:, :2].T.reshape(-1)


def convert_rpy_units(rpy, rpy_unit):
    if rpy_unit == "deg1000":
        return rpy * 1000.0, True
    if rpy_unit == "deg":
        return rpy, True
    if rpy_unit == "rad":
        return rpy, False
    raise ValueError(f"Unsupported rpy_unit: {rpy_unit}")


def rpy_pose_to_quat_pose(ee_pose, rpy_order, rpy_unit):
    ee_pose = np.asarray(ee_pose, dtype=np.float64)
    xyz = ee_pose[:, :3]
    rpy, degrees = convert_rpy_units(ee_pose[:, 3:6], rpy_unit)
    quat = R.from_euler(rpy_order, rpy, degrees=degrees).as_quat()
    return np.concatenate([xyz, quat], axis=1)


def pose_to_10d(arm_pose, gripper):
    position = arm_pose[:3]
    rotation_6d = quaternion_to_rotation_6d(arm_pose[3:7])
    return np.concatenate([position, rotation_6d, np.array([gripper])], axis=0)


def pose_pair_to_20d(left_arm, left_gripper, right_arm, right_gripper):
    left_pose = pose_to_10d(left_arm, left_gripper)
    right_pose = pose_to_10d(right_arm, right_gripper)
    return np.concatenate([left_pose, right_pose], axis=0)


def cal_cts(end_pose_vector, qa_last):
    pa = (end_pose_vector[8:11] + end_pose_vector[:3]) / 2.0
    pr = end_pose_vector[8:11] - end_pose_vector[:3]
    r1_true = R.from_quat(end_pose_vector[3:7]).as_matrix()
    r2_true = R.from_quat(end_pose_vector[11:15]).as_matrix()
    rr = r1_true.T @ r2_true
    qr = R.from_matrix(rr).as_quat()

    q1 = R.from_matrix(r1_true).as_quat()
    q2 = R.from_matrix(r2_true).as_quat()
    if np.dot(q1, q2) < 0.0:
        q2 = -q2
    qa = q1
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


def compute_mode_ratio_labels(
    cts_states,
    rotation_scale=DEFAULT_ROTATION_SCALE,
    motion_eps=DEFAULT_MOTION_EPS,
    mode_threshold=DEFAULT_MODE_THRESHOLD,
    chunk_horizon=DEFAULT_CHUNK_HORIZON,
    tail_hold_steps=DEFAULT_TAIL_HOLD_STEPS,
):
    mode_labels = []
    ratios = []
    ratio_masks = []
    motion_energies = []

    for j in range(len(cts_states) - 1):
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

    return (
        np.asarray(mode_labels, dtype=np.int32),
        np.asarray(ratios, dtype=np.float32).reshape(-1, len(RATIO_LABELS)),
        np.asarray(ratio_masks, dtype=np.float32),
        np.asarray(motion_energies, dtype=np.float32).reshape(-1, len(RATIO_LABELS)),
    )


def visualize_action_dimensions(actions, save_path, labels):
    actions = np.asarray(actions, dtype=np.float32)
    if actions.size == 0:
        return

    fig, axes = plt.subplots(5, 4, figsize=(20, 16), sharex=True)
    axes = axes.flatten()
    time_index = np.arange(actions.shape[0])

    for dim_idx, ax in enumerate(axes):
        ax.plot(time_index, actions[:, dim_idx], linewidth=1.0)
        ax.set_title(labels[dim_idx])
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    for ax in axes[-4:]:
        ax.set_xlabel("step")

    fig.suptitle("Action Dimension Trajectories", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def eef_dim_labels():
    labels = []
    for arm_name in ARM_LABELS:
        for local_dim in range(ARM_DIM):
            if local_dim < 3:
                dim_name = f"pos_{local_dim}"
            elif local_dim < 3 + ROT6D_DIM:
                dim_name = f"rot6d_{local_dim - 3}"
            else:
                dim_name = "gripper"
            labels.append(f"{arm_name}_{dim_name}")
    return labels


def read_config(source_dir):
    config_path = source_dir / "config.json"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_instructions(instructions_json, instructions, task_name):
    if instructions_json is not None:
        with Path(instructions_json).open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            loaded = payload
        elif "instructions" in payload:
            loaded = payload["instructions"]
        elif "seen" in payload:
            loaded = payload["seen"]
        else:
            raise ValueError(f"Unsupported instructions JSON format: {instructions_json}")
        return [str(item) for item in loaded]

    if instructions:
        return instructions

    return [task_name.replace("_", " ")]


def check_required_datasets(root, episode_path):
    missing = [key for key in REQUIRED_DATASETS if key not in root]
    if missing:
        raise KeyError(f"{episode_path} is missing required datasets: {missing}")


def get_episode_length(root, episode_path, allow_length_mismatch):
    lengths = {key: int(root[key].shape[0]) for key in REQUIRED_DATASETS}
    unique_lengths = sorted(set(lengths.values()))
    if len(unique_lengths) == 1:
        return unique_lengths[0]
    if not allow_length_mismatch:
        raise ValueError(f"{episode_path} has mismatched dataset lengths: {lengths}")
    trimmed_length = min(unique_lengths)
    print(f"[warn] {episode_path} has mismatched lengths; trimming to {trimmed_length}: {lengths}")
    return trimmed_length


def encode_image(image):
    if image.shape[:2] != (TARGET_IMAGE_SIZE[1], TARGET_IMAGE_SIZE[0]):
        image = cv2.resize(image, TARGET_IMAGE_SIZE)
    success, encoded_image = cv2.imencode(".jpg", image)
    if not success:
        raise RuntimeError("Failed to encode image as JPEG.")
    return encoded_image.tobytes()


def encode_camera_datasets(root, frame_count):
    encoded = {}
    max_lengths = {}
    for target_name, raw_key in CAMERA_MAP.items():
        camera_data = root[raw_key]
        encoded_camera = []
        max_len = 0
        for frame_idx in range(frame_count):
            jpeg_data = encode_image(camera_data[frame_idx])
            encoded_camera.append(jpeg_data)
            max_len = max(max_len, len(jpeg_data))
        encoded[target_name] = encoded_camera
        max_lengths[target_name] = max_len
    return encoded, max_lengths


def load_episode_payload(
    episode_path,
    max_frames,
    rpy_order,
    rpy_unit,
    allow_length_mismatch,
):
    with h5py.File(episode_path, "r") as root:
        check_required_datasets(root, episode_path)
        episode_length = get_episode_length(root, episode_path, allow_length_mismatch)
        if max_frames is not None:
            episode_length = min(episode_length, max_frames)
        if episode_length < 2:
            raise ValueError(f"{episode_path} must contain at least 2 frames, got {episode_length}.")

        left_arm = rpy_pose_to_quat_pose(root["/slave_left_arm/ee_pose"][:episode_length], rpy_order, rpy_unit)
        right_arm = rpy_pose_to_quat_pose(root["/slave_right_arm/ee_pose"][:episode_length], rpy_order, rpy_unit)
        left_gripper = np.asarray(root["/slave_left_arm/gripper"][:episode_length], dtype=np.float64) * GRIPPER_SCALE
        right_gripper = np.asarray(root["/slave_right_arm/gripper"][:episode_length], dtype=np.float64) * GRIPPER_SCALE

        eef_states = []
        end_pose_vectors = []
        for frame_idx in range(episode_length):
            eef_states.append(
                pose_pair_to_20d(
                    left_arm[frame_idx],
                    left_gripper[frame_idx],
                    right_arm[frame_idx],
                    right_gripper[frame_idx],
                ).astype(np.float32)
            )
            end_pose_vectors.append(
                np.concatenate(
                    [
                        left_arm[frame_idx],
                        np.array([left_gripper[frame_idx]]),
                        right_arm[frame_idx],
                        np.array([right_gripper[frame_idx]]),
                    ],
                    axis=0,
                )
            )

        cts_states = []
        cts_10d_states = []
        qa_last = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        for end_pose_vector in end_pose_vectors:
            cts_state = cal_cts(end_pose_vector, qa_last)
            qa_last = cts_state[3:7]
            cts_states.append(cts_state)
            cts_10d_states.append(cts_quat_to_cts_10d(cts_state).astype(np.float32))

        encoded_images, image_max_lengths = encode_camera_datasets(root, episode_length - 1)

    return {
        "eef_states": np.asarray(eef_states, dtype=np.float32),
        "cts_states": cts_states,
        "cts_10d_states": np.asarray(cts_10d_states, dtype=np.float32),
        "encoded_images": encoded_images,
        "image_max_lengths": image_max_lengths,
    }


def split_state_action(states):
    return states[:-1].astype(np.float32), states[1:].astype(np.float32)


def write_episode_hdf5(
    save_dir,
    episode_idx,
    qpos,
    actions,
    encoded_images,
    image_max_lengths,
    instructions,
    mode_payload=None,
    visualize_actions=True,
    labels=None,
):
    episode_save_dir = save_dir / f"episode_{episode_idx}"
    episode_save_dir.mkdir(parents=True, exist_ok=True)

    with (episode_save_dir / "instructions.json").open("w", encoding="utf-8") as f:
        json.dump({"instructions": instructions}, f, indent=2)

    hdf5_path = episode_save_dir / f"episode_{episode_idx}.hdf5"
    with h5py.File(hdf5_path, "w") as f:
        f.create_dataset("action", data=actions.astype(np.float32))

        if mode_payload is not None:
            f.create_dataset("mode", data=mode_payload["mode"])
            f.create_dataset("ratio", data=mode_payload["ratio"])
            f.create_dataset("ratio_mask", data=mode_payload["ratio_mask"])
            f.create_dataset("motion_energy", data=mode_payload["motion_energy"])
            f.attrs["mode_names"] = json.dumps(MODE_NAMES)
            f.attrs["ratio_labels"] = json.dumps(RATIO_LABELS)
            f.attrs["rotation_scale"] = float(mode_payload["rotation_scale"])
            f.attrs["motion_eps"] = float(mode_payload["motion_eps"])
            f.attrs["mode_threshold"] = float(mode_payload["mode_threshold"])
            f.attrs["chunk_horizon"] = int(mode_payload["chunk_horizon"])
            f.attrs["tail_hold_steps"] = int(mode_payload["tail_hold_steps"])

        obs = f.create_group("observations")
        obs.create_dataset("qpos", data=qpos.astype(np.float32))
        obs.create_dataset("left_arm_dim", data=np.full(actions.shape[0], ARM_DIM, dtype=np.int32))
        obs.create_dataset("right_arm_dim", data=np.full(actions.shape[0], ARM_DIM, dtype=np.int32))

        image_group = obs.create_group("images")
        for camera_name in ("cam_high", "cam_right_wrist", "cam_left_wrist"):
            image_group.create_dataset(
                camera_name,
                data=encoded_images[camera_name],
                dtype=f"S{image_max_lengths[camera_name]}",
            )

    if visualize_actions and labels is not None:
        visualize_action_dimensions(actions, episode_save_dir / "action_dims.png", labels)


def validate_mode_ratio_args(args):
    if args.rotation_scale < 0.0:
        raise ValueError("--rotation-scale must be non-negative.")
    if args.motion_eps < 0.0:
        raise ValueError("--motion-eps must be non-negative.")
    if not 0.0 < args.mode_threshold <= 1.0:
        raise ValueError("--mode-threshold must be in (0, 1].")
    if args.chunk_horizon <= 0:
        raise ValueError("--chunk-horizon must be positive.")
    if args.tail_hold_steps < 0:
        raise ValueError("--tail-hold-steps must be non-negative.")


def selected_modes(mode):
    if mode == "all":
        return ("eef-10d", "cts-10d", "cts-10d-mode-ratio")
    return (mode,)


def prepare_output_dirs(output_root, output_name, modes, overwrite):
    output_dirs = {}
    for mode in modes:
        save_dir = output_root / TARGET_ROOTS[mode] / output_name
        if save_dir.exists():
            if not overwrite:
                raise FileExistsError(f"{save_dir} already exists. Pass --overwrite to replace it.")
            shutil.rmtree(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        output_dirs[mode] = save_dir
    return output_dirs


def convert_dataset(args):
    validate_mode_ratio_args(args)
    if args.rpy_degrees:
        args.rpy_unit = "deg"

    source_dir = args.source_dir
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    config = read_config(source_dir)
    task_name = args.task_name or config.get("task_name") or source_dir.name.replace("training_data_realworld_", "")
    instructions = load_instructions(args.instructions_json, args.instruction, task_name)

    episode_files = sorted(source_dir.glob("*.hdf5"), key=numeric_sort_key)
    if args.episode_start:
        episode_files = episode_files[args.episode_start :]
    if args.num_episodes is not None:
        episode_files = episode_files[: args.num_episodes]
    if not episode_files:
        raise FileNotFoundError(f"No .hdf5 episode files found in {source_dir}")

    modes = selected_modes(args.mode)
    output_name = args.output_name or f"{task_name}-{args.setting}-{len(episode_files)}"
    output_dirs = prepare_output_dirs(args.output_root, output_name, modes, args.overwrite)

    print(f"source_dir: {source_dir}")
    print(f"episodes: {len(episode_files)}")
    print(f"task_name: {task_name}")
    print(f"rpy_order: {args.rpy_order}")
    print(f"rpy_unit: {args.rpy_unit}")
    print(f"instructions: {instructions}")
    for mode, save_dir in output_dirs.items():
        print(f"{mode} -> {save_dir}")

    eef_labels = eef_dim_labels()
    cts_labels = list(CTS_DIM_LABELS)

    for out_episode_idx, episode_path in enumerate(episode_files):
        payload = load_episode_payload(
            episode_path,
            max_frames=args.max_frames,
            rpy_order=args.rpy_order,
            rpy_unit=args.rpy_unit,
            allow_length_mismatch=args.allow_length_mismatch,
        )

        if "eef-10d" in output_dirs:
            qpos, actions = split_state_action(payload["eef_states"])
            write_episode_hdf5(
                output_dirs["eef-10d"],
                out_episode_idx,
                qpos,
                actions,
                payload["encoded_images"],
                payload["image_max_lengths"],
                instructions,
                visualize_actions=not args.skip_action_plot,
                labels=eef_labels,
            )

        if "cts-10d" in output_dirs:
            qpos, actions = split_state_action(payload["cts_10d_states"])
            write_episode_hdf5(
                output_dirs["cts-10d"],
                out_episode_idx,
                qpos,
                actions,
                payload["encoded_images"],
                payload["image_max_lengths"],
                instructions,
                visualize_actions=not args.skip_action_plot,
                labels=cts_labels,
            )

        if "cts-10d-mode-ratio" in output_dirs:
            qpos, actions = split_state_action(payload["cts_10d_states"])
            mode, ratio, ratio_mask, motion_energy = compute_mode_ratio_labels(
                payload["cts_states"],
                rotation_scale=args.rotation_scale,
                motion_eps=args.motion_eps,
                mode_threshold=args.mode_threshold,
                chunk_horizon=args.chunk_horizon,
                tail_hold_steps=args.tail_hold_steps,
            )
            mode_payload = {
                "mode": mode,
                "ratio": ratio,
                "ratio_mask": ratio_mask,
                "motion_energy": motion_energy,
                "rotation_scale": args.rotation_scale,
                "motion_eps": args.motion_eps,
                "mode_threshold": args.mode_threshold,
                "chunk_horizon": args.chunk_horizon,
                "tail_hold_steps": args.tail_hold_steps,
            }
            write_episode_hdf5(
                output_dirs["cts-10d-mode-ratio"],
                out_episode_idx,
                qpos,
                actions,
                payload["encoded_images"],
                payload["image_max_lengths"],
                instructions,
                mode_payload=mode_payload,
                visualize_actions=not args.skip_action_plot,
                labels=cts_labels,
            )

        print(f"process {episode_path.name} -> episode_{out_episode_idx} success")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Convert real-world dual-arm HDF5 files with xyz+rpy ee_pose into the pi0/RoboTwin "
            "10D, CTS-10D, and CTS-10D-mode-ratio HDF5 layouts."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("training_data_realworld_sweep_table"),
        help="Directory containing flat real-world episode files such as 0.hdf5, 1.hdf5, ...",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("."),
        help="Directory under which training_data_50_tasks_* target roots are created.",
    )
    parser.add_argument(
        "--mode",
        choices=("all", "eef-10d", "cts-10d", "cts-10d-mode-ratio"),
        default="all",
        help="Which target format to write.",
    )
    parser.add_argument("--task-name", type=str, default=None, help="Task name used in the output directory name.")
    parser.add_argument("--setting", type=str, default="realworld", help="Setting name used in the output directory.")
    parser.add_argument("--output-name", type=str, default=None, help="Override target subdirectory name.")
    parser.add_argument("--num-episodes", type=int, default=None, help="Convert only the first N selected episodes.")
    parser.add_argument("--episode-start", type=int, default=0, help="Skip this many sorted source episodes first.")
    parser.add_argument("--max-frames", type=int, default=None, help="Debug option: cap frames per episode.")
    parser.add_argument(
        "--instruction",
        action="append",
        default=None,
        help="Instruction string to write. Can be passed multiple times.",
    )
    parser.add_argument(
        "--instructions-json",
        type=Path,
        default=None,
        help="JSON list, {'instructions': [...]}, or {'seen': [...]} used for every converted episode.",
    )
    parser.add_argument("--rpy-order", type=str, default="xyz", help="Euler axis order for ee_pose[:, 3:6].")
    parser.add_argument(
        "--rpy-unit",
        choices=("deg1000", "deg", "rad"),
        default="deg1000",
        help=(
            "Unit of ee_pose[:, 3:6]. Real Piper data is typically degree/1000: "
            "0.18 means 180 degrees."
        ),
    )
    parser.add_argument(
        "--rpy-degrees",
        action="store_true",
        help="Deprecated alias for --rpy-unit deg. Kept for old commands.",
    )
    parser.add_argument("--allow-length-mismatch", action="store_true", help="Trim datasets to the shortest length.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing target subdirectory.")
    parser.add_argument("--skip-action-plot", action="store_true", help="Skip saving per-episode action plots.")
    parser.add_argument("--rotation-scale", type=float, default=DEFAULT_ROTATION_SCALE)
    parser.add_argument("--motion-eps", type=float, default=DEFAULT_MOTION_EPS)
    parser.add_argument("--mode-threshold", type=float, default=DEFAULT_MODE_THRESHOLD)
    parser.add_argument("--chunk-horizon", type=int, default=DEFAULT_CHUNK_HORIZON)
    parser.add_argument("--tail-hold-steps", type=int, default=DEFAULT_TAIL_HOLD_STEPS)
    return parser


if __name__ == "__main__":
    convert_dataset(build_parser().parse_args())
