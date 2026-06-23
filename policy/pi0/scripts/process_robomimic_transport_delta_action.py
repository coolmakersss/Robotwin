import argparse
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R


DEFAULT_INPUT = "/mnt/afs/huangdi/xiangenda/robomimic/datasets/transport/ph/image_v15.hdf5"
DEFAULT_OUTPUT = "training_data_robomimic_transport_delta_action"
DEFAULT_INSTRUCTIONS = [
    "Transport the payload from the start bin to the target bin, move the payload into the target bin while clearing the obstacle, and use both arms to complete the transport task.",
]

ARM_DIM = 7
IMAGE_SIZE = (640, 480)
GRIPPER_OPEN_WIDTH = 0.08


def gripper_qpos_to_open_amount(qpos: np.ndarray) -> np.ndarray:
    width = qpos[:, 0] - qpos[:, 1]
    return np.clip(width / GRIPPER_OPEN_WIDTH, 0.0, 1.0)


def arm_state_to_7d(position: np.ndarray, quaternion: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    rotation_vector = R.from_quat(quaternion).as_rotvec()
    return np.concatenate([position, rotation_vector, gripper[:, None]], axis=1).astype(np.float32)


def demo_state_to_14d(demo: h5py.Group, robot0_is_left: bool) -> np.ndarray:
    obs = demo["obs"]
    robot0 = arm_state_to_7d(
        obs["robot0_eef_pos"][()],
        obs["robot0_eef_quat"][()],
        gripper_qpos_to_open_amount(obs["robot0_gripper_qpos"][()]),
    )
    robot1 = arm_state_to_7d(
        obs["robot1_eef_pos"][()],
        obs["robot1_eef_quat"][()],
        gripper_qpos_to_open_amount(obs["robot1_gripper_qpos"][()]),
    )
    arms = (robot0, robot1) if robot0_is_left else (robot1, robot0)
    return np.concatenate(arms, axis=1).astype(np.float32)


def demo_delta_action_to_14d(demo: h5py.Group, robot0_is_left: bool) -> np.ndarray:
    actions = np.asarray(demo["actions"][()], dtype=np.float32)
    expected_dim = 2 * ARM_DIM
    if actions.ndim != 2 or actions.shape[1] != expected_dim:
        raise ValueError(f"Expected raw robomimic actions with shape [T, {expected_dim}], got {actions.shape}.")

    robot0 = actions[:, :ARM_DIM]
    robot1 = actions[:, ARM_DIM:]
    arms = (robot0, robot1) if robot0_is_left else (robot1, robot0)
    return np.concatenate(arms, axis=1).astype(np.float32)


def encode_images(images: np.ndarray) -> tuple[list[bytes], int]:
    encoded_images = []
    max_len = 0
    for image in images:
        resized = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
        success, encoded = cv2.imencode(".jpg", resized)
        if not success:
            raise RuntimeError("Failed to encode image as JPEG.")
        data = encoded.tobytes()
        encoded_images.append(data)
        max_len = max(max_len, len(data))
    return encoded_images, max_len


def sorted_demo_names(root: h5py.File, mask: str | None) -> list[str]:
    if mask is not None:
        if "mask" not in root or mask not in root["mask"]:
            raise KeyError(f"Mask '{mask}' not found in {root.filename}.")
        names = [
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in root["mask"][mask][()]
        ]
    else:
        names = list(root["data"].keys())
    return sorted(names, key=lambda name: int(name.split("_")[-1]))


def write_instructions(path: Path, instructions: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump({"instructions": instructions}, f, indent=2)


def write_episode(
    demo: h5py.Group,
    episode_index: int,
    output_dir: Path,
    instructions: list[str],
    robot0_is_left: bool,
) -> int:
    state = demo_state_to_14d(demo, robot0_is_left=robot0_is_left)
    actions = demo_delta_action_to_14d(demo, robot0_is_left=robot0_is_left)
    if state.shape != actions.shape:
        raise ValueError(f"State/action shape mismatch in {demo.name}: state={state.shape}, action={actions.shape}.")

    obs = demo["obs"]
    if robot0_is_left:
        left_wrist = obs["robot0_eye_in_hand_image"][()]
        right_wrist = obs["robot1_eye_in_hand_image"][()]
    else:
        left_wrist = obs["robot1_eye_in_hand_image"][()]
        right_wrist = obs["robot0_eye_in_hand_image"][()]
    images = {
        "cam_high": obs["shouldercamera0_image"][()],
        "cam_left_wrist": left_wrist,
        "cam_right_wrist": right_wrist,
    }
    for camera_name, image_array in images.items():
        if image_array.shape[0] != state.shape[0]:
            raise ValueError(
                f"State/image length mismatch in {demo.name}/{camera_name}: "
                f"state={state.shape[0]}, images={image_array.shape[0]}."
            )

    episode_dir = output_dir / f"episode_{episode_index}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    write_instructions(episode_dir / "instructions.json", instructions)

    hdf5_path = episode_dir / f"episode_{episode_index}.hdf5"
    with h5py.File(hdf5_path, "w") as f:
        f.create_dataset("action", data=actions)
        observations = f.create_group("observations")
        observations.create_dataset("qpos", data=state)
        observations.create_dataset("left_arm_dim", data=np.full(state.shape[0], ARM_DIM, dtype=np.int32))
        observations.create_dataset("right_arm_dim", data=np.full(state.shape[0], ARM_DIM, dtype=np.int32))

        image_group = observations.create_group("images")
        for target_name, image_array in images.items():
            encoded, max_len = encode_images(image_array)
            image_group.create_dataset(target_name, data=encoded, dtype=f"S{max_len}")

    return state.shape[0]


def parse_instructions(raw: str | None) -> list[str]:
    if raw is None:
        return DEFAULT_INSTRUCTIONS
    path = Path(raw)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            loaded = loaded.get("instructions", loaded.get("seen", loaded.get("prompts")))
        if not isinstance(loaded, list) or not all(isinstance(item, str) for item in loaded):
            raise ValueError("Instruction file must contain a string list or an object with an instructions list.")
        return loaded
    return [item.strip() for item in raw.split("|") if item.strip()]


def convert(args: argparse.Namespace) -> None:
    input_hdf5 = Path(args.input_hdf5)
    output_dir = Path(args.save_dir)
    instructions = parse_instructions(args.instructions)

    if not input_hdf5.is_file():
        raise FileNotFoundError(input_hdf5)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    with h5py.File(input_hdf5, "r") as root:
        demo_names = sorted_demo_names(root, args.mask)
        if args.max_episodes is not None:
            demo_names = demo_names[: args.max_episodes]

        for episode_index, demo_name in enumerate(demo_names):
            frames = write_episode(
                root["data"][demo_name],
                episode_index=episode_index,
                output_dir=output_dir,
                instructions=instructions,
                robot0_is_left=args.robot0_is_left,
            )
            total_frames += frames
            print(f"processed {demo_name} -> episode_{episode_index} ({frames} frames)")

    print(f"wrote {len(demo_names)} episodes and {total_frames} frames to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert robomimic Transport image_v15.hdf5 to 14d EEF state plus the original "
            "14d robomimic delta action intermediate format."
        )
    )
    parser.add_argument("--input-hdf5", default=DEFAULT_INPUT)
    parser.add_argument("--save-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--instructions", default=None, help="A JSON file or 'prompt one|prompt two' string.")
    parser.add_argument("--mask", default=None, help="Optional robomimic mask to use, e.g. train or valid.")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--robot0-is-left",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Map robomimic robot0 actions/states/camera to the left arm and robot1 to the right arm.",
    )
    args = parser.parse_args()
    convert(args)


if __name__ == "__main__":
    main()
