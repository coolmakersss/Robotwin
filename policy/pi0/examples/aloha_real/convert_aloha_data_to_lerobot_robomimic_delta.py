"""
Convert processed robomimic Transport episodes to LeRobot v2 format.

Both observation.state and action are 14d:
  state  = per-arm absolute xyz + rotation-vector + gripper-open amount
  action = per-arm original robomimic delta xyz + delta rotation-vector + gripper command
"""

import fnmatch
import os
from pathlib import Path
import shutil
from typing import Literal

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tyro

from convert_aloha_data_to_lerobot_robotwin import DEFAULT_DATASET_CONFIG
from convert_aloha_data_to_lerobot_robotwin import DatasetConfig
from convert_aloha_data_to_lerobot_robotwin import populate_dataset


STATE_NAMES = [
    "left_x",
    "left_y",
    "left_z",
    "left_rotation_vector_x",
    "left_rotation_vector_y",
    "left_rotation_vector_z",
    "left_gripper_open_amount",
    "right_x",
    "right_y",
    "right_z",
    "right_rotation_vector_x",
    "right_rotation_vector_y",
    "right_rotation_vector_z",
    "right_gripper_open_amount",
]

ACTION_NAMES = [
    "left_delta_x",
    "left_delta_y",
    "left_delta_z",
    "left_delta_rotation_x",
    "left_delta_rotation_y",
    "left_delta_rotation_z",
    "left_gripper_command",
    "right_delta_x",
    "right_delta_y",
    "right_delta_z",
    "right_delta_rotation_x",
    "right_delta_rotation_y",
    "right_delta_rotation_z",
    "right_gripper_command",
]

CAMERAS = [
    "cam_high",
    "cam_left_wrist",
    "cam_right_wrist",
]


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "image",
    *,
    fps: int = 20,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(STATE_NAMES),),
            "names": [STATE_NAMES],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(ACTION_NAMES),),
            "names": [ACTION_NAMES],
        },
    }
    for camera in CAMERAS:
        features[f"observation.images.{camera}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        }

    if Path(HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def episode_sort_key(path: Path) -> tuple[int, str]:
    try:
        episode_index = int(path.parent.name.rsplit("_", 1)[-1])
    except ValueError:
        episode_index = 0
    return episode_index, str(path)


def port_aloha(
    raw_dir: Path,
    repo_id: str,
    task: str = "DEBUG",
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "image",
    fps: int = 20,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> None:
    if not raw_dir.exists():
        raise FileNotFoundError(raw_dir)

    hdf5_files = []
    for root, _, files in os.walk(raw_dir):
        for filename in fnmatch.filter(files, "*.hdf5"):
            hdf5_files.append(Path(root) / filename)
    hdf5_files.sort(key=episode_sort_key)
    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 episodes found under {raw_dir}.")

    dataset = create_empty_dataset(
        repo_id,
        robot_type="mobile_aloha" if is_mobile else "aloha",
        mode=mode,
        fps=fps,
        dataset_config=dataset_config,
    )
    dataset = populate_dataset(dataset, hdf5_files, task=task, episodes=episodes)

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    tyro.cli(port_aloha)
