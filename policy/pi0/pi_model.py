#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
import json
import sys
import jax
import numpy as np
from openpi.models import model as _model
from openpi.policies import aloha_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

import cv2, os
from PIL import Image

from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


class PI0:

    def __init__(self, train_config_name, model_name, checkpoint_id, pi0_step):
        self.train_config_name = train_config_name
        self.model_name = model_name
        self.checkpoint_id = checkpoint_id

        config = _config.get_config(self.train_config_name)
        checkpoint_dir = self._resolve_checkpoint_dir()
        specified_path = os.path.join(checkpoint_dir, "assets")
        entries = os.listdir(specified_path)
        assets_id = entries[0]

        self.policy = _policy_config.create_trained_policy(
            config,
            checkpoint_dir,
            robotwin_repo_id=assets_id)
        print("loading model success!")
        self.img_size = (224, 224)
        self.observation_window = None
        self.pi0_step = pi0_step

    def _resolve_checkpoint_dir(self):
        checkpoint_root = os.path.join(
            "policy", "pi0", "checkpoints", self.train_config_name, self.model_name
        )
        requested_dir = os.path.join(checkpoint_root, str(self.checkpoint_id))

        if os.path.isdir(os.path.join(requested_dir, "assets")):
            return requested_dir

        available_steps = sorted(
            [
                step_name
                for step_name in os.listdir(checkpoint_root)
                if step_name.isdigit() and os.path.isdir(os.path.join(checkpoint_root, step_name, "assets"))
            ],
            key=int,
        )

        if not available_steps:
            raise FileNotFoundError(
                f"No valid checkpoint with assets found under '{checkpoint_root}'."
            )

        fallback_step = available_steps[-1]
        print(
            f"checkpoint {self.checkpoint_id} not found under '{checkpoint_root}', "
            f"falling back to latest available checkpoint {fallback_step}."
        )
        self.checkpoint_id = int(fallback_step)
        return os.path.join(checkpoint_root, fallback_step)

    # set img_size
    def set_img_size(self, img_size):
        self.img_size = img_size

    # set language randomly
    def set_language(self, instruction):
        self.instruction = instruction
        print(f"successfully set instruction:{instruction}")

    # Update the observation window buffer
    def update_observation_window(self, img_arr, state):
        img_front, img_right, img_left, puppet_arm = (
            img_arr[0],
            img_arr[1],
            img_arr[2],
            state,
        )
        img_front = np.transpose(img_front, (2, 0, 1))
        img_right = np.transpose(img_right, (2, 0, 1))
        img_left = np.transpose(img_left, (2, 0, 1))

        self.observation_window = {
            "state": state,
            "images": {
                "cam_high": img_front,
                "cam_left_wrist": img_left,
                "cam_right_wrist": img_right,
            },
            "prompt": self.instruction,
        }

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first!"
        return self.policy.infer(self.observation_window)["actions"]

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        print("successfully unset obs and language intruction")
