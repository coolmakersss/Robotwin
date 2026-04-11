from typing import Dict
import numba
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler,
    get_val_mask,
    downsample_mask,
)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer
import pdb


class RobotImageDataset(BaseImageDataset):

    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        batch_size=128,
        max_train_episodes=None,
    ):

        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            # keys=['head_camera', 'front_camera', 'left_camera', 'right_camera', 'state', 'action'],
            # keys=["head_camera", "state", "action"],
            keys=['head_camera', 'left_camera', 'right_camera', 'joint_state', 'end_state', 'cts_state', 'joint_action', 'action', 'cts_action', 'delta_action', 'delta_cts_action'],
        )

        val_mask = get_val_mask(n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        self.batch_size = batch_size
        sequence_length = self.sampler.sequence_length
        self.buffers = {
            k: np.zeros((batch_size, sequence_length, *v.shape[1:]), dtype=v.dtype)
            for k, v in self.sampler.replay_buffer.items()
        }
        self.buffers_torch = {k: torch.from_numpy(v) for k, v in self.buffers.items()}
        for v in self.buffers_torch.values():
            v.pin_memory()

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def _build_chunk_delta_action(self, ref_offset: int = 0) -> np.ndarray:
        """
        chunk_delta[t] = action[t] - agent_pos[t0]
        t0 = window_start + ref_offset
        ref_offset=0 表示用窗口第一帧作基准
        """
        action = self.replay_buffer["action"]          # [N, Da]
        agent_pos = self.replay_buffer["end_state"]    # [N, Ds]
        episode_ends = self.replay_buffer.episode_ends # [E]

        Da = action.shape[-1]
        assert agent_pos.shape[-1] >= Da, \
            f"end_state dim({agent_pos.shape[-1]}) < action dim({Da})"

        H = self.horizon
        chunks = []
        ep_start = 0
        for ep_end in episode_ends:
            ep_action = action[ep_start:ep_end]                 # [L, Da]
            ep_pos = agent_pos[ep_start:ep_end, :Da]            # [L, Da]
            L = ep_action.shape[0]

            if L >= H:
                for s in range(0, L - H + 1):
                    t0 = s + ref_offset
                    if t0 >= s + H:
                        continue
                    base = ep_pos[t0]                           # [Da]
                    chunk = ep_action[s:s + H] - base[None, :] # [H, Da]
                    chunks.append(chunk)

            ep_start = ep_end

        if len(chunks) == 0:
            raise RuntimeError("No valid chunks found for chunk_delta_action normalizer.")

        # 展平成 [M*H, Da]，方便 LinearNormalizer(last_n_dims=1) 统计
        return np.concatenate(chunks, axis=0)

    def _build_chunk_mixed_action(self, ref_offset: int = 0) -> np.ndarray:
        """
        mixed_action[t]:
        - left xyz  (0:3)   = action[t, 0:3]   - agent_pos[t0, 0:3]
        - right xyz (8:11)  = action[t, 8:11]  - agent_pos[t0, 8:11]
        - 其余维度保持 action 绝对值 (quat + gripper)
        """
        action = self.replay_buffer["action"]          # [N, 16]
        agent_pos = self.replay_buffer["end_state"]    # [N, >=16]
        episode_ends = self.replay_buffer.episode_ends # [E]

        Da = action.shape[-1]
        assert Da == 16, f"Expected action dim=16, got {Da}"
        assert agent_pos.shape[-1] >= Da, \
            f"end_state dim({agent_pos.shape[-1]}) < action dim({Da})"

        H = self.horizon
        chunks = []
        ep_start = 0
        for ep_end in episode_ends:
            ep_action = action[ep_start:ep_end]         # [L,16]
            ep_pos = agent_pos[ep_start:ep_end, :Da]    # [L,16]
            L = ep_action.shape[0]

            if L >= H:
                for s in range(0, L - H + 1):
                    t0 = s + ref_offset
                    if t0 >= s + H:
                        continue
                    base = ep_pos[t0]                   # [16]

                    chunk = ep_action[s:s + H].copy()   # [H,16]
                    chunk[:, 0:3] -= base[0:3]          # left xyz relative
                    chunk[:, 8:11] -= base[8:11]        # right xyz relative
                    chunks.append(chunk)

            ep_start = ep_end

        if len(chunks) == 0:
            raise RuntimeError("No valid chunks found for chunk_mixed_action normalizer.")

        return np.concatenate(chunks, axis=0)  # [M*H,16]


    def _build_chunk_mixed_cts_action(self, ref_offset: int = 0) -> np.ndarray:
        """
        mixed_action[t]:
        - left xyz  (0:3)   = action[t, 0:3]   - agent_pos[t0, 0:3]
        - right xyz (8:11)  = action[t, 8:11]  - agent_pos[t0, 8:11]
        - 其余维度保持 action 绝对值 (quat + gripper)
        """
        action = self.replay_buffer["cts_action"]          # [N, 16]
        agent_pos = self.replay_buffer["cts_state"]    # [N, >=16]
        episode_ends = self.replay_buffer.episode_ends # [E]

        Da = action.shape[-1]
        assert Da == 16, f"Expected action dim=16, got {Da}"
        assert agent_pos.shape[-1] >= Da, \
            f"end_state dim({agent_pos.shape[-1]}) < action dim({Da})"

        H = self.horizon
        chunks = []
        ep_start = 0
        for ep_end in episode_ends:
            ep_action = action[ep_start:ep_end]         # [L,16]
            ep_pos = agent_pos[ep_start:ep_end, :Da]    # [L,16]
            L = ep_action.shape[0]

            if L >= H:
                for s in range(0, L - H + 1):
                    t0 = s + ref_offset
                    if t0 >= s + H:
                        continue
                    base = ep_pos[t0]                   # [16]

                    chunk = ep_action[s:s + H].copy()   # [H,16]
                    chunk[:, 0:3] -= base[0:3]          # left xyz relative
                    chunk[:, 7:10] -= base[7:10]        # right xyz relative
                    chunks.append(chunk)

            ep_start = ep_end

        if len(chunks) == 0:
            raise RuntimeError("No valid chunks found for chunk_mixed_action normalizer.")

        return np.concatenate(chunks, axis=0)  # [M*H,16]

        
    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "joint_action": self.replay_buffer["joint_action"],
            "action": self.replay_buffer["action"],
            "delta_action": self.replay_buffer["delta_action"],
            "chunk_delta_action": self._build_chunk_mixed_action(ref_offset=0),   # 新增
            "chunk_delta_cts_action": self._build_chunk_mixed_cts_action(ref_offset=0),   # 新增
            "cts_action": self.replay_buffer["cts_action"],
            "delta_cts_action": self.replay_buffer["delta_cts_action"],
            "joint_pos": self.replay_buffer["joint_state"],
            "agent_pos": self.replay_buffer["end_state"],
            "cts_pos": self.replay_buffer["cts_state"]
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer["head_cam"] = get_image_range_normalizer()
        normalizer["front_cam"] = get_image_range_normalizer()
        normalizer["left_cam"] = get_image_range_normalizer()
        normalizer["right_cam"] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        joint_pos = sample["joint_state"].astype(np.float32)  # (agent_posx2, block_posex3)
        agent_pos = sample["end_state"].astype(np.float32)  # (agent_posx2, block_posex3)
        cts_pos = sample["cts_state"].astype(np.float32)  # (agent_posx2, block_posex3)
        head_cam = np.moveaxis(sample["head_camera"], -1, 1) / 255
        # front_cam = np.moveaxis(sample['front_camera'],-1,1)/255
        left_cam = np.moveaxis(sample['left_camera'],-1,1)/255
        right_cam = np.moveaxis(sample['right_camera'],-1,1)/255

        data = {
            "obs": {
                "head_cam": head_cam,  # T, 3, H, W
                # 'front_cam': front_cam, # T, 3, H, W
                'left_cam': left_cam, # T, 3, H, W
                'right_cam': right_cam, # T, 3, H, W
                'joint_pos': joint_pos, # T, D
                "agent_pos": agent_pos,  # T, D
                "cts_pos": cts_pos
            },
            "joint_action": sample["joint_action"].astype(np.float32),  # T, D
            "action": sample["action"].astype(np.float32),  # T, D
            "delta_action": sample["delta_action"].astype(np.float32),  # T, D
            "cts_action": sample["cts_action"].astype(np.float32),  # T, D
            "delta_cts_action": sample["delta_cts_action"].astype(np.float32),  # T, D
        }
        return data

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        if isinstance(idx, slice):
            raise NotImplementedError  # Specialized
        elif isinstance(idx, int):
            sample = self.sampler.sample_sequence(idx)
            sample = dict_apply(sample, torch.from_numpy)
            return sample
        elif isinstance(idx, np.ndarray):
            assert len(idx) == self.batch_size
            for k, v in self.sampler.replay_buffer.items():
                batch_sample_sequence(
                    self.buffers[k],
                    v,
                    self.sampler.indices,
                    idx,
                    self.sampler.sequence_length,
                )
            return self.buffers_torch
        else:
            raise ValueError(idx)

    def postprocess(self, samples, device):
        joint_pos = samples["joint_state"].to(device, non_blocking=True)
        agent_pos = samples["end_state"].to(device, non_blocking=True)
        cts_pos = samples["cts_state"].to(device, non_blocking=True)
        head_cam = samples["head_camera"].to(device, non_blocking=True) / 255.0
        # front_cam = samples['front_camera'].to(device, non_blocking=True) / 255.0
        left_cam = samples['left_camera'].to(device, non_blocking=True) / 255.0
        right_cam = samples['right_camera'].to(device, non_blocking=True) / 255.0
        joint_action = samples["joint_action"].to(device, non_blocking=True)
        action = samples["action"].to(device, non_blocking=True)
        delta_action = samples["delta_action"].to(device, non_blocking=True)
        cts_action = samples["cts_action"].to(device, non_blocking=True)
        delta_cts_action = samples["delta_cts_action"].to(device, non_blocking=True)
        return {
            "obs": {
                "head_cam": head_cam,  # B, T, 3, H, W
                # 'front_cam': front_cam, # B, T, 3, H, W
                'left_cam': left_cam, # B, T, 3, H, W
                'right_cam': right_cam, # B, T, 3, H, W
                'joint_pos': joint_pos, # T, D
                "agent_pos": agent_pos,  # B, T, D
                "cts_pos": cts_pos,  # B, T, D
            },
            "joint_action": joint_action,  # B, T, D
            "action": action,  # B, T, D
            "delta_action": delta_action,  # B, T, D
            "cts_action": cts_action,  # B, T, D
            "delta_cts_action": delta_cts_action,  # B, T, D
        }


def _batch_sample_sequence(
    data: np.ndarray,
    input_arr: np.ndarray,
    indices: np.ndarray,
    idx: np.ndarray,
    sequence_length: int,
):
    for i in numba.prange(len(idx)):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = indices[idx[i]]
        data[i, sample_start_idx:sample_end_idx] = input_arr[buffer_start_idx:buffer_end_idx]
        if sample_start_idx > 0:
            data[i, :sample_start_idx] = data[i, sample_start_idx]
        if sample_end_idx < sequence_length:
            data[i, sample_end_idx:] = data[i, sample_end_idx - 1]


_batch_sample_sequence_sequential = numba.jit(_batch_sample_sequence, nopython=True, parallel=False)
_batch_sample_sequence_parallel = numba.jit(_batch_sample_sequence, nopython=True, parallel=True)


def batch_sample_sequence(
    data: np.ndarray,
    input_arr: np.ndarray,
    indices: np.ndarray,
    idx: np.ndarray,
    sequence_length: int,
):
    batch_size = len(idx)
    assert data.shape == (batch_size, sequence_length, *input_arr.shape[1:])
    if batch_size >= 16 and data.nbytes // batch_size >= 2**16:
        _batch_sample_sequence_parallel(data, input_arr, indices, idx, sequence_length)
    else:
        _batch_sample_sequence_sequential(data, input_arr, indices, idx, sequence_length)
