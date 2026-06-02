from __future__ import annotations

import argparse
import base64
import dataclasses
import json
from pathlib import Path
import signal
import socket
import time
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from challenge_deploy.config import load_config, set_by_dotted_path
from challenge_deploy.piper import DualPiperSystem
from challenge_deploy.realsense import RealSenseRig
from challenge_deploy.recording import OpenPiRolloutRecorder, RecordingSchema, preview_until_continue, save_frame1_image
from challenge_deploy.runtime import DualPiperObservationSource


DEPLOY_ROOT = Path(__file__).resolve().parent
INIT_JOINTS = np.array(
    [
        -0.05918411,
        0.00076794,
        -0.12870058,
        -0.13548991,
        0.29586821,
        0.13372713,
        0.0,
        0.08932595,
        0.00970403,
        -0.21027726,
        -0.08838347,
        0.39285615,
        0.08686504,
        0.0,
    ],
    dtype=np.float64,
)
SOCKET_CHUNK_SIZE = 1 << 20
GRIPPER_SCALE = 70.0 / 100.0
CAMERA_NAMES = ("cam_high", "cam_right_wrist", "cam_left_wrist")
EEF_10D_NAMES = (
    "left_pos_x",
    "left_pos_y",
    "left_pos_z",
    "left_rot6d_0",
    "left_rot6d_1",
    "left_rot6d_2",
    "left_rot6d_3",
    "left_rot6d_4",
    "left_rot6d_5",
    "left_gripper",
    "right_pos_x",
    "right_pos_y",
    "right_pos_z",
    "right_rot6d_0",
    "right_rot6d_1",
    "right_rot6d_2",
    "right_rot6d_3",
    "right_rot6d_4",
    "right_rot6d_5",
    "right_gripper",
)
CTS_10D_NAMES = (
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


class _EvalServerNumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return {
                "__numpy_array__": True,
                "data": base64.b64encode(obj.tobytes()).decode("ascii"),
                "dtype": str(obj.dtype),
                "shape": obj.shape,
            }
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _eval_server_json_to_numpy(json_str: str) -> Any:
    def object_hook(obj: dict[str, Any]) -> Any:
        if obj.get("__numpy_array__"):
            raw = base64.b64decode(obj["data"])
            return np.frombuffer(raw, dtype=np.dtype(obj["dtype"])).reshape(obj["shape"])
        return obj

    return json.loads(json_str, object_hook=object_hook)


class EvalServerModelClient:
    """Client for script/policy_model_server.py, launched by eval_server.sh."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        timeout_s: float = 30.0,
        connect_retries: int = 1000,
        retry_delay_s: float = 5.0,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.connect_retries = connect_retries
        self.retry_delay_s = retry_delay_s
        self.sock: socket.socket | None = None
        self._connect()

    def _connect(self) -> None:
        for attempt in range(1, self.connect_retries + 1):
            sock: socket.socket | None = None
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.settimeout(self.timeout_s)
                sock.connect((self.host, self.port))
                self.sock = sock
                print(f'{{"eval_server": "connected", "host": "{self.host}", "port": {self.port}}}', flush=True)
                return
            except OSError as exc:
                if attempt >= self.connect_retries:
                    raise ConnectionError(
                        f"Failed to connect to eval server {self.host}:{self.port} after {attempt} attempts"
                    ) from exc
                print(
                    f'{{"eval_server": "waiting", "host": "{self.host}", "port": {self.port}, '
                    f'"attempt": {attempt}, "retry_delay_s": {self.retry_delay_s}}}',
                    flush=True,
                )
                if sock is not None:
                    try:
                        sock.close()
                    except Exception:
                        pass
                if self.retry_delay_s > 0:
                    import time

                    time.sleep(self.retry_delay_s)

    def call(self, func_name: str, obs: Any | None = None) -> Any:
        if self.sock is None:
            self._connect()
        assert self.sock is not None
        payload = json.dumps({"cmd": func_name, "obs": obs}, cls=_EvalServerNumpyEncoder).encode("utf-8")
        try:
            self.sock.sendall(len(payload).to_bytes(4, "big"))
            self.sock.sendall(payload)
            response = self._recv_response()
        except OSError as exc:
            self.close()
            raise ConnectionError(f"eval server communication failed: {exc}") from exc
        if "error" in response:
            traceback = response.get("traceback", "")
            raise RuntimeError(f"eval server returned error: {response['error']}\n{traceback}")
        return response["res"]

    def _recv_response(self) -> Any:
        assert self.sock is not None
        len_bytes = self._recv_exact(4)
        msg_length = int.from_bytes(len_bytes, "big")
        chunks: list[bytes] = []
        remaining = msg_length
        while remaining > 0:
            chunk = self.sock.recv(min(remaining, SOCKET_CHUNK_SIZE))
            if not chunk:
                raise ConnectionError("Incomplete response received from eval server")
            chunks.append(chunk)
            remaining -= len(chunk)
        return _eval_server_json_to_numpy(b"".join(chunks).decode("utf-8"))

    def _recv_exact(self, size: int) -> bytes:
        assert self.sock is not None
        chunks: list[bytes] = []
        remaining = size
        while remaining > 0:
            chunk = self.sock.recv(remaining)
            if not chunk:
                raise ConnectionError("Connection closed by eval server")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def close(self) -> None:
        if self.sock is None:
            return
        try:
            self.sock.close()
        finally:
            self.sock = None


def _available_image_names(images: Any) -> list[str]:
    if isinstance(images, dict):
        return sorted(str(name) for name in images)
    return sorted(name for name in dir(images) if not name.startswith("_"))


def _lookup_snapshot_image(images: Any, names: tuple[str, ...]) -> np.ndarray:
    for name in names:
        if isinstance(images, dict) and name in images:
            return np.asarray(images[name])
        get = getattr(images, "get", None)
        if callable(get):
            value = get(name)
            if value is not None:
                return np.asarray(value)
        if hasattr(images, name):
            return np.asarray(getattr(images, name))
    raise KeyError(f"Missing image for any of {names}; available images: {_available_image_names(images)}")


def _as_eval_server_hwc_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        raise ValueError(f"Expected HWC/CHW RGB image, got shape {arr.shape}")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return np.ascontiguousarray(arr)


def _eval_server_image_triplet(snapshot: Any) -> list[np.ndarray]:
    images = getattr(snapshot, "images", None)
    if images is None:
        raise AttributeError("Snapshot has no images field; eval_server backend requires cameras.")
    front = _lookup_snapshot_image(images, ("cam_high", "head_camera", "front_camera", "front", "head"))
    right = _lookup_snapshot_image(images, ("cam_right_wrist", "right_camera", "right_wrist", "right"))
    left = _lookup_snapshot_image(images, ("cam_left_wrist", "left_camera", "left_wrist", "left"))
    return [_as_eval_server_hwc_rgb(front), _as_eval_server_hwc_rgb(right), _as_eval_server_hwc_rgb(left)]


def _normalize_action_mode(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized == "ee_10d":
        normalized = "eef_10d"
    if normalized not in {"eef_10d", "cts_10d"}:
        raise argparse.ArgumentTypeError("--action-mode must be one of eef_10d/ee_10d/cts_10d")
    return normalized


def quaternion_to_rotation_6d(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64).reshape(4)
    rotation_matrix = R.from_quat(quaternion).as_matrix()
    return rotation_matrix[:, :2].T.reshape(-1)


def rotation_6d_to_quaternion(rotation_6d: np.ndarray) -> np.ndarray:
    rotation_6d = np.asarray(rotation_6d, dtype=np.float64)
    single_action = rotation_6d.ndim == 1
    rotation_6d = rotation_6d.reshape(-1, 6)

    first_column = rotation_6d[:, :3]
    second_column = rotation_6d[:, 3:]
    first_column = first_column / np.clip(np.linalg.norm(first_column, axis=1, keepdims=True), 1e-8, None)
    second_column = second_column - np.sum(first_column * second_column, axis=1, keepdims=True) * first_column
    second_column = second_column / np.clip(np.linalg.norm(second_column, axis=1, keepdims=True), 1e-8, None)
    third_column = np.cross(first_column, second_column, axis=1)

    rotation_matrix = np.stack((first_column, second_column, third_column), axis=-1)
    quaternion = R.from_matrix(rotation_matrix).as_quat()
    return quaternion[0] if single_action else quaternion


def _pose_to_quat_pose(pose: Any, *, rpy_order: str, rpy_unit: str) -> np.ndarray:
    pose_arr = np.asarray(pose, dtype=np.float64).reshape(-1)
    if pose_arr.shape[0] == 7:
        quat = pose_arr[3:7]
        quat_norm = np.linalg.norm(quat)
        if quat_norm < 1e-8:
            raise ValueError(f"Invalid zero quaternion pose: {pose_arr}")
        return np.concatenate([pose_arr[:3], quat / quat_norm], axis=0)
    if pose_arr.shape[0] != 6:
        raise ValueError(f"Expected end-effector pose with 6 xyz+rpy or 7 xyz+quat values, got {pose_arr.shape}")

    rpy = pose_arr[3:6]
    degrees = False
    if rpy_unit == "deg1000":
        rpy = rpy * 1000.0
        degrees = True
    elif rpy_unit == "deg":
        degrees = True
    elif rpy_unit == "rad":
        degrees = False
    else:
        raise ValueError(f"Unsupported rpy unit: {rpy_unit}")
    quat = R.from_euler(rpy_order, rpy, degrees=degrees).as_quat()
    return np.concatenate([pose_arr[:3], quat], axis=0)


def pose_to_10d(arm_pose: np.ndarray, gripper: float) -> np.ndarray:
    arm_pose = np.asarray(arm_pose, dtype=np.float64).reshape(7)
    return np.concatenate([arm_pose[:3], quaternion_to_rotation_6d(arm_pose[3:7]), np.array([gripper])], axis=0)


def pose_pair_to_20d(left_arm: np.ndarray, left_gripper: float, right_arm: np.ndarray, right_gripper: float) -> np.ndarray:
    return np.concatenate([pose_to_10d(left_arm, left_gripper), pose_to_10d(right_arm, right_gripper)], axis=0)


def ee_10d_to_ee(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action)
    single_action = action.ndim == 1
    action = action.reshape(-1, 20)

    left_position = action[:, :3]
    left_quaternion = rotation_6d_to_quaternion(action[:, 3:9])
    left_gripper = action[:, 9:10]
    right_position = action[:, 10:13]
    right_quaternion = rotation_6d_to_quaternion(action[:, 13:19])
    right_gripper = action[:, 19:20]

    ee_action = np.concatenate(
        [left_position, left_quaternion, left_gripper, right_position, right_quaternion, right_gripper],
        axis=1,
    )
    ee_action = ee_action.astype(action.dtype, copy=False)
    return ee_action[0] if single_action else ee_action


def cal_cts(end_pose_vector: np.ndarray, qa_last: np.ndarray) -> np.ndarray:
    end_pose_vector = np.asarray(end_pose_vector, dtype=np.float64).reshape(16)
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


def cts_quat_to_cts_10d(cts_pose_state: np.ndarray) -> np.ndarray:
    cts_pose_state = np.asarray(cts_pose_state, dtype=np.float64).reshape(16)
    pa = cts_pose_state[:3]
    qa_6d = quaternion_to_rotation_6d(cts_pose_state[3:7])
    left_gripper = cts_pose_state[7:8]
    pr = cts_pose_state[8:11]
    qr_6d = quaternion_to_rotation_6d(cts_pose_state[11:15])
    right_gripper = cts_pose_state[15:16]
    return np.concatenate([pa, qa_6d, left_gripper, pr, qr_6d, right_gripper], axis=0)


def cts_10d_to_cts_quat(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action)
    single_action = action.ndim == 1
    action = action.reshape(-1, 20)

    pa = action[:, :3]
    qa = rotation_6d_to_quaternion(action[:, 3:9])
    left_gripper = action[:, 9:10]
    pr = action[:, 10:13]
    qr = rotation_6d_to_quaternion(action[:, 13:19])
    right_gripper = action[:, 19:20]

    cts_action = np.concatenate([pa, qa, left_gripper, pr, qr, right_gripper], axis=1)
    cts_action = cts_action.astype(action.dtype, copy=False)
    return cts_action[0] if single_action else cts_action


def cts_quat_to_ee(cts_action: np.ndarray) -> np.ndarray:
    cts_action = np.asarray(cts_action)
    single_action = cts_action.ndim == 1
    cts_action = cts_action.reshape(-1, 16)

    pa = cts_action[:, :3]
    qa = cts_action[:, 3:7]
    left_gripper = cts_action[:, 7:8]
    pr = cts_action[:, 8:11]
    qr = cts_action[:, 11:15]
    right_gripper = cts_action[:, 15:16]

    left_position = pa - 0.5 * pr
    right_position = pa + 0.5 * pr
    left_quaternion = qa
    right_quaternion = []
    for qa_i, qr_i in zip(qa, qr, strict=False):
        right_quaternion.append((R.from_quat(qa_i) * R.from_quat(qr_i)).as_quat())
    right_quaternion = np.asarray(right_quaternion, dtype=cts_action.dtype)

    ee_action = np.concatenate(
        [left_position, left_quaternion, left_gripper, right_position, right_quaternion, right_gripper],
        axis=1,
    )
    return ee_action[0] if single_action else ee_action


def cts_10d_to_ee(action: np.ndarray) -> np.ndarray:
    return cts_quat_to_ee(cts_10d_to_cts_quat(action))


def cts_quat_to_sim_cts_order(cts_action: np.ndarray) -> np.ndarray:
    cts_action = np.asarray(cts_action)
    single_action = cts_action.ndim == 1
    cts_action = cts_action.reshape(-1, 16)
    idx = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 7, 15]
    reordered = cts_action[:, idx]
    return reordered[0] if single_action else reordered


def _get_value(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    value = getattr(obj, key, None)
    if callable(value) and key.startswith("get_"):
        return value()
    return value


def _get_path(obj: Any, path: tuple[str, ...]) -> Any:
    current = obj
    for key in path:
        current = _get_value(current, key)
        if current is None:
            return None
    return current


def _available_fields(obj: Any) -> list[str]:
    if isinstance(obj, dict):
        return sorted(str(key) for key in obj)
    return sorted(name for name in dir(obj) if not name.startswith("_"))


def _extract_first(snapshot: Any, paths: tuple[tuple[str, ...], ...], label: str) -> Any:
    for path in paths:
        value = _get_path(snapshot, path)
        if value is not None:
            return value
    raise KeyError(f"Unable to find {label}; top-level snapshot fields: {_available_fields(snapshot)}")


_LEFT_POSE_PATHS = (
    ("left_ee_pose",),
    ("left_tcp_pose",),
    ("left_endpose",),
    ("left_pose",),
    ("left", "ee_pose"),
    ("left", "tcp_pose"),
    ("left", "endpose"),
    ("left", "pose"),
    ("state", "left_ee_pose"),
    ("state", "left_tcp_pose"),
    ("state", "left_endpose"),
    ("state", "left_pose"),
    ("state", "left", "ee_pose"),
    ("state", "left", "tcp_pose"),
    ("state", "left", "endpose"),
    ("robot", "left_ee_pose"),
    ("robot", "left_tcp_pose"),
    ("robot", "left_endpose"),
    ("robot", "get_left_ee_pose"),
    ("robot", "get_left_tcp_pose"),
    ("robot", "left", "ee_pose"),
    ("robot", "left", "tcp_pose"),
)
_RIGHT_POSE_PATHS = tuple(tuple(part.replace("left", "right") for part in path) for path in _LEFT_POSE_PATHS)
_LEFT_GRIPPER_PATHS = (
    ("left_gripper",),
    ("left_gripper_position",),
    ("left_gripper_val",),
    ("left", "gripper"),
    ("left", "gripper_position"),
    ("left", "gripper_val"),
    ("state", "left_gripper"),
    ("state", "left_gripper_position"),
    ("state", "left_gripper_val"),
    ("state", "left", "gripper"),
    ("state", "left", "gripper_position"),
    ("robot", "left_gripper"),
    ("robot", "left_gripper_position"),
    ("robot", "left_gripper_val"),
    ("robot", "get_left_gripper_val"),
    ("robot", "left", "gripper"),
)
_RIGHT_GRIPPER_PATHS = tuple(tuple(part.replace("left", "right") for part in path) for path in _LEFT_GRIPPER_PATHS)


def _as_scalar(value: Any) -> float:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("Expected scalar gripper value, got empty value")
    return float(arr[0])


def build_eef_quat_state(
    snapshot: Any,
    *,
    rpy_order: str,
    rpy_unit: str,
    gripper_state_scale: float,
) -> np.ndarray:
    left_pose = _pose_to_quat_pose(_extract_first(snapshot, _LEFT_POSE_PATHS, "left ee pose"), rpy_order=rpy_order, rpy_unit=rpy_unit)
    right_pose = _pose_to_quat_pose(_extract_first(snapshot, _RIGHT_POSE_PATHS, "right ee pose"), rpy_order=rpy_order, rpy_unit=rpy_unit)
    left_gripper = _as_scalar(_extract_first(snapshot, _LEFT_GRIPPER_PATHS, "left gripper")) * gripper_state_scale
    right_gripper = _as_scalar(_extract_first(snapshot, _RIGHT_GRIPPER_PATHS, "right gripper")) * gripper_state_scale
    return np.concatenate([left_pose, np.array([left_gripper]), right_pose, np.array([right_gripper])], axis=0)


def _state_action_names(action_mode: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    names = CTS_10D_NAMES if action_mode == "cts_10d" else EEF_10D_NAMES
    return names, names


class EvalServerPiperClient:
    """Inference client for eval_server.sh plus local 10D state/action conversion."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        action_mode: str,
        rpy_order: str,
        rpy_unit: str,
        gripper_state_scale: float,
        timeout_s: float,
        connect_retries: int,
        retry_delay_s: float,
    ) -> None:
        self.action_mode = action_mode
        self.rpy_order = rpy_order
        self.rpy_unit = rpy_unit
        self.gripper_state_scale = gripper_state_scale
        self.action_names, self.state_names = _state_action_names(action_mode)
        self.image_ids = CAMERA_NAMES
        self._qa_last = np.zeros(4, dtype=np.float64)
        self._eval_server_client = EvalServerModelClient(
            host=host,
            port=port,
            timeout_s=timeout_s,
            connect_retries=connect_retries,
            retry_delay_s=retry_delay_s,
        )

    def build_state(self, snapshot: Any) -> np.ndarray:
        eef_quat_state = build_eef_quat_state(
            snapshot,
            rpy_order=self.rpy_order,
            rpy_unit=self.rpy_unit,
            gripper_state_scale=self.gripper_state_scale,
        )
        if self.action_mode == "eef_10d":
            return pose_pair_to_20d(
                eef_quat_state[:7],
                float(eef_quat_state[7]),
                eef_quat_state[8:15],
                float(eef_quat_state[15]),
            ).astype(np.float32)

        cts_state = cal_cts(eef_quat_state, self._qa_last)
        self._qa_last = cts_state[3:7]
        return cts_quat_to_cts_10d(cts_state).astype(np.float32)

    def infer_actions(self, snapshot: Any, *, prompt: str) -> np.ndarray:
        input_rgb_arr = _eval_server_image_triplet(snapshot)
        input_state = self.build_state(snapshot)
        actions = self._eval_server_client.call(
            "infer_action",
            obs=(prompt, input_rgb_arr, input_state),
        )
        return np.asarray(actions, dtype=np.float32).reshape(-1, 20)

    def update_observation_window(self, snapshot: Any) -> None:
        self._eval_server_client.call(
            "update_observation_window",
            obs=(_eval_server_image_triplet(snapshot), self.build_state(snapshot)),
        )

    def decode_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(20)
        if self.action_mode == "cts_10d":
            return cts_10d_to_ee(action).astype(np.float32)
        return ee_10d_to_ee(action).astype(np.float32)

    def sim_cts_action(self, action: np.ndarray) -> np.ndarray:
        if self.action_mode != "cts_10d":
            raise ValueError("sim_cts_action is only valid for cts_10d")
        return cts_quat_to_sim_cts_order(cts_10d_to_cts_quat(action)).astype(np.float32)

    def reset(self) -> None:
        self._qa_last = np.zeros(4, dtype=np.float64)
        self._eval_server_client.call("reset_obsrvationwindows")

    def close(self) -> None:
        self._eval_server_client.close()


def _make_recording_schema(client: EvalServerPiperClient) -> RecordingSchema:
    return RecordingSchema(
        camera_names=CAMERA_NAMES,
        action_names=tuple(client.action_names),
        state_names=tuple(client.state_names),
        used_action_names=frozenset(client.action_names),
    )


def _record_name_prefix(args: argparse.Namespace) -> str:
    run_name = Path(args.run_name).name if args.run_name else f"{args.action_mode}_{args.host}_{args.port}"
    return f"{run_name}_{args.execution_mode}"


def _install_record_signal_handlers() -> None:
    def _raise_keyboard_interrupt(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    for signal_name in ("SIGINT", "SIGTERM", "SIGHUP"):
        signal_value = getattr(signal, signal_name, None)
        if signal_value is None:
            continue
        try:
            signal.signal(signal_value, _raise_keyboard_interrupt)
        except (OSError, ValueError):
            pass


def _ignore_record_signal_handlers() -> None:
    for signal_name in ("SIGINT", "SIGTERM", "SIGHUP"):
        signal_value = getattr(signal, signal_name, None)
        if signal_value is None:
            continue
        try:
            signal.signal(signal_value, signal.SIG_IGN)
        except (OSError, ValueError):
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RoboTwin pi0 eval-server client: capture real Piper snapshots, infer 10D chunks, and command dual arms."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--eval-server-timeout", type=float, default=30.0)
    parser.add_argument("--eval-server-connect-retries", type=int, default=1000)
    parser.add_argument("--eval-server-retry-delay", type=float, default=5.0)
    parser.add_argument("--prompt", "--instruction", dest="prompt", required=True)
    parser.add_argument("--action-mode", type=_normalize_action_mode, default="cts_10d", help="Policy IO mode: eef_10d/ee_10d or cts_10d.")
    parser.add_argument("--rpy-order", default="xyz", help="Euler axis order if runtime ee poses are xyz+rpy.")
    parser.add_argument(
        "--rpy-unit",
        choices=["deg1000", "deg", "rad"],
        default="deg1000",
        help="Unit for runtime xyz+rpy ee poses. Ignored when runtime poses are xyz+quat.",
    )
    parser.add_argument(
        "--gripper-state-scale",
        type=float,
        default=GRIPPER_SCALE,
        help="Scale live raw gripper state before sending qpos to policy; real conversion uses 0.7.",
    )
    parser.add_argument("--joint-speed-percent", type=int, default=50)
    parser.add_argument("--ee-speed-percent", type=int, default=50)
    parser.add_argument(
        "--gripper_threshold",
        type=float,
        default=None,
        help="Optional executable-scale gripper threshold. Final gripper values below this are clipped to 0.",
    )
    for side in ("left", "right"):
        parser.add_argument(f"--{side}_gripper_threshold", *([f"--{side}_gripper_thrshold"] if side == "left" else []), dest=f"{side}_gripper_threshold", type=float, default=None)
        parser.add_argument(f"--{side}_gripper_lower", type=float, default=None); parser.add_argument(f"--{side}_gripper_upper", type=float, default=None)
    parser.add_argument("--gripper_lower", type=float, default=None)
    parser.add_argument("--gripper_upper", type=float, default=None)
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=1000,
        help="Number of action frames to command; 0 means run until Ctrl-C.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=20,
        help="Actions to execute from each policy chunk.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Action command frequency in Hz; 0 sends the chunk as fast as possible.",
    )
    parser.add_argument(
        "--execution-mode",
        choices=["chunk_sync"],
        default="chunk_sync",
        help="Only chunk_sync is supported for eval_server real-world execution.",
    )
    parser.add_argument(
        "--update-server-obs-every-action",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mirror deploy_policy.py by refreshing eval_server observation_window after each executed action.",
    )
    parser.add_argument("--metrics-json", default=None, help="Optional path to save rollout timing metrics as JSON.")
    parser.add_argument("--record", action="store_true", help="Record cameras, actions, and states into one deploy video.")
    parser.add_argument("--record-dir", default=str(DEPLOY_ROOT / "artifacts" / "openpi_records"))
    parser.add_argument("--run-name", default=None, help="Optional prefix used for recording filenames.")
    parser.add_argument("--config", default=str(DEPLOY_ROOT / "configs" / "dual_piper_example.yaml"))
    parser.add_argument("--left-can", default=None)
    parser.add_argument("--right-can", default=None)
    parser.add_argument("--camera-front-serial", default=None)
    parser.add_argument("--camera-left-serial", default=None)
    parser.add_argument("--camera-right-serial", default=None)
    parser.add_argument("--no-cameras", action="store_true")
    parser.add_argument("--window", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Infer and decode the first action, but do not command Piper.")
    parser.add_argument("--ready-timeout", type=float, default=15.0)
    return parser


def _apply_runtime_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.left_can:
        set_by_dotted_path(config, "robot.left.can_name", args.left_can)
    if args.right_can:
        set_by_dotted_path(config, "robot.right.can_name", args.right_can)
    if args.camera_front_serial:
        set_by_dotted_path(config, "cameras.serials.cam_high", args.camera_front_serial)
    if args.camera_right_serial:
        set_by_dotted_path(config, "cameras.serials.cam_right_wrist", args.camera_right_serial)
    if args.camera_left_serial:
        set_by_dotted_path(config, "cameras.serials.cam_left_wrist", args.camera_left_serial)
    if args.no_cameras:
        set_by_dotted_path(config, "cameras.enabled", False)
    return config


def _make_runtime(config: dict[str, Any], *, commands_enabled: bool) -> tuple[Any, Any, Any]:
    robot = DualPiperSystem(
        left_can_name=config["robot"]["left"]["can_name"],
        right_can_name=config["robot"]["right"]["can_name"],
        commands_enabled=commands_enabled,
        name="openpi_piper_client",
    )
    cameras = None
    if config["cameras"]["enabled"]:
        cameras = RealSenseRig(
            config["cameras"]["serials"],
            width=int(config["cameras"]["width"]),
            height=int(config["cameras"]["height"]),
            fps=int(config["cameras"]["fps"]),
            warmup_frames=int(config["cameras"]["warmup_frames"]),
        )
    return robot, cameras, DualPiperObservationSource(robot=robot, cameras=cameras)


def _normalized_prompt(value: str | None) -> str | None:
    if value is None:
        return None
    prompt = value.strip()
    return prompt or None


def _decoded_action_summary(ee_action: np.ndarray) -> dict[str, Any]:
    ee_action = np.asarray(ee_action, dtype=np.float64).reshape(16)
    return {
        "left": {
            "position": ee_action[:3].round(5).tolist(),
            "quaternion_xyzw": ee_action[3:7].round(5).tolist(),
            "gripper": round(float(ee_action[7]), 5),
        },
        "right": {
            "position": ee_action[8:11].round(5).tolist(),
            "quaternion_xyzw": ee_action[11:15].round(5).tolist(),
            "gripper": round(float(ee_action[15]), 5),
        },
    }


def _clip_gripper(value: float, *, side: str, args: argparse.Namespace) -> float:
    lower = getattr(args, f"{side}_gripper_lower")
    upper = getattr(args, f"{side}_gripper_upper")
    threshold = getattr(args, f"{side}_gripper_threshold")
    if lower is None:
        lower = args.gripper_lower
    if upper is None:
        upper = args.gripper_upper
    if threshold is None:
        threshold = args.gripper_threshold

    clipped = float(value)
    if threshold is not None and clipped < threshold:
        clipped = 0.0
    if lower is not None:
        clipped = max(float(lower), clipped)
    if upper is not None:
        clipped = min(float(upper), clipped)
    return clipped


def _call_first_method(
    obj: Any,
    method_names: tuple[str, ...],
    variants: list[tuple[tuple[Any, ...], dict[str, Any]]],
) -> bool:
    return _call_first_method_index(obj, method_names, variants) is not None


def _call_first_method_index(
    obj: Any,
    method_names: tuple[str, ...],
    variants: list[tuple[tuple[Any, ...], dict[str, Any]]],
) -> int | None:
    for method_name in method_names:
        method = getattr(obj, method_name, None)
        if not callable(method):
            continue
        for variant_index, (call_args, call_kwargs) in enumerate(variants):
            try:
                method(*call_args, **call_kwargs)
                return variant_index
            except TypeError:
                pass
    return None


def _command_ee_action(robot: Any, ee_action: np.ndarray, args: argparse.Namespace) -> None:
    ee_action = np.asarray(ee_action, dtype=np.float64).reshape(16)
    left_pose = ee_action[:7]
    right_pose = ee_action[8:15]
    left_gripper = _clip_gripper(float(ee_action[7]), side="left", args=args)
    right_gripper = _clip_gripper(float(ee_action[15]), side="right", args=args)
    ee_speed = int(args.ee_speed_percent)

    full_action = np.concatenate([left_pose, [left_gripper], right_pose, [right_gripper]])
    action_methods = (
        "execute_ee_action",
        "command_ee_action",
        "execute_action",
        "command_action",
    )
    action_variants = [
        ((full_action,), {"speed_percent": ee_speed}),
        ((full_action,), {"action_type": "ee", "speed_percent": ee_speed}),
        ((full_action, "ee"), {"speed_percent": ee_speed}),
    ]
    if _call_first_method(robot, action_methods, action_variants):
        return

    combined_pose_methods = (
        "move_to_ee_pose",
        "move_to_tcp_pose",
        "move_to_end_effector_pose",
        "command_ee_pose",
        "command_tcp_pose",
        "command_end_effector_pose",
        "move_to_ee_poses",
        "move_to_tcp_poses",
        "move_to_end_effector_poses",
        "command_ee_poses",
        "command_tcp_poses",
        "command_end_effector_poses",
    )
    combined_pose_variants = [
        ((left_pose, right_pose), {"speed_percent": ee_speed}),
        ((left_pose, right_pose, left_gripper, right_gripper), {"speed_percent": ee_speed}),
        ((), {"left_pose": left_pose, "right_pose": right_pose, "left_gripper": left_gripper, "right_gripper": right_gripper, "speed_percent": ee_speed}),
    ]
    combined_variant_index = _call_first_method_index(robot, combined_pose_methods, combined_pose_variants)
    moved_by_combined = combined_variant_index is not None
    if combined_variant_index in {1, 2}:
        return

    if not moved_by_combined:
        side_pose_methods = (
            "move_to_ee_pose",
            "move_to_tcp_pose",
            "move_to_end_effector_pose",
            "command_ee_pose",
            "command_tcp_pose",
            "command_end_effector_pose",
        )
        left_variants = [
            (("left", left_pose), {"speed_percent": ee_speed}),
            ((left_pose, "left"), {"speed_percent": ee_speed}),
            ((left_pose,), {"side": "left", "speed_percent": ee_speed}),
            ((), {"side": "left", "pose": left_pose, "speed_percent": ee_speed}),
        ]
        right_variants = [
            (("right", right_pose), {"speed_percent": ee_speed}),
            ((right_pose, "right"), {"speed_percent": ee_speed}),
            ((right_pose,), {"side": "right", "speed_percent": ee_speed}),
            ((), {"side": "right", "pose": right_pose, "speed_percent": ee_speed}),
        ]
        left_ok = _call_first_method(
            robot,
            ("move_left_to_ee_pose", "left_move_to_ee_pose", "command_left_ee_pose", *side_pose_methods),
            [((left_pose,), {"speed_percent": ee_speed}), *left_variants],
        )
        right_ok = _call_first_method(
            robot,
            ("move_right_to_ee_pose", "right_move_to_ee_pose", "command_right_ee_pose", *side_pose_methods),
            [((right_pose,), {"speed_percent": ee_speed}), *right_variants],
        )
        if not (left_ok and right_ok):
            raise RuntimeError(
                "DualPiperSystem does not expose a supported ee pose command method. "
                "Expected one of execute_ee_action/move_to_ee_poses/move_to_ee_pose or side-specific variants."
            )

    gripper_methods = ("set_grippers", "move_grippers", "command_grippers")
    gripper_variants = [
        ((left_gripper, right_gripper), {}),
        ((), {"left": left_gripper, "right": right_gripper}),
        ((), {"left_gripper": left_gripper, "right_gripper": right_gripper}),
    ]
    if _call_first_method(robot, gripper_methods, gripper_variants):
        return

    per_side_gripper_methods = ("set_gripper", "move_gripper", "command_gripper")
    left_ok = _call_first_method(
        robot,
        ("set_left_gripper", "move_left_gripper", "command_left_gripper", *per_side_gripper_methods),
        [((left_gripper,), {}), ((left_gripper, "left"), {}), (("left", left_gripper), {}), ((), {"side": "left", "value": left_gripper})],
    )
    right_ok = _call_first_method(
        robot,
        ("set_right_gripper", "move_right_gripper", "command_right_gripper", *per_side_gripper_methods),
        [((right_gripper,), {}), ((right_gripper, "right"), {}), (("right", right_gripper), {}), ((), {"side": "right", "value": right_gripper})],
    )
    if not (left_ok and right_ok):
        raise RuntimeError("DualPiperSystem does not expose a supported gripper command method.")


@dataclasses.dataclass
class RolloutMetrics:
    executed_steps: int = 0
    inference_calls: int = 0
    interrupted: bool = False
    started_at_s: float = dataclasses.field(default_factory=time.monotonic)
    finished_at_s: float | None = None

    def summary(self) -> dict[str, Any]:
        finished = self.finished_at_s if self.finished_at_s is not None else time.monotonic()
        duration = finished - self.started_at_s
        return {
            "executed_steps": self.executed_steps,
            "inference_calls": self.inference_calls,
            "interrupted": self.interrupted,
            "duration_s": duration,
            "avg_fps": self.executed_steps / duration if duration > 0 else None,
        }


def _save_metrics(metrics: RolloutMetrics, metrics_json_path: str | None) -> None:
    if metrics_json_path is None:
        return
    path = Path(metrics_json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics.summary(), indent=2), encoding="utf-8")
    print(f"Rollout metrics saved to {path}", flush=True)


def _print_rollout_chunk_summary(
    *,
    client: EvalServerPiperClient,
    chunk_index: int,
    action_count: int,
    executed_steps: int,
    rollout_steps: int,
    first_action: np.ndarray,
) -> None:
    summary = _decoded_action_summary(client.decode_action(first_action))
    target = "unlimited" if rollout_steps == 0 else str(rollout_steps)
    print(
        json.dumps(
            {
                "rollout_chunk": chunk_index,
                "actions_in_chunk": action_count,
                "executed_steps": executed_steps,
                "target_steps": target,
                "first_action": summary,
            },
            indent=2,
        ),
        flush=True,
    )


def run_once(args: argparse.Namespace) -> None:
    cli_prompt = _normalized_prompt(args.prompt)
    if cli_prompt is None:
        raise ValueError("--prompt/--instruction must be a non-empty string")
    if args.rollout_steps < 0:
        raise ValueError("--rollout-steps must be non-negative")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if args.fps < 0.0:
        raise ValueError("--fps must be non-negative")
    if args.gripper_state_scale < 0.0:
        raise ValueError("--gripper-state-scale must be non-negative")
    if args.eval_server_timeout <= 0.0:
        raise ValueError("--eval-server-timeout must be positive")
    if args.eval_server_connect_retries < 1:
        raise ValueError("--eval-server-connect-retries must be positive")
    if args.eval_server_retry_delay < 0.0:
        raise ValueError("--eval-server-retry-delay must be non-negative")
    if args.gripper_threshold is not None and args.gripper_threshold < 0.0:
        raise ValueError("--gripper_threshold must be non-negative")
    if args.gripper_threshold is not None and (args.gripper_lower is not None or args.gripper_upper is not None):
        raise ValueError("--gripper_threshold cannot be combined with --gripper_lower/--gripper_upper")

    print(
        json.dumps(
            {
                "policy_client": {
                    "backend": "eval_server",
                    "server": f"{args.host}:{args.port}",
                    "action_mode": args.action_mode,
                    "prompt": cli_prompt,
                    "rpy_order": args.rpy_order,
                    "rpy_unit": args.rpy_unit,
                    "gripper_state_scale": args.gripper_state_scale,
                }
            },
            indent=2,
        ),
        flush=True,
    )

    client = EvalServerPiperClient(
        host=args.host,
        port=args.port,
        action_mode=args.action_mode,
        rpy_order=args.rpy_order,
        rpy_unit=args.rpy_unit,
        gripper_state_scale=args.gripper_state_scale,
        timeout_s=args.eval_server_timeout,
        connect_retries=args.eval_server_connect_retries,
        retry_delay_s=args.eval_server_retry_delay,
    )
    runtime_config = _apply_runtime_overrides(load_config(args.config), args)
    robot, cameras, source = _make_runtime(runtime_config, commands_enabled=not args.dry_run)
    recording_schema = _make_recording_schema(client)
    saved_actions: list[np.ndarray] | None = [] if args.record else None
    recorder = (
        OpenPiRolloutRecorder(
            output_dir=args.record_dir,
            schema=recording_schema,
            fps=args.fps,
            name_prefix=_record_name_prefix(args),
        )
        if args.record
        else None
    )
    if recorder is not None:
        _install_record_signal_handlers()

    first_obs_snapshot = None
    frame1_compare_path: Path | None = None
    metrics = RolloutMetrics()
    robot.connect(read_only=args.dry_run)
    try:
        if cameras is not None:
            cameras.start()
        if not source.wait_until_ready(timeout_s=args.ready_timeout):
            raise RuntimeError("Timed out waiting for Piper/RealSense data")
        client.reset()

        if args.dry_run:
            snapshot = source.capture_snapshot()
            first_obs_snapshot = snapshot
            raw_actions = client.infer_actions(snapshot, prompt=cli_prompt)[: args.chunk_size]
            if raw_actions.shape[0] == 0:
                raise RuntimeError("eval_server returned an empty action chunk")
            decoded_first_action = client.decode_action(raw_actions[0])
            if recorder is not None:
                recorder.record(
                    images=snapshot.images,
                    action=raw_actions[0],
                    state=client.build_state(snapshot),
                    timestamp_s=snapshot.timestamp_s,
                )
            if saved_actions is not None:
                saved_actions.append(raw_actions[0].copy())
            if recorder is not None:
                frame1_compare_path = save_frame1_image(
                    recorder,
                    snapshot,
                    distribution_image_path=None,
                )
                if frame1_compare_path is not None:
                    print(f"Frame1 comparison saved to {frame1_compare_path}", flush=True)
            if args.window:
                preview_until_continue(source, distribution_image_path=None)
            print(json.dumps({"first_decoded_ee_action": _decoded_action_summary(decoded_first_action)}, indent=2))
            return

        print('{"hardware_init": "enable_dual_piper"}', flush=True)
        if not robot.enable():
            print("Warning: Piper arm enable check did not report success; continuing anyway.", flush=True)

        print(json.dumps({"initial_pose": {"qpos": INIT_JOINTS.tolist()}}, indent=2), flush=True)
        robot.move_to_joint_positions(INIT_JOINTS, speed_percent=args.joint_speed_percent)
        first_obs_snapshot = source.capture_snapshot()
        if recorder is not None:
            frame1_compare_path = save_frame1_image(
                recorder,
                first_obs_snapshot,
                distribution_image_path=None,
            )
            if frame1_compare_path is not None:
                print(f"Frame1 comparison saved to {frame1_compare_path}", flush=True)
        if args.window:
            preview_until_continue(source, distribution_image_path=None)

        print(
            json.dumps(
                {
                    "rollout": {
                        "execution_mode": args.execution_mode,
                        "rollout_steps": args.rollout_steps,
                        "chunk_size": args.chunk_size,
                        "fps": args.fps,
                        "update_server_obs_every_action": args.update_server_obs_every_action,
                        "joint_speed_percent": args.joint_speed_percent,
                        "ee_speed_percent": args.ee_speed_percent,
                        "gripper_threshold": args.gripper_threshold,
                    }
                },
                indent=2,
            ),
            flush=True,
        )

        def log_chunk(chunk_index: int, action_count: int, executed_steps: int, first_action: np.ndarray) -> None:
            _print_rollout_chunk_summary(
                client=client,
                chunk_index=chunk_index,
                action_count=action_count,
                executed_steps=executed_steps,
                rollout_steps=args.rollout_steps,
                first_action=first_action,
            )

        sleep_s = 0.0 if args.fps == 0.0 else 1.0 / args.fps
        snapshot = first_obs_snapshot
        chunk_index = 0
        while args.rollout_steps == 0 or metrics.executed_steps < args.rollout_steps:
            raw_actions = client.infer_actions(snapshot, prompt=cli_prompt)[: args.chunk_size]
            if raw_actions.shape[0] == 0:
                raise RuntimeError("eval_server returned an empty action chunk")
            metrics.inference_calls += 1
            log_chunk(chunk_index, raw_actions.shape[0], metrics.executed_steps, raw_actions[0])
            chunk_index += 1

            for raw_action in raw_actions:
                if args.rollout_steps != 0 and metrics.executed_steps >= args.rollout_steps:
                    break
                if recorder is not None:
                    recorder.record(
                        images=snapshot.images,
                        action=raw_action,
                        state=client.build_state(snapshot),
                        timestamp_s=snapshot.timestamp_s,
                    )
                if saved_actions is not None:
                    saved_actions.append(raw_action.copy())

                decoded_ee_action = client.decode_action(raw_action)
                _command_ee_action(robot, decoded_ee_action, args)
                metrics.executed_steps += 1

                if sleep_s > 0.0:
                    time.sleep(sleep_s)

                if args.update_server_obs_every_action:
                    snapshot = source.capture_snapshot()
                    client.update_observation_window(snapshot)

            snapshot = source.capture_snapshot()

        metrics.finished_at_s = time.monotonic()
        if metrics.interrupted:
            print("Interrupted by user; stopping rollout.", flush=True)
        print(json.dumps({"rollout_metrics": metrics.summary()}, indent=2), flush=True)
        _save_metrics(metrics, args.metrics_json)
    except KeyboardInterrupt:
        metrics.interrupted = True
        metrics.finished_at_s = time.monotonic()
        print("Interrupted by user; stopping rollout.", flush=True)
        print(json.dumps({"rollout_metrics": metrics.summary()}, indent=2), flush=True)
        _save_metrics(metrics, args.metrics_json)
    finally:
        if recorder is not None:
            _ignore_record_signal_handlers()
        if cameras is not None:
            try:
                cameras.stop()
            except Exception as exc:
                print(f"Failed to stop cameras cleanly: {exc}", flush=True)
        close_client = getattr(client, "close", None)
        if callable(close_client):
            try:
                close_client()
            except Exception as exc:
                print(f"Failed to close policy client cleanly: {exc}", flush=True)
        try:
            robot.disconnect()
        except Exception as exc:
            print(f"Failed to disconnect robot cleanly: {exc}", flush=True)
        if recorder is not None:
            try:
                action_path = recorder.run_dir / f"{recorder.record_stem}_actions.npz"
                action_trajectory = np.stack(saved_actions, axis=0) if saved_actions else np.empty((0, len(recording_schema.action_names)), dtype=np.float64)
                np.savez_compressed(action_path, action_mean_trajectory=action_trajectory, action_names=np.asarray(recording_schema.action_names))
                print(f"Actions saved to {action_path}", flush=True)
            except Exception as exc:
                print(f"Failed to save actions: {exc}", flush=True)
            output_path = None
            try:
                output_path = recorder.finalize()
            except Exception as exc:
                print(f"Failed to finalize recording: {exc}", flush=True)
            if output_path is not None:
                print(f"Recording saved to {output_path}", flush=True)
                try:
                    if frame1_compare_path is None:
                        frame1_compare_path = save_frame1_image(
                            recorder,
                            first_obs_snapshot,
                            distribution_image_path=None,
                        )
                    if frame1_compare_path is not None:
                        print(f"Frame1 comparison saved to {frame1_compare_path}", flush=True)
                except Exception as exc:
                    print(f"Failed to save frame1 comparison image: {exc}", flush=True)


def main() -> None:
    run_once(build_parser().parse_args())


if __name__ == "__main__":
    main()
