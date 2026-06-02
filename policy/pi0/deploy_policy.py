import numpy as np
import torch
import dill
import os, sys

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(parent_directory)

from pi_model import *


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _action_chunk_steps(default=20):
    try:
        return max(1, int(os.environ.get("PI0_ACTION_CHUNK_STEPS", default)))
    except ValueError:
        return default


def _update_obs_every_action(task_env):
    if task_env.eval_video_path is not None:
        return True
    return _env_flag("PI0_UPDATE_OBS_EVERY_ACTION", True)


def _infer_action_chunk(model, instruction, input_rgb_arr, input_state):
    actions = model.call(
        func_name="infer_action",
        obs=(instruction, input_rgb_arr, input_state),
    )
    return actions[:_action_chunk_steps()]


def rotation_6d_to_quaternion(rotation_6d):
    from scipy.spatial.transform import Rotation as R

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


def ee_10d_to_ee(action):
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


def quaternion_to_rotation_6d(quaternion):
    from scipy.spatial.transform import Rotation as R

    quaternion = np.asarray(quaternion, dtype=np.float64)
    single_pose = quaternion.ndim == 1
    quaternion = quaternion.reshape(-1, 4)

    rotation_6d = R.from_quat(quaternion).as_matrix()[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)
    rotation_6d = rotation_6d.astype(quaternion.dtype, copy=False)
    return rotation_6d[0] if single_pose else rotation_6d


def ee_to_ee_10d(state):
    state = np.asarray(state)
    single_state = state.ndim == 1
    state = state.reshape(-1, 16)

    left_position = state[:, :3]
    left_rotation_6d = quaternion_to_rotation_6d(state[:, 3:7])
    left_gripper = state[:, 7:8]
    right_position = state[:, 8:11]
    right_rotation_6d = quaternion_to_rotation_6d(state[:, 11:15])
    right_gripper = state[:, 15:16]

    state_10d = np.concatenate(
        [left_position, left_rotation_6d, left_gripper, right_position, right_rotation_6d, right_gripper],
        axis=1,
    )
    state_10d = state_10d.astype(state.dtype, copy=False)
    return state_10d[0] if single_state else state_10d



def cal_cts(end_pose_vector):
    from scipy.spatial.transform import Rotation as R
    from scipy.spatial.transform import Slerp
    Pa = (end_pose_vector[8:11] + end_pose_vector[:3]) / 2.0
    Pr = end_pose_vector[8:11] - end_pose_vector[:3]
    R1_true = R.from_quat(end_pose_vector[3:7]).as_matrix()
    R2_true = R.from_quat(end_pose_vector[11:15]).as_matrix()
    # 相对旋转
    Rr = R1_true.T @ R2_true    
    Qr = R.from_matrix(Rr).as_quat()
    # 计算中点 Qa（与前面的定义一致）
    def mid_rotation_scipy(R1, R2, t=0.5):
        """
        用 scipy 的 Slerp 在 SO(3) 上插值。
        R1, R2: (3,3) 旋转矩阵或可以转换为矩阵的 array-like
        t: 插值参数，0 -> R1, 1 -> R2，默认 0.5（中点）
        返回: (3,3) 旋转矩阵
        """
        # 把两个矩阵打包为 Rotation 对象
        rots = R.from_matrix(np.stack([R1, R2], axis=0))  # shape (2,)
        key_times = np.array([0.0, 1.0])                  # 对应两个关键帧的时间点
        slerp = Slerp(key_times, rots)
        Q_mid = slerp([t]).as_quat()[0]
        return Q_mid
    #Qa = mid_rotation_scipy(R1_true, R2_true, t=0.5)
    Qa = R.from_matrix(R1_true).as_quat()
    cts_pose_state = np.concatenate([Pa, Qa, end_pose_vector[7:8], Pr, Qr, end_pose_vector[15:16]])
    #print(cts_pose_state)
    return cts_pose_state

# Encode observation for the model
def encode_obs(observation):
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    #input_state = observation["joint_action"]["vector"]
    input_state = np.concatenate([observation["endpose"]["left_endpose"],
                        observation["endpose"]["left_gripper"].reshape(1),
                        observation["endpose"]["right_endpose"],
                        observation["endpose"]["right_gripper"].reshape(1)])

    return input_rgb_arr, input_state


def get_model(usr_args):
    train_config_name, model_name, checkpoint_id, pi0_step = (usr_args["train_config_name"], usr_args["model_name"],
                                                              usr_args["checkpoint_id"], usr_args["pi0_step"])
    return PI0(train_config_name, model_name, checkpoint_id, pi0_step)


def eval(TASK_ENV, model, observation):
    action_type = os.environ.get("PI0_ACTION_TYPE", "cts_10d")
    
    if action_type == "ee":
        #if model.observation_window is None:
        instruction = TASK_ENV.get_instruction()

        input_rgb_arr, input_state = encode_obs(observation)

        # ======== Get Action ========

        #actions = model.call(func_name='get_action')[:model.pi0_step]
        actions = _infer_action_chunk(model, instruction, input_rgb_arr, input_state)
        print(actions[0])

        for action in actions:
            TASK_ENV.take_action(action, action_type="ee")
            #TASK_ENV.take_action(action, action_type="cts")
            if _update_obs_every_action(TASK_ENV):
                observation = TASK_ENV.get_obs()
                input_rgb_arr, input_state = encode_obs(observation)
                #model.update_observation_window(input_rgb_arr, input_state)
                model.call(func_name="update_observation_window", obs = (input_rgb_arr, input_state))

    if action_type == "ee_10d":
        #if model.observation_window is None:
        instruction = TASK_ENV.get_instruction()

        input_rgb_arr, input_state = encode_obs(observation)
        input_state = ee_to_ee_10d(input_state)

        # ======== Get Action ========

        #actions = model.call(func_name='get_action')[:model.pi0_step]
        actions = _infer_action_chunk(model, instruction, input_rgb_arr, input_state)
        actions = ee_10d_to_ee(actions)
        print(actions[0])

        for action in actions:
            TASK_ENV.take_action(action, action_type="ee")
            #TASK_ENV.take_action(action, action_type="cts")
            if _update_obs_every_action(TASK_ENV):
                observation = TASK_ENV.get_obs()
                input_rgb_arr, input_state = encode_obs(observation)
                input_state = ee_to_ee_10d(input_state)
                #model.update_observation_window(input_rgb_arr, input_state)
                model.call(func_name="update_observation_window", obs = (input_rgb_arr, input_state))

    elif action_type == "cts":
        last_Qa = None
        #if model.observation_window is None:
        instruction = TASK_ENV.get_instruction()

        input_rgb_arr, input_state = encode_obs(observation)
        input_state = cal_cts(input_state)
        last_Qa = input_state[3:7]

        # ======== Get Action ========

        #actions = model.call(func_name='get_action')[:model.pi0_step]
        actions = _infer_action_chunk(model, instruction, input_rgb_arr, input_state)
        print(actions[0])

        for action in actions:
            #TASK_ENV.take_action(action, action_type="ee")
            TASK_ENV.take_action(action, action_type="cts")
            if _update_obs_every_action(TASK_ENV):
                observation = TASK_ENV.get_obs()
                input_rgb_arr, input_state = encode_obs(observation)
                input_state = cal_cts(input_state)
                print("@")
                if np.dot(last_Qa, input_state[3:7]) < -1e-4:
                    print(last_Qa)
                    print(input_state[3:7])
                    input_state[3:7] = -input_state[3:7]
                    #
                #model.update_observation_window(input_rgb_arr, input_state)
                model.call(func_name="update_observation_window", obs = (input_rgb_arr, input_state))
                last_Qa = input_state[3:7]

    elif action_type == "cts_10d":
        last_Qa = None
        #if model.observation_window is None:
        instruction = TASK_ENV.get_instruction()

        input_rgb_arr, input_state = encode_obs(observation)
        input_state = cal_cts(input_state)
        input_state = ee_to_ee_10d(input_state)
        #last_Qa = input_state[3:7]

        # ======== Get Action ========

        #actions = model.call(func_name='get_action')[:model.pi0_step]
        actions = _infer_action_chunk(model, instruction, input_rgb_arr, input_state)
        actions = ee_10d_to_ee(actions)
        # 兼容action_type="cts"控制
        idx = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 7, 15]
        actions = actions[:, idx]

        print(actions[0])

        for action in actions:
            #TASK_ENV.take_action(action, action_type="ee")
            TASK_ENV.take_action(action, action_type="cts")
            if _update_obs_every_action(TASK_ENV):
                observation = TASK_ENV.get_obs()
                input_rgb_arr, input_state = encode_obs(observation)
                input_state = cal_cts(input_state)
                input_state = ee_to_ee_10d(input_state)
                #if np.dot(last_Qa, input_state[3:7]) < -1e-4:
                    #print(last_Qa)
                    #print(input_state[3:7])
                    #input_state[3:7] = -input_state[3:7]
                    #
                #model.update_observation_window(input_rgb_arr, input_state)
                model.call(func_name="update_observation_window", obs = (input_rgb_arr, input_state))
                #last_Qa = input_state[3:7]
    # ============================


def reset_model(model):
    model.reset_obsrvationwindows()
