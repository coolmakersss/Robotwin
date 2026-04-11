import numpy as np
import torch
import dill
import os, sys

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(parent_directory)

from pi_model import *



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
    cts_pose_state = np.concatenate([Pa, Qa, Pr, Qr, end_pose_vector[7:8], end_pose_vector[15:16]])
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
    action_type = "cts"
    
    if action_type == "ee":
        #if model.observation_window is None:
        instruction = TASK_ENV.get_instruction()
        model.call(func_name="set_language",obs=instruction)

        input_rgb_arr, input_state = encode_obs(observation)
        model.call(func_name="update_observation_window", obs = (input_rgb_arr, input_state))

        # ======== Get Action ========

        #actions = model.call(func_name='get_action')[:model.pi0_step]
        actions = model.call(func_name='get_action')[:20]
        print(actions[0])

        for action in actions:
            TASK_ENV.take_action(action, action_type="ee")
            #TASK_ENV.take_action(action, action_type="cts")
            observation = TASK_ENV.get_obs()
            input_rgb_arr, input_state = encode_obs(observation)
            #model.update_observation_window(input_rgb_arr, input_state)
            model.call(func_name="update_observation_window", obs = (input_rgb_arr, input_state))

    elif action_type == "cts":
        last_Qa = None
        #if model.observation_window is None:
        instruction = TASK_ENV.get_instruction()
        model.call(func_name="set_language",obs=instruction)

        input_rgb_arr, input_state = encode_obs(observation)
        input_state = cal_cts(input_state)
        model.call(func_name="update_observation_window", obs = (input_rgb_arr, input_state))
        last_Qa = input_state[3:7]

        # ======== Get Action ========

        #actions = model.call(func_name='get_action')[:model.pi0_step]
        actions = model.call(func_name='get_action')[:20]
        print(actions[0])

        for action in actions:
            #TASK_ENV.take_action(action, action_type="ee")
            TASK_ENV.take_action(action, action_type="cts")
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

    # ============================


def reset_model(model):
    model.reset_obsrvationwindows()
