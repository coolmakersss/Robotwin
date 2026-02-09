import numpy as np
from .dp_model import DP
import yaml

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

def encode_obs(observation):
    head_cam = (np.moveaxis(observation["observation"]["head_camera"]["rgb"], -1, 0) / 255)
    left_cam = (np.moveaxis(observation["observation"]["left_camera"]["rgb"], -1, 0) / 255)
    right_cam = (np.moveaxis(observation["observation"]["right_camera"]["rgb"], -1, 0) / 255)
    obs = dict(
        head_cam=head_cam,
        left_cam=left_cam,
        right_cam=right_cam,
    )
    obs["joint_pos"] = observation["joint_action"]["vector"]
    obs["agent_pos"] = np.concatenate([observation["endpose"]["left_endpose"],
                        observation["endpose"]["left_gripper"].reshape(1),
                        observation["endpose"]["right_endpose"],
                        observation["endpose"]["right_gripper"].reshape(1)])
    obs["cts_pos"] = cal_cts(obs["agent_pos"])
    return obs


def get_model(usr_args):
    ckpt_file = f"./policy/DP/checkpoints/{usr_args['task_name']}-{usr_args['ckpt_setting']}-{usr_args['expert_data_num']}-{usr_args['seed']}/{usr_args['mode']}/{usr_args['checkpoint_num']}.ckpt"
    #action_dim = usr_args['left_arm_dim'] + usr_args['right_arm_dim'] + 2 # 2 gripper
    action_dim = usr_args['action_dim']
    #print(action_dim)
    if usr_args['mode'] == "cts":
        load_config_path = f'./policy/DP/diffusion_policy/config/robot_dp_{action_dim}_cts.yaml'
    elif usr_args['mode'] == "joint":
        load_config_path = f'./policy/DP/diffusion_policy/config/robot_dp_{action_dim}_joint.yaml'
    else:
        load_config_path = f'./policy/DP/diffusion_policy/config/robot_dp_{action_dim}.yaml'
    with open(load_config_path, "r", encoding="utf-8") as f:
        model_training_config = yaml.safe_load(f)
    
    n_obs_steps = model_training_config['n_obs_steps']
    n_action_steps = model_training_config['n_action_steps']
    
    return DP(ckpt_file, n_obs_steps=n_obs_steps, n_action_steps=n_action_steps)


def eval(TASK_ENV, model, observation):
    """
    TASK_ENV: Task Environment Class, you can use this class to interact with the environment
    model: The model from 'get_model()' function
    observation: The observation about the environment
    """
    obs = encode_obs(observation)
    instruction = TASK_ENV.get_instruction()

    # ======== Get Action ========
    if len(model.runner.obs) == 0:
        actions = model.get_action(obs)
    else:
        actions = model.get_action()

    for action in actions:
        TASK_ENV.take_action(action=action, action_type="ee")
        #TASK_ENV.take_action(action=action, action_type="cts")
        #TASK_ENV.take_action(action=action)
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs)

def reset_model(model):
    model.reset_obs()
