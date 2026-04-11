

import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from glob import glob
#import nibabel as nib
import h5py, os
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

def summarize_var(X, axis=0):
    X = np.asarray(X)
    if X.size == 0:
        return np.array([]), 0.0, 0.0
    per_dim = np.var(X, axis=axis, ddof=0)
    mean_dim = float(np.mean(per_dim))
    total = float(np.sum(per_dim))
    return per_dim, mean_dim, total

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
        # Create 16-dimensional vector: left_endpose(7) + left_gripper(1) + right_endpose(7) + right_gripper(1)
        joint_vector = root["/joint_action/vector"][()]
        end_pose_vector = np.concatenate([
            left_arm,  # 7 dims: xyz + quaternion
            left_gripper.reshape(-1, 1),   # 1 dim: gripper
            right_arm, # 7 dims: xyz + quaternion  
            right_gripper.reshape(-1, 1)  # 1 dim: gripper
        ], axis=1)

        image_dict = dict()
        for cam_name in root[f"/observation/"].keys():
            image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]

    return left_gripper, left_arm, right_gripper, right_arm, joint_vector, end_pose_vector, image_dict

def cal_cts(end_pose_vector, Qa_last):
    Pa = (end_pose_vector[8:11] + end_pose_vector[:3]) / 2.0
    Pr = end_pose_vector[8:11] - end_pose_vector[:3]
    R1_true = R.from_quat(end_pose_vector[3:7]).as_matrix()
    R2_true = R.from_quat(end_pose_vector[11:15]).as_matrix()
    # 相对旋转
    Rr = R1_true.T @ R2_true    
    Qr = R.from_matrix(Rr).as_quat()
    # 计算中点 Qa（与前面的定义一致）
    def mid_rotation_scipy_before(R1, R2, t=0.5):
        """
        用 scipy 的 Slerp 在 SO(3) 上插值。
        R1, R2: (3,3) 旋转矩阵或可以转换为矩阵的 array-like
        t: 插值参数，0 -> R1, 1 -> R2，默认 0.5（中点）
        返回: (3,3) 旋转矩阵
        """
        rots = R.from_matrix(np.stack([R1, R2], axis=0))  # shape (2,)
        key_times = np.array([0.0, 1.0])                  # 对应两个关键帧的时间点
        slerp = Slerp(key_times, rots)
        Q_mid = slerp([t]).as_quat()[0]
        return Q_mid
    def mid_rotation_scipy(R1, R2, t=0.5):
        """
        用 scipy 的 Slerp 在 SO(3) 上插值。
        R1, R2: (3,3) 旋转矩阵或可以转换为矩阵的 array-like
        t: 插值参数，0 -> R1, 1 -> R2，默认 0.5（中点）
        返回: (3,3) 旋转矩阵
        """
        # 把两个矩阵打包为 Rotation 对象
        q1 = R.from_matrix(R1).as_quat()
        q2 = R.from_matrix(R2).as_quat()

        #if np.dot(q1, q2) < 0.0:
        #    q2 = -q2
        Q_mid = q1
        return Q_mid
    Qa = mid_rotation_scipy(R1_true, R2_true, t=0.5)
    if np.dot(Qa, Qa_last) < -1e-4:
        Qa = -Qa
    Qa_before = mid_rotation_scipy_before(R1_true, R2_true, t=0.5)
    #if not np.allclose(Qa, Qa_before, atol=1e-5):
    #    print("@: ", Qa, Qa_before)
    #print("qa: ", Qa)
    #print("qr: ", Qr)
    #print("q1: ", end_pose_vector[3:7])
    #print("q2: ", end_pose_vector[11:15])
    #print()
    cts_pose_state = np.concatenate([Pa, Qa, Pr, Qr, end_pose_vector[7:8], end_pose_vector[15:16]])
    #print(cts_pose_state)
    return cts_pose_state


task_name = "lift_pot"
#task_name = "grab_roller"
#task_name = "handover_mic" 
#task_name = "place_bread_skillet" 
#task_name = "place_cans_plasticbox"
task_config = "aloha-agilex_clean_50"
#task_config = "arx-x5_clean_50"
num = 50
current_ep = 0
total_count = 0
load_dir = "./data/" + str(task_name) + "/" + str(task_config)

joint_action_arrays, end_action_arrays, cts_action_arrays, delta_end_action_arrays, action_array_labels, delta_cts_action_arrays= (
    [],
    [],
    [],
    [],
    [],
    [],
)

while current_ep < 1:
    print(f"processing episode: {current_ep + 1} / {num}", end="\r")
    load_path = os.path.join(load_dir, f"data/episode{current_ep}.hdf5")
    (
        left_gripper_all,
        left_arm_all,
        right_gripper_all,
        right_arm_all,
        joint_vector_all,
        end_pose_vector_all,
        image_dict_all,
    ) = load_hdf5(load_path)

    for j in range(0, left_gripper_all.shape[0]):

        joint_state = joint_vector_all[j]
        end_pose_state = end_pose_vector_all[j]
        if j == 0:
            Qa_last = np.array([0,0,0,0])
        cts_pose_state = cal_cts(end_pose_vector_all[j], Qa_last)
        Qa_last = cts_pose_state[3:7]

        joint_action_arrays.append(joint_state)
        end_action_arrays.append(end_pose_state)
        cts_action_arrays.append(cts_pose_state)
        

        if j != 0 :
            delta_cts_action_arrays.append(cts_action_arrays[-1] - cts_action_arrays[-2])
            delta_end_action_arrays.append(end_action_arrays[-1] - end_action_arrays[-2])
            if j == 1:
                delta_cts_action_arrays.append(cts_action_arrays[-1] - cts_action_arrays[-2])
                delta_end_action_arrays.append(end_action_arrays[-1] - end_action_arrays[-2])
            action_array_labels.append(np.linalg.norm((end_action_arrays[-1][:3] - end_action_arrays[-2][:3]) - (end_action_arrays[-1][8:11] - end_action_arrays[-2][8:11])))
        else :
            
            action_array_labels.append(0)
    current_ep += 1
    total_count += left_gripper_all.shape[0] - 1

joint_action_arrays = np.asarray(joint_action_arrays)
end_action_arrays = np.asarray(end_action_arrays)
cts_action_arrays = np.asarray(cts_action_arrays)
delta_end_action_arrays = np.asarray(delta_end_action_arrays)
delta_cts_action_arrays = np.asarray(delta_cts_action_arrays)

def plot_actions_over_time(mode):
    """
    可视化 (time_step, action_dim) 形状的数据。
    每个 action 维度占据一个独立的子图。
    """
    if mode == "cts":
        action_data = np.array(np.delete(cts_action_arrays, [14, 15], axis=1))  # [6200, 57600]
    elif mode == "delta_cts":
        action_data = np.array(np.delete(delta_cts_action_arrays, [14, 15], axis=1))  # [6200, 57600]
    elif mode == "eef":
        action_data = np.array(np.delete(end_action_arrays, [7, 15], axis=1))
    elif mode == "delta_eef":
        action_data = np.array(np.delete(delta_end_action_arrays, [7, 15], axis=1))
    elif mode == "joint":
        action_data = np.array(np.delete(joint_action_arrays, [6, 13], axis=1))
    
    t_steps, a_dim = action_data.shape
    time_axis = np.arange(t_steps)

    # 创建子图
    # nrows=a_dim: 有多少个动作维度就创建多少行
    # sharex=True: 所有子图共享 X 轴，这样拖动或缩放时会同步，且只在最底部显示时间标签
    # figsize: 根据维度数量动态调整总图高度，避免太拥挤
    fig, axs = plt.subplots(nrows=a_dim, ncols=1, figsize=(12, 2.5 * a_dim), sharex=True)

    # 处理 action_dim 为 1 的特殊情况（此时 axs 不是列表，而是单个对象）
    if a_dim == 1:
        axs = [axs]

    # 遍历每个维度进行绘图
    for i, ax in enumerate(axs):
        # 提取当前维度的数据：所有时间步的第 i 列
        dim_data = action_data[:, i]
        # 绘制曲线
        # 可以根据需要修改颜色、线型等，例如 color='royalblue', linewidth=1.5
        ax.plot(time_axis, dim_data, label=f'Dim {i}')
        # 设置 Y 轴标签，表明是第几个维度
        ax.set_ylabel(f'Action Dim {i}', fontsize=10)
        # 添加网格线，便于观察数值
        ax.grid(True, linestyle='--', alpha=0.6)
        # 可选：在每个子图中显示图例（如果曲线很简单，也可以省略）
        # ax.legend(loc='upper right')

    # 只在最后一个子图设置 X 轴标签
    axs[-1].set_xlabel('Time Step', fontsize=12)
    
    # 设置整张大图的标题
    fig.suptitle(f'Visualization of Action Dimensions Over Time (Shape: {t_steps}x{a_dim})', fontsize=14, y=0.99)
    
    # 自动调整子图间距，防止标签重叠
    plt.tight_layout()
    plt.savefig(f"./view_dim_{task_name}_{task_config}_{mode}.png")
    # 显示图像
    #plt.show()
# 运行可视化函数
mode = "eef"
mode = "cts"
plot_actions_over_time(mode)

exit()


_,joint_mean_dim_left,_ = summarize_var(joint_action_arrays[:, :6])
_,joint_mean_dim_right,_ = summarize_var(joint_action_arrays[:, 7:13])
_,eef_mean_dim_left,_ = summarize_var(end_action_arrays[:, :7])
_,eef_mean_dim_right,_ = summarize_var(end_action_arrays[:, 8:15])
_,cts_mean_dim_abs,_ = summarize_var(cts_action_arrays[:, :7])
_,cts_mean_dim_rel,_ = summarize_var(cts_action_arrays[:, 7:14])
_,delta_eef_mean_dim_left,_ = summarize_var(delta_end_action_arrays[:, :7])
_,delta_eef_mean_dim_right,_ = summarize_var(delta_end_action_arrays[:, 8:15])
_,delta_cts_mean_dim_abs,_ = summarize_var(delta_cts_action_arrays[:, :7])
_,delta_cts_mean_dim_rel,_ = summarize_var(delta_cts_action_arrays[:, 7:14])

print("joint: ")
print(joint_mean_dim_left)
print(joint_mean_dim_right)
print("eef: ")
print(eef_mean_dim_left)
print(eef_mean_dim_right)
print("cts: ")
print(cts_mean_dim_abs)
print(cts_mean_dim_rel)
print("delta_eef: ")
print(delta_eef_mean_dim_left)
print(delta_eef_mean_dim_right)
print("delta_cts: ")
print(delta_cts_mean_dim_abs)
print(delta_cts_mean_dim_rel)

modes = ["joint","eef","cts","delta_eef","delta_cts"]

