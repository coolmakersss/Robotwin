

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


#task_name = "lift_pot"
#task_name = "grab_roller"
#task_name = "handover_mic" 
#task_name = "place_bread_skillet" 
task_name = "place_cans_plasticbox"
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

while current_ep < num:
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
mode = "delta_eef"
for mode in modes:
    plt.clf()
    if mode == "cts":
        brats_array = np.array(cts_action_arrays)  # [6200, 57600]
    elif mode == "delta_cts":
        brats_array = np.array(delta_cts_action_arrays)  # [6200, 57600]
    elif mode == "eef":
        brats_array = np.array(end_action_arrays)
    elif mode == "delta_eef":
        brats_array = np.array(delta_end_action_arrays)
    elif mode == "joint":
        brats_array = np.array(joint_action_arrays)
    #brats_label_array = np.array(action_array_labels, dtype='uint8')   # [6200]


    delta_mag_full = np.array(action_array_labels)
    #print(delta_mag_full)
    #delta_mag_full = np.concatenate([[0.0], delta_mag])  # shape (T,)
    n_classes = 8

    # 推荐：用分位数（quantile）分箱使每一类样本数较均衡
    # 若想用等宽分箱，替换下面一行：bins = np.linspace(delta_mag_full.min(), delta_mag_full.max(), n_classes+1)
    percentiles = np.linspace(0, 100, n_classes+1)
    bins = np.percentile(delta_mag_full, percentiles)  # length n_classes+1

    # 处理极端情况：所有值都相同则直接归为类0
    if np.allclose(bins[0], bins[-1]):
        class_labels = np.zeros_like(delta_mag_full, dtype=int)
    else:
        # np.digitize 将值放入 bins 区间，返回 1..len(bins)
        # 我们希望得到 0..n_classes-1
        # 注意 np.digitize 的 right 参数可改分界方向，这里选择默认左闭右开
        class_idx = np.digitize(delta_mag_full, bins[1:-1], right=False)
        class_labels = class_idx.astype(int)  # 0..n_classes-1

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=42).fit_transform(brats_array)
    cmap = plt.get_cmap('inferno')  # 也可选 'plasma','inferno','magma','cividis' 等
    colors_list = cmap(np.linspace(0, 1, n_classes))

    plt.figure(figsize=(8,8))
    s = 4  # 点大小，可调

    for i in range(n_classes):
        idxs = (class_labels == i)
        if np.sum(idxs) == 0:
            continue
        plt.scatter(tsne[idxs,0], tsne[idxs,1],
                    s=s, color=colors_list[i], label=f'class {i}')


    #x_min, x_max = tsne.min(0), tsne.max(0)
    #tsne_norm = (tsne - x_min) / (x_max - x_min)

    #normal_idxs = (brats_label_array == 0)
    #abnorm_idxs = (brats_label_array == 1)
    #tsne_normal = tsne[normal_idxs]
    #tsne_abnormal = tsne[abnorm_idxs]

    #plt.figure(figsize=(8, 8))
    #plt.scatter(tsne_normal[:, 0], tsne_normal[:, 1], 1, color='green', label='uncoolaberate actions')
    # tsne_normal[i, 0]为横坐标，X_norm[i, 1]为纵坐标，1为散点图的面积， color给每个类别设定颜色
    #plt.scatter(tsne_abnormal[:, 0], tsne_abnormal[:, 1], 1, color='red', label='coolaberate actions')
    plt.legend(loc='upper left')
    plt.savefig(f"./tsne_{task_name}_{task_config}_{mode}.png")
