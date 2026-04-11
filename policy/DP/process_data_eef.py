import pickle, os
import numpy as np
import pdb
from copy import deepcopy
import zarr
import shutil
import argparse
import yaml
import cv2
import h5py
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


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

        # 最短路径检查：如果点积为负，则翻转 q2
        #（四元数 q 和 -q 表示相同旋转，但 slerp 走向会不同，点积 < 0 表明它们在球面上相反方向）
        if np.dot(q1, q2) < 0.0:
            q2 = -q2
        #if np.dot(q1, q2) > 1.0 - 1e-4:
        #    Q_mid = (q1 + q2) / 2
        #    return Q_mid
        #rots = R.from_quat(np.stack([q1, q2], axis=0))  # shape (2,)
        #key_times = np.array([0.0, 1.0])                  # 对应两个关键帧的时间点
        #slerp = Slerp(key_times, rots)
        #Q_mid = slerp([t]).as_quat()[0]
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

def main():
    parser = argparse.ArgumentParser(description="Process some episodes.")
    parser.add_argument(
        "task_name",
        type=str,
        help="The name of the task (e.g., beat_block_hammer)",
    )
    parser.add_argument("task_config", type=str)
    parser.add_argument(
        "expert_data_num",
        type=int,
        help="Number of episodes to process (e.g., 50)",
    )
    args = parser.parse_args()

    task_name = args.task_name
    num = args.expert_data_num
    task_config = args.task_config

    load_dir = "../../data/" + str(task_name) + "/" + str(task_config)

    total_count = 0

    save_dir = f"./data/{task_name}-{task_config}-{num}.zarr"

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    current_ep = 0

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    head_camera_arrays, front_camera_arrays, left_camera_arrays, right_camera_arrays = (
        [],
        [],
        [],
        [],
    )
    episode_ends_arrays, action_arrays, joint_state_arrays, end_state_arrays, cts_state_arrays, joint_action_arrays, end_action_arrays, cts_action_arrays, delta_action_arrays, delta_cts_action_arrays = (
        [],
        [],
        [],
        [],
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
        Qa_last = np.array([0.,0.,0.,0.])
        previous_cts = None
        previous_end = None

        for j in range(0, left_gripper_all.shape[0]):

            head_img_bit = image_dict_all["head_camera"][j]
            left_img_bit = image_dict_all["left_camera"][j]
            right_img_bit = image_dict_all["right_camera"][j]
            joint_state = joint_vector_all[j]
            end_pose_state = end_pose_vector_all[j]
            if j == 0:
                Qa_last = np.array([0,0,0,0])
            cts_pose_state = cal_cts(end_pose_vector_all[j], Qa_last)
            Qa_last = cts_pose_state[3:7]
            if previous_cts is not None:
                delta_cts_action_arrays.append(cts_pose_state - previous_cts)
                delta_action_arrays.append(end_pose_state - previous_end)
            previous_cts = cts_pose_state
            previous_end = end_pose_state
            if j == 0:
                previous_cts = cts_pose_state
                previous_end = end_pose_state            
            if j != left_gripper_all.shape[0] - 1:
                head_img = cv2.imdecode(np.frombuffer(head_img_bit, np.uint8), cv2.IMREAD_COLOR)
                left_img = cv2.imdecode(np.frombuffer(left_img_bit, np.uint8), cv2.IMREAD_COLOR)
                right_img = cv2.imdecode(np.frombuffer(right_img_bit, np.uint8), cv2.IMREAD_COLOR)
                head_camera_arrays.append(head_img)
                left_camera_arrays.append(left_img)
                right_camera_arrays.append(right_img)
                joint_state_arrays.append(joint_state)
                end_state_arrays.append(end_pose_state)
                cts_state_arrays.append(cts_pose_state)
            if j != 0:
                joint_action_arrays.append(joint_state)
                end_action_arrays.append(end_pose_state)
                cts_action_arrays.append(cts_pose_state)

        current_ep += 1
        total_count += left_gripper_all.shape[0] - 1
        episode_ends_arrays.append(total_count)

    print()
    episode_ends_arrays = np.array(episode_ends_arrays)
    # action_arrays = np.array(action_arrays)
    joint_state_arrays = np.array(joint_state_arrays)
    end_state_arrays = np.array(end_state_arrays)
    cts_state_arrays = np.array(cts_state_arrays)
    head_camera_arrays = np.array(head_camera_arrays)
    left_camera_arrays = np.array(left_camera_arrays)
    right_camera_arrays = np.array(right_camera_arrays)
    joint_action_arrays = np.array(joint_action_arrays)
    end_action_arrays = np.array(end_action_arrays)
    cts_action_arrays = np.array(cts_action_arrays)

    delta_action_arrays = np.array(delta_action_arrays)
    delta_cts_action_arrays = np.array(delta_cts_action_arrays)
    print(len(cts_action_arrays))
    print(len(delta_cts_action_arrays))

    head_camera_arrays = np.moveaxis(head_camera_arrays, -1, 1)  # NHWC -> NCHW
    left_camera_arrays = np.moveaxis(left_camera_arrays, -1, 1)  # NHWC -> NCHW
    right_camera_arrays = np.moveaxis(right_camera_arrays, -1, 1)  # NHWC -> NCHW

    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    # action_chunk_size = (100, action_arrays.shape[1])
    joint_state_chunk_size = (100, joint_state_arrays.shape[1])
    end_state_chunk_size = (100, end_state_arrays.shape[1])
    cts_state_chunk_size = (100, cts_state_arrays.shape[1])
    joint_chunk_size = (100, joint_action_arrays.shape[1])
    end_chunk_size = (100, end_action_arrays.shape[1])
    cts_chunk_size = (100, cts_action_arrays.shape[1])
    delta_chunk_size = (100, delta_action_arrays.shape[1])
    delta_cts_chunk_size = (100, delta_cts_action_arrays.shape[1])
    
    head_camera_chunk_size = (100, *head_camera_arrays.shape[1:])
    left_camera_chunk_size = (100, *left_camera_arrays.shape[1:])
    right_camera_chunk_size = (100, *right_camera_arrays.shape[1:])

    zarr_data.create_dataset(
        "head_camera",
        data=head_camera_arrays,
        chunks=head_camera_chunk_size,
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "left_camera",
        data=left_camera_arrays,
        chunks=left_camera_chunk_size,
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "right_camera",
        data=right_camera_arrays,
        chunks=right_camera_chunk_size,
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "joint_state",
        data=joint_state_arrays,
        chunks=joint_state_chunk_size,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "end_state",
        data=end_state_arrays,
        chunks=end_state_chunk_size,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "cts_state",
        data=cts_state_arrays,
        chunks=cts_state_chunk_size,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "joint_action",
        data=joint_action_arrays,
        chunks=joint_chunk_size,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "action",
        data=end_action_arrays,
        chunks=end_chunk_size,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "cts_action",
        data=cts_action_arrays,
        chunks=cts_chunk_size,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "delta_action",
        data=delta_action_arrays,
        chunks=delta_chunk_size,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "delta_cts_action",
        data=delta_cts_action_arrays,
        chunks=delta_cts_chunk_size,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_meta.create_dataset(
        "episode_ends",
        data=episode_ends_arrays,
        dtype="int64",
        overwrite=True,
        compressor=compressor,
    )


if __name__ == "__main__":
    main()
