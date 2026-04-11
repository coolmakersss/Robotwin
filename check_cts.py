from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import numpy as np

def cal_cts(end_pose_vector):
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
    Qa = mid_rotation_scipy(R1_true, R2_true, t=0.5)
    cts_pose_state = np.concatenate([Pa, Qa, Pr, Qr, end_pose_vector[7:8], end_pose_vector[14:15]])
    #print(cts_pose_state)
    return cts_pose_state

def q1_q2_from_qa_qr(qa, qr, eps=1e-9):
    Ra = R.from_quat(qa).as_matrix()
    rot_r = R.from_quat(qr)
    rotvec = rot_r.as_rotvec()

    half_rotvec = 0.5 * rotvec
    S = R.from_rotvec(half_rotvec).as_matrix()

    R1 = Ra @ S.T
    R2 = Ra @ S
    print(R1)
    print(R1.reshape(-1,3,3))
    
    q1 = R.from_matrix(R1.reshape(-1,3,3)).as_quat()
    q2 = R.from_matrix(R2.reshape(-1,3,3)).as_quat()

    return q1, q2

end_pose_vector = [-3.9955231e-04, -2.5228554e-01,  9.4769859e-01,  7.0962626e-01,
  -1.4301739e-02,  1.5875442e-02,  7.1335441e-01,  5.9318870e-01,
   1.3637423e-04,  4.7534392e-03, -4.7723465e-02,  8.0093294e-02,
   5.3119943e-03,  9.9665850e-01,  9.9943471e-01,  7.2879243e-01]
print(np.linalg.norm(end_pose_vector[10:14], ord=2))
p1 = [-3.9955231e-04, -2.5228554e-01,  9.4769859e-01]
#R1 = R.from_euler('xyz', [40,30,10], degrees=True).as_quat()
R1 = [7.0962626e-01, -1.4301739e-02,  1.5875442e-02,  7.1335441e-01]
p2 = [4.7534392e-03, -4.7723465e-02,  8.0093294e-02]
#R2 = R.from_euler('xyz', [30,20,50], degrees=True).as_quat()
R2 = [-7.0962626e-01, -1.4301739e-02,  1.5875442e-02,  7.1335441e-01]
print(np.linalg.norm(R1, ord=2))
print(np.linalg.norm(R2, ord=2))
print(R1,R2)
cts_pose_state = cal_cts(np.concatenate((p1,R1,[1.0],p2,R2,[1.0])))
print(cts_pose_state)
print(np.linalg.norm(cts_pose_state[10:14], ord=2))
print(q1_q2_from_qa_qr(cts_pose_state[3:7],cts_pose_state[10:14]))