"""
几何工具：四元数、位姿、内参、动作计算

三个采集脚本（r2r / panoramic / heatmap）共用的几何变换函数。
"""
import math
import numpy as np
from typing import Dict


def quaternion_to_rotation_matrix(q) -> np.ndarray:
    """
    将 Quaternion 转换为 3x3 旋转矩阵。
    支持 Magnum Quaternion、numpy-quaternion 和 array-like 三种输入。
    """
    if hasattr(q, "scalar") and hasattr(q, "vector"):
        w, x, y, z = q.scalar, q.vector.x, q.vector.y, q.vector.z
    elif hasattr(q, "w") and hasattr(q, "x"):
        w, x, y, z = q.w, q.x, q.y, q.z
    elif hasattr(q, "__getitem__"):
        w, x, y, z = q[0], q[1], q[2], q[3]
    else:
        raise ValueError(f"Unknown quaternion type: {type(q)}")

    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


def compute_camera_pose(agent_state, T_agent_cam: np.ndarray) -> np.ndarray:
    """计算相机到世界的 4x4 变换矩阵  T_world_cam = T_world_agent @ T_agent_cam"""
    R_agent = quaternion_to_rotation_matrix(agent_state.rotation)
    T_w_agent = np.eye(4, dtype=np.float32)
    T_w_agent[:3, :3] = R_agent
    T_w_agent[:3, 3] = agent_state.position
    return T_w_agent @ T_agent_cam


def get_sensor_extrinsics(config) -> np.ndarray:
    """
    从 Habitat 配置获取 RGB 传感器外参矩阵 T_agent_cam (4x4)。
    支持 3-element euler (roll, pitch, yaw) 和 4-element quaternion (x, y, z, w) 两种 ORIENTATION 格式。
    """
    sensor_cfg = config.SIMULATOR.RGB_SENSOR
    sensor_position = np.array(sensor_cfg.POSITION, dtype=np.float32)

    sensor_rotation = np.eye(3, dtype=np.float32)
    if hasattr(sensor_cfg, "ORIENTATION"):
        orientation = sensor_cfg.ORIENTATION
        if isinstance(orientation, (list, tuple)):
            if len(orientation) == 3:
                roll, pitch, yaw = orientation
                cy, sy = np.cos(yaw), np.sin(yaw)
                cp, sp = np.cos(pitch), np.sin(pitch)
                cr, sr = np.cos(roll), np.sin(roll)
                sensor_rotation = np.array([
                    [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                    [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                    [-sp, cp * sr, cp * cr],
                ], dtype=np.float32)
            elif len(orientation) == 4:
                x, y, z, w = orientation
                norm = math.sqrt(w * w + x * x + y * y + z * z)
                if norm > 0:
                    w, x, y, z = w / norm, x / norm, y / norm, z / norm
                    sensor_rotation = np.array([
                        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
                    ], dtype=np.float32)
        else:
            sensor_rotation = quaternion_to_rotation_matrix(orientation)

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = sensor_rotation
    T[:3, 3] = sensor_position
    return T


def compute_intrinsics(config) -> Dict:
    """
    从 Habitat 配置计算 Pinhole 相机内参。
    返回包含 width / height / hfov / vfov / fx / fy / cx / cy / K 的字典。
    """
    rgb_cfg = config.SIMULATOR.RGB_SENSOR
    width = int(rgb_cfg.WIDTH)
    height = int(rgb_cfg.HEIGHT)
    hfov_deg = float(getattr(rgb_cfg, "HFOV", 90.0))
    hfov_rad = math.radians(hfov_deg)

    fx = width / (2.0 * math.tan(hfov_rad / 2.0))
    fy = fx
    cx, cy = width / 2.0, height / 2.0
    vfov_deg = math.degrees(2.0 * math.atan(height / (2.0 * fy)))

    return {
        "projection": "pinhole",
        "width": width, "height": height,
        "hfov": hfov_deg, "vfov": vfov_deg,
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "K": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
    }


def compute_2d_action(pose_before, pose_after) -> np.ndarray:
    """
    从相邻位姿计算 agent-local 2D 连续动作 (dx, dy)。

    坐标系约定（Habitat）: X 向右, Y 向上, -Z 向前
    返回: dx（右/左）, dy（前/后，前为正，即 -dz）
    """
    T_rel = np.linalg.inv(np.asarray(pose_before, dtype=np.float32)) \
        @ np.asarray(pose_after, dtype=np.float32)
    dx = T_rel[0, 3]
    dy = -T_rel[2, 3]
    return np.array([dx, dy], dtype=np.float32)


def discrete_action_to_name(action: int) -> str:
    """将 HabitatSimActions 枚举值转换为名称"""
    names = {0: "STOP", 1: "MOVE_FORWARD", 2: "TURN_LEFT", 3: "TURN_RIGHT"}
    return names.get(action, f"UNKNOWN({action})")
