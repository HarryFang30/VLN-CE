"""
多方向（全景）采集：常量定义 + capture_multiview 函数

支持 front/right/back/left 四方向，以及可选的 front_down 俯视观测。
"""
import numpy as np
from types import SimpleNamespace
from .geometry import compute_camera_pose

# ==================== 方向常量 ====================

DIRECTIONS = ["front", "right", "back", "left"]

DIRECTION_YAW_OFFSETS = {
    "front": 0.0,
    "right": -np.pi / 2,
    "back": np.pi,
    "left": np.pi / 2,
}

LOOKDOWN_PITCH_DEG = 30.0
LOOKDOWN_DIRECTION = "front_down"


# ==================== 四元数辅助 ====================

def ensure_np_quaternion(q):
    """将 Magnum/Habitat 四元数转为 numpy-quaternion，保证可做四元数乘法。"""
    if isinstance(q, np.quaternion):
        return q
    if hasattr(q, "scalar") and hasattr(q, "vector"):
        return np.quaternion(q.scalar, q.vector.x, q.vector.y, q.vector.z)
    if hasattr(q, "w"):
        return np.quaternion(q.w, q.x, q.y, q.z)
    raise ValueError(f"Cannot convert {type(q)} to np.quaternion")


# ==================== 多视角采集 ====================

def capture_multiview(
    sim,
    T_agent_cam: np.ndarray,
    lookdown_pitch_deg: float = 0.0,
) -> dict:
    """
    在当前 agent 位置采集 4 方向 RGB+Depth+Pose，以及可选的 front_down 视角。

    完成后恢复 agent 的原始位姿，不影响路径跟随器。

    Args:
        sim: Habitat 仿真器
        T_agent_cam: 传感器外参矩阵 [4, 4]
        lookdown_pitch_deg: >0 时额外采集一个向下倾斜的 front_down 观测

    Returns:
        {direction: {"rgb": ndarray, "depth": ndarray, "pose": ndarray[4,4]}}
    """
    agent_state = sim.get_agent_state()
    orig_pos = agent_state.position.copy()
    orig_rot = ensure_np_quaternion(agent_state.rotation)

    results = {}

    for d in DIRECTIONS:
        yaw = DIRECTION_YAW_OFFSETS[d]
        if abs(yaw) < 1e-6:
            rot = orig_rot
        else:
            q_yaw = np.quaternion(np.cos(yaw / 2), 0, np.sin(yaw / 2), 0)
            rot = orig_rot * q_yaw

        sim.set_agent_state(orig_pos, rot)
        obs = sim.get_sensor_observations()
        fake_state = SimpleNamespace(position=orig_pos, rotation=rot)
        pose = compute_camera_pose(fake_state, T_agent_cam)
        results[d] = {"rgb": obs["rgb"].copy(), "depth": obs["depth"].copy(), "pose": pose}

    if lookdown_pitch_deg > 0:
        pitch_rad = np.radians(lookdown_pitch_deg)
        q_pitch = np.quaternion(np.cos(pitch_rad / 2), np.sin(pitch_rad / 2), 0, 0)
        rot_down = orig_rot * q_pitch
        sim.set_agent_state(orig_pos, rot_down)
        obs = sim.get_sensor_observations()
        fake_state = SimpleNamespace(position=orig_pos, rotation=rot_down)
        pose = compute_camera_pose(fake_state, T_agent_cam)
        results[LOOKDOWN_DIRECTION] = {
            "rgb": obs["rgb"].copy(),
            "depth": obs["depth"].copy(),
            "pose": pose,
        }

    sim.set_agent_state(orig_pos, orig_rot)
    return results
