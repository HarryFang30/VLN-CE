#!/usr/bin/env python3
"""
数据采集脚本 - 包含动作记录
基于原始 collect.py，新增：
1. 离散动作记录 (discrete_actions.npy)
2. 2D 连续动作计算 (actions.npy) - 从相邻位姿推导 (dx, dy)

动作语义（重要）：
- action[i] = 从 frame[i] 到 frame[i+1] 的动作
- 最后一帧 action[T-1] = (0, 0) 或 STOP

动作格式：
- discrete_actions: [T] int array, 值为 HabitatSimActions 枚举
  - 0: STOP, 1: MOVE_FORWARD, 2: TURN_LEFT, 3: TURN_RIGHT
- actions: [T, 2] float32 array, 2D 连续动作 (dx, dy)
  - dx: agent-local X 方向位移（右为正）
  - dy: agent-local Z 方向位移（前为正）

训练用法：
- 给定 frame[i]，预测 action[i]，然后到达 frame[i+1]
"""
import habitat
import cv2
import numpy as np
import json
import math
import time
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import concurrent.futures
import habitat_extensions
from habitat_extensions.config.default import get_extended_config
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.sims.habitat_simulator.actions import HabitatSimActions


# ==================== 动作相关函数 ====================

def compute_2d_action_from_poses(pose_before: np.ndarray, pose_after: np.ndarray) -> np.ndarray:
    """
    从相邻位姿计算 2D 连续动作 (dx, dy)
    
    坐标系约定（Habitat）：
    - Agent-local 坐标系：X 向右，Y 向上，Z 向后（-Z 是前方）
    - 返回的动作：dx（右/左），dy（前/后，前为正）
    
    Args:
        pose_before: 4x4 相机位姿矩阵（执行动作前）
        pose_after: 4x4 相机位姿矩阵（执行动作后）
    
    Returns:
        np.ndarray: [dx, dy] 2D 动作向量
    """
    # 计算相对变换：T_rel = T_before^{-1} @ T_after
    T_before = np.array(pose_before, dtype=np.float32)
    T_after = np.array(pose_after, dtype=np.float32)
    T_rel = np.linalg.inv(T_before) @ T_after
    
    # 提取平移分量（在 before 帧的局部坐标系下）
    # Habitat 坐标系：X 右，Y 上，Z 后
    dx = T_rel[0, 3]  # X 方向（右/左）
    dz = T_rel[2, 3]  # Z 方向（后/前）
    
    # 转换为 (dx, dy) 其中 dy 表示前进方向
    # 由于 -Z 是前方，前进时 dz < 0，所以 dy = -dz
    dy = -dz
    
    return np.array([dx, dy], dtype=np.float32)


def discrete_action_to_name(action: int) -> str:
    """将离散动作转换为名称"""
    names = {
        0: "STOP",
        1: "MOVE_FORWARD", 
        2: "TURN_LEFT",
        3: "TURN_RIGHT"
    }
    return names.get(action, f"UNKNOWN({action})")


# ==================== 辅助函数 ====================

def save_image_async(path: Path, image: np.ndarray) -> bool:
    """异步保存图像（用于线程池）"""
    try:
        success = cv2.imwrite(str(path), image)
        return success
    except Exception as e:
        print(f"  ⚠️  Image save failed: {path.name}, error: {e}")
        return False

def quaternion_to_rotation_matrix(q) -> np.ndarray:
    """
    将 Quaternion 转换为 3×3 旋转矩阵
    支持 Magnum Quaternion 和 numpy-quaternion 两种类型
    """
    if hasattr(q, 'scalar') and hasattr(q, 'vector'):
        w = q.scalar
        x = q.vector.x
        y = q.vector.y
        z = q.vector.z
    elif hasattr(q, 'w') and hasattr(q, 'x'):
        w = q.w
        x = q.x
        y = q.y
        z = q.z
    elif hasattr(q, '__getitem__'):
        w, x, y, z = q[0], q[1], q[2], q[3]
    else:
        raise ValueError(f"Unknown quaternion type: {type(q)}")

    norm = math.sqrt(w*w + x*x + y*y + z*z)
    if norm == 0:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm

    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ], dtype=np.float32)

    return R


def get_sensor_extrinsics(config) -> np.ndarray:
    """获取传感器外参矩阵 T_agent_cam"""
    sensor_cfg = config.SIMULATOR.RGB_SENSOR
    sensor_position = np.array(sensor_cfg.POSITION, dtype=np.float32)

    if hasattr(sensor_cfg, 'ORIENTATION'):
        orientation = sensor_cfg.ORIENTATION
        if isinstance(orientation, (list, tuple)):
            if len(orientation) == 3:
                roll, pitch, yaw = orientation
                cy = np.cos(yaw)
                sy = np.sin(yaw)
                cp = np.cos(pitch)
                sp = np.sin(pitch)
                cr = np.cos(roll)
                sr = np.sin(roll)

                sensor_rotation = np.array([
                    [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                    [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                    [-sp, cp*sr, cp*cr]
                ], dtype=np.float32)
            elif len(orientation) == 4:
                x, y, z, w = orientation
                norm = math.sqrt(w*w + x*x + y*y + z*z)
                if norm == 0:
                    sensor_rotation = np.eye(3, dtype=np.float32)
                else:
                    w, x, y, z = w/norm, x/norm, y/norm, z/norm
                    sensor_rotation = np.array([
                        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
                    ], dtype=np.float32)
            else:
                sensor_rotation = np.eye(3, dtype=np.float32)
        else:
            sensor_rotation = quaternion_to_rotation_matrix(orientation)
    else:
        sensor_rotation = np.eye(3, dtype=np.float32)

    T_agent_cam = np.eye(4, dtype=np.float32)
    T_agent_cam[:3, :3] = sensor_rotation
    T_agent_cam[:3, 3] = sensor_position
    return T_agent_cam


def compute_intrinsics(config) -> Dict:
    """计算Equirectangular传感器的基础参数"""
    rgb_cfg = config.SIMULATOR.RGB_SENSOR
    width = int(rgb_cfg.WIDTH)
    height = int(rgb_cfg.HEIGHT)
    subtype = getattr(rgb_cfg, "SENSOR_SUBTYPE", "PINHOLE")
    subtype_str = str(subtype).upper()
    if "EQUIRECT" not in subtype_str:
        raise ValueError(
            "collect.py 现在要求 RGB_SENSOR.SENSOR_SUBTYPE 为 EQUIRECTANGULAR，"
            f" 当前值为 {subtype}"
        )

    horizontal_fov = float(getattr(rgb_cfg, "HFOV", 360.0))
    vertical_fov = 180.0
    pixels_per_rad_h = width / (2.0 * math.pi)
    pixels_per_rad_v = height / math.pi

    return {
        "projection": "equirectangular",
        "width": width,
        "height": height,
        "hfov": horizontal_fov,
        "vfov": vertical_fov,
        "pixels_per_radian_horizontal": pixels_per_rad_h,
        "pixels_per_radian_vertical": pixels_per_rad_v
    }


def project_point_equirect(p_cam: np.ndarray, width: int, height: int) -> Tuple[float, float]:
    """将相机坐标系下的3D点投影到Equirectangular图像坐标"""
    x, y, z = float(p_cam[0]), float(p_cam[1]), float(p_cam[2])
    r = math.sqrt(x * x + y * y + z * z)
    if r < 1e-6:
        return None

    phi = math.atan2(x, -z)
    theta = math.asin(np.clip(y / r, -1.0, 1.0))

    u = (phi + math.pi) / (2.0 * math.pi) * width
    v = (0.5 - (theta / math.pi)) * height

    u = u % width
    v = np.clip(v, 0.0, height - 1e-6)
    return float(u), float(v)


def compute_camera_pose(agent_state, T_agent_cam: np.ndarray) -> np.ndarray:
    """计算相机到世界的变换矩阵"""
    agent_position = agent_state.position
    agent_rotation = agent_state.rotation

    R_agent = quaternion_to_rotation_matrix(agent_rotation)
    T_w_agent = np.eye(4, dtype=np.float32)
    T_w_agent[:3, :3] = R_agent
    T_w_agent[:3, 3] = agent_position

    T_w_c = T_w_agent @ T_agent_cam
    return T_w_c


def match_keyframes_to_trajectory(
    trajectory: List[np.ndarray],
    reference_path: List[List[float]]
) -> Tuple[List[int], List[float]]:
    """为reference_path中的每个节点找到最近的轨迹帧"""
    keyframe_indices = []
    keyframe_distances = []

    for ref_point in reference_path:
        ref_pos = np.array(ref_point)
        distances = [np.linalg.norm(pos - ref_pos) for pos in trajectory]
        closest_idx = np.argmin(distances)
        closest_dist = distances[closest_idx]

        keyframe_indices.append(int(closest_idx))
        keyframe_distances.append(float(closest_dist))

    return keyframe_indices, keyframe_distances


def select_keyframes_motion_based(
    poses: List[List[float]],
    min_dist: float = 0.5,
    min_angle_deg: float = 15.0
) -> List[int]:
    """基于运动量的贪婪关键帧选择策略"""
    num_frames = len(poses)
    if num_frames == 0:
        return []

    selected_indices = [0]
    last_idx = 0

    for curr_idx in range(1, num_frames):
        T_last = np.array(poses[last_idx], dtype=np.float32)
        T_curr = np.array(poses[curr_idx], dtype=np.float32)

        pos_last = T_last[:3, 3]
        pos_curr = T_curr[:3, 3]
        dist = float(np.linalg.norm(pos_curr - pos_last))

        R_last = T_last[:3, :3]
        R_curr = T_curr[:3, :3]
        R_diff = R_curr @ R_last.T
        trace = float(np.trace(R_diff))
        trace = float(np.clip(trace, -1.0, 3.0))
        angle_rad = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
        angle_deg = float(np.degrees(angle_rad))

        is_last_frame = (curr_idx == num_frames - 1)
        if dist > min_dist or angle_deg > min_angle_deg or is_last_frame:
            selected_indices.append(curr_idx)
            last_idx = curr_idx

    return selected_indices


def draw_nerf_ripple_point(
    heatmap: np.ndarray,
    center: Tuple[float, float],
    sigma: float,
    frame_rank: int,
    base_freq: float = 1.0
) -> None:
    """基于 NeRF 位置编码思想的波纹绘制"""
    H, W = heatmap.shape
    u, v = center

    radius = max(1, int(np.ceil(3.0 * sigma)))
    x_min = max(0, int(np.floor(u - radius)))
    x_max = min(W, int(np.ceil(u + radius)))
    y_min = max(0, int(np.floor(v - radius)))
    y_max = min(H, int(np.ceil(v + radius)))

    if x_min >= x_max or y_min >= y_max:
        return

    xs = np.arange(x_min, x_max, dtype=np.float32)
    ys = np.arange(y_min, y_max, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    dist = np.sqrt((xx - u) ** 2 + (yy - v) ** 2)

    envelope = np.exp(-(dist ** 2) / (2.0 * sigma ** 2))

    if frame_rank <= 0:
        modulation = 1.0
    else:
        norm_dist = dist / sigma 
        signal_accum = 0.0
        weight_accum = 0.0
        L = min(frame_rank, 8) 
        
        for i in range(L):
            freq = base_freq * (2.0 ** i) * np.pi
            wavelength = 2 * np.pi / (freq / sigma)
            if wavelength < 2.0:
                break

            w = 1.0 / (2.0 ** (i * 0.5)) 
            signal_accum += w * np.cos(freq * norm_dist)
            weight_accum += w
            
        if weight_accum > 0:
            modulation = 0.5 + 0.5 * (signal_accum / weight_accum)
        else:
            modulation = 1.0

    blob = (envelope * modulation).astype(np.float32, copy=False)
    roi = heatmap[y_min:y_max, x_min:x_max]
    np.add(roi, blob, out=roi)


def compute_adaptive_min_valid_ratio(
    keyframe_distances: List[float]
) -> Tuple[float, str]:
    """基于轨迹质量动态计算MIN_VALID_RATIO阈值"""
    if len(keyframe_distances) == 0:
        return 0.4, "low"

    avg_dist = float(np.mean(keyframe_distances))

    if avg_dist <= 0.3:
        threshold = 0.70
        quality_tier = "excellent"
    elif avg_dist >= 2.5:
        threshold = 0.40
        quality_tier = "low"
    else:
        threshold = 0.70 - (avg_dist - 0.3) / (2.5 - 0.3) * (0.70 - 0.40)
        threshold = np.clip(threshold, 0.40, 0.70)

        if avg_dist <= 0.8:
            quality_tier = "high"
        elif avg_dist <= 1.5:
            quality_tier = "medium"
        else:
            quality_tier = "acceptable"

    return float(threshold), quality_tier


def compute_adaptive_sigma(
    distance: float,
    object_size_3d: float = 0.3,
    heatmap_width: int = 64,
    min_sigma: float = 0.5,
    max_sigma: float = 5.0
) -> float:
    """计算Equirectangular坐标系下的自适应sigma"""
    if distance <= 1e-4:
        return float(max_sigma)

    angular_radius = math.atan2(object_size_3d, distance)
    pixels_per_rad = heatmap_width / (2.0 * math.pi)
    projected_radius_heatmap = angular_radius * pixels_per_rad
    sigma = projected_radius_heatmap / 3.0
    sigma = np.clip(sigma, min_sigma, max_sigma)
    return float(sigma)


# ==================== 命令行参数 ====================
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="VLN-CE 数据采集脚本（包含动作）")
    parser.add_argument('--config', type=str, default="habitat_extensions/config/vlnce_collect.yaml",
                        help='Habitat 配置文件路径')
    parser.add_argument('--output', type=str, default="/root/autodl-tmp/dataset_with_actions",
                        help='输出目录')
    parser.add_argument('--split', type=str, default="train", choices=['train', 'val', 'val_seen', 'val_unseen', 'test'],
                        help='数据集划分')
    parser.add_argument('--num-clips', type=int, default=1000,
                        help='采集的 clip 数量')
    parser.add_argument('--max-steps', type=int, default=100,
                        help='每个方向的最大步数')
    parser.add_argument('--num-workers', type=int, default=16,
                        help='IO worker 数量')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU 设备 ID')
    return parser.parse_args()

args = parse_args()

# ==================== 主程序配置 ====================

CONFIG_PATH = args.config
OUTPUT_ROOT = args.output
SPLIT = args.split
NUM_CLIPS = args.num_clips
MAX_STEPS = args.max_steps
NUM_WORKERS = args.num_workers

print(f"🚀 Starting data collection (WITH ACTIONS)")
print(f"   Output: {OUTPUT_ROOT}")
print(f"   Split: {SPLIT}")
print(f"   Clips: {NUM_CLIPS}")
print(f"   Max steps per direction: {MAX_STEPS}")
print(f"   ✨ NEW: Recording discrete and 2D continuous actions")

config = get_extended_config(CONFIG_PATH)
config.defrost()
config.DATASET.SPLIT = SPLIT
config.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = args.gpu
config.freeze()

print("Creating environment...")
env = habitat.Env(config=config)
sim = env.sim
follower = ShortestPathFollower(sim, goal_radius=0.2, return_one_hot=False)

print("✅ Environment created!")

T_agent_cam = get_sensor_extrinsics(config)
print(f"📐 Sensor extrinsics computed")

depth_cfg = getattr(config.SIMULATOR, "DEPTH_SENSOR", None)
if depth_cfg is None:
    depth_normalize, depth_min, depth_max = False, 0.0, 10.0
    print("⚠️  No DEPTH_SENSOR config node; assume metric depth in meters.")
else:
    depth_normalize = getattr(depth_cfg, "NORMALIZE_DEPTH", False)
    depth_min = float(getattr(depth_cfg, "MIN_DEPTH", 0.0))
    depth_max = float(getattr(depth_cfg, "MAX_DEPTH", 10.0))
print(f"📐 Depth config: normalize={depth_normalize}, range=[{depth_min}, {depth_max}]")

dataset = env._dataset
print(f"📊 Dataset: {len(dataset.episodes)} episodes in total")

episodes_by_scene = {}
for i, ep in enumerate(dataset.episodes):
    scene_name = ep.scene_id.split("/")[-1].replace(".glb", "")
    if scene_name not in episodes_by_scene:
        episodes_by_scene[scene_name] = []
    episodes_by_scene[scene_name].append(i)

print(f"🏠 Found {len(episodes_by_scene)} scenes")
print(f"   Top 5 scenes: {list(episodes_by_scene.keys())[:5]}")

# 🚀 优化：按场景分组采集，减少场景切换次数
all_episode_indices = []
scene_names = list(episodes_by_scene.keys())
random.seed(42)
random.shuffle(scene_names)

# 按场景顺序添加所有 episode（同一场景的 episode 连续排列）
for scene in scene_names:
    scene_episodes = list(episodes_by_scene[scene])
    random.shuffle(scene_episodes)  # 场景内随机顺序
    all_episode_indices.extend(scene_episodes)

print(f"✅ Prepared {len(all_episode_indices)} episode indices (grouped by scene for faster collection)")
print(f"   Scenes order: {scene_names[:5]}... ({len(scene_names)} total)")

output_root = Path(OUTPUT_ROOT)
output_root.mkdir(parents=True, exist_ok=True)
progress_file = output_root / "progress.json"

start_clip = 1
start_episode_attempt = None
if progress_file.exists():
    with open(progress_file, "r") as f:
        progress = json.load(f)
        start_clip = progress.get("next_clip_to_try", 1)
        start_episode_attempt = progress.get("next_episode_attempt")
    print(f"📂 Resuming from clip {start_clip}")

# 🆕 扫描已采集的 episode_id，避免重复采集
collected_episode_ids = set()
split_dir = output_root / SPLIT
if split_dir.exists():
    for meta_file in split_dir.rglob("meta.json"):
        try:
            with open(meta_file, "r") as f:
                meta_data = json.load(f)
                ep_id = meta_data.get("episode_id")
                if ep_id:
                    collected_episode_ids.add(str(ep_id))
        except:
            pass
    print(f"📋 Found {len(collected_episode_ids)} already collected episode_ids (will skip duplicates)")

stats = {
    "successful": 0,
    "failed": 0,
    "failed_clips": [],
    "scenes": {},
    "total_frames": 0,
    "total_actions": 0,  # 新增：动作统计
    "action_distribution": {  # 新增：动作分布
        "STOP": 0,
        "MOVE_FORWARD": 0,
        "TURN_LEFT": 0,
        "TURN_RIGHT": 0
    },
    "missing_fields": {
        "goals_missing": 0,
        "goals_empty": 0,
        "reference_path_missing": 0,
        "reference_path_empty": 0,
        "instruction_missing": 0,
        "instruction_text_missing": 0,
        "trajectory_id_missing": 0
    }
}

collected = []
start_time = time.time()

executor = concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS)
io_futures = []

clip_id = start_clip
if start_episode_attempt is None:
    episode_attempt = clip_id - 1
else:
    episode_attempt = start_episode_attempt

while clip_id <= NUM_CLIPS:
    if episode_attempt >= len(all_episode_indices):
        print(f"\n⚠️  Warning: Ran out of prepared episodes")
        break

    print(f"\n📁 Collecting clip {clip_id}/{NUM_CLIPS}")

    try:
        clip_failed = False
        failure_reason = ""
        failure_stage = ""
        failed_frame_count = 0
        max_failed_frames = 2

        episode_idx = all_episode_indices[episode_attempt]
        episode_attempt += 1
        episode = dataset.episodes[episode_idx]

        # 🆕 检查是否已采集过该 episode，跳过重复
        if str(episode.episode_id) in collected_episode_ids:
            print(f"  ⏭️  Skipping already collected episode {episode.episode_id}")
            continue

        current_scene = episode.scene_id.split("/")[-1].replace(".glb", "")
        
        # 检查是否切换场景
        if not hasattr(env, '_last_scene') or env._last_scene != current_scene:
            print(f"  🔄 Loading scene: {current_scene}")
        
        env._current_episode = episode
        observations = env.reset()
        
        # 只在场景变化时重建 follower
        if not hasattr(env, '_last_scene') or env._last_scene != current_scene:
            sim = env.sim
            follower = ShortestPathFollower(sim, goal_radius=0.2, return_one_hot=False)
            env._last_scene = current_scene

        missing = [k for k in ("rgb", "depth") if k not in observations]
        if missing:
            raise RuntimeError(f"Missing observations: {missing}.")

        scene_name = current_scene

        clip_dir = output_root / SPLIT / scene_name / f"clip_{clip_id:06d}"
        rgb_dir = clip_dir / "rgb"
        depth_dir = clip_dir / "depth"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)

        # I/O健康检查
        try:
            test_file = rgb_dir / ".io_health_check"
            with open(test_file, "wb") as f:
                f.write(b"health_check_test")
            with open(test_file, "rb") as f:
                if f.read() != b"health_check_test":
                    raise IOError("I/O verification mismatch")
            test_file.unlink()
        except Exception as io_error:
            print(f"  ❌ I/O health check failed: {io_error}")
            try:
                shutil.rmtree(clip_dir)
            except:
                pass
            stats["failed"] += 1
            stats["failed_clips"].append({
                "clip_id": clip_id,
                "episode_id": episode.episode_id,
                "error": f"I/O health check: {str(io_error)}",
                "stage": "io_check"
            })
            continue

        print(f"  Scene: {scene_name}, Episode: {episode.episode_id}")

        # Episode字段验证
        validation_errors = []

        if episode.goals is None:
            validation_errors.append("goals is None")
            stats["missing_fields"]["goals_missing"] += 1
        elif len(episode.goals) == 0:
            validation_errors.append("goals is empty list")
            stats["missing_fields"]["goals_empty"] += 1
        elif not hasattr(episode.goals[0], 'position'):
            validation_errors.append("goals[0] missing position attribute")

        if episode.reference_path is None:
            validation_errors.append("reference_path is None")
            stats["missing_fields"]["reference_path_missing"] += 1
        elif len(episode.reference_path) == 0:
            validation_errors.append("reference_path is empty")
            stats["missing_fields"]["reference_path_empty"] += 1

        if episode.instruction is None:
            validation_errors.append("instruction is None")
            stats["missing_fields"]["instruction_missing"] += 1
        elif not hasattr(episode.instruction, 'instruction_text'):
            validation_errors.append("instruction missing instruction_text attribute")
            stats["missing_fields"]["instruction_text_missing"] += 1

        if validation_errors:
            print(f"  ❌ Episode validation failed: {', '.join(validation_errors)}")
            stats["failed"] += 1
            stats["failed_clips"].append({
                "clip_id": clip_id,
                "episode_id": episode.episode_id,
                "error": f"Invalid episode data: {', '.join(validation_errors)}",
                "stage": "episode_validation"
            })
            continue

        goal_pos = episode.goals[0].position
        start_pos = episode.start_position
        reference_path = episode.reference_path

        if episode.instruction is None or episode.instruction.instruction_text is None:
            instruction_text = ""
            stats["missing_fields"]["instruction_text_missing"] += 1
            print(f"  ⚠️  Warning: instruction_text is None, using empty string")
        else:
            instruction_text = episode.instruction.instruction_text

        if episode.trajectory_id is None:
            trajectory_id = "unknown"
            stats["missing_fields"]["trajectory_id_missing"] += 1
            print(f"  ⚠️  Warning: trajectory_id is None, using 'unknown'")
        else:
            trajectory_id = episode.trajectory_id

        print(f"  ✅ Episode validated: {len(reference_path)} waypoints, "
              f"goal at ({goal_pos[0]:.2f}, {goal_pos[1]:.2f}, {goal_pos[2]:.2f})")

        # ==================== 初始化存储（新增动作列表）====================
        poses = []
        trajectory_positions = []
        depth_images = []
        discrete_actions = []  # 🆕 离散动作列表：action[i] = 从 frame[i] 到 frame[i+1] 的动作
        pending_action = None  # 🆕 暂存待记录的动作（用于下一帧）
        frame_id = 0

        print("  Walking to goal")
        start_frame = frame_id

        # 先记录起始帧（使用异步 IO 加速）
        if "rgb" in observations:
            rgb = observations["rgb"]
            if rgb.shape[2] == 4:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
            elif rgb.shape[2] == 3:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            # 🚀 异步保存 RGB（JPEG 格式，质量 95%，速度更快）
            rgb_path = rgb_dir / f"{frame_id:06d}.jpg"
            io_futures.append(executor.submit(cv2.imwrite, str(rgb_path), rgb.copy(), [cv2.IMWRITE_JPEG_QUALITY, 95]))

        if "depth" in observations:
            depth = observations["depth"]
            depth_images.append(depth.copy())
            # 🚀 异步保存 depth
            depth_path = depth_dir / f"{frame_id:06d}.npy"
            io_futures.append(executor.submit(np.save, str(depth_path), depth.copy().astype(np.float16)))  # float16 节省空间

        agent_state = sim.get_agent_state()
        trajectory_positions.append(agent_state.position.copy())
        T_w_c = compute_camera_pose(agent_state, T_agent_cam)
        poses.append(T_w_c.tolist())
        
        # 🆕 第一帧的动作将在下一步执行时记录（pending_action 机制）
        # discrete_actions[0] 将是从 frame[0] 到 frame[1] 的动作
        
        frame_id += 1

        target_idx = 0
        steps_taken = 0
        max_steps_total = MAX_STEPS * max(1, len(reference_path))

        while not clip_failed and target_idx < len(reference_path):
            goal_point = reference_path[target_idx]
            action = follower.get_next_action(goal_point)
            if action is None:
                action = HabitatSimActions.STOP

            if action == HabitatSimActions.STOP:
                target_idx += 1
                continue

            # 🆕 记录从当前帧出发的动作（在执行动作之前记录）
            # discrete_actions[i] = 从 frame[i] 到 frame[i+1] 的动作
            discrete_actions.append(int(action))

            observations = env.step(action)
            steps_taken += 1

            # 记录移动后的帧（使用异步 IO 加速）
            if "rgb" in observations:
                rgb = observations["rgb"]
                if rgb.shape[2] == 4:
                    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
                elif rgb.shape[2] == 3:
                    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                # 🚀 异步保存 RGB（JPEG 格式，质量 95%，速度更快）
                rgb_path = rgb_dir / f"{frame_id:06d}.jpg"
                io_futures.append(executor.submit(cv2.imwrite, str(rgb_path), rgb.copy(), [cv2.IMWRITE_JPEG_QUALITY, 95]))

            if "depth" in observations:
                depth = observations["depth"]
                depth_images.append(depth.copy())
                # 🚀 异步保存 depth
                depth_path = depth_dir / f"{frame_id:06d}.npy"
                depth_copy = depth.copy()
                io_futures.append(executor.submit(np.save, str(depth_path), depth_copy.astype(np.float16)))  # float16 节省空间

            if clip_failed:
                break

            agent_state = sim.get_agent_state()
            trajectory_positions.append(agent_state.position.copy())
            T_w_c = compute_camera_pose(agent_state, T_agent_cam)
            poses.append(T_w_c.tolist())
            
            frame_id += 1

            if steps_taken >= max_steps_total:
                clip_failed = True
                failure_reason = f"Exceeded steps limit ({steps_taken}/{max_steps_total})"
                failure_stage = "trajectory"
                break
        
        # 🆕 最后一帧没有后续动作，记录为 STOP (0)
        discrete_actions.append(0)  # STOP

        end_frame = max(start_frame, frame_id - 1)
        num_frames = end_frame - start_frame + 1
        print(f"    {num_frames} frames, {len(discrete_actions)} actions recorded")

        if clip_failed:
            print(f"  ⏩ Skipping due to failure: {failure_reason}")
            try:
                shutil.rmtree(clip_dir)
            except:
                pass
            stats["failed"] += 1
            stats["failed_clips"].append({
                "clip_id": clip_id,
                "episode_id": episode.episode_id,
                "error": failure_reason,
                "stage": failure_stage
            })
            continue

        MIN_FRAMES = 5
        if num_frames < MIN_FRAMES:
            print(f"  ⚠️  Skipping: trajectory too short ({num_frames} frames < {MIN_FRAMES})")
            shutil.rmtree(clip_dir)
            stats["failed"] += 1
            stats["failed_clips"].append({
                "clip_id": clip_id,
                "episode_id": episode.episode_id,
                "error": f"Trajectory too short ({num_frames} frames)",
                "stage": "trajectory_length"
            })
            continue

        # ==================== 计算 2D 连续动作 ====================
        # 🆕 修改语义：action[i] = 从 frame[i] 到 frame[i+1] 的动作
        print("  Computing 2D continuous actions from poses...")
        actions_2d = []
        for i in range(len(poses)):
            if i == len(poses) - 1:
                # 最后一帧：没有后续帧，动作为 (0, 0)
                actions_2d.append(np.array([0.0, 0.0], dtype=np.float32))
            else:
                # 从当前帧到下一帧的位移
                action_2d = compute_2d_action_from_poses(poses[i], poses[i+1])
                actions_2d.append(action_2d)
        
        actions_2d = np.array(actions_2d, dtype=np.float32)  # [T, 2]
        discrete_actions_arr = np.array(discrete_actions, dtype=np.int32)  # [T]
        
        # 统计动作分布
        for act in discrete_actions:
            act_name = discrete_action_to_name(act)
            if act_name in stats["action_distribution"]:
                stats["action_distribution"][act_name] += 1
        
        # 动作统计
        action_stats = {
            "dx": {"min": float(actions_2d[:, 0].min()), "max": float(actions_2d[:, 0].max()), "mean": float(actions_2d[:, 0].mean())},
            "dy": {"min": float(actions_2d[:, 1].min()), "max": float(actions_2d[:, 1].max()), "mean": float(actions_2d[:, 1].mean())}
        }
        print(f"    2D actions: dx=[{action_stats['dx']['min']:.3f}, {action_stats['dx']['max']:.3f}], "
              f"dy=[{action_stats['dy']['min']:.3f}, {action_stats['dy']['max']:.3f}]")

        # 关键帧匹配
        positions = trajectory_positions[start_frame:end_frame+1]
        keyframe_indices, keyframe_distances = match_keyframes_to_trajectory(
            positions, reference_path
        )
        keyframe_indices = [idx + start_frame for idx in keyframe_indices]

        # 验证帧数
        total_frames = len(poses)
        if total_frames == 0:
            clip_failed = True
            failure_reason = "No valid frames collected"
            failure_stage = "trajectory"
            raise Exception(failure_reason)
        
        # 计算内参（用于 meta.json）
        intrinsics_data = compute_intrinsics(config)
        img_width = intrinsics_data['width']
        img_height = intrinsics_data['height']
        
        h_obs, w_obs = observations["rgb"].shape[:2]
        if (w_obs != img_width) or (h_obs != img_height):
            img_width, img_height = w_obs, h_obs
            intrinsics_data["width"] = w_obs
            intrinsics_data["height"] = h_obs
            intrinsics_data["pixels_per_radian_horizontal"] = w_obs / (2.0 * math.pi)
            intrinsics_data["pixels_per_radian_vertical"] = h_obs / math.pi
        
        print(f"  ✅ Collected {total_frames} frames (热力图将在训练时动态计算)")

        # ==================== 保存数据 ====================
        # 1. 保存位姿
        with open(clip_dir / "poses.json", "w") as f:
            json.dump(poses, f, indent=2)

        # 2. 保存内参
        intrinsics_out = {
            "projection": "equirectangular",
            "width": int(img_width),
            "height": int(img_height),
            "hfov": float(intrinsics_data["hfov"]),
            "vfov": float(intrinsics_data["vfov"]),
            "pixels_per_radian": {
                "horizontal": float(intrinsics_data["pixels_per_radian_horizontal"]),
                "vertical": float(intrinsics_data["pixels_per_radian_vertical"])
            }
        }
        with open(clip_dir / "intrinsics.json", "w") as f:
            json.dump(intrinsics_out, f, indent=2)

        # 3. 保存动作数据
        np.save(clip_dir / "actions.npy", actions_2d)  # [T, 2] float32
        np.save(clip_dir / "discrete_actions.npy", discrete_actions_arr)  # [T] int32
        print(f"  ✨ Saved actions.npy [{actions_2d.shape}] and discrete_actions.npy [{discrete_actions_arr.shape}]")

        # 5. 保存元数据
        MIN_VALID_RATIO, quality_tier = compute_adaptive_min_valid_ratio(keyframe_distances)
        avg_keyframe_dist = float(np.mean(keyframe_distances))

        meta = {
            "episode_id": episode.episode_id,
            "trajectory_id": trajectory_id,
            "scene_id": scene_name,
            "instruction": instruction_text,
            "sampling_strategy": "walk_to_goal",
            "num_frames": frame_id,
            "trajectory": {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "num_frames": num_frames
            },
            "reference_path": reference_path,
            "keyframe_indices": keyframe_indices,
            "keyframe_distances": keyframe_distances,
            "max_keyframe_distance": float(np.max(keyframe_distances)),
            "mean_keyframe_distance": float(np.mean(keyframe_distances)),
            # 动作信息
            "actions": {
                "num_actions": len(actions_2d),
                "action_dim": 2,
                "action_semantic": "action[i] = from frame[i] to frame[i+1]",
                "action_format": "(dx, dy) - agent-local 2D displacement",
                "discrete_action_format": "HabitatSimActions (0=STOP, 1=FORWARD, 2=LEFT, 3=RIGHT)",
                "last_action_note": "action[T-1] = (0,0) / STOP since no next frame",
                "stats": action_stats,
                "files": {
                    "continuous": "actions.npy",
                    "discrete": "discrete_actions.npy"
                }
            },
            "quality_control": {
                "min_valid_ratio_used": float(MIN_VALID_RATIO),
                "avg_keyframe_distance": float(avg_keyframe_dist),
                "quality_tier": quality_tier
            }
        }

        with open(clip_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        # 更新统计
        stats["successful"] += 1
        stats["total_frames"] += frame_id
        stats["total_actions"] += len(actions_2d)
        stats["scenes"][scene_name] = stats["scenes"].get(scene_name, 0) + 1

        # 🆕 记录已采集的 episode_id，避免同次运行中重复
        collected_episode_ids.add(str(episode.episode_id))

        # 🚀 等待当前 clip 的 IO 完成，然后清理 futures
        if len(io_futures) > 100:
            concurrent.futures.wait(io_futures)
            io_futures.clear()

        collected.append(meta)
        print(f"✅ Clip {clip_id} done: {frame_id} frames, {len(actions_2d)} actions")

        clip_id += 1

    except Exception as e:
        print(f"❌ Clip {clip_id} failed: {e}")
        if 'clip_dir' in locals() and clip_dir.exists():
            try:
                shutil.rmtree(clip_dir)
            except:
                pass

        stats["failed"] += 1
        failure_info = {"clip_id": clip_id, "error": str(e)}
        if 'episode' in locals():
            failure_info["episode_id"] = episode.episode_id
        stats["failed_clips"].append(failure_info)
        continue

    finally:
        with open(progress_file, "w") as f:
            json.dump({
                "next_clip_to_try": clip_id,
                "next_episode_attempt": episode_attempt
            }, f)

# 🚀 等待所有异步 IO 完成
if io_futures:
    print(f"⏳ Waiting for {len(io_futures)} pending IO operations...")
    concurrent.futures.wait(io_futures)
    io_futures.clear()
    print("✅ All IO completed")

env.close()

# 输出最终统计
elapsed = time.time() - start_time
print("\n" + "="*60)
print("🎉 Data collection completed (WITH ACTIONS)!")
print("="*60)
print(f"✅ Successful: {stats['successful']}/{NUM_CLIPS}")
print(f"❌ Failed: {stats['failed']}/{NUM_CLIPS}")
print(f"📊 Total frames: {stats['total_frames']}")
print(f"🎮 Total actions: {stats['total_actions']}")
print(f"⏱️  Time elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
print(f"🏠 Scenes collected: {len(stats['scenes'])}")

print("\n📊 Action distribution:")
for action_name, count in stats["action_distribution"].items():
    pct = count / stats["total_actions"] * 100 if stats["total_actions"] > 0 else 0
    print(f"  {action_name}: {count} ({pct:.1f}%)")

print("\nScene distribution:")
for scene, count in sorted(stats["scenes"].items(), key=lambda x: -x[1]):
    print(f"  {scene}: {count} clips")

# 保存统计
with open(output_root / "collection_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

index_path = output_root / f"{SPLIT}_index.json"
with open(index_path, "w") as f:
    json.dump(collected, f, indent=2)

print(f"\n📝 Statistics saved to {output_root / 'collection_stats.json'}")
print(f"📄 Index saved to {index_path}")

