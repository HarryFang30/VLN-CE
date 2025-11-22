#!/usr/bin/env python3
"""
工作版本的数据采集脚本
基于test_minimal_collect_v2.py，硬编码参数避免初始化问题
优化版：修复episode done bug，添加错误处理和进度保存
多场景版：从所有可用场景中均匀采集数据
完整版：包含所有VLN训练必需的数据（相机内外参、关键帧、指令等）
并行版：使用多进程加速数据采集
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

    Args:
        q: Quaternion对象 (可能是Magnum或numpy-quaternion)

    Returns:
        np.ndarray: 3×3 旋转矩阵
    """
    # 尝试Magnum Quaternion格式 (.scalar, .vector)
    if hasattr(q, 'scalar') and hasattr(q, 'vector'):
        w = q.scalar
        x = q.vector.x
        y = q.vector.y
        z = q.vector.z
    # 尝试numpy-quaternion格式 (.w, .x, .y, .z)
    elif hasattr(q, 'w') and hasattr(q, 'x'):
        w = q.w
        x = q.x
        y = q.y
        z = q.z
    # 尝试作为数组访问
    elif hasattr(q, '__getitem__'):
        w, x, y, z = q[0], q[1], q[2], q[3]
    else:
        raise ValueError(f"Unknown quaternion type: {type(q)}")

    # 归一化四元数（防止脏数据导致姿态偏差）
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    if norm == 0:
        # 退化情况：返回单位矩阵
        return np.eye(3, dtype=np.float32)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm

    # 四元数到旋转矩阵
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

    # ORIENTATION可能是list、Quaternion对象，或者是[0,0,0]表示无旋转
    if hasattr(sensor_cfg, 'ORIENTATION'):
        orientation = sensor_cfg.ORIENTATION
        # 如果是list，需要手动构造旋转矩阵
        if isinstance(orientation, (list, tuple)):
            # 检查长度：[roll, pitch, yaw] 或 [x,y,z,w]
            if len(orientation) == 3:
                # [roll, pitch, yaw] 欧拉角（弧度）- ZYX旋转顺序
                roll, pitch, yaw = orientation

                # 使用ZYX旋转顺序将欧拉角转换为旋转矩阵
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
                # [x, y, z, w] 四元数
                x, y, z, w = orientation

                # 归一化四元数（防止脏数据导致姿态偏差）
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
                # 其他情况默认无旋转
                sensor_rotation = np.eye(3, dtype=np.float32)
        else:
            # 如果是Magnum Quaternion对象
            sensor_rotation = quaternion_to_rotation_matrix(orientation)
    else:
        sensor_rotation = np.eye(3, dtype=np.float32)

    T_agent_cam = np.eye(4, dtype=np.float32)
    T_agent_cam[:3, :3] = sensor_rotation
    T_agent_cam[:3, 3] = sensor_position
    return T_agent_cam


def compute_intrinsics(config) -> Dict:
    """
    从配置计算相机内参（确保 RGB/Depth 一致）

    重要：Habitat-Sim使用方形像素，但仅指定HFOV。
    对于非方形分辨率，VFOV由HFOV和宽高比通过反三角关系计算：
    VFOV = 2 × arctan((H/W) × tan(HFOV/2))

    注意：VFOV ≠ HFOV × (H/W)，后者仅在小角度时近似成立。

    参考：https://github.com/facebookresearch/habitat-lab/issues/474
    """
    rgb_cfg = config.SIMULATOR.RGB_SENSOR

    width = rgb_cfg.WIDTH
    height = rgb_cfg.HEIGHT
    hfov_degrees = rgb_cfg.HFOV

    # 计算焦距
    # fx: 使用水平FOV
    hfov_rad = math.radians(hfov_degrees)
    fx = width / (2.0 * math.tan(hfov_rad / 2.0))

    # fy: 使用推导的垂直FOV（反三角关系，非线性比例）
    # Habitat使用方形像素，VFOV = 2*atan((H/W)*tan(HFOV/2))
    vfov_rad = 2.0 * math.atan((height / width) * math.tan(hfov_rad / 2.0))
    fy = height / (2.0 * math.tan(vfov_rad / 2.0))
    vfov_degrees = math.degrees(vfov_rad)

    # 主点（光心）通常在图像中心
    cx = width / 2.0
    cy = height / 2.0

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    return {
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "K": K.tolist(),
        "width": width,
        "height": height,
        "hfov": hfov_degrees,
        "vfov": vfov_degrees  # 记录推导的VFOV
    }


def compute_camera_pose(agent_state, T_agent_cam: np.ndarray) -> np.ndarray:
    """
    计算相机到世界的变换矩阵

    坐标系约定：
    - Habitat世界坐标系：右手系，Y-up
    - Habitat相机坐标系：OpenGL约定（-Z朝前，+Y向上，+X向右）
    - 因此相机前方的点在相机坐标系中 Z < 0

    Args:
        agent_state: Habitat AgentState
        T_agent_cam: Agent到相机的外参矩阵

    Returns:
        T_w_c: 4×4 相机到世界变换矩阵
    """
    # Agent 到世界的变换
    agent_position = agent_state.position
    agent_rotation = agent_state.rotation

    R_agent = quaternion_to_rotation_matrix(agent_rotation)
    T_w_agent = np.eye(4, dtype=np.float32)
    T_w_agent[:3, :3] = R_agent
    T_w_agent[:3, 3] = agent_position

    # 相机到世界 = Agent到世界 × Agent到相机
    T_w_c = T_w_agent @ T_agent_cam

    return T_w_c


def match_keyframes_to_trajectory(
    trajectory: List[np.ndarray],
    reference_path: List[List[float]]
) -> Tuple[List[int], List[float]]:
    """
    为reference_path中的每个节点找到最近的轨迹帧

    Args:
        trajectory: agent在每一帧的位置
        reference_path: R2R的关键节点

    Returns:
        keyframe_indices: 关键帧索引列表
        keyframe_distances: 匹配距离列表
    """
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
    """
    基于运动量（位移+旋转）的贪婪关键帧选择策略

    作用：
        - 过滤掉原地不动的冗余帧
        - 捕获原地旋转带来的视觉剧烈变化

    Args:
        poses: 每帧相机位姿（4x4矩阵列表）
        min_dist: 最小位移阈值（米）
        min_angle_deg: 最小旋转阈值（度）

    Returns:
        关键帧索引列表（升序）
    """
    num_frames = len(poses)
    if num_frames == 0:
        return []

    selected_indices = [0]  # 第一帧总是保留
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
        trace = float(np.clip(trace, -1.0, 3.0))  # 防止浮点误差导致NaN
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
    base_freq: float = 1.0  # 基础频率，控制最内层波纹的大小
) -> None:
    """
    基于 NeRF 位置编码思想的波纹绘制：
    使用 2^k 的指数频率叠加，即保留了热力图的整体性，又在高 Rank 时增加了精细纹理。
    """
    H, W = heatmap.shape
    u, v = center

    # 3倍 sigma 截断
    radius = max(1, int(np.ceil(3.0 * sigma)))
    x_min = max(0, int(np.floor(u - radius)))
    x_max = min(W, int(np.ceil(u + radius)))
    y_min = max(0, int(np.floor(v - radius)))
    y_max = min(H, int(np.ceil(v + radius)))

    if x_min >= x_max or y_min >= y_max:
        return

    # 坐标生成
    xs = np.arange(x_min, x_max, dtype=np.float32)
    ys = np.arange(y_min, y_max, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    dist = np.sqrt((xx - u) ** 2 + (yy - v) ** 2)

    # 1. 高斯包络 (Envelope) - 保持热力图的“热点”特性
    envelope = np.exp(-(dist ** 2) / (2.0 * sigma ** 2))

    # 2. NeRF 风格的调制 (Modulation)
    if frame_rank <= 0:
        modulation = 1.0
    else:
        # 归一化距离：将 3*sigma 的范围映射到大约 [0, 1.5] 左右的相空间
        # 这样保证频率是相对的，不受 sigma 大小影响
        norm_dist = dist / sigma 
        
        signal_accum = 0.0
        weight_accum = 0.0
        
        # 限制最大频带数，防止计算量过大
        L = min(frame_rank, 8) 
        
        for i in range(L):
            # NeRF 核心：指数级频率 2^i
            freq = base_freq * (2.0 ** i) * np.pi
            
            # 简单的抗混叠 (Mip-NeRF 思想): 
            # 如果波长小于2个像素，这个频率就画不出来，反而会产生噪点，跳过
            wavelength = 2 * np.pi / (freq / sigma) # 换算回像素单位波长
            if wavelength < 2.0:
                break

            # 权重衰减：高频分量的振幅应该小一些，否则看起来太乱
            # 1.0 / (2**i) 是分形噪声常用的衰减，也可以用 1.0 / (i+1)
            w = 1.0 / (2.0 ** (i * 0.5)) 
            
            signal_accum += w * np.cos(freq * norm_dist)
            weight_accum += w
            
        # 归一化到 [0, 1] 并调整基准值
        if weight_accum > 0:
            # 将 [-total, total] 压缩回 [0, 1]
            # 0.5 + 0.5 * cos...
            modulation = 0.5 + 0.5 * (signal_accum / weight_accum)
        else:
            modulation = 1.0

    # 合成
    blob = (envelope * modulation).astype(np.float32, copy=False)
    roi = heatmap[y_min:y_max, x_min:x_max]
    np.add(roi, blob, out=roi)

def compute_adaptive_min_valid_ratio(
    forward_keyframe_distances: List[float],
    backward_keyframe_distances: List[float]
) -> Tuple[float, str]:
    """
    基于轨迹质量动态计算MIN_VALID_RATIO阈值（线性插值）

    策略：keyframe距离越小（轨迹越准确），要求热力图有效率越高

    原理：
    - 高质量轨迹（avg_dist ≤ 0.3m）：agent精确跟随参考路径，
      reference_path节点应该有很高的可见性 → threshold = 0.70
    - 低质量轨迹（avg_dist ≥ 2.5m）：agent明显偏离参考路径，
      部分节点不可见是正常的，适度放宽 → threshold = 0.40
    - 中间质量：线性插值

    Args:
        forward_keyframe_distances: 正向轨迹关键帧距离列表（米）
        backward_keyframe_distances: 反向轨迹关键帧距离列表（米）

    Returns:
        (threshold, quality_tier): 自适应阈值[0.40, 0.70]和质量等级标签

    Quality Tiers:
        - excellent: avg_dist ≤ 0.3m  → threshold = 0.70
        - high:      0.3m < avg_dist ≤ 0.8m → threshold = 0.67-0.61
        - medium:    0.8m < avg_dist ≤ 1.5m → threshold = 0.60-0.51
        - acceptable: 1.5m < avg_dist ≤ 2.5m → threshold = 0.50-0.40
        - low:       avg_dist > 2.5m → threshold = 0.40
    """
    forward_mean = np.mean(forward_keyframe_distances)
    backward_mean = np.mean(backward_keyframe_distances)
    avg_dist = (forward_mean + backward_mean) / 2.0

    # 线性插值：[0.3m → 0.70, 2.5m → 0.40]
    if avg_dist <= 0.3:
        threshold = 0.70
        quality_tier = "excellent"
    elif avg_dist >= 2.5:
        threshold = 0.40
        quality_tier = "low"
    else:
        # 线性递减：threshold = 0.70 - slope * (avg_dist - 0.3)
        # slope = (0.70 - 0.40) / (2.5 - 0.3) = 0.30 / 2.2 ≈ 0.1364
        threshold = 0.70 - (avg_dist - 0.3) / (2.5 - 0.3) * (0.70 - 0.40)
        threshold = np.clip(threshold, 0.40, 0.70)

        # 质量分级（基于细粒度分段点）
        if avg_dist <= 0.8:
            quality_tier = "high"
        elif avg_dist <= 1.5:
            quality_tier = "medium"
        else:
            quality_tier = "acceptable"

    return float(threshold), quality_tier


def compute_adaptive_sigma(
    distance: float,
    fx: float,
    object_size_3d: float = 0.3,
    heatmap_width: int = 64,
    img_width: int = 640,
    min_sigma: float = 0.5,
    max_sigma: float = 5.0
) -> float:
    """
    计算基于投影尺度的自适应sigma

    原理：Sigma应该代表固定3D不确定区域的投影大小。
    一个object_size_3d米的物体在不同距离下投影到不同的像素数量。

    Args:
        distance: 光轴深度/Z深度（米，非欧氏距离）
                 因为成像比例尺 ~ fx/Z，使用Z深度更准确
        fx: 焦距（原始图像分辨率下的像素单位）
        object_size_3d: 3D不确定半径（米），默认0.3m
        heatmap_width: 热力图分辨率宽度
        img_width: 原始图像宽度
        min_sigma: 最小sigma（热力图像素单位）
        max_sigma: 最大sigma（热力图像素单位）

    Returns:
        sigma: 自适应sigma（热力图像素单位）

    示例（基于Z深度）：
        - 2m距离: sigma ≈ 1.6 像素
        - 5m距离: sigma ≈ 0.64 像素
        - 15m距离: sigma ≈ 0.5 像素（被min_sigma限制）
    """
    # 计算3D物体在原始图像分辨率下的投影半径（像素）
    projected_radius_pixels = (fx * object_size_3d) / distance

    # 缩放到热力图分辨率
    projected_radius_heatmap = projected_radius_pixels * (heatmap_width / img_width)

    # 使用3-sigma规则：半径约等于3σ（99.7%覆盖）
    sigma = projected_radius_heatmap / 3.0

    # 限制在合理范围内，防止极端值
    sigma = np.clip(sigma, min_sigma, max_sigma)

    return sigma


def find_best_frame_for_point(
    ref_point: List[float],
    poses: List[List],
    K_mat: np.ndarray,
    img_width: int,
    img_height: int,
    heatmap_width: int,
    heatmap_height: int,
    sigma: float,
    depth_images: List[np.ndarray] = None,
    occlusion_threshold: float = 0.15,
    preferred_distance_range: Tuple[float, float] = (0.5, 20.0),
    z_forward_negative: bool = True,
    depth_normalize: bool = False,
    depth_min: float = 0.0,
    depth_max: float = 10.0,
    poses_inv: List[np.ndarray] = None
) -> Tuple[int, float, float, bool, float]:
    """
    为3D参考点找到最佳观察帧（改进版）

    注意：Habitat使用OpenGL相机坐标系约定（-Z朝前，+Y向上，+X向右）

    策略:
    1. 点必须在相机前方 (z < 0, 因为Habitat使用-Z朝前)
    2. 点可以在图像边缘外一定距离（±margin）
    3. 遮挡检测：使用深度图检查点是否被遮挡（可选）
    4. 多准则评分：距离、视角、像面位置
    5. 如果有多个候选帧，选择综合评分最高的

    Args:
        ref_point: [x, y, z] 3D参考点
        poses: 所有帧的位姿列表 (List of 4×4 matrices)
        K_mat: 相机内参矩阵
        img_width: 图像宽度
        img_height: 图像高度
        heatmap_width: 热力图宽度（用于计算margin_x）
        heatmap_height: 热力图高度（用于计算margin_y）
        sigma: 高斯标准差（像素单位，热力图分辨率）
        depth_images: 深度图列表（用于遮挡检测，可选）
        occlusion_threshold: 遮挡判定阈值（米），默认0.15m
        preferred_distance_range: (min_dist, max_dist) 优先距离范围

    Returns:
        (best_frame_idx, u, v, is_valid, distance)
        - best_frame_idx: 最佳帧索引 (-1 if 没有找到)
        - u, v: 投影坐标
        - is_valid: 是否有效
        - distance: 光轴深度/Z深度（米，非欧氏距离），用于自适应sigma和距离评分
                   因为成像比例尺 ~ fx/Z，使用Z深度更准确
    """
    p_world = np.array([ref_point[0], ref_point[1], ref_point[2], 1.0], dtype=np.float32)

    # 计算允许的边界裕度（基于高斯分布的3σ范围）
    # sigma是热力图分辨率下的标准差，需要换算到原始图像分辨率
    # 使用各向异性缩放，避免宽高比不同时的误过滤
    margin_x = 3 * sigma * (img_width / heatmap_width)
    margin_y = 3 * sigma * (img_height / heatmap_height)

    candidates = []  # (frame_idx, distance, u, v, viewing_angle, center_distance)
    debug_stats = {"total": len(poses), "behind": 0, "oob": 0, "occluded": 0, "valid": 0}

    for frame_idx, pose in enumerate(poses):
        # 转换到相机坐标系 (需要使用world→camera的逆变换)
        # 使用预计算的逆矩阵（性能优化）
        if poses_inv is not None:
            T_c_w = poses_inv[frame_idx]
        else:
            T_w_c = np.array(pose, dtype=np.float32)
            T_c_w = np.linalg.inv(T_w_c)
        p_cam = T_c_w @ p_world

        # 检查是否在相机前方（自适应Z方向）
        # z_forward_negative=True: 前方为Z<0 (OpenGL约定)
        # z_forward_negative=False: 前方为Z>0 (计算机视觉约定)
        is_front = (p_cam[2] < 0) if z_forward_negative else (p_cam[2] > 0)
        if not is_front:
            debug_stats["behind"] += 1
            continue

        # 投影到图像平面
        # 使用原始矩阵乘法方式
        p_img = K_mat @ p_cam[:3]
        # 检查深度是否过小以避免除零
        if abs(p_img[2]) < 0.01:  # 如果深度<1cm，跳过
            debug_stats["behind"] += 1  # 算作behind
            continue
        u = p_img[0] / p_img[2]
        v = p_img[1] / p_img[2]

        # 放宽边界检查：允许点在图像边缘外±margin像素（各向异性）
        if not (-margin_x <= u < img_width + margin_x and -margin_y <= v < img_height + margin_y):
            debug_stats["oob"] += 1
            if debug_stats["oob"] == 1:  # print first OOB case
                print(f"      [TEMP DEBUG] First OOB: u={u:.1f}, v={v:.1f}, bounds=[{-margin_x:.1f}, {img_width+margin_x:.1f}] x [{-margin_y:.1f}, {img_height+margin_y:.1f}]")
                print(f"      [TEMP DEBUG] p_cam={p_cam[:3]}, p_img[2]={p_img[2]:.2f}, fx={K_mat[0,0]:.1f}, fy={K_mat[1,1]:.1f}, cx={K_mat[0,2]:.1f}, cy={K_mat[1,2]:.1f}")
            continue

        # 遮挡检测（如果提供了深度图）
        passed_oob_check = True  # reached here, so passed behind and OOB checks
        if depth_images is not None and frame_idx < len(depth_images):
            # 计算期望深度（相机平面距离，即Z分量的绝对值）
            # Habitat的depth默认表示光轴方向距离，不是欧氏范数
            expected_depth = (-p_cam[2]) if z_forward_negative else (p_cam[2])
            expected_depth = float(abs(expected_depth))

            # 获取深度图中的观测深度
            depth_img = depth_images[frame_idx]

            # 处理RGB与Depth分辨率不一致的情况（需要缩放坐标）
            dh, dw = depth_img.shape[0], depth_img.shape[1]
            if (dw != img_width) or (dh != img_height):
                u_d = u * (dw / img_width)
                v_d = v * (dh / img_height)
            else:
                u_d, v_d = u, v

            v_int = int(np.clip(v_d, 0, dh - 1))
            u_int = int(np.clip(u_d, 0, dw - 1))

            # 处理不同的深度图格式（可能是(H,W)或(H,W,1)）
            if depth_img.ndim == 3 and depth_img.shape[-1] == 1:
                observed_depth = float(depth_img[v_int, u_int, 0])
            else:
                observed_depth = float(depth_img[v_int, u_int])

            # 反归一化深度值（如果配置了NORMALIZE_DEPTH）
            if depth_normalize:
                observed_depth = depth_min + observed_depth * (depth_max - depth_min)

            # Debug: print first frame that reaches occlusion check
            if debug_stats["occluded"] == 0 and debug_stats["valid"] == 0:
                print(f"      [TEMP DEBUG] First frame reaching occlusion check: expected={expected_depth:.2f}m, observed={observed_depth:.2f}m, valid={np.isfinite(observed_depth) and observed_depth > 0}")

            # 处理无效深度（0/NaN/inf）：直接视为遮挡以避免误用不可见点
            if not np.isfinite(observed_depth) or observed_depth <= 0:
                debug_stats["occluded"] += 1
                continue

            # 如果观测深度明显小于期望深度，说明被遮挡
            if observed_depth < expected_depth - occlusion_threshold:
                debug_stats["occluded"] += 1
                continue  # 跳过被遮挡的点

        # 计算评分所需的几何指标
        # 使用光轴深度（Z分量），因为成像比例尺 ~ fx/Z
        z_depth = (-p_cam[2]) if z_forward_negative else (p_cam[2])
        z_depth = float(abs(z_depth))
        distance = z_depth  # 用于自适应sigma计算和距离评分

        # 视轴夹角：相机光轴与点方向的夹角（根据Z方向自适配）
        camera_forward = np.array([0.0, 0.0, -1.0], dtype=np.float32) if z_forward_negative \
                         else np.array([0.0, 0.0, 1.0], dtype=np.float32)
        point_direction = p_cam[:3] / np.linalg.norm(p_cam[:3])
        cos_angle = np.clip(np.dot(camera_forward, point_direction), -1.0, 1.0)
        viewing_angle = np.arccos(cos_angle)  # 弧度，[0, π]

        # 像面位置：距离图像中心的归一化距离
        center_u = img_width / 2.0
        center_v = img_height / 2.0
        pixel_distance = np.sqrt((u - center_u)**2 + (v - center_v)**2)
        max_distance = np.sqrt(center_u**2 + center_v**2)  # 对角线距离
        center_distance = pixel_distance / max_distance  # [0, 1]

        # 通过所有检查，添加到候选列表
        debug_stats["valid"] += 1
        if debug_stats["valid"] == 1:  # print first valid case
            print(f"      [TEMP DEBUG] First VALID frame: frame_idx={frame_idx}, u={u:.1f}, v={v:.1f}, dist={distance:.2f}m")
        candidates.append((frame_idx, distance, u, v, viewing_angle, center_distance))

    # Temp debug: print first heatmap stats
    if len(candidates) == 0:
        valid_count = debug_stats["total"] - debug_stats["behind"] - debug_stats["oob"] - debug_stats["occluded"]
        print(f"      [TEMP DEBUG] Rejection stats: total={debug_stats['total']}, behind={debug_stats['behind']}, oob={debug_stats['oob']}, occluded={debug_stats['occluded']}, valid={valid_count}")

    if not candidates:
        # 没有找到有效帧
        return -1, 0.0, 0.0, False, 0.0  # 返回distance=0.0（无效）

    # 多准则评分函数
    def score_candidate(candidate):
        """
        多准则评分：结合距离、视轴夹角、像面位置

        权重分配：
        - 距离 (50%): 优先preferred_distance_range范围（室内场景最佳）
                     注：distance为Z深度（非欧氏距离），更符合成像比例尺
        - 视角 (35%): 优先正对观察（小夹角）
        - 位置 (15%): 优先图像中心（减少畸变）
        """
        frame_idx, distance, u, v, viewing_angle, center_distance = candidate

        # === 1. 距离评分 ===
        # 使用传入的preferred_distance_range参数（基于Z深度）
        ideal_min, ideal_max = preferred_distance_range
        if ideal_min <= distance <= ideal_max:
            dist_score = 1.0  # 理想范围内满分
        elif distance < ideal_min:
            dist_score = distance / ideal_min  # 太近线性衰减
        else:
            dist_score = np.exp(-(distance - ideal_max) / 10.0)  # 太远指数衰减

        # === 2. 视轴夹角评分 ===
        # 夹角越小越好：cos(0°)=1.0, cos(90°)=0.0
        angle_score = np.cos(viewing_angle)

        # === 3. 像面位置评分 ===
        # 中心位置更好：中心=1.0, 角落=0.0
        position_score = 1.0 - center_distance**2

        # === 加权组合 ===
        overall_score = (
            0.50 * dist_score +      # 距离最重要
            0.35 * angle_score +     # 视角次之
            0.15 * position_score    # 位置权重最小
        )

        return overall_score

    # 选择最佳候选
    best_candidate = max(candidates, key=score_candidate)
    frame_idx, distance, u, v, viewing_angle, center_distance = best_candidate

    return frame_idx, u, v, True, distance  # 返回distance用于自适应sigma计算


# 硬编码参数（避免argparse可能的问题）
CONFIG_PATH = "habitat_extensions/config/vlnce_collect.yaml"
OUTPUT_ROOT = "/root/autodl-tmp/dataset_train"  # 输出目录
SPLIT = "train"  # 数据集split
NUM_CLIPS = 5  # 总共采集的clips数量
MAX_STEPS = 50
NUM_WORKERS = 4  # 线程池大小（用于异步I/O）

print(f"🚀 Starting data collection (Optimized Version with Keyframe Selection)")
print(f"   Output: {OUTPUT_ROOT}")
print(f"   Split: {SPLIT}")
print(f"   Clips: {NUM_CLIPS}")
print(f"   Max steps per direction: {MAX_STEPS}")

# 使用与test成功案例完全相同的配置方式
config = get_extended_config(CONFIG_PATH)
config.defrost()
config.DATASET.SPLIT = SPLIT
# 删除 config.TASK.SENSORS = [] - 这可能截断观测pipeline
# 明确声明Agent级别传感器（强烈推荐）
config.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = 0
config.freeze()

print("Creating environment...")
env = habitat.Env(config=config)
sim = env.sim
follower = ShortestPathFollower(sim, goal_radius=0.2, return_one_hot=False)

print("✅ Environment created!")

# 获取传感器外参（只需计算一次）
T_agent_cam = get_sensor_extrinsics(config)
print(f"📐 Sensor extrinsics computed")

# 获取深度传感器配置（用于深度归一化）- 健壮读取
depth_cfg = getattr(config.SIMULATOR, "DEPTH_SENSOR", None)
if depth_cfg is None:
    depth_normalize, depth_min, depth_max = False, 0.0, 10.0
    print("⚠️  No DEPTH_SENSOR config node; assume metric depth in meters.")
else:
    depth_normalize = getattr(depth_cfg, "NORMALIZE_DEPTH", False)
    depth_min = float(getattr(depth_cfg, "MIN_DEPTH", 0.0))
    depth_max = float(getattr(depth_cfg, "MAX_DEPTH", 10.0))
print(f"📐 Depth config: normalize={depth_normalize}, range=[{depth_min}, {depth_max}]")

# 获取数据集中所有的episodes并按场景分组
dataset = env._dataset
print(f"📊 Dataset: {len(dataset.episodes)} episodes in total")

# 按场景分组episodes
episodes_by_scene = {}
for i, ep in enumerate(dataset.episodes):
    scene_name = ep.scene_id.split("/")[-1].replace(".glb", "")
    if scene_name not in episodes_by_scene:
        episodes_by_scene[scene_name] = []
    episodes_by_scene[scene_name].append(i)

print(f"🏠 Found {len(episodes_by_scene)} scenes")
print(f"   Top 5 scenes: {list(episodes_by_scene.keys())[:5]}")

# 创建一个episode索引列表，从所有场景中随机采样（无放回）
# 使用加权随机采样，确保场景分布均匀
all_episode_indices = []
scene_names = list(episodes_by_scene.keys())
random.seed(42)  # 固定随机种子，确保可重复性
random.shuffle(scene_names)  # 打乱场景顺序

# 为每个场景创建可用episode池（无放回采样）
episodes_available_by_scene = {scene: list(eps) for scene, eps in episodes_by_scene.items()}
scene_exhausted_count = {}  # 记录场景耗尽次数

# 轮流从每个场景中选择episodes（无放回）
scene_idx = 0
for _ in range(NUM_CLIPS * 5):  # 多准备一些以防失败（成功率~33%，所以准备5倍更安全）
    if len(all_episode_indices) >= NUM_CLIPS * 5:
        break
    scene = scene_names[scene_idx % len(scene_names)]

    # 如果该场景还有可用episode
    if episodes_available_by_scene[scene]:
        ep_idx = random.choice(episodes_available_by_scene[scene])
        episodes_available_by_scene[scene].remove(ep_idx)  # 无放回：移除已选episode
        all_episode_indices.append(ep_idx)

        # 如果场景的episode池用完，补充（允许第二轮采样，但会警告）
        if not episodes_available_by_scene[scene]:
            episodes_available_by_scene[scene] = list(episodes_by_scene[scene])
            scene_exhausted_count[scene] = scene_exhausted_count.get(scene, 0) + 1
            if scene_exhausted_count[scene] == 1:  # 只在第一次耗尽时警告
                print(f"  ⚠️  Scene '{scene}' exhausted ({len(episodes_by_scene[scene])} episodes), replenishing for 2nd pass")

    scene_idx += 1

print(f"✅ Prepared {len(all_episode_indices)} episode indices for collection")

output_root = Path(OUTPUT_ROOT)
output_root.mkdir(parents=True, exist_ok=True)  # 确保目录存在（防止progress.json写入失败）
progress_file = output_root / "progress.json"

# 加载进度（如果存在）
start_clip = 1
start_episode_attempt = None
if progress_file.exists():
    with open(progress_file, "r") as f:
        progress = json.load(f)
        # 语义：保存的是"下一个要尝试的clip"，直接使用无需+1
        start_clip = progress.get("next_clip_to_try", 1)
        start_episode_attempt = progress.get("next_episode_attempt")
    print(f"📂 Resuming from clip {start_clip}")

# 统计信息
stats = {
    "successful": 0,
    "failed": 0,
    "failed_clips": [],
    "scenes": {},
    "total_frames": 0,
    "missing_fields": {  # 字段缺失统计（用于数据质量监控）
        "goals_missing": 0,
        "goals_empty": 0,
        "reference_path_missing": 0,
        "reference_path_empty": 0,
        "instruction_missing": 0,
        "instruction_text_missing": 0,
        "trajectory_id_missing": 0
    }
}

# 采集记录（用于生成索引文件）
collected = []

start_time = time.time()

# 创建线程池用于异步I/O
executor = concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS)
io_futures = []  # 存储异步I/O任务

clip_id = start_clip
if start_episode_attempt is None:
    episode_attempt = clip_id - 1  # 尝试的episode索引（从0开始）
else:
    episode_attempt = start_episode_attempt

while clip_id <= NUM_CLIPS:
    # 确保有足够的episodes可用
    if episode_attempt >= len(all_episode_indices):
        print(f"\n⚠️  Warning: Ran out of prepared episodes")
        break

    print(f"\n📁 Collecting clip {clip_id}/{NUM_CLIPS}")

    try:
        # ==================== 失败标志变量 ====================
        clip_failed = False  # 失败标志，用于提前跳出
        failure_reason = ""  # 失败原因描述
        failure_stage = ""   # 失败阶段标识（io_check/forward_init/forward_trajectory/backward_trajectory/heatmap）

        # 帧写入失败容忍（最多5%）
        failed_frame_count = 0  # 累计失败帧数
        max_failed_frames = 2   # 追踪遇到的最大允许失败数（初始化为最小值2）

        # 使用预先准备的episode索引，确保场景多样性
        episode_idx = all_episode_indices[episode_attempt]
        episode_attempt += 1  # 无论是否成功，都尝试下一个episode
        episode = dataset.episodes[episode_idx]

        # 重置环境到指定的episode
        env._current_episode = episode
        observations = env.reset()

        # ==================== 观测存在性检查 ====================
        # 确保rgb和depth在observations中（防止config.TASK.SENSORS=[]导致缺失）
        missing = [k for k in ("rgb", "depth") if k not in observations]
        if missing:
            raise RuntimeError(
                f"Missing observations: {missing}. "
                "Check SIMULATOR.AGENT_0.SENSORS includes 'RGB_SENSOR' and 'DEPTH_SENSOR'. "
                "If caused by: config.TASK.SENSORS = [] → comment it out or ensure AGENT_0.SENSORS is set."
            )

        # ==================== 深度归一化sanity check（仅第一次）====================
        # 在第一个clip的第一次reset后检查深度格式，避免配置与实际不符
        if clip_id == start_clip and "depth" in observations:
            sample_depth = observations["depth"]
            sample_flat = sample_depth.reshape(-1).astype(np.float32)
            finite_mask = np.isfinite(sample_flat)
            if np.any(finite_mask):
                finite_values = sample_flat[finite_mask]
                sample_min = float(np.min(finite_values))
                sample_max = float(np.max(finite_values))

                # 允许0~1的小容差，避免浮点抖动
                normalized_hint = (sample_min >= -1e-3) and (sample_max <= 1.02)
                metric_hint = sample_max > depth_max + 1e-3

                if depth_normalize:
                    if metric_hint:
                        print(f"  ⚠️  Depth sanity check: max={sample_max:.2f} > depth_max={depth_max:.2f}")
                        print(f"      Depth appears to be metric already. Disabling normalization globally.")
                        depth_normalize = False
                else:
                    if normalized_hint:
                        print(f"  ⚠️  Depth sanity check: range≈[{sample_min:.3f}, {sample_max:.3f}] ⊂ [0,1]")
                        print(f"      Depth appears to be normalized. Enabling normalization globally.")
                        depth_normalize = True

        # 重新加载场景（如果场景改变了）
        current_scene = episode.scene_id.split("/")[-1].replace(".glb", "")
        if hasattr(env, '_last_scene') and env._last_scene != current_scene:
            # 场景改变了，需要重新创建follower
            sim = env.sim
            follower = ShortestPathFollower(sim, goal_radius=0.2, return_one_hot=False)
        env._last_scene = current_scene

        scene_name = current_scene

        clip_dir = output_root / SPLIT / scene_name / f"clip_{clip_id:06d}"
        rgb_dir = clip_dir / "rgb"
        depth_dir = clip_dir / "depth"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)

        # ==================== 早期I/O健康检查 ====================
        # 在轨迹采集前检测磁盘满、权限问题、网络存储失效等
        # 成本：0秒，避免4-10秒的无效计算
        try:
            test_file = rgb_dir / ".io_health_check"
            with open(test_file, "wb") as f:
                f.write(b"health_check_test")
            # 验证读取
            with open(test_file, "rb") as f:
                if f.read() != b"health_check_test":
                    raise IOError("I/O verification mismatch")
            test_file.unlink()
        except Exception as io_error:
            print(f"  ❌ I/O health check failed: {io_error}")
            # 清理失败的目录
            try:
                shutil.rmtree(clip_dir)
            except:
                pass  # 清理失败也继续

            # 记录失败统计
            stats["failed"] += 1
            stats["failed_clips"].append({
                "clip_id": clip_id,
                "episode_id": episode.episode_id,
                "error": f"I/O health check: {str(io_error)}",
                "stage": "io_check"
            })
            continue  # 跳过本clip，尝试下一个

        print(f"  Scene: {scene_name}, Episode: {episode.episode_id}")

        # ==================== Episode字段验证（健壮性检查）====================
        validation_errors = []

        # 验证goals（必需字段）
        if episode.goals is None:
            validation_errors.append("goals is None")
            stats["missing_fields"]["goals_missing"] += 1
        elif len(episode.goals) == 0:
            validation_errors.append("goals is empty list")
            stats["missing_fields"]["goals_empty"] += 1
        elif not hasattr(episode.goals[0], 'position'):
            validation_errors.append("goals[0] missing position attribute")

        # 验证reference_path（必需字段）
        if episode.reference_path is None:
            validation_errors.append("reference_path is None")
            stats["missing_fields"]["reference_path_missing"] += 1
        elif len(episode.reference_path) == 0:
            validation_errors.append("reference_path is empty")
            stats["missing_fields"]["reference_path_empty"] += 1

        # 验证instruction（链式检查）
        if episode.instruction is None:
            validation_errors.append("instruction is None")
            stats["missing_fields"]["instruction_missing"] += 1
        elif not hasattr(episode.instruction, 'instruction_text'):
            validation_errors.append("instruction missing instruction_text attribute")
            stats["missing_fields"]["instruction_text_missing"] += 1

        # 如果有致命错误，立即跳过（严格模式）
        if validation_errors:
            print(f"  ❌ Episode validation failed: {', '.join(validation_errors)}")
            stats["failed"] += 1
            stats["failed_clips"].append({
                "clip_id": clip_id,
                "episode_id": episode.episode_id,
                "error": f"Invalid episode data: {', '.join(validation_errors)}",
                "stage": "episode_validation"
            })
            continue  # 跳过此episode

        # ==================== 安全提取Episode字段 ====================
        # 验证通过后，安全访问必需字段
        goal_pos = episode.goals[0].position
        start_pos = episode.start_position
        reference_path = episode.reference_path

        # 可选字段：instruction_text（使用空字符串作为默认值）
        if episode.instruction is None or episode.instruction.instruction_text is None:
            instruction_text = ""
            stats["missing_fields"]["instruction_text_missing"] += 1
            print(f"  ⚠️  Warning: instruction_text is None, using empty string")
        else:
            instruction_text = episode.instruction.instruction_text

        # 可选字段：trajectory_id（使用"unknown"作为默认值）
        if episode.trajectory_id is None:
            trajectory_id = "unknown"
            stats["missing_fields"]["trajectory_id_missing"] += 1
            print(f"  ⚠️  Warning: trajectory_id is None, using 'unknown'")
        else:
            trajectory_id = episode.trajectory_id

        print(f"  ✅ Episode validated: {len(reference_path)} waypoints, "
              f"goal at ({goal_pos[0]:.2f}, {goal_pos[1]:.2f}, {goal_pos[2]:.2f})")

        # 初始化存储
        poses = []
        trajectory_positions = []
        depth_images = []  # 用于遮挡检测
        frame_id = 0

        # Forward：沿reference_path逐点行走
        print("  Phase 1: Forward")
        forward_start_frame = frame_id

        # 先记录起始帧
        if "rgb" in observations:
            rgb = observations["rgb"]
            if rgb.shape[2] == 4:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
            elif rgb.shape[2] == 3:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            try:
                rgb_path = rgb_dir / f"{frame_id:06d}.png"
                success = cv2.imwrite(str(rgb_path), rgb)
                if not success:
                    raise IOError(f"cv2.imwrite returned False for frame {frame_id}")
                if frame_id == 0:
                    if not rgb_path.exists() or rgb_path.stat().st_size == 0:
                        raise IOError("First frame RGB write verification failed: file missing or empty")
            except Exception as rgb_error:
                failed_frame_count += 1
                print(f"    ⚠️  Frame {frame_id} RGB write failed: {rgb_error}")
                allowed_failures = max(2, int(0.05 * (frame_id + 1)))
                max_failed_frames = max(max_failed_frames, allowed_failures)
                if failed_frame_count > allowed_failures:
                    print(f"  ❌ Too many failed frames ({failed_frame_count}/{frame_id+1} > {allowed_failures})")
                    clip_failed = True
                    failure_reason = f"Exceeded failed frame tolerance: {failed_frame_count} frames > {allowed_failures} allowed"
                    failure_stage = "forward_trajectory"
                elif frame_id == 0:
                    print(f"  ❌ First frame write failed, aborting clip")
                    clip_failed = True
                    failure_reason = f"First frame RGB write failed: {str(rgb_error)}"
                    failure_stage = "forward_init"

        if "depth" in observations:
            depth = observations["depth"]
            depth_images.append(depth.copy())
            try:
                depth_path = depth_dir / f"{frame_id:06d}.npy"
                np.save(str(depth_path), depth)
                if frame_id == 0:
                    if not depth_path.exists() or depth_path.stat().st_size == 0:
                        raise IOError("First frame depth write verification failed: file missing or empty")
            except Exception as depth_error:
                failed_frame_count += 1
                print(f"    ⚠️  Frame {frame_id} depth write failed: {depth_error}")
                allowed_failures = max(2, int(0.05 * (frame_id + 1)))
                max_failed_frames = max(max_failed_frames, allowed_failures)
                if failed_frame_count > allowed_failures:
                    print(f"  ❌ Too many failed frames ({failed_frame_count}/{frame_id+1} > {allowed_failures})")
                    clip_failed = True
                    failure_reason = f"Exceeded failed frame tolerance: {failed_frame_count} frames > {allowed_failures} allowed"
                    failure_stage = "forward_trajectory"
                elif frame_id == 0:
                    print(f"  ❌ First frame depth write failed, aborting clip")
                    clip_failed = True
                    failure_reason = f"First frame depth write failed: {str(depth_error)}"
                    failure_stage = "forward_init"

        if clip_failed:
            continue

        agent_state = sim.get_agent_state()
        trajectory_positions.append(agent_state.position.copy())
        T_w_c = compute_camera_pose(agent_state, T_agent_cam)
        poses.append(T_w_c.tolist())
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

            observations = env.step(action)
            steps_taken += 1

            # 记录移动后的帧
            if "rgb" in observations:
                rgb = observations["rgb"]
                if rgb.shape[2] == 4:
                    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
                elif rgb.shape[2] == 3:
                    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                try:
                    rgb_path = rgb_dir / f"{frame_id:06d}.png"
                    success = cv2.imwrite(str(rgb_path), rgb)
                    if not success:
                        raise IOError(f"cv2.imwrite returned False for frame {frame_id}")
                except Exception as rgb_error:
                    failed_frame_count += 1
                    print(f"    ⚠️  Frame {frame_id} RGB write failed: {rgb_error}")
                    allowed_failures = max(2, int(0.05 * (frame_id + 1)))
                    max_failed_frames = max(max_failed_frames, allowed_failures)
                    if failed_frame_count > allowed_failures:
                        print(f"  ❌ Too many failed frames ({failed_frame_count}/{frame_id+1} > {allowed_failures})")
                        clip_failed = True
                        failure_reason = f"Exceeded failed frame tolerance: {failed_frame_count} frames > {allowed_failures} allowed"
                        failure_stage = "forward_trajectory"
                        break

            if "depth" in observations:
                depth = observations["depth"]
                depth_images.append(depth.copy())
                try:
                    depth_path = depth_dir / f"{frame_id:06d}.npy"
                    np.save(str(depth_path), depth)
                except Exception as depth_error:
                    failed_frame_count += 1
                    print(f"    ⚠️  Frame {frame_id} depth write failed: {depth_error}")
                    allowed_failures = max(2, int(0.05 * (frame_id + 1)))
                    max_failed_frames = max(max_failed_frames, allowed_failures)
                    if failed_frame_count > allowed_failures:
                        print(f"  ❌ Too many failed frames ({failed_frame_count}/{frame_id+1} > {allowed_failures})")
                        clip_failed = True
                        failure_reason = f"Exceeded failed frame tolerance: {failed_frame_count} frames > {allowed_failures} allowed"
                        failure_stage = "forward_trajectory"
                        break

            if clip_failed:
                break

            agent_state = sim.get_agent_state()
            trajectory_positions.append(agent_state.position.copy())
            T_w_c = compute_camera_pose(agent_state, T_agent_cam)
            poses.append(T_w_c.tolist())
            frame_id += 1

            if steps_taken >= max_steps_total:
                clip_failed = True
                failure_reason = (
                    f"Exceeded forward steps limit ({steps_taken}/{max_steps_total}) "
                    f"before reaching waypoint {target_idx+1}/{len(reference_path)}"
                )
                failure_stage = "forward_trajectory"
                break

        forward_end_frame = max(forward_start_frame, frame_id - 1)
        forward_frames = forward_end_frame - forward_start_frame + 1
        print(f"    {forward_frames} frames")

        # ==================== 检查点1：Forward阶段失败检测 ====================
        if clip_failed:
            print(f"  ⏩ Skipping backward phase due to forward failure: {failure_reason}")
            # 清理已保存的数据
            try:
                shutil.rmtree(clip_dir)
            except Exception as cleanup_error:
                print(f"  ⚠️  Cleanup failed: {cleanup_error}")

            # 记录失败统计
            stats["failed"] += 1
            stats["failed_clips"].append({
                "clip_id": clip_id,
                "episode_id": episode.episode_id,
                "error": failure_reason,
                "stage": failure_stage,
                "failed_frame_count": failed_frame_count,
                "total_frames_attempted": frame_id
            })
            continue  # 跳过本clip

        # Backward：沿reference_path逆序返回
        print("  Phase 2: Backward")
        backward_start_frame = frame_id
        backward_targets = list(reversed(reference_path))
        target_idx = 0
        steps_taken = 0
        max_steps_total = MAX_STEPS * max(1, len(backward_targets))

        while not clip_failed and target_idx < len(backward_targets):
            goal_point = backward_targets[target_idx]
            action = follower.get_next_action(goal_point)
            if action is None:
                action = HabitatSimActions.STOP

            if action == HabitatSimActions.STOP:
                target_idx += 1
                continue

            observations = env.step(action)
            steps_taken += 1

            if "rgb" in observations:
                rgb = observations["rgb"]
                if rgb.shape[2] == 4:
                    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
                elif rgb.shape[2] == 3:
                    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                try:
                    rgb_path = rgb_dir / f"{frame_id:06d}.png"
                    success = cv2.imwrite(str(rgb_path), rgb)
                    if not success:
                        raise IOError(f"cv2.imwrite returned False for frame {frame_id}")
                except Exception as rgb_error:
                    failed_frame_count += 1
                    print(f"    ⚠️  Frame {frame_id} RGB write failed: {rgb_error}")
                    allowed_failures = max(2, int(0.05 * (frame_id + 1)))
                    max_failed_frames = max(max_failed_frames, allowed_failures)
                    if failed_frame_count > allowed_failures:
                        print(f"  ❌ Too many failed frames ({failed_frame_count}/{frame_id+1} > {allowed_failures})")
                        clip_failed = True
                        failure_reason = f"Exceeded failed frame tolerance: {failed_frame_count} frames > {allowed_failures} allowed"
                        failure_stage = "backward_trajectory"
                        break

            if "depth" in observations:
                depth = observations["depth"]
                depth_images.append(depth.copy())
                try:
                    depth_path = depth_dir / f"{frame_id:06d}.npy"
                    np.save(str(depth_path), depth)
                except Exception as depth_error:
                    failed_frame_count += 1
                    print(f"    ⚠️  Frame {frame_id} depth write failed: {depth_error}")
                    allowed_failures = max(2, int(0.05 * (frame_id + 1)))
                    max_failed_frames = max(max_failed_frames, allowed_failures)
                    if failed_frame_count > allowed_failures:
                        print(f"  ❌ Too many failed frames ({failed_frame_count}/{frame_id+1} > {allowed_failures})")
                        clip_failed = True
                        failure_reason = f"Exceeded failed frame tolerance: {failed_frame_count} frames > {allowed_failures} allowed"
                        failure_stage = "backward_trajectory"
                        break

            if clip_failed:
                break

            agent_state = sim.get_agent_state()
            trajectory_positions.append(agent_state.position.copy())
            T_w_c = compute_camera_pose(agent_state, T_agent_cam)
            poses.append(T_w_c.tolist())
            frame_id += 1

            if steps_taken >= max_steps_total:
                clip_failed = True
                failure_reason = (
                    f"Exceeded backward steps limit ({steps_taken}/{max_steps_total}) "
                    f"before reaching waypoint {target_idx+1}/{len(backward_targets)}"
                )
                failure_stage = "backward_trajectory"
                break

        backward_end_frame = max(backward_start_frame, frame_id - 1)
        backward_frames = backward_end_frame - backward_start_frame + 1
        print(f"    {backward_frames} frames")

        # ==================== 检查点2：Backward阶段失败检测 ====================
        if clip_failed:
            print(f"  ⏩ Skipping remaining processing due to backward failure: {failure_reason}")
            # 清理已保存的数据
            try:
                shutil.rmtree(clip_dir)
            except Exception as cleanup_error:
                print(f"  ⚠️  Cleanup failed: {cleanup_error}")

            # 记录失败统计
            stats["failed"] += 1
            stats["failed_clips"].append({
                "clip_id": clip_id,
                "episode_id": episode.episode_id,
                "error": failure_reason,
                "stage": failure_stage,
                "failed_frame_count": failed_frame_count,
                "total_frames_attempted": frame_id
            })
            continue  # 跳过本clip

        # ==================== 过滤短轨迹和不平衡轨迹 ====================
        MIN_FRAMES = 5  # 单方向最小帧数要求

        # 检查总帧数
        if frame_id < MIN_FRAMES:
            print(f"  ⚠️  Skipping: trajectory too short ({frame_id} frames < {MIN_FRAMES})")
            # 清理已保存的RGB图像
            import shutil
            shutil.rmtree(clip_dir)
            stats["failed"] += 1
            stats["failed_clips"].append({
                "clip_id": clip_id,
                "episode_id": episode.episode_id,
                "error": f"Trajectory too short ({frame_id} frames < {MIN_FRAMES})",
                "stage": "trajectory_length"
            })
            continue  # 跳过这个episode，不计入成功数

        # 检查单方向帧数（方案E：防止F=50,B=1这种不平衡情况）
        if forward_frames < MIN_FRAMES or backward_frames < MIN_FRAMES:
            print(f"  ⚠️  Skipping: trajectory imbalanced (Forward={forward_frames}, Backward={backward_frames}, both need ≥{MIN_FRAMES})")
            # 清理已保存的RGB图像
            import shutil
            shutil.rmtree(clip_dir)
            stats["failed"] += 1
            stats["failed_clips"].append({
                "clip_id": clip_id,
                "episode_id": episode.episode_id,
                "error": f"Trajectory imbalanced (Forward={forward_frames}, Backward={backward_frames}, need ≥{MIN_FRAMES})",
                "stage": "trajectory_balance"
            })
            continue  # 跳过这个episode，不计入成功数

        # ==================== 关键帧匹配 ====================
        # 正向轨迹匹配reference_path
        forward_positions = trajectory_positions[forward_start_frame:forward_end_frame+1]
        forward_keyframe_indices, forward_keyframe_distances = match_keyframes_to_trajectory(
            forward_positions, reference_path
        )
        # 转换为全局索引
        forward_keyframe_indices = [idx + forward_start_frame for idx in forward_keyframe_indices]

        # 反向轨迹匹配reference_path（逆序）
        backward_positions = trajectory_positions[backward_start_frame:backward_end_frame+1]
        backward_keyframe_indices, backward_keyframe_distances = match_keyframes_to_trajectory(
            backward_positions, reference_path[::-1]  # 逆序匹配
        )
        # 转换为全局索引，并reverse回来对齐到reference_path顺序
        backward_keyframe_indices = [idx + backward_start_frame for idx in backward_keyframe_indices][::-1]
        backward_keyframe_distances = backward_keyframe_distances[::-1]

        # ==================== 检查点3：热力图生成前检查 ====================
        if clip_failed:
            print(f"  ⏩ Skipping heatmap generation due to earlier failure: {failure_reason}")
            raise Exception(failure_reason)

        # ==================== 生成逐帧热力图 ====================
        total_frames = len(poses)
        if total_frames == 0:
            print(f"  ❌ No valid frames collected (frame_id={frame_id}, poses_count={len(poses)})")
            clip_failed = True
            failure_reason = "No valid frames collected for heatmap generation"
            failure_stage = "trajectory"
            raise Exception(failure_reason)

        HEATMAP_SIZE = (64, 64)
        HEATMAP_OCCLUSION_TOLERANCE = 0.25  # 深度容差，避免误判遮挡
        LOCAL_WINDOW = 5  # 局部补帧窗口，确保短期邻居也可被投影
        PROJECTION_MARGIN = 20  # 像素级边界裕度
        MAX_VISIBLE_DISTANCE = 15.0  # 超过该距离的点忽略（减少远距离噪声）
        heatmaps_history = np.zeros((total_frames, HEATMAP_SIZE[0], HEATMAP_SIZE[1]), dtype=np.float32)
        heatmaps_future = np.zeros_like(heatmaps_history)
        mask_history = np.zeros(total_frames, dtype=np.float32)
        mask_future = np.zeros(total_frames, dtype=np.float32)
        mask = np.zeros(total_frames, dtype=np.float32)  # union: 有任意历史/未来信息即记为1
        visibility_counts_history = np.zeros(total_frames, dtype=np.int32)
        visibility_counts_future = np.zeros(total_frames, dtype=np.int32)
        visibility_counts = np.zeros(total_frames, dtype=np.int32)  # union: 历史+未来数量

        # 获取相机内参矩阵
        intrinsics_data = compute_intrinsics(config)
        K_mat = np.array(intrinsics_data['K'], dtype=np.float32)
        img_width = intrinsics_data['width']
        img_height = intrinsics_data['height']

        # ==================== 内参尺寸校验（用实测分辨率）====================
        # 防止YAML配置与实际观测不符
        h_obs, w_obs = observations["rgb"].shape[:2]
        if (w_obs != img_width) or (h_obs != img_height):
            print(f"  ⚠️  Intrinsics size ({img_width}x{img_height}) != RGB obs ({w_obs}x{h_obs}). "
                  "Recomputing intrinsics with observed size.")
            # 重新计算K矩阵（使用相同的HFOV，但用实测尺寸）
            hfov_rad = math.radians(intrinsics_data['hfov'])
            fx = w_obs / (2.0 * math.tan(hfov_rad / 2.0))
            vfov_rad = 2.0 * math.atan((h_obs / w_obs) * math.tan(hfov_rad / 2.0))
            fy = h_obs / (2.0 * math.tan(vfov_rad / 2.0))
            K_mat = np.array([[fx, 0, w_obs/2], [0, fy, h_obs/2], [0, 0, 1]], dtype=np.float32)
            img_width, img_height = w_obs, h_obs

        # ==================== 自动推断Z轴方向（投票机制）====================
        # 使用前N帧×头尾M个参考点投票，更健壮（防止起点朝向背离）
        def robust_infer_z_sign(poses, K_mat, img_w, img_h, ref_path, N=3, M=2):
            """使用多帧多点投票推断Z方向符号"""
            frames = list(range(min(N, len(poses))))
            pts = [ref_path[i] for i in [0, min(len(ref_path)-1, 1)][:M]]
            score_neg = score_pos = 0

            for f in frames:
                T_c_w = np.linalg.inv(np.array(poses[f], dtype=np.float32))
                for p in pts:
                    p_cam = T_c_w @ np.array([p[0], p[1], p[2], 1.0], dtype=np.float32)

                    # 测试两种Z方向约定
                    for zneg in (True, False):
                        front = (p_cam[2] < 0) if zneg else (p_cam[2] > 0)
                        if not front:
                            continue

                        # 检查投影是否在画内
                        p_img = K_mat @ p_cam[:3]
                        if abs(p_img[2]) < 1e-3:
                            continue
                        u, v = p_img[0]/p_img[2], p_img[1]/p_img[2]
                        inbound = (0 <= u < img_w) and (0 <= v < img_h)

                        if inbound:
                            if zneg:
                                score_neg += 1
                            else:
                                score_pos += 1

            return True if score_neg >= score_pos else False

        z_forward_negative = robust_infer_z_sign(poses, K_mat, img_width, img_height, reference_path)
        print(f"  [Auto-Infer] Z-axis convention: front={'Z<0 (OpenGL)' if z_forward_negative else 'Z>0 (CV)'}")

        # ==================== Sanity Check：诊断Z轴和投影 ====================
        # 取样检查：2帧 × 2个参考点，统计Z前向与投影可见率
        probe_frame_indices = [forward_start_frame, min(forward_end_frame, forward_start_frame + 1)]
        num_waypoints = len(reference_path)
        probe_points = [reference_path[0], reference_path[-1]] if num_waypoints >= 2 else [reference_path[0]]

        front_neg = front_pos = inbound = 0
        total_probes = len(probe_frame_indices) * len(probe_points)

        for probe_idx in probe_frame_indices:
            T_w_c_probe = np.array(poses[probe_idx], dtype=np.float32)
            T_c_w_probe = np.linalg.inv(T_w_c_probe)

            for probe_point in probe_points:
                p_world_probe = np.array([probe_point[0], probe_point[1], probe_point[2], 1.0], dtype=np.float32)
                p_cam_probe = T_c_w_probe @ p_world_probe

                # 统计Z方向
                if p_cam_probe[2] < 0:
                    front_neg += 1
                elif p_cam_probe[2] > 0:
                    front_pos += 1

                # 统计投影边界
                p_img_probe = K_mat @ p_cam_probe[:3]
                if abs(p_img_probe[2]) > 0.01:
                    u_probe = p_img_probe[0] / p_img_probe[2]
                    v_probe = p_img_probe[1] / p_img_probe[2]
                    if 0 <= u_probe < img_width and 0 <= v_probe < img_height:
                        inbound += 1

        print(f"  [Sanity Check] Probe: Zneg={front_neg}, Zpos={front_pos}, inbounds={inbound}/{total_probes}")
        print(f"  [Sanity Check] Camera intrinsics: fx={K_mat[0,0]:.1f}, fy={K_mat[1,1]:.1f}, cx={K_mat[0,2]:.1f}, cy={K_mat[1,2]:.1f}")

        # ==================== 预计算所有poses的逆矩阵（性能优化）====================
        print(f"  Precomputing inverse pose matrices for {len(poses)} frames...")
        pose_mats = [np.array(pose, dtype=np.float32) for pose in poses]
        poses_inv = [np.linalg.inv(mat) for mat in pose_mats]
        camera_centers = [mat[:3, 3].copy() for mat in pose_mats]

        # ==================== 选择关键帧（运动量驱动）====================
        # 根据位移/旋转选择贪婪关键帧，确保原地转向被捕获
        motion_keyframes = select_keyframes_motion_based(
            poses=poses,
            min_dist=0.5,
            min_angle_deg=15.0
        )

        # 合并R2R路径关键帧，避免导航目标丢失
        base_keyframes = list(set(forward_keyframe_indices + backward_keyframe_indices))
        keyframe_indices = sorted(set(motion_keyframes + base_keyframes))

        reduction_factor = total_frames / len(keyframe_indices) if len(keyframe_indices) > 0 else 1.0
        print(f"  Selected {len(keyframe_indices)}/{total_frames} keyframes (motion-based, reduction {reduction_factor:.1f}×)")

        print(f"  Generating frame-to-frame visibility heatmaps ({total_frames} frames, {len(keyframe_indices)} keyframes)...")
        empty_heatmap_indices = []

        for frame_idx in range(total_frames):
            depth_img = depth_images[frame_idx] if frame_idx < len(depth_images) else None
            if depth_img is not None and depth_img.ndim == 3 and depth_img.shape[-1] == 1:
                depth_plane = depth_img[:, :, 0]
            else:
                depth_plane = depth_img
            depth_h = depth_plane.shape[0] if depth_plane is not None else None
            depth_w = depth_plane.shape[1] if depth_plane is not None else None

            T_c_w = poses_inv[frame_idx]
            heatmap_history_local = np.zeros(HEATMAP_SIZE, dtype=np.float32)
            heatmap_future_local = np.zeros(HEATMAP_SIZE, dtype=np.float32)
            visibility_history = 0
            visibility_future = 0
            stats_reject = {"behind": 0, "oob": 0, "occluded": 0, "dist": 0}

            candidates_to_project = set(keyframe_indices)
            start_local = max(0, frame_idx - LOCAL_WINDOW)
            end_local = min(total_frames, frame_idx + LOCAL_WINDOW + 1)
            for l_idx in range(start_local, end_local):
                candidates_to_project.add(l_idx)

            for other_idx in sorted(candidates_to_project):
                if other_idx == frame_idx:
                    continue

                other_center = np.array([camera_centers[other_idx][0],
                                         camera_centers[other_idx][1],
                                         camera_centers[other_idx][2],
                                         1.0], dtype=np.float32)
                p_cam = T_c_w @ other_center

                is_front = (p_cam[2] < 0) if z_forward_negative else (p_cam[2] > 0)
                if not is_front:
                    stats_reject["behind"] += 1
                    continue

                p_img = K_mat @ p_cam[:3]
                if abs(p_img[2]) < 1e-3:
                    continue
                u = p_img[0] / p_img[2]
                v = p_img[1] / p_img[2]
                if not (-PROJECTION_MARGIN <= u < img_width + PROJECTION_MARGIN and
                        -PROJECTION_MARGIN <= v < img_height + PROJECTION_MARGIN):
                    stats_reject["oob"] += 1
                    continue

                expected_depth = float(abs((-p_cam[2]) if z_forward_negative else p_cam[2]))
                if expected_depth > MAX_VISIBLE_DISTANCE:
                    stats_reject["dist"] += 1
                    continue

                if depth_plane is not None:
                    u_d = u * (depth_w / img_width)
                    v_d = v * (depth_h / img_height)
                    u_int = int(np.clip(u_d, 0, depth_w - 1))
                    v_int = int(np.clip(v_d, 0, depth_h - 1))
                    observed_depth = float(depth_plane[v_int, u_int])
                    if depth_normalize:
                        observed_depth = depth_min + observed_depth * (depth_max - depth_min)
                    if not np.isfinite(observed_depth) or observed_depth <= 0:
                        stats_reject["occluded"] += 1
                        continue
                    if observed_depth < expected_depth - HEATMAP_OCCLUSION_TOLERANCE:
                        stats_reject["occluded"] += 1
                        continue

                distance = expected_depth
                adaptive_sigma = compute_adaptive_sigma(
                    distance=distance,
                    fx=K_mat[0, 0],
                    object_size_3d=0.5,
                    heatmap_width=HEATMAP_SIZE[1],
                    img_width=img_width,
                    min_sigma=0.8,
                    max_sigma=6.0
                )
                u_hm = u * HEATMAP_SIZE[1] / img_width
                v_hm = v * HEATMAP_SIZE[0] / img_height
                if other_idx in keyframe_indices:
                    temporal_rank = max(0, abs(frame_idx - other_idx) // 5)
                else:
                    temporal_rank = 0
                if other_idx < frame_idx:
                    target_heatmap = heatmap_history_local
                    visibility_history += 1
                else:
                    target_heatmap = heatmap_future_local
                    visibility_future += 1
                draw_nerf_ripple_point(
                    heatmap=target_heatmap,
                    center=(u_hm, v_hm),
                    sigma=adaptive_sigma,
                    frame_rank=temporal_rank
                )

            max_hist = heatmap_history_local.max()
            if visibility_history > 0 and max_hist > 0:
                heatmap_history_local /= max_hist
                mask_history[frame_idx] = 1.0
            elif visibility_history > 0:
                mask_history[frame_idx] = 1.0
            else:
                mask_history[frame_idx] = 0.0

            max_future = heatmap_future_local.max()
            if visibility_future > 0 and max_future > 0:
                heatmap_future_local /= max_future
                mask_future[frame_idx] = 1.0
            elif visibility_future > 0:
                mask_future[frame_idx] = 1.0
            else:
                mask_future[frame_idx] = 0.0

            heatmaps_history[frame_idx] = heatmap_history_local
            heatmaps_future[frame_idx] = heatmap_future_local
            visibility_counts_history[frame_idx] = visibility_history
            visibility_counts_future[frame_idx] = visibility_future

            visibility = visibility_history + visibility_future
            visibility_counts[frame_idx] = visibility
            if mask_history[frame_idx] > 0 or mask_future[frame_idx] > 0:
                mask[frame_idx] = 1.0
            else:
                mask[frame_idx] = 0.0
                empty_heatmap_indices.append(frame_idx)
                if len(empty_heatmap_indices) <= 5:
                    print(f"    ⚠️ Frame {frame_idx} is EMPTY. Rejects: {stats_reject}")

            if frame_idx < 5:
                print(f"    Frame {frame_idx}: history_neighbors={visibility_history}, "
                      f"future_neighbors={visibility_future}")

        valid_history = int(mask_history.sum())
        valid_future = int(mask_future.sum())
        valid_count = int(mask.sum())
        print(f"  ✅ Generated {total_frames} heatmaps "
              f"(history valid={valid_history}, future valid={valid_future}, either={valid_count})")
        if empty_heatmap_indices:
            print(f"     Empty heatmaps: {len(empty_heatmap_indices)}/{total_frames} "
                  f"(sample indices: {empty_heatmap_indices[:10]})")

        # ==================== 保存数据 ====================
        # 1. 保存位姿
        with open(clip_dir / "poses.json", "w") as f:
            json.dump(poses, f, indent=2)

        # 2. 保存内参（使用最终实际使用的K_mat，而不是重新计算）
        # 关键：如果前面重算过K_mat（尺寸校验），这里必须用最终版本
        hfov_deg_out = math.degrees(2.0 * math.atan(img_width / (2.0 * K_mat[0, 0])))
        vfov_deg_out = math.degrees(2.0 * math.atan(img_height / (2.0 * K_mat[1, 1])))
        intrinsics_out = {
            "fx": float(K_mat[0, 0]),
            "fy": float(K_mat[1, 1]),
            "cx": float(K_mat[0, 2]),
            "cy": float(K_mat[1, 2]),
            "K": K_mat.tolist(),
            "width": int(img_width),
            "height": int(img_height),
            "hfov": float(hfov_deg_out),
            "vfov": float(vfov_deg_out),
        }
        with open(clip_dir / "intrinsics.json", "w") as f:
            json.dump(intrinsics_out, f, indent=2)

        # 3. 保存热力图、掩码和可见性计数
        np.save(clip_dir / "heatmaps_history.npy", heatmaps_history)
        np.save(clip_dir / "heatmaps_future.npy", heatmaps_future)
        np.save(clip_dir / "mask_history.npy", mask_history)
        np.save(clip_dir / "mask_future.npy", mask_future)
        np.save(clip_dir / "mask.npy", mask)
        np.save(clip_dir / "visibility_counts_history.npy", visibility_counts_history)
        np.save(clip_dir / "visibility_counts_future.npy", visibility_counts_future)
        np.save(clip_dir / "visibility_counts.npy", visibility_counts)

        # 4. 保存完整元数据
        meta = {
            "episode_id": episode.episode_id,
            "trajectory_id": trajectory_id,
            "scene_id": scene_name,
            "instruction": instruction_text,

            "sampling_strategy": "bidirectional_walk",
            "num_frames": frame_id,

            # 正反向分段信息
            "forward_segment": {
                "start_frame": forward_start_frame,
                "end_frame": forward_end_frame,
                "num_frames": forward_frames
            },
            "backward_segment": {
                "start_frame": backward_start_frame,
                "end_frame": backward_end_frame,
                "num_frames": backward_frames
            },

            # R2R关键节点信息
            "reference_path": reference_path,
            "forward_keyframe_indices": forward_keyframe_indices,
            "forward_keyframe_distances": forward_keyframe_distances,
            "backward_keyframe_indices": backward_keyframe_indices,
            "backward_keyframe_distances": backward_keyframe_distances,

            # 质量指标
            "forward_max_keyframe_distance": float(np.max(forward_keyframe_distances)),
            "forward_mean_keyframe_distance": float(np.mean(forward_keyframe_distances)),
            "backward_max_keyframe_distance": float(np.max(backward_keyframe_distances)),
            "backward_mean_keyframe_distance": float(np.mean(backward_keyframe_distances)),

            # 热力图信息
            "num_heatmaps": total_frames,
            "heatmap_size": list(HEATMAP_SIZE),
            "valid_heatmaps": {
                "history": valid_history,
                "future": valid_future,
                "either": valid_count
            },
            "heatmap_type": "frame_to_frame_visibility_split",
            "heatmap_files": {
                "history": "heatmaps_history.npy",
                "future": "heatmaps_future.npy"
            },
            "mask_files": {
                "history": "mask_history.npy",
                "future": "mask_future.npy",
                "combined": "mask.npy"
            },
            "visibility_counts": {
                "history": visibility_counts_history.tolist(),
                "future": visibility_counts_future.tolist(),
                "combined": visibility_counts.tolist()
            },

            # 自适应sigma配置
            "adaptive_sigma": {
                "method": "projection_based",
                "object_size_3d": 0.3,  # 3D不确定半径（米）
                "min_sigma": 0.5,  # 最小sigma（热力图像素）
                "max_sigma": 5.0,  # 最大sigma（热力图像素）
                "formula": "sigma = (fx * object_size_3d / distance) * (heatmap_width / img_width) / 3.0"
            },
        }

        # ==================== 计算质量控制信息（自适应阈值）====================
        # 在保存meta.json之前计算，以便记录到元数据中
        MIN_VALID_RATIO, quality_tier = compute_adaptive_min_valid_ratio(
            forward_keyframe_distances,
            backward_keyframe_distances
        )

        # 计算平均关键帧距离
        forward_mean_dist = float(np.mean(forward_keyframe_distances))
        backward_mean_dist = float(np.mean(backward_keyframe_distances))
        avg_keyframe_dist = (forward_mean_dist + backward_mean_dist) / 2.0

        valid_ratio_history = valid_history / total_frames if total_frames > 0 else 0.0
        valid_ratio_future = valid_future / total_frames if total_frames > 0 else 0.0
        valid_ratio = valid_count / total_frames if total_frames > 0 else 0.0

        def calc_visibility_stats(arr):
            if total_frames == 0:
                return {"min": 0, "max": 0, "mean": 0.0}
            return {
                "min": int(np.min(arr)),
                "max": int(np.max(arr)),
                "mean": float(np.mean(arr))
            }

        visibility_stats = {
            "history": calc_visibility_stats(visibility_counts_history),
            "future": calc_visibility_stats(visibility_counts_future),
            "combined": calc_visibility_stats(visibility_counts)
        }

        # 添加质量控制信息到元数据
        meta["quality_control"] = {
            "valid_heatmap_count": {
                "history": valid_history,
                "future": valid_future,
                "either": valid_count
            },
            "total_heatmap_count": total_frames,
            "valid_ratio": {
                "history": float(valid_ratio_history),
                "future": float(valid_ratio_future),
                "either": float(valid_ratio)
            },
            "min_valid_ratio_used": float(MIN_VALID_RATIO),
            "avg_keyframe_distance": float(avg_keyframe_dist),
            "quality_tier": quality_tier,
            "visibility_stats": visibility_stats
        }

        # 添加I/O警告信息（如果有失败帧）
        if failed_frame_count > 0:
            meta["io_warnings"] = {
                "failed_frame_count": failed_frame_count,
                "total_frames": frame_id,
                "failed_frame_ratio": float(failed_frame_count / frame_id) if frame_id > 0 else 0.0,
                "max_failed_frames_allowed": max_failed_frames,
                "note": "Some frames failed to write but were within tolerance (5%)"
            }
            print(f"  ⚠️  I/O warnings: {failed_frame_count}/{frame_id} frames failed to write "
                  f"({100*failed_frame_count/frame_id:.1f}%, within {max_failed_frames} tolerance)")

        with open(clip_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  Quality: tier={quality_tier}, avg_keyframe_dist={avg_keyframe_dist:.3f}m, "
              f"visibility_ratio={valid_ratio:.2f} (threshold={MIN_VALID_RATIO:.2f})")

        # 更新统计信息
        stats["successful"] += 1
        stats["total_frames"] += frame_id
        stats["scenes"][scene_name] = stats["scenes"].get(scene_name, 0) + 1

        # 添加到采集记录
        collected.append(meta)

        print(f"✅ Clip {clip_id} done: {frame_id} frames")

        # 成功采集后递增clip_id
        clip_id += 1

    except Exception as e:
        print(f"❌ Clip {clip_id} failed: {e}")

        # 清理部分数据（如果clip_dir存在）
        if 'clip_dir' in locals() and clip_dir.exists():
            try:
                shutil.rmtree(clip_dir)
            except Exception as cleanup_error:
                print(f"  ⚠️  Cleanup failed: {cleanup_error}")

        # 记录详细失败信息
        stats["failed"] += 1
        failure_info = {
            "clip_id": clip_id,
            "error": str(e)
        }

        # 添加episode_id（如果episode已加载）
        if 'episode' in locals():
            failure_info["episode_id"] = episode.episode_id

        # 添加阶段信息（如果failure_stage已设置）
        if 'failure_stage' in locals() and failure_stage:
            failure_info["stage"] = failure_stage

        # 添加失败帧统计（如果有）
        if 'failed_frame_count' in locals() and failed_frame_count > 0:
            failure_info["failed_frame_count"] = failed_frame_count
            if 'frame_id' in locals():
                failure_info["total_frames_attempted"] = frame_id

        stats["failed_clips"].append(failure_info)
        continue

    finally:
        # 保存进度：clip_id 表示"下一个要尝试的clip"
        # 成功时已递增（line 728），失败时保持不变（重启后重试）
        with open(progress_file, "w") as f:
            json.dump({
                "next_clip_to_try": clip_id,
                "next_episode_attempt": episode_attempt
            }, f)

env.close()

# 输出最终统计
elapsed = time.time() - start_time
print("\n" + "="*60)
print("🎉 Data collection completed!")
print("="*60)
print(f"✅ Successful: {stats['successful']}/{NUM_CLIPS}")
print(f"❌ Failed: {stats['failed']}/{NUM_CLIPS}")
print(f"📊 Total frames: {stats['total_frames']}")
print(f"⏱️  Time elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
print(f"🏠 Scenes collected: {len(stats['scenes'])}")

# 输出缺失字段统计
if any(stats["missing_fields"].values()):
    print("\n📊 Missing fields statistics:")
    for field, count in stats["missing_fields"].items():
        if count > 0:
            print(f"  {field}: {count}")
else:
    print("\n✅ No missing fields detected")

print("\nScene distribution:")
for scene, count in sorted(stats["scenes"].items(), key=lambda x: -x[1]):
    print(f"  {scene}: {count} clips")

if stats["failed"] > 0:
    print("\n⚠️  Failed clips:")
    for failed in stats["failed_clips"]:
        print(f"  Clip {failed['clip_id']}: {failed['error']}")

# 保存最终统计
with open(output_root / "collection_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

# 保存采集索引
index_path = output_root / f"{SPLIT}_index.json"
with open(index_path, "w") as f:
    json.dump(collected, f, indent=2)

print(f"\n📝 Statistics saved to {output_root / 'collection_stats.json'}")
print(f"📄 Index saved to {index_path}")
