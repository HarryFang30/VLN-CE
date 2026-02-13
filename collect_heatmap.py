#!/usr/bin/env python3
"""
多点往返巡逻数据采集脚本 (Multi-point Patrolling)

目标任务：输入当前第一视角图像，输出热力图指示哪些像素区域是过去走过的位置

采集策略：
1. 在场景中随机采样 N 个可导航点 (A, B, C, ...)
2. 规划往返路径 A→B→C→...→A（形成闭环）
3. 当从后续点返回时，视野中会大量包含之前走过的位置

数据格式：
- rgb/{front,right,back,left}/: 四个方向的 RGB 图像 (JPG)
- depth/{front,right,back,left}/: 四个方向的深度图 (NPY, float16)
- poses.json: 相机位姿，每帧包含 4 个方向的 4x4 变换矩阵
- trajectory_3d.npy: 3D 轨迹点 [T, 3]
- intrinsics.json: 相机内参
- meta.json: 元数据

注意：热力图 (heatmaps) 在训练时动态生成，不预先保存以节省磁盘空间
"""
import habitat
import cv2
import numpy as np
import json
import math
import time
import random
import shutil
import quaternion as _quaternion  # numpy-quaternion，用于四元数旋转
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple, Optional
import concurrent.futures
import habitat_extensions
from habitat_extensions.config.default import get_extended_config
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.utils.visualizations import maps as habitat_maps


# ==================== 多方向采集配置 ====================

DIRECTIONS = ["front", "right", "back", "left"]
DIRECTION_YAW_OFFSETS = {
    "front": 0.0,
    "right": -np.pi / 2,   # 顺时针 90°（向右看）
    "back": np.pi,          # 180°（向后看）
    "left": np.pi / 2,      # 逆时针 90°（向左看）
}


# ==================== 核心函数：热力图生成 ====================

def project_3d_to_2d_pinhole(
    points_3d: np.ndarray,  # [N, 3] 世界坐标系下的 3D 点
    T_world_cam: np.ndarray,  # [4, 4] 相机到世界的变换矩阵
    K: np.ndarray,  # [3, 3] 内参矩阵
    width: int,
    height: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将世界坐标系下的 3D 点投影到 Pinhole 相机图像
    
    Returns:
        pixels: [N, 2] 像素坐标 (u, v)
        valid_mask: [N] bool，标记哪些点投影成功（在视野内且在相机前方）
        z_depths: [N] float，每个点在相机坐标系中的 Z 深度（用于遮挡检测）
    """
    if len(points_3d) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float32)
    
    # 计算世界到相机的变换
    T_cam_world = np.linalg.inv(T_world_cam)
    
    # 转换为齐次坐标
    ones = np.ones((len(points_3d), 1), dtype=np.float32)
    points_homo = np.hstack([points_3d, ones])  # [N, 4]
    
    # 变换到相机坐标系
    points_cam = (T_cam_world @ points_homo.T).T[:, :3]  # [N, 3]
    
    # Habitat 坐标系：X 右，Y 上，-Z 前
    # 相机前方是 -Z 方向，所以 z < 0 才是在相机前方
    z = points_cam[:, 2]
    in_front = z < -0.1  # 在相机前方至少 0.1m
    
    # 投影到图像平面
    # Habitat 相机坐标系：X 右，Y 上，-Z 前
    # 标准图像坐标系：u 向右，v 向下
    x_cam = points_cam[:, 0]
    y_cam = points_cam[:, 1]
    z_cam = -points_cam[:, 2]  # 转换为正值（深度）
    
    # 避免除零
    z_cam_safe = np.maximum(z_cam, 1e-6)
    
    # 投影（注意 Y 轴方向：相机 Y 向上，图像 v 向下，所以取负）
    u = K[0, 0] * x_cam / z_cam_safe + K[0, 2]
    v = K[1, 1] * (-y_cam) / z_cam_safe + K[1, 2]  # Y 轴翻转
    
    # 检查是否在图像范围内
    in_image = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    
    # 组合有效性
    valid_mask = in_front & in_image
    
    pixels = np.stack([u, v], axis=1).astype(np.float32)
    return pixels, valid_mask, z_cam  # 返回 Z 深度用于遮挡检测


def generate_visited_heatmap(
    past_positions: np.ndarray,  # [M, 3] 过去走过的 3D 位置（Agent 地面位置）
    current_pose: np.ndarray,  # [4, 4] 当前相机位姿
    K: np.ndarray,  # [3, 3] 内参矩阵
    width: int,
    height: int,
    depth_image: Optional[np.ndarray] = None,  # [H, W] 当前帧深度图，用于遮挡检测
    sigma: float = 15.0,  # 高斯核大小
    occlusion_margin: float = 0.5,  # 遮挡判断容差（米）
) -> np.ndarray:
    """
    生成热力图：标记当前视野中哪些区域对应过去走过的位置
    
    遮挡检测：利用深度图判断是否被墙壁/楼板等物体遮挡
    - 深度遮挡可以正确处理楼层间遮挡（楼板会阻挡视线）
    - 楼梯口/挑空区域可以正确看到其他楼层
    
    Args:
        past_positions: 过去走过的 3D 位置（Agent 地面位置）
        current_pose: 当前相机位姿 [4, 4]
        K: 相机内参矩阵 [3, 3]
        width, height: 图像尺寸
        depth_image: 当前帧深度图，用于遮挡检测
        sigma: 高斯核大小（用于远处点的上限）
        occlusion_margin: 遮挡判断容差
    
    Returns:
        heatmap: [H, W] float32，值域 [0, 1]
    """
    heatmap = np.zeros((height, width), dtype=np.float32)
    
    if len(past_positions) == 0:
        return heatmap
    
    # 将过去的地面位置调整到相机可见的高度（Agent 眼睛高度约 1.25m）
    adjusted_positions = past_positions.copy()
    adjusted_positions[:, 1] += 1.25  # Y 轴向上调整到眼睛高度
    
    # 投影到当前视野（同时获取 Z 深度用于遮挡检测）
    pixels, valid_mask, z_depths = project_3d_to_2d_pinhole(
        adjusted_positions, current_pose, K, width, height
    )
    
    # 深度遮挡检测：利用深度图判断是否被遮挡
    # 这可以正确处理：墙壁遮挡、楼板遮挡、家具遮挡等
    if depth_image is not None:
        # 确保深度图是 2D
        if len(depth_image.shape) == 3:
            depth_image = depth_image[:, :, 0]
        
        # Habitat 深度图是归一化的 [0, 1]，需要转换回实际米数
        max_depth = 10.0
        depth_image_meters = depth_image * max_depth
        
        # 对于每个有效投影点，检查是否被遮挡
        for i in range(len(pixels)):
            if not valid_mask[i]:
                continue
            
            u, v = int(pixels[i, 0]), int(pixels[i, 1])
            if 0 <= u < width and 0 <= v < height:
                depth_at_pixel = depth_image_meters[v, u]
                point_z_depth = z_depths[i]
                
                # 如果深度图中的深度 < 点的Z深度 - 容差，说明被遮挡
                if depth_at_pixel > 0 and depth_at_pixel < point_z_depth - occlusion_margin:
                    valid_mask[i] = False
    
    valid_pixels = pixels[valid_mask]
    valid_z_depths = z_depths[valid_mask]
    
    if len(valid_pixels) == 0:
        return heatmap
    
    # 获取焦距（用于计算自适应 sigma）
    fx = K[0, 0]
    
    # 在投影位置绘制高斯斑点（近大远小）
    for idx, (uv, z_depth) in enumerate(zip(valid_pixels, valid_z_depths)):
        u, v = int(uv[0]), int(uv[1])
        
        # 自适应 sigma：近大远小
        # 假设一个 0.5m 的物体，投影后的像素大小 = object_size * fx / z_depth
        object_size_3d = 0.5  # 假设"走过的位置"对应一个 0.5m 的区域
        projected_size = object_size_3d * fx / max(z_depth, 0.1)
        # sigma 约为投影大小的 1/3
        adaptive_sigma = max(2.0, min(projected_size / 3.0, sigma))
        
        radius = int(3 * adaptive_sigma)
        y_min = max(0, v - radius)
        y_max = min(height, v + radius + 1)
        x_min = max(0, u - radius)
        x_max = min(width, u + radius + 1)
        
        if y_min >= y_max or x_min >= x_max:
            continue
        
        # 创建高斯核
        yy, xx = np.meshgrid(
            np.arange(y_min, y_max) - v,
            np.arange(x_min, x_max) - u,
            indexing='ij'
        )
        gaussian = np.exp(-(xx**2 + yy**2) / (2 * adaptive_sigma**2))
        
        # 累加到热力图
        heatmap[y_min:y_max, x_min:x_max] += gaussian.astype(np.float32)
    
    # 归一化到 [0, 1]
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    return heatmap


# ==================== 上帝视角轨迹图 ====================

def draw_arrow(img, pt1, pt2, color, thickness=2, arrow_size=8):
    """绘制带箭头的线段"""
    cv2.line(img, pt1, pt2, color, thickness)
    
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    length = np.sqrt(dx*dx + dy*dy)
    
    if length < 1:
        return
    
    ux, uy = dx / length, dy / length
    arrow_angle = 0.5
    left_x = pt2[0] - arrow_size * (ux * np.cos(arrow_angle) + uy * np.sin(arrow_angle))
    left_y = pt2[1] - arrow_size * (uy * np.cos(arrow_angle) - ux * np.sin(arrow_angle))
    right_x = pt2[0] - arrow_size * (ux * np.cos(arrow_angle) - uy * np.sin(arrow_angle))
    right_y = pt2[1] - arrow_size * (uy * np.cos(arrow_angle) + ux * np.sin(arrow_angle))
    
    pts = np.array([[pt2[0], pt2[1]], [int(left_x), int(left_y)], [int(right_x), int(right_y)]], np.int32)
    cv2.fillPoly(img, [pts], color)


def generate_topdown_trajectory_map(
    sim,
    trajectory_3d: np.ndarray,  # [T, 3] 3D 轨迹点
    waypoints: List[np.ndarray],  # 巡逻点列表
    current_frame: Optional[int] = None,
    output_size: int = 512,
    padding_meters: float = 5.0
) -> Tuple[np.ndarray, Dict]:
    """
    生成上帝视角（俯视图）的轨迹图，使用导航网格作为背景
    
    如需真实纹理俯视图，请使用 /root/habitat_tools 中的工具
    """
    # 计算轨迹的中心点和范围
    all_points = np.vstack([trajectory_3d] + [wp.reshape(1, 3) for wp in waypoints])
    
    # 计算边界框
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()
    
    # 添加边距
    x_min -= padding_meters
    x_max += padding_meters
    z_min -= padding_meters
    z_max += padding_meters
    
    # 计算视野范围（取较大的边）
    view_range = max(x_max - x_min, z_max - z_min)
    
    # 使用导航网格生成俯视图
    agent_height = sim.get_agent_state().position[1]
    
    # 高分辨率地图
    top_down_map = habitat_maps.get_topdown_map(
        sim.pathfinder, agent_height, 2048, False, 0.015
    )
    
    def world_to_map(pos_3d):
        grid_x, grid_y = habitat_maps.to_grid(
            pos_3d[2], pos_3d[0], top_down_map.shape[:2], sim
        )
        return int(grid_y), int(grid_x)
    
    # 计算裁剪区域
    map_points = [world_to_map(p) for p in all_points]
    xs = [p[0] for p in map_points]
    ys = [p[1] for p in map_points]
    
    padding = int(padding_meters / 0.015)
    x_min_px = max(0, min(xs) - padding)
    x_max_px = min(top_down_map.shape[1], max(xs) + padding)
    y_min_px = max(0, min(ys) - padding)
    y_max_px = min(top_down_map.shape[0], max(ys) + padding)
    
    size = max(x_max_px - x_min_px, y_max_px - y_min_px)
    x_center_px = (x_min_px + x_max_px) // 2
    y_center_px = (y_min_px + y_max_px) // 2
    
    x_min_px = max(0, x_center_px - size // 2)
    x_max_px = min(top_down_map.shape[1], x_center_px + size // 2)
    y_min_px = max(0, y_center_px - size // 2)
    y_max_px = min(top_down_map.shape[0], y_center_px + size // 2)
    
    cropped = top_down_map[y_min_px:y_max_px, x_min_px:x_max_px]
    
    # 创建清晰的彩色地图
    color_map = np.ones((cropped.shape[0], cropped.shape[1], 3), dtype=np.uint8) * 255
    
    # 可导航区域：浅色
    color_map[cropped == 1] = [235, 230, 225]
    # 墙壁/障碍：深色
    color_map[cropped == 0] = [120, 115, 110]
    # 边界：中灰
    color_map[cropped == 2] = [180, 175, 170]
    
    color_map = cv2.resize(color_map, (output_size, output_size), interpolation=cv2.INTER_AREA)
    
    def world_to_pixel(pos_3d):
        col, row = world_to_map(pos_3d)
        px = int((col - x_min_px) / max(1, size) * output_size)
        py = int((row - y_min_px) / max(1, size) * output_size)
        return px, py
    
    # 降采样轨迹
    step = max(1, len(trajectory_3d) // 60)
    sampled_traj = trajectory_3d[::step]
    if len(sampled_traj) > 1 and not np.array_equal(sampled_traj[-1], trajectory_3d[-1]):
        sampled_traj = np.vstack([sampled_traj, trajectory_3d[-1]])
    
    # 绘制轨迹（黄色带箭头）
    trajectory_color = (0, 220, 255)  # 黄色 BGR
    outline_color = (0, 100, 150)  # 深黄色轮廓
    
    if len(sampled_traj) > 1:
        pts = [world_to_pixel(p) for p in sampled_traj]
        
        # 绘制轨迹轮廓（深色）
        for i in range(len(pts) - 1):
            cv2.line(color_map, pts[i], pts[i+1], outline_color, thickness=5)
        
        # 绘制轨迹主线（亮色）
        for i in range(len(pts) - 1):
            cv2.line(color_map, pts[i], pts[i+1], trajectory_color, thickness=3)
        
        # 添加箭头
        arrow_interval = max(1, len(pts) // 6)
        for i in range(arrow_interval, len(pts), arrow_interval):
            draw_arrow(color_map, pts[i-1], pts[i], trajectory_color, thickness=3, arrow_size=10)
        
        # 终点箭头
        if len(pts) > 1:
            draw_arrow(color_map, pts[-2], pts[-1], trajectory_color, thickness=3, arrow_size=12)
    
    # 绘制巡逻点
    for i, wp in enumerate(waypoints):
        pt = world_to_pixel(wp)
        
        if i == 0:
            # 起点：绿色
            cv2.circle(color_map, pt, radius=18, color=(0, 200, 0), thickness=-1)
            cv2.circle(color_map, pt, radius=18, color=(0, 0, 0), thickness=2)
            label = "S"
        else:
            # 其他点：红色
            cv2.circle(color_map, pt, radius=15, color=(0, 0, 230), thickness=-1)
            cv2.circle(color_map, pt, radius=15, color=(0, 0, 0), thickness=2)
            label = str(i)
        
        cv2.putText(color_map, label, (pt[0] - 7, pt[1] + 7), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 添加比例尺
    scale_meters = 5.0
    # 根据视野范围计算比例尺像素
    scale_pixels = int(scale_meters / view_range * output_size)
    scale_pixels = max(20, min(scale_pixels, output_size // 3))
    
    scale_x, scale_y = 20, output_size - 25
    cv2.line(color_map, (scale_x, scale_y), (scale_x + scale_pixels, scale_y), (0, 0, 0), 4)
    cv2.line(color_map, (scale_x, scale_y), (scale_x + scale_pixels, scale_y), (255, 255, 255), 2)
    cv2.putText(color_map, f"{scale_meters:.0f}m", (scale_x + scale_pixels + 5, scale_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(color_map, f"{scale_meters:.0f}m", (scale_x + scale_pixels + 5, scale_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 计算所有轨迹点的像素坐标（用于可视化脚本）
    trajectory_pixels = [world_to_pixel(p) for p in trajectory_3d]
    waypoints_pixels = [world_to_pixel(wp) for wp in waypoints]
    
    # 返回坐标转换信息
    transform_info = {
        "trajectory_pixels": trajectory_pixels,  # 每帧在俯视图中的像素坐标
        "waypoints_pixels": waypoints_pixels,    # 每个waypoint的像素坐标
        "output_size": output_size,
    }
    
    return color_map, transform_info


# ==================== 辅助函数 ====================

def quaternion_to_rotation_matrix(q) -> np.ndarray:
    """将 Quaternion 转换为 3×3 旋转矩阵"""
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


def capture_multiview(sim, T_agent_cam: np.ndarray) -> Dict:
    """
    在当前位置采集前后左右四个方向的 RGB、Depth 图像和相机位姿
    
    通过临时旋转 Agent 朝向来采集不同方向的观测，采集完成后恢复原始状态。
    
    Args:
        sim: Habitat 仿真器
        T_agent_cam: 传感器外参矩阵 [4, 4]
    
    Returns:
        dict: {direction: {"rgb": ndarray, "depth": ndarray, "pose": ndarray[4,4]}}
              direction ∈ ["front", "right", "back", "left"]
    """
    agent_state = sim.get_agent_state()
    original_position = agent_state.position.copy()
    original_rotation = agent_state.rotation
    
    # 确保原始旋转是 np.quaternion 类型
    if not isinstance(original_rotation, np.quaternion):
        if hasattr(original_rotation, 'scalar') and hasattr(original_rotation, 'vector'):
            original_rotation = np.quaternion(
                original_rotation.scalar,
                original_rotation.vector.x,
                original_rotation.vector.y,
                original_rotation.vector.z
            )
        elif hasattr(original_rotation, 'w'):
            original_rotation = np.quaternion(
                original_rotation.w, original_rotation.x,
                original_rotation.y, original_rotation.z
            )
    
    results = {}
    
    for direction in DIRECTIONS:
        yaw = DIRECTION_YAW_OFFSETS[direction]
        
        if abs(yaw) < 1e-6:
            # Front: 使用原始朝向
            rotated_rotation = original_rotation
        else:
            # 创建绕 Y 轴旋转的四元数，并与原始朝向复合
            q_yaw = np.quaternion(np.cos(yaw / 2), 0, np.sin(yaw / 2), 0)
            rotated_rotation = original_rotation * q_yaw
        
        # 设置旋转后的 Agent 状态
        sim.set_agent_state(original_position, rotated_rotation)
        
        # 获取传感器观测
        obs = sim.get_sensor_observations()
        
        # 计算该方向的相机位姿
        fake_state = SimpleNamespace(position=original_position, rotation=rotated_rotation)
        pose = compute_camera_pose(fake_state, T_agent_cam)
        
        results[direction] = {
            "rgb": obs["rgb"].copy(),
            "depth": obs["depth"].copy(),
            "pose": pose,
        }
    
    # 恢复原始 Agent 状态（确保后续路径跟随不受影响）
    sim.set_agent_state(original_position, original_rotation)
    
    return results


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


def compute_intrinsics(config) -> Dict:
    """计算 Pinhole 相机内参"""
    rgb_cfg = config.SIMULATOR.RGB_SENSOR
    width = int(rgb_cfg.WIDTH)
    height = int(rgb_cfg.HEIGHT)
    hfov_deg = float(getattr(rgb_cfg, "HFOV", 90.0))
    hfov_rad = math.radians(hfov_deg)
    
    fx = width / (2.0 * math.tan(hfov_rad / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    
    vfov_rad = 2.0 * math.atan(height / (2.0 * fy))
    vfov_deg = math.degrees(vfov_rad)
    
    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    return {
        "projection": "pinhole",
        "width": width,
        "height": height,
        "hfov": hfov_deg,
        "vfov": vfov_deg,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "K": K.tolist()
    }


def check_path_exists(sim, start: np.ndarray, end: np.ndarray, max_detour_ratio: float = 5.0) -> bool:
    """
    检查两点之间是否存在可达路径
    
    Args:
        sim: Habitat 仿真器
        start: 起点 3D 坐标
        end: 终点 3D 坐标
        max_detour_ratio: 最大绕路比例（测地距离 / 直线距离）
    
    Returns:
        bool: 路径是否存在且合理
    """
    import habitat_sim
    
    # 使用 ShortestPath 对象检查可达性
    path = habitat_sim.ShortestPath()
    path.requested_start = start
    path.requested_end = end
    
    found = sim.pathfinder.find_path(path)
    
    # 检查是否找到路径
    if not found:
        return False
    
    # 检查测地距离是否有效
    geodesic_dist = path.geodesic_distance
    if not np.isfinite(geodesic_dist):
        return False
    
    # 路径长度不应该过长（避免绕太远的路径）
    straight_dist = np.linalg.norm(end - start)
    if straight_dist > 0 and geodesic_dist > straight_dist * max_detour_ratio:
        return False
    
    return True


def sample_navigable_points(
    sim, 
    num_points: int, 
    start_position: Optional[np.ndarray] = None,
    min_distance: float = 2.0,
    max_distance: float = 15.0,
    require_reachable: bool = True
) -> List[np.ndarray]:
    """
    在场景中采样可导航点，确保点之间有一定距离且互相可达
    
    Args:
        sim: Habitat 仿真器
        num_points: 需要采样的点数
        start_position: 起始位置（如果提供，第一个采样点需要与起始位置可达）
        min_distance: 点之间的最小距离（米）
        max_distance: 点之间的最大距离（米），避免采样太远的点
        require_reachable: 是否要求点之间可达
    
    Returns:
        List of 3D positions（不包含 start_position）
    """
    points = []
    max_attempts = num_points * 200  # 增加尝试次数，因为可达性检查更严格
    attempts = 0
    
    # 如果提供了起始位置，将其作为参考点（用于可达性检查，但不加入返回列表）
    reference_points = [start_position] if start_position is not None else []
    
    while len(points) < num_points and attempts < max_attempts:
        attempts += 1
        point = sim.pathfinder.get_random_navigable_point()
        
        if not np.isfinite(point).all():
            continue
        
        # 检查与所有参考点和已采样点的距离和可达性
        all_check_points = reference_points + points
        valid = True
        
        for existing_point in all_check_points:
            dist = np.linalg.norm(point - existing_point)
            
            # 距离检查：不能太近也不能太远
            if dist < min_distance or dist > max_distance:
                valid = False
                break
            
            # 可达性检查：确保两点之间存在路径
            if require_reachable and not check_path_exists(sim, existing_point, point):
                valid = False
                break
        
        if valid:
            points.append(point)
    
    return points


def plan_patrol_path(
    sim,
    waypoints: List[np.ndarray],
    return_to_start: bool = True
) -> List[np.ndarray]:
    """
    规划巡逻路径：按顺序访问所有路径点
    
    Args:
        sim: Habitat 仿真器
        waypoints: 路径点列表
        return_to_start: 是否返回起点形成闭环
    
    Returns:
        按顺序排列的路径点（如果 return_to_start=True，最后一个点是起点）
    """
    if len(waypoints) == 0:
        return []
    
    path = list(waypoints)
    
    if return_to_start and len(path) > 1:
        path.append(waypoints[0])  # 返回起点
    
    return path


# ==================== 命令行参数 ====================
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="多点往返巡逻数据采集")
    parser.add_argument('--config', type=str, default="habitat_extensions/config/vlnce_collect.yaml",
                        help='Habitat 配置文件路径')
    parser.add_argument('--output', type=str, default="/root/autodl-tmp/heatmap_train_data",
                        help='输出目录')
    parser.add_argument('--num-clips', type=int, default=1000,
                        help='采集的 clip 数量')
    parser.add_argument('--num-waypoints', type=int, default=4,
                        help='每个 clip 的巡逻点数量')
    parser.add_argument('--min-waypoint-dist', type=float, default=3.0,
                        help='巡逻点之间的最小距离（米）')
    parser.add_argument('--max-waypoint-dist', type=float, default=15.0,
                        help='巡逻点之间的最大距离（米）')
    parser.add_argument('--max-steps', type=int, default=500,
                        help='每个 clip 的最大步数')
    parser.add_argument('--num-workers', type=int, default=16,
                        help='IO worker 数量')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU 设备 ID')
    parser.add_argument('--return-to-start', action='store_true', default=True,
                        help='是否返回起点形成闭环')
    return parser.parse_args()


# ==================== 主程序 ====================

def main():
    args = parse_args()
    
    print("="*60)
    print("🚀 多点往返巡逻数据采集 (Multi-point Patrolling)")
    print("="*60)
    print(f"   输出目录: {args.output}")
    print(f"   Clips 数量: {args.num_clips}")
    print(f"   每个 clip 巡逻点: {args.num_waypoints}")
    print(f"   巡逻点距离: {args.min_waypoint_dist}m ~ {args.max_waypoint_dist}m")
    print(f"   最大步数: {args.max_steps}")
    print(f"   返回起点: {args.return_to_start}")
    print(f"   💾 热力图: 不保存 (训练时动态生成)")
    print(f"   ✅ 路径可达性检查: 已启用")
    print("="*60)
    
    # 加载配置
    config = get_extended_config(args.config)
    config.defrost()
    config.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
    config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = args.gpu
    config.freeze()
    
    # 验证是 Pinhole 模式
    subtype = getattr(config.SIMULATOR.RGB_SENSOR, "SENSOR_SUBTYPE", "PINHOLE")
    if "EQUIRECT" in str(subtype).upper():
        raise ValueError("此脚本需要 PINHOLE 投影模式，请检查配置文件")
    
    print("📷 创建环境...")
    env = habitat.Env(config=config)
    sim = env.sim
    
    # 计算内参
    intrinsics = compute_intrinsics(config)
    K = np.array(intrinsics["K"], dtype=np.float32)
    img_width = intrinsics["width"]
    img_height = intrinsics["height"]
    print(f"📐 相机参数: {img_width}x{img_height}, HFOV={intrinsics['hfov']}°")
    
    # 传感器外参
    T_agent_cam = get_sensor_extrinsics(config)
    
    # 获取场景列表
    dataset = env._dataset
    print(f"📊 数据集: {len(dataset.episodes)} episodes")
    
    # 按场景分组
    scenes = {}
    for i, ep in enumerate(dataset.episodes):
        scene_name = ep.scene_id.split("/")[-1].replace(".glb", "")
        if scene_name not in scenes:
            scenes[scene_name] = []
        scenes[scene_name].append(i)
    
    print(f"🏠 场景数量: {len(scenes)}")
    
    # 准备输出
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    
    # 统计
    stats = {
        "successful": 0,
        "failed": 0,
        "total_frames": 0,
        "scenes": {}
    }
    
    # 线程池
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers)
    io_futures = []
    
    start_time = time.time()
    
    # 断点续采：扫描现有 clips 找到最大 clip_id
    existing_clips = list(output_root.glob("*/clip_*/meta.json"))
    if existing_clips:
        max_existing_id = 0
        for meta_path in existing_clips:
            clip_name = meta_path.parent.name  # e.g., "clip_001234"
            try:
                clip_num = int(clip_name.split("_")[1])
                max_existing_id = max(max_existing_id, clip_num)
            except (ValueError, IndexError):
                pass
        clip_id = max_existing_id + 1
        print(f"📂 断点续采: 发现 {len(existing_clips)} 个已完成 clips，从 clip_{clip_id:06d} 继续")
    else:
        clip_id = 1
        print(f"📂 全新采集: 从 clip_{clip_id:06d} 开始")
    
    scene_list = list(scenes.keys())
    random.shuffle(scene_list)
    scene_idx = 0
    
    while clip_id <= args.num_clips:
        if scene_idx >= len(scene_list):
            scene_idx = 0
            random.shuffle(scene_list)
        
        scene_name = scene_list[scene_idx]
        scene_episodes = scenes[scene_name]
        episode_idx = random.choice(scene_episodes)
        episode = dataset.episodes[episode_idx]
        scene_idx += 1
        
        print(f"\n📁 Clip {clip_id}/{args.num_clips} - Scene: {scene_name}")
        
        try:
            # 重置环境
            env._current_episode = episode
            observations = env.reset()
            sim = env.sim
            
            # 创建路径跟随器
            follower = ShortestPathFollower(sim, goal_radius=0.5, return_one_hot=False)
            
            # 获取起始位置
            start_pos = sim.get_agent_state().position.copy()
            
            # 采样巡逻点（传入起始位置，确保第一个点与起始位置可达）
            waypoints = sample_navigable_points(
                sim, 
                args.num_waypoints, 
                start_position=start_pos,
                min_distance=args.min_waypoint_dist,
                max_distance=args.max_waypoint_dist,
                require_reachable=True
            )
            
            if len(waypoints) < 2:
                print(f"  ⚠️  无法采样足够的可达巡逻点，跳过")
                stats["failed"] += 1
                continue
            
            # 将起始位置加入巡逻路径开头
            waypoints.insert(0, start_pos)
            
            # 规划巡逻路径
            patrol_path = plan_patrol_path(sim, waypoints, return_to_start=args.return_to_start)
            print(f"  📍 巡逻路径: {len(patrol_path)} 个点")
            
            # 创建输出目录
            clip_dir = output_root / scene_name / f"clip_{clip_id:06d}"
            rgb_dir = clip_dir / "rgb"
            depth_dir = clip_dir / "depth"
            for direction in DIRECTIONS:
                (rgb_dir / direction).mkdir(parents=True, exist_ok=True)
                (depth_dir / direction).mkdir(parents=True, exist_ok=True)
            
            # 数据存储
            poses = []
            trajectory_3d = []  # 3D 轨迹点
            frame_id = 0
            
            # 遍历巡逻路径
            for target_idx, target_point in enumerate(patrol_path[1:], 1):
                print(f"    → 前往点 {target_idx}/{len(patrol_path)-1}")
                
                steps_to_target = 0
                max_steps_per_target = args.max_steps // len(patrol_path)
                
                while steps_to_target < max_steps_per_target:
                    # 获取当前状态
                    agent_state = sim.get_agent_state()
                    current_pos = agent_state.position.copy()
                    
                    # 检查是否到达目标
                    dist_to_target = np.linalg.norm(current_pos - target_point)
                    if dist_to_target < 0.5:
                        break
                    
                    # 记录当前帧（前后左右四个方向）
                    multiview = capture_multiview(sim, T_agent_cam)
                    
                    frame_poses = {}
                    for direction in DIRECTIONS:
                        obs_d = multiview[direction]
                        
                        # 1. 保存 RGB
                        rgb = obs_d["rgb"]
                        if rgb.shape[2] == 4:
                            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
                        elif rgb.shape[2] == 3:
                            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                        rgb_path = rgb_dir / direction / f"{frame_id:06d}.jpg"
                        io_futures.append(executor.submit(
                            cv2.imwrite, str(rgb_path), rgb.copy(), 
                            [cv2.IMWRITE_JPEG_QUALITY, 95]
                        ))
                        
                        # 2. 保存 Depth
                        depth = obs_d["depth"]
                        depth_path = depth_dir / direction / f"{frame_id:06d}.npy"
                        io_futures.append(executor.submit(
                            np.save, str(depth_path), depth.copy().astype(np.float16)
                        ))
                        
                        # 3. 记录该方向的相机位姿
                        frame_poses[direction] = obs_d["pose"].tolist()
                    
                    poses.append(frame_poses)
                    
                    # 4. 记录 3D 位置（Agent 地面位置，与方向无关）
                    trajectory_3d.append(current_pos.copy())
                    
                    frame_id += 1
                    
                    # 获取下一个动作
                    action = follower.get_next_action(target_point)
                    if action is None or action == HabitatSimActions.STOP:
                        break
                    
                    # 执行动作
                    observations = env.step(action)
                    steps_to_target += 1
                
                # 检查是否超过最大步数
                if frame_id >= args.max_steps:
                    print(f"    ⚠️  达到最大步数限制")
                    break
            
            # 最低帧数要求
            MIN_FRAMES = 20
            if frame_id < MIN_FRAMES:
                print(f"  ⚠️  帧数太少 ({frame_id} < {MIN_FRAMES})，跳过")
                shutil.rmtree(clip_dir)
                continue
            
            # 保存元数据
            
            # 保存 poses
            with open(clip_dir / "poses.json", "w") as f:
                json.dump(poses, f, indent=2)
            
            # 保存 3D 轨迹
            trajectory_3d_arr = np.array(trajectory_3d, dtype=np.float32)
            np.save(clip_dir / "trajectory_3d.npy", trajectory_3d_arr)
            
            # 生成并保存上帝视角轨迹图（带真实场景纹理）
            topdown_transform = None
            try:
                topdown_map, topdown_transform = generate_topdown_trajectory_map(
                    sim,
                    trajectory_3d_arr,
                    waypoints,
                    current_frame=None,
                    output_size=512,
                    padding_meters=5.0
                )
                topdown_path = clip_dir / "topdown_trajectory.jpg"
                io_futures.append(executor.submit(
                    cv2.imwrite, str(topdown_path), topdown_map,
                    [cv2.IMWRITE_JPEG_QUALITY, 90]
                ))
                
                # 保存坐标转换信息（用于可视化脚本）
                with open(clip_dir / "topdown_transform.json", "w") as f:
                    json.dump(topdown_transform, f)
            except Exception as e:
                print(f"    ⚠️  上帝视角图生成失败: {e}")
            
            # 保存内参
            with open(clip_dir / "intrinsics.json", "w") as f:
                json.dump(intrinsics, f, indent=2)
            
            # 保存巡逻点
            waypoints_list = [wp.tolist() for wp in waypoints]
            
            # 保存元数据
            meta = {
                "scene_id": scene_name,
                "episode_id": episode.episode_id,
                "num_frames": frame_id,
                "num_waypoints": len(waypoints),
                "waypoints": waypoints_list,
                "return_to_start": args.return_to_start,
                "data_format": {
                    "rgb": "JPG images in rgb/{front,right,back,left}/ folders (4 views per frame)",
                    "depth": "NPY float16 in depth/{front,right,back,left}/ folders (4 views per frame)",
                    "poses": "Per-frame dict with 4 directions, each a 4x4 camera-to-world transform",
                    "trajectory_3d": "NPY float32 [T,3] world positions",
                    "topdown_trajectory": "JPG bird's eye view trajectory map",
                    "directions": DIRECTIONS
                }
            }
            
            with open(clip_dir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)
            
            # 更新统计
            stats["successful"] += 1
            stats["total_frames"] += frame_id
            stats["scenes"][scene_name] = stats["scenes"].get(scene_name, 0) + 1
            
            print(f"  ✅ 完成: {frame_id} 帧")
            
            # 清理 IO futures
            if len(io_futures) > 100:
                concurrent.futures.wait(io_futures)
                io_futures.clear()
            
            clip_id += 1
            
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            stats["failed"] += 1
            if 'clip_dir' in locals() and clip_dir.exists():
                try:
                    shutil.rmtree(clip_dir)
                except:
                    pass
            continue
    
    # 等待所有 IO 完成
    if io_futures:
        print(f"\n⏳ 等待 {len(io_futures)} 个 IO 操作完成...")
        concurrent.futures.wait(io_futures)
    
    env.close()
    
    # 输出统计
    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print("🎉 采集完成!")
    print("="*60)
    print(f"✅ 成功: {stats['successful']}/{args.num_clips}")
    print(f"❌ 失败: {stats['failed']}")
    print(f"📊 总帧数: {stats['total_frames']}")
    print(f"⏱️  耗时: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"🏠 场景分布: {len(stats['scenes'])} 个场景")
    
    # 保存统计
    with open(output_root / "collection_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n📝 统计信息已保存到 {output_root / 'collection_stats.json'}")


if __name__ == "__main__":
    main()
