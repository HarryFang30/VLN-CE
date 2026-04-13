"""
3D→2D 投影 与 热力图生成

提供 Pinhole 投影和 visited-heatmap 生成，供训练时动态调用。
采集阶段不保存热力图（节省磁盘），训练时按需生成。
"""
import numpy as np
from typing import Tuple, Optional


def project_3d_to_2d_pinhole(
    points_3d: np.ndarray,
    T_world_cam: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将世界坐标系下的 3D 点投影到 Pinhole 相机图像。

    Args:
        points_3d: [N, 3] 世界坐标系 3D 点
        T_world_cam: [4, 4] 相机到世界变换
        K: [3, 3] 内参矩阵
        width, height: 图像尺寸

    Returns:
        pixels: [N, 2] 像素坐标 (u, v)
        valid_mask: [N] bool
        z_depths: [N] float，相机坐标系 Z 深度
    """
    if len(points_3d) == 0:
        return (np.zeros((0, 2), dtype=np.float32),
                np.zeros(0, dtype=bool),
                np.zeros(0, dtype=np.float32))

    T_cam_world = np.linalg.inv(T_world_cam)

    ones = np.ones((len(points_3d), 1), dtype=np.float32)
    points_homo = np.hstack([points_3d, ones])
    points_cam = (T_cam_world @ points_homo.T).T[:, :3]

    # Habitat: X 右, Y 上, -Z 前  →  z < -0.1 才在相机前方
    z = points_cam[:, 2]
    in_front = z < -0.1

    x_cam = points_cam[:, 0]
    y_cam = points_cam[:, 1]
    z_cam = -points_cam[:, 2]
    z_cam_safe = np.maximum(z_cam, 1e-6)

    u = K[0, 0] * x_cam / z_cam_safe + K[0, 2]
    v = K[1, 1] * (-y_cam) / z_cam_safe + K[1, 2]

    in_image = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    valid_mask = in_front & in_image

    pixels = np.stack([u, v], axis=1).astype(np.float32)
    return pixels, valid_mask, z_cam


def generate_visited_heatmap(
    past_positions: np.ndarray,
    current_pose: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    depth_image: Optional[np.ndarray] = None,
    occlusion_margin: float = 0.5,
    max_visible_distance: float = 15.0,
    use_distance_decay: bool = True,
    distance_decay_ref: float = 5.0,
    min_peak_value: float = 0.3,
) -> np.ndarray:
    """
    生成热力图：标记当前视野中哪些区域对应过去走过的位置。

    使用 max 合并（避免重叠饱和）+ 距离衰减，与训练时 HeatmapVLN 保持一致。

    Args:
        past_positions: [M, 3] 过去走过的 Agent 地面位置
        current_pose: [4, 4] 当前相机位姿
        K: [3, 3] 内参矩阵
        depth_image: [H, W] 当前帧深度图（可选，用于遮挡检测）

    Returns:
        heatmap: [H, W] float32, 值域 [0, 1]
    """
    heatmap = np.zeros((height, width), dtype=np.float32)
    if len(past_positions) == 0:
        return heatmap

    adjusted = past_positions.copy()
    adjusted[:, 1] += 1.25  # 调整到眼睛高度

    pixels, valid_mask, z_depths = project_3d_to_2d_pinhole(
        adjusted, current_pose, K, width, height,
    )

    # 深度遮挡检测
    if depth_image is not None:
        if len(depth_image.shape) == 3:
            depth_image = depth_image[:, :, 0]
        depth_meters = depth_image * 10.0
        for i in range(len(pixels)):
            if not valid_mask[i]:
                continue
            u, v = int(pixels[i, 0]), int(pixels[i, 1])
            if 0 <= u < width and 0 <= v < height:
                d = depth_meters[v, u]
                if 0 < d < z_depths[i] - occlusion_margin:
                    valid_mask[i] = False

    valid_pixels = pixels[valid_mask]
    valid_z = z_depths[valid_mask]
    if len(valid_pixels) == 0:
        return heatmap

    fx = K[0, 0]
    cam_pos = current_pose[:3, 3]
    valid_positions = adjusted[valid_mask]
    valid_distances = np.linalg.norm(valid_positions - cam_pos, axis=1)

    hm_width = 64
    img_to_hm_scale = hm_width / width

    for uv, z_depth, distance in zip(valid_pixels, valid_z, valid_distances):
        u, v = int(uv[0]), int(uv[1])
        if z_depth > max_visible_distance:
            continue

        projected_size_hm = 0.5 * fx / max(z_depth, 0.1) * img_to_hm_scale
        sigma_hm = np.clip(projected_size_hm / 3.0, 1.5, 6.0)
        adaptive_sigma = sigma_hm / img_to_hm_scale

        if use_distance_decay:
            decay = 1.0 / (1.0 + distance / distance_decay_ref)
            peak_value = min_peak_value + (1.0 - min_peak_value) * decay
        else:
            peak_value = 1.0

        radius = int(3 * adaptive_sigma)
        y_min, y_max = max(0, v - radius), min(height, v + radius + 1)
        x_min, x_max = max(0, u - radius), min(width, u + radius + 1)
        if y_min >= y_max or x_min >= x_max:
            continue

        yy, xx = np.meshgrid(
            np.arange(y_min, y_max) - v,
            np.arange(x_min, x_max) - u,
            indexing="ij",
        )
        gaussian = peak_value * np.exp(-(xx ** 2 + yy ** 2) / (2 * adaptive_sigma ** 2))

        np.maximum(
            heatmap[y_min:y_max, x_min:x_max],
            gaussian.astype(np.float32),
            out=heatmap[y_min:y_max, x_min:x_max],
        )

    return heatmap
