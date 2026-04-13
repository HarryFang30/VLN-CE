"""
俯视图 / 轨迹可视化

生成上帝视角轨迹图（基于导航网格），用于数据质量检查。
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from habitat.utils.visualizations import maps as habitat_maps


def draw_arrow(img, pt1, pt2, color, thickness=2, arrow_size=8):
    """绘制带箭头的线段"""
    cv2.line(img, pt1, pt2, color, thickness)
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    length = np.sqrt(dx * dx + dy * dy)
    if length < 1:
        return

    ux, uy = dx / length, dy / length
    a = 0.5
    pts = np.array([
        [pt2[0], pt2[1]],
        [int(pt2[0] - arrow_size * (ux * np.cos(a) + uy * np.sin(a))),
         int(pt2[1] - arrow_size * (uy * np.cos(a) - ux * np.sin(a)))],
        [int(pt2[0] - arrow_size * (ux * np.cos(a) - uy * np.sin(a))),
         int(pt2[1] - arrow_size * (uy * np.cos(a) + ux * np.sin(a)))],
    ], np.int32)
    cv2.fillPoly(img, [pts], color)


def generate_topdown_trajectory_map(
    sim,
    trajectory_3d: np.ndarray,
    waypoints: List[np.ndarray],
    current_frame: Optional[int] = None,
    output_size: int = 512,
    padding_meters: float = 5.0,
) -> Tuple[np.ndarray, Dict]:
    """
    生成上帝视角（俯视图）轨迹图，使用导航网格作为背景。

    Returns:
        color_map: [output_size, output_size, 3] uint8 BGR 图像
        transform_info: 包含像素坐标映射信息的字典
    """
    all_points = np.vstack([trajectory_3d] + [wp.reshape(1, 3) for wp in waypoints])

    x_min, x_max = all_points[:, 0].min() - padding_meters, all_points[:, 0].max() + padding_meters
    z_min, z_max = all_points[:, 2].min() - padding_meters, all_points[:, 2].max() + padding_meters
    view_range = max(x_max - x_min, z_max - z_min)

    agent_height = sim.get_agent_state().position[1]
    top_down_map = habitat_maps.get_topdown_map(
        sim.pathfinder, agent_height, 2048, False, 0.015,
    )

    def world_to_map(pos_3d):
        gx, gy = habitat_maps.to_grid(pos_3d[2], pos_3d[0], top_down_map.shape[:2], sim)
        return int(gy), int(gx)

    map_points = [world_to_map(p) for p in all_points]
    xs = [p[0] for p in map_points]
    ys = [p[1] for p in map_points]
    padding_px = int(padding_meters / 0.015)

    x_min_px = max(0, min(xs) - padding_px)
    x_max_px = min(top_down_map.shape[1], max(xs) + padding_px)
    y_min_px = max(0, min(ys) - padding_px)
    y_max_px = min(top_down_map.shape[0], max(ys) + padding_px)

    size = max(x_max_px - x_min_px, y_max_px - y_min_px)
    cx_px = (x_min_px + x_max_px) // 2
    cy_px = (y_min_px + y_max_px) // 2
    x_min_px = max(0, cx_px - size // 2)
    x_max_px = min(top_down_map.shape[1], cx_px + size // 2)
    y_min_px = max(0, cy_px - size // 2)
    y_max_px = min(top_down_map.shape[0], cy_px + size // 2)

    cropped = top_down_map[y_min_px:y_max_px, x_min_px:x_max_px]

    color_map = np.ones((cropped.shape[0], cropped.shape[1], 3), dtype=np.uint8) * 255
    color_map[cropped == 1] = [235, 230, 225]
    color_map[cropped == 0] = [120, 115, 110]
    color_map[cropped == 2] = [180, 175, 170]
    color_map = cv2.resize(color_map, (output_size, output_size), interpolation=cv2.INTER_AREA)

    def world_to_pixel(pos_3d):
        col, row = world_to_map(pos_3d)
        px = int((col - x_min_px) / max(1, size) * output_size)
        py = int((row - y_min_px) / max(1, size) * output_size)
        return px, py

    # 降采样轨迹
    step = max(1, len(trajectory_3d) // 60)
    sampled = trajectory_3d[::step]
    if len(sampled) > 1 and not np.array_equal(sampled[-1], trajectory_3d[-1]):
        sampled = np.vstack([sampled, trajectory_3d[-1]])

    trajectory_color = (0, 220, 255)
    outline_color = (0, 100, 150)

    if len(sampled) > 1:
        pts = [world_to_pixel(p) for p in sampled]
        for i in range(len(pts) - 1):
            cv2.line(color_map, pts[i], pts[i + 1], outline_color, thickness=5)
        for i in range(len(pts) - 1):
            cv2.line(color_map, pts[i], pts[i + 1], trajectory_color, thickness=3)
        arrow_interval = max(1, len(pts) // 6)
        for i in range(arrow_interval, len(pts), arrow_interval):
            draw_arrow(color_map, pts[i - 1], pts[i], trajectory_color, thickness=3, arrow_size=10)
        if len(pts) > 1:
            draw_arrow(color_map, pts[-2], pts[-1], trajectory_color, thickness=3, arrow_size=12)

    # 巡逻点标记
    for i, wp in enumerate(waypoints):
        pt = world_to_pixel(wp)
        if i == 0:
            cv2.circle(color_map, pt, 18, (0, 200, 0), -1)
            cv2.circle(color_map, pt, 18, (0, 0, 0), 2)
            label = "S"
        else:
            cv2.circle(color_map, pt, 15, (0, 0, 230), -1)
            cv2.circle(color_map, pt, 15, (0, 0, 0), 2)
            label = str(i)
        cv2.putText(color_map, label, (pt[0] - 7, pt[1] + 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 比例尺
    scale_m = 5.0
    scale_px = max(20, min(int(scale_m / view_range * output_size), output_size // 3))
    sx, sy = 20, output_size - 25
    cv2.line(color_map, (sx, sy), (sx + scale_px, sy), (0, 0, 0), 4)
    cv2.line(color_map, (sx, sy), (sx + scale_px, sy), (255, 255, 255), 2)
    cv2.putText(color_map, f"{scale_m:.0f}m", (sx + scale_px + 5, sy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(color_map, f"{scale_m:.0f}m", (sx + scale_px + 5, sy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    transform_info = {
        "trajectory_pixels": [world_to_pixel(p) for p in trajectory_3d],
        "waypoints_pixels": [world_to_pixel(wp) for wp in waypoints],
        "output_size": output_size,
    }
    return color_map, transform_info
