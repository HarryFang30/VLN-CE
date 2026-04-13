"""
R2R 采集专用工具函数：关键帧匹配、质量控制、NeRF 波纹等
"""
import numpy as np
from typing import List, Tuple


def match_keyframes_to_trajectory(
    trajectory: List[np.ndarray],
    reference_path: List[List[float]],
) -> Tuple[List[int], List[float]]:
    """为 reference_path 中的每个节点找到最近的轨迹帧索引及距离"""
    keyframe_indices = []
    keyframe_distances = []
    for ref_point in reference_path:
        ref_pos = np.array(ref_point)
        distances = [np.linalg.norm(pos - ref_pos) for pos in trajectory]
        closest_idx = int(np.argmin(distances))
        keyframe_indices.append(closest_idx)
        keyframe_distances.append(float(distances[closest_idx]))
    return keyframe_indices, keyframe_distances


def compute_adaptive_min_valid_ratio(
    keyframe_distances: List[float],
) -> Tuple[float, str]:
    """基于轨迹质量动态计算 MIN_VALID_RATIO 阈值"""
    if len(keyframe_distances) == 0:
        return 0.4, "low"

    avg_dist = float(np.mean(keyframe_distances))

    if avg_dist <= 0.3:
        threshold, quality_tier = 0.70, "excellent"
    elif avg_dist >= 2.5:
        threshold, quality_tier = 0.40, "low"
    else:
        threshold = 0.70 - (avg_dist - 0.3) / (2.5 - 0.3) * (0.70 - 0.40)
        threshold = float(np.clip(threshold, 0.40, 0.70))
        if avg_dist <= 0.8:
            quality_tier = "high"
        elif avg_dist <= 1.5:
            quality_tier = "medium"
        else:
            quality_tier = "acceptable"

    return threshold, quality_tier


def select_keyframes_motion_based(
    poses: List[List[float]],
    min_dist: float = 0.5,
    min_angle_deg: float = 15.0,
) -> List[int]:
    """基于运动量的贪婪关键帧选择策略"""
    num_frames = len(poses)
    if num_frames == 0:
        return []

    selected = [0]
    last_idx = 0

    for curr_idx in range(1, num_frames):
        T_last = np.array(poses[last_idx], dtype=np.float32)
        T_curr = np.array(poses[curr_idx], dtype=np.float32)

        dist = float(np.linalg.norm(T_curr[:3, 3] - T_last[:3, 3]))

        R_diff = T_curr[:3, :3] @ T_last[:3, :3].T
        trace = float(np.clip(np.trace(R_diff), -1.0, 3.0))
        angle_deg = float(np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))))

        if dist > min_dist or angle_deg > min_angle_deg or curr_idx == num_frames - 1:
            selected.append(curr_idx)
            last_idx = curr_idx

    return selected


def draw_nerf_ripple_point(
    heatmap: np.ndarray,
    center: Tuple[float, float],
    sigma: float,
    frame_rank: int,
    base_freq: float = 1.0,
) -> None:
    """基于 NeRF 位置编码思想的波纹绘制"""
    H, W = heatmap.shape
    u, v = center

    radius = max(1, int(np.ceil(3.0 * sigma)))
    x_min, x_max = max(0, int(np.floor(u - radius))), min(W, int(np.ceil(u + radius)))
    y_min, y_max = max(0, int(np.floor(v - radius))), min(H, int(np.ceil(v + radius)))
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
        modulation = 0.5 + 0.5 * (signal_accum / weight_accum) if weight_accum > 0 else 1.0

    blob = (envelope * modulation).astype(np.float32, copy=False)
    roi = heatmap[y_min:y_max, x_min:x_max]
    np.add(roi, blob, out=roi)
