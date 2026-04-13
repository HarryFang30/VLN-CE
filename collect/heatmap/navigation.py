"""
路径规划与可导航点采样

提供巡逻路径规划所需的可达性检查和随机巡逻点采样。
"""
import numpy as np
from typing import List, Optional


def check_path_exists(
    sim,
    start: np.ndarray,
    end: np.ndarray,
    max_detour_ratio: float = 5.0,
) -> bool:
    """
    检查两点之间是否存在可达路径（测地距离 / 直线距离 <= max_detour_ratio）。
    """
    import habitat_sim

    path = habitat_sim.ShortestPath()
    path.requested_start = start
    path.requested_end = end

    if not sim.pathfinder.find_path(path):
        return False
    if not np.isfinite(path.geodesic_distance):
        return False

    straight_dist = np.linalg.norm(end - start)
    if straight_dist > 0 and path.geodesic_distance > straight_dist * max_detour_ratio:
        return False
    return True


def sample_navigable_points(
    sim,
    num_points: int,
    start_position: Optional[np.ndarray] = None,
    min_distance: float = 2.0,
    max_distance: float = 15.0,
    require_reachable: bool = True,
) -> List[np.ndarray]:
    """
    在场景中采样可导航点，确保点间距离合理且互相可达。

    Args:
        sim: Habitat 仿真器
        num_points: 需要采样的点数
        start_position: 起始位置（不加入返回列表，但作为距离/可达参考）
        min_distance: 点间最小距离（米）
        max_distance: 点间最大距离（米）
        require_reachable: 是否要求可达

    Returns:
        采样到的 3D 坐标列表（不含 start_position）
    """
    points: List[np.ndarray] = []
    reference = [start_position] if start_position is not None else []
    max_attempts = num_points * 200

    for _ in range(max_attempts):
        if len(points) >= num_points:
            break

        pt = sim.pathfinder.get_random_navigable_point()
        if not np.isfinite(pt).all():
            continue

        ok = True
        for existing in reference + points:
            dist = np.linalg.norm(pt - existing)
            if dist < min_distance or dist > max_distance:
                ok = False
                break
            if require_reachable and not check_path_exists(sim, existing, pt):
                ok = False
                break
        if ok:
            points.append(pt)

    return points


def plan_patrol_path(
    waypoints: List[np.ndarray],
    return_to_start: bool = True,
) -> List[np.ndarray]:
    """
    生成巡逻路径：按顺序访问所有路径点，可选返回起点形成闭环。
    """
    if not waypoints:
        return []
    path = list(waypoints)
    if return_to_start and len(path) > 1:
        path.append(waypoints[0])
    return path
