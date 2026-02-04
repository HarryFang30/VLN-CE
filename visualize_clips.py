#!/usr/bin/env python3
"""
可视化巡逻数据采集的clips

为每个clip的每一帧创建完整可视化，包含：
- Top-Down: 导航网格 + 过去轨迹 + 当前位置箭头
- RGB: 当前第一视角图像
- Heatmap: 热力图（动态生成，显示覆盖率百分比）
- Overlay: RGB + 热力图叠加

用法:
    python visualize_clips.py --input /path/to/patrol_data --output /path/to/output
    python visualize_clips.py --input /path/to/patrol_data --output /path/to/output --interval 3
    python visualize_clips.py --input /path/to/patrol_data --output /path/to/output --random 10
"""

import argparse
import json
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ==================== 热力图动态生成函数 (历史帧相机位置) ====================

def project_point_pinhole(
    p_cam: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int
):
    """
    将相机坐标系下的3D点投影到Pinhole图像坐标
    
    Habitat 相机坐标系：X 右，Y 上，-Z 前
    因此相机前方是 z < 0
    """
    x, y, z = float(p_cam[0]), float(p_cam[1]), float(p_cam[2])
    
    # 相机前方是 -Z 方向，所以 z < 0 才是在相机前方
    if z >= -0.1:
        return None
    
    z_depth = -z
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    u = fx * x / z_depth + cx
    v = fy * (-y) / z_depth + cy  # Y 轴翻转
    
    if not (0 <= u < width and 0 <= v < height):
        return None
    
    return float(u), float(v), float(z_depth)


def generate_history_heatmap(
    history_poses: list,
    current_pose: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    depth_image: Optional[np.ndarray] = None,
    occlusion_margin: float = 0.5,
    max_visible_distance: float = 15.0,
    # 新增参数：与训练逻辑一致
    use_max_merge: bool = True,           # 使用 max 合并（避免累加饱和）
    use_distance_decay: bool = True,      # 启用距离衰减
    distance_decay_ref: float = 5.0,      # 距离衰减参考值（米）
    min_peak_value: float = 0.3,          # 最远处的最小峰值
) -> np.ndarray:
    """
    生成热力图：标记当前视野中历史帧相机位置的投影
    
    设计原则（便于模型学习）：
    1. 使用 max 合并而非累加，避免重叠区域饱和
    2. 峰值随距离衰减，体现"近处更重要"
    3. 增大 min_sigma，让远处的点也清晰可见
    4. 值范围 [0, 1]，不依赖历史帧数量
    
    与 HeatmapVLN 训练时使用的逻辑完全一致
    """
    heatmap = np.zeros((height, width), dtype=np.float32)
    
    if len(history_poses) == 0:
        return heatmap
    
    # 当前帧位姿的逆
    T_current = np.array(current_pose, dtype=np.float32)
    T_current_inv = np.linalg.inv(T_current)
    
    # 处理深度图
    depth_plane = None
    if depth_image is not None:
        if len(depth_image.shape) == 3:
            depth_plane = depth_image[:, :, 0]
        else:
            depth_plane = depth_image
    
    fx = K[0, 0]
    
    for hist_pose in history_poses:
        T_hist = np.array(hist_pose, dtype=np.float32)
        
        # 历史帧相机中心（世界坐标系）
        hist_center_world = np.array([
            T_hist[0, 3], T_hist[1, 3], T_hist[2, 3], 1.0
        ], dtype=np.float32)
        
        # 转换到当前帧相机坐标系
        p_cam = T_current_inv @ hist_center_world
        
        # 计算距离
        distance = float(np.linalg.norm(p_cam[:3]))
        if distance < 1e-4 or distance > max_visible_distance:
            continue
        
        # 投影到图像坐标 (Pinhole)
        projection = project_point_pinhole(p_cam, K, width, height)
        if projection is None:
            continue
        u, v, z_depth = projection
        
        # 遮挡检测
        if depth_plane is not None:
            depth_h, depth_w = depth_plane.shape
            u_d = u * (depth_w / width)
            v_d = v * (depth_h / height)
            u_int = int(np.clip(u_d, 0, depth_w - 1))
            v_int = int(np.clip(v_d, 0, depth_h - 1))
            observed_depth = float(depth_plane[v_int, u_int])
            
            # 深度归一化转换
            observed_depth = observed_depth * 10.0  # Habitat 默认 max_depth=10
            
            if observed_depth > 0 and observed_depth < z_depth - occlusion_margin:
                continue  # 被遮挡
        
        # 计算自适应 sigma（与训练时一致的相对比例）
        # 训练时：在 64x64 热力图上，sigma 范围 1.5-6.0（增大了 min_sigma）
        # 可视化时：需要按比例放大到原图尺寸
        hm_width = 64  # 训练时热力图宽度
        scale = width / hm_width  # 缩放因子 = 640/64 = 10
        
        object_size_3d = 0.5
        # 先在热力图尺寸下计算
        fx_hm = fx / scale  # 等效焦距
        projected_size_hm = object_size_3d * fx_hm / max(z_depth, 0.1)
        sigma_hm = max(1.5, min(projected_size_hm / 3.0, 6.0))  # min_sigma 从 0.8 增加到 1.5
        # 转换到原图尺寸
        sigma = sigma_hm * scale
        
        # 计算距离衰减的峰值
        if use_distance_decay:
            decay_factor = 1.0 / (1.0 + distance / distance_decay_ref)
            peak_value = min_peak_value + (1.0 - min_peak_value) * decay_factor
        else:
            peak_value = 1.0
        
        # 绘制高斯点
        u_int, v_int = int(u), int(v)
        radius = int(3 * sigma)
        y_min = max(0, v_int - radius)
        y_max = min(height, v_int + radius + 1)
        x_min = max(0, u_int - radius)
        x_max = min(width, u_int + radius + 1)
        
        if y_min >= y_max or x_min >= x_max:
            continue
        
        yy, xx = np.meshgrid(
            np.arange(y_min, y_max) - v_int,
            np.arange(x_min, x_max) - u_int,
            indexing='ij'
        )
        gaussian = peak_value * np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        
        if use_max_merge:
            # Max 合并：避免重叠区域饱和
            heatmap[y_min:y_max, x_min:x_max] = np.maximum(
                heatmap[y_min:y_max, x_min:x_max],
                gaussian.astype(np.float32)
            )
        else:
            # 累加模式（旧版本）
            heatmap[y_min:y_max, x_min:x_max] += gaussian.astype(np.float32)
    
    # 不再做全局归一化，保持值范围 [0, 1]
    # 热力图值有明确的物理意义：表示"这个位置的历史帧有多近/多重要"
    
    return heatmap


def create_full_clip_visualization(
    clip_path: Path,
    output_dir: Path,
    sample_interval: int = 5,
    frames_per_page: int = 8,
) -> int:
    """为clip的每一帧创建完整可视化（动态生成热力图）

    Args:
        clip_path: clip数据目录路径
        output_dir: 输出目录路径
        sample_interval: 采样间隔（每隔多少帧采一帧）
        frames_per_page: 每页显示的帧数

    Returns:
        生成的页数
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    with open(clip_path / "poses.json", "r") as f:
        poses = json.load(f)
    with open(clip_path / "meta.json", "r") as f:
        meta = json.load(f)
    with open(clip_path / "intrinsics.json", "r") as f:
        intrinsics = json.load(f)
    
    # 相机内参
    K = np.array(intrinsics["K"], dtype=np.float32)
    img_width = intrinsics["width"]
    img_height = intrinsics["height"]
    
    # 将 poses 转换为 numpy 数组
    poses_np = [np.array(p, dtype=np.float32) for p in poses]

    num_frames = len(poses)

    # 读取导航网格俯视图
    navmesh_img = cv2.imread(str(clip_path / "topdown_trajectory.jpg"))
    if navmesh_img is None:
        print(f"  警告: 无法读取俯视图 {clip_path / 'topdown_trajectory.jpg'}")
        return 0

    # 加载坐标转换信息（像素坐标已预计算）
    transform_path = clip_path / "topdown_transform.json"
    if transform_path.exists():
        with open(transform_path, "r") as f:
            transform = json.load(f)
        trajectory_pixels = transform["trajectory_pixels"]  # 每帧的像素坐标
    else:
        # 兼容旧数据：使用简单线性映射（可能不准确）
        print(f"  警告: 未找到 topdown_transform.json，使用近似坐标")
        trajectory_3d = np.load(str(clip_path / "trajectory_3d.npy"))
        x_coords = trajectory_3d[:, 0]
        z_coords = trajectory_3d[:, 2]
        padding = 5.0
        x_min, x_max = x_coords.min() - padding, x_coords.max() + padding
        z_min, z_max = z_coords.min() - padding, z_coords.max() + padding
        map_h, map_w = navmesh_img.shape[:2]
        view_range = max(x_max - x_min, z_max - z_min)
        x_center = (x_min + x_max) / 2
        z_center = (z_min + z_max) / 2
        trajectory_pixels = []
        for i in range(len(trajectory_3d)):
            x, z = trajectory_3d[i, 0], trajectory_3d[i, 2]
            px = int((x - (x_center - view_range / 2)) / view_range * map_w)
            py = int((z - (z_center - view_range / 2)) / view_range * map_h)
            trajectory_pixels.append([max(0, min(map_w - 1, px)), max(0, min(map_h - 1, py))])

    # 采样帧
    frames_to_process = list(range(0, num_frames, sample_interval))
    if frames_to_process[-1] != num_frames - 1:
        frames_to_process.append(num_frames - 1)

    vis_frames = []
    for frame_idx in frames_to_process:
        # 1. 俯视图 + 轨迹
        topdown = navmesh_img.copy()

        # 绘制过去轨迹（蓝色）- 使用预计算的像素坐标
        for i in range(min(frame_idx, len(trajectory_pixels) - 1)):
            p1 = tuple(trajectory_pixels[i])
            p2 = tuple(trajectory_pixels[i + 1])
            cv2.line(topdown, p1, p2, (255, 150, 0), 2)

        # 当前位置（红色圆点 + 朝向箭头）
        if frame_idx < len(trajectory_pixels):
            curr = tuple(trajectory_pixels[frame_idx])
            cv2.circle(topdown, curr, 8, (0, 0, 255), -1)

            # 相机朝向：使用下一帧位置作为朝向参考
            if frame_idx + 1 < len(trajectory_pixels):
                next_pt = tuple(trajectory_pixels[frame_idx + 1])
                # 计算方向向量
                dx = next_pt[0] - curr[0]
                dy = next_pt[1] - curr[1]
                length = max(1, (dx**2 + dy**2)**0.5)
                # 归一化并延长
                arrow_len = 20
                fwd_px = (int(curr[0] + dx / length * arrow_len), int(curr[1] + dy / length * arrow_len))
                cv2.arrowedLine(topdown, curr, fwd_px, (0, 0, 200), 2, tipLength=0.3)

        cv2.putText(
            topdown,
            f"Frame {frame_idx}/{num_frames-1}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )

        # 缩放俯视图
        topdown = cv2.resize(topdown, (256, 256))

        # 2. RGB
        rgb_path = clip_path / "rgb" / f"{frame_idx:06d}.jpg"
        rgb = cv2.imread(str(rgb_path))
        if rgb is None:
            print(f"  警告: 无法读取 {rgb_path}")
            continue
        rgb = cv2.resize(rgb, (320, 240))

        # 3. 动态生成热力图（历史帧相机位置）
        if frame_idx > 0 and frame_idx < len(poses_np):
            # 获取当前相机位姿
            current_pose = poses_np[frame_idx]
            # 历史帧的位姿
            history_poses = poses_np[:frame_idx]
            
            # 加载深度图用于遮挡检测
            depth_path = clip_path / "depth" / f"{frame_idx:06d}.npy"
            depth_image = None
            if depth_path.exists():
                depth_image = np.load(str(depth_path)).astype(np.float32)
            
            # 生成热力图（与训练时逻辑一致）
            hm = generate_history_heatmap(
                history_poses, current_pose, K,
                img_width, img_height,
                depth_image=depth_image,
                occlusion_margin=0.5,
                max_visible_distance=15.0
            )
        else:
            # 第一帧没有历史帧
            hm = np.zeros((img_height, img_width), dtype=np.float32)
        
        hm_color = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        hm_color = cv2.resize(hm_color, (320, 240))

        # 覆盖率标注
        cov = np.count_nonzero(hm > 0.01) / hm.size * 100  # 使用阈值避免噪声
        cv2.putText(
            hm_color,
            f"{cov:.1f}%",
            (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # 4. 叠加
        overlay = cv2.addWeighted(rgb, 0.5, hm_color, 0.5, 0)

        # 拼接：俯视图 | RGB | 热力图 | 叠加
        # 调整俯视图高度以匹配其他图像
        topdown_padded = np.ones((240, 256, 3), dtype=np.uint8) * 200
        topdown_resized = cv2.resize(topdown, (256, 256))[:240, :]
        topdown_padded[:240, :256] = topdown_resized

        row = np.hstack([topdown_padded, rgb, hm_color, overlay])
        vis_frames.append(row)

    if len(vis_frames) == 0:
        print("  警告: 没有有效帧")
        return 0

    # 分页保存
    num_pages = (len(vis_frames) + frames_per_page - 1) // frames_per_page

    for page_idx in range(num_pages):
        start = page_idx * frames_per_page
        end = min(start + frames_per_page, len(vis_frames))

        page_frames = vis_frames[start:end]
        # 填充到整数行
        while len(page_frames) < frames_per_page:
            page_frames.append(np.ones_like(page_frames[0]) * 255)

        # 垂直拼接
        page = np.vstack(page_frames)

        # 添加标题行
        title_height = 30
        title = np.ones((title_height, page.shape[1], 3), dtype=np.uint8) * 255
        labels = ["Top-Down", "RGB", "Heatmap", "Overlay"]
        x_offsets = [100, 256 + 120, 256 + 320 + 100, 256 + 640 + 100]
        for label, x in zip(labels, x_offsets):
            cv2.putText(
                title, label, (x, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
            )

        final = np.vstack([title, page])

        output_path = output_dir / f"page_{page_idx+1:02d}.jpg"
        cv2.imwrite(str(output_path), final, [cv2.IMWRITE_JPEG_QUALITY, 85])

    return num_pages


def find_all_clips(data_dir: Path) -> list:
    """查找所有clip目录"""
    clips = []
    for scene_dir in sorted(data_dir.iterdir()):
        if scene_dir.is_dir() and not scene_dir.name.startswith("."):
            # 跳过非场景目录
            if scene_dir.name in ["collection_stats.json"]:
                continue
            for clip_dir in sorted(scene_dir.iterdir()):
                if clip_dir.is_dir() and clip_dir.name.startswith("clip"):
                    clips.append(clip_dir)
    return clips


def main():
    parser = argparse.ArgumentParser(
        description="可视化巡逻数据采集的clips",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python visualize_clips.py --input /root/autodl-tmp/patrol_data --output /root/autodl-tmp/vis
    python visualize_clips.py --input /root/autodl-tmp/patrol_data --output /root/autodl-tmp/vis --interval 3 --frames-per-page 6
        """,
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入数据目录（包含场景子目录的目录）",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出目录",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="采样间隔（每隔多少帧采一帧），默认5",
    )
    parser.add_argument(
        "--frames-per-page",
        type=int,
        default=8,
        help="每页显示的帧数，默认8",
    )
    parser.add_argument(
        "--clip",
        type=str,
        default=None,
        help="只处理指定的clip（格式: scene_name/clip_xxx）",
    )
    parser.add_argument(
        "--random",
        type=int,
        default=None,
        help="随机选取指定数量的clips进行可视化",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认42）",
    )

    args = parser.parse_args()

    data_dir = Path(args.input)
    output_base = Path(args.output)

    if not data_dir.exists():
        print(f"错误: 输入目录不存在: {data_dir}")
        return

    output_base.mkdir(parents=True, exist_ok=True)

    # 查找所有clips
    if args.clip:
        # 处理指定的clip
        clip_path = data_dir / args.clip
        if not clip_path.exists():
            print(f"错误: clip目录不存在: {clip_path}")
            return
        clips = [clip_path]
    else:
        clips = find_all_clips(data_dir)
        
        # 随机选取
        if args.random and args.random < len(clips):
            random.seed(args.seed)
            clips = random.sample(clips, args.random)
            print(f"随机选取 {args.random} 个 clips (seed={args.seed})")

    if len(clips) == 0:
        print("未找到任何clip目录")
        return

    print(f"找到 {len(clips)} 个 clips")
    print(f"采样间隔: {args.interval} 帧")
    print(f"每页帧数: {args.frames_per_page}")
    print()

    total_pages = 0
    for clip_path in clips:
        clip_name = f"{clip_path.parent.name}_{clip_path.name}"
        print(f"处理 {clip_name}...")
        output_dir = output_base / clip_name
        num_pages = create_full_clip_visualization(
            clip_path,
            output_dir,
            sample_interval=args.interval,
            frames_per_page=args.frames_per_page,
        )
        if num_pages > 0:
            print(f"  生成 {num_pages} 页可视化")
            total_pages += num_pages

    print()
    print(f"完成! 共生成 {total_pages} 页可视化")
    print(f"输出目录: {output_base}")


if __name__ == "__main__":
    main()
