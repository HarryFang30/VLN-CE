#!/usr/bin/env python3
"""
可视化巡逻数据采集的clips

为每个clip的每一帧创建完整可视化，包含：
- Top-Down: 导航网格 + 过去轨迹 + 当前位置箭头
- RGB: 当前第一视角图像
- Heatmap: 热力图（显示覆盖率百分比）
- Overlay: RGB + 热力图叠加

用法:
    python visualize_clips.py --input /path/to/patrol_data --output /path/to/output
    python visualize_clips.py --input /path/to/patrol_data --output /path/to/output --interval 3
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def create_full_clip_visualization(
    clip_path: Path,
    output_dir: Path,
    sample_interval: int = 5,
    frames_per_page: int = 8,
) -> int:
    """为clip的每一帧创建完整可视化

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

        # 3. 热力图
        hm_path = clip_path / "heatmaps" / f"{frame_idx:06d}.npy"
        hm = np.load(str(hm_path))
        hm_color = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        hm_color = cv2.resize(hm_color, (320, 240))

        # 覆盖率标注
        cov = np.count_nonzero(hm) / hm.size * 100
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
