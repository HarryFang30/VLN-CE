#!/usr/bin/env python3
"""
多点往返巡逻数据采集 (Multi-point Patrolling)

在场景中随机采样可导航点，规划往返闭环路径，每步采集四方向 RGB+Depth+Pose。
热力图在训练时动态生成，不预先保存以节省磁盘空间。

输出格式 (per clip, chunks 模式):
  chunks/chunk_*.npz       — 分块 4 方向 RGB(JPEG)+Depth+Pose
  trajectory_3d.npy        — [T, 3] float32 agent 世界坐标
  topdown_trajectory.jpg   — 上帝视角轨迹图
  topdown_transform.json   — 坐标映射信息
  intrinsics.json / meta.json
"""
import argparse
import json
import random
import shutil
import time
import concurrent.futures
from pathlib import Path
from typing import List

import cv2
import numpy as np

import habitat
import habitat_extensions  # noqa: F401
from habitat_extensions.config.default import get_extended_config
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.sims.habitat_simulator.actions import HabitatSimActions

from collect.common.geometry import (
    get_sensor_extrinsics,
    compute_intrinsics,
)
from collect.common.io_utils import (
    submit_io_task,
    save_chunk_npz,
    drain_io_futures,
)
from collect.common.multiview import DIRECTIONS, capture_multiview
from collect.heatmap.navigation import sample_navigable_points, plan_patrol_path
from collect.heatmap.visualization import generate_topdown_trajectory_map


def parse_args():
    p = argparse.ArgumentParser(description="多点往返巡逻数据采集")
    p.add_argument("--config", default="habitat_extensions/config/vlnce_collect.yaml")
    p.add_argument("--output", default="/root/autodl-tmp/heatmap_train_data")
    p.add_argument("--num-clips", type=int, default=1000)
    p.add_argument("--num-waypoints", type=int, default=4)
    p.add_argument("--min-waypoint-dist", type=float, default=3.0)
    p.add_argument("--max-waypoint-dist", type=float, default=15.0)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--max-pending-io", type=int, default=512)
    p.add_argument("--jpg-quality", type=int, default=90)
    p.add_argument("--storage-format", default="chunks", choices=["frames", "chunks"])
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--return-to-start", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Multi-point Patrolling Data Collection")
    print("=" * 60)
    print(f"  Output:       {args.output}")
    print(f"  Clips:        {args.num_clips}")
    print(f"  Waypoints:    {args.num_waypoints}")
    print(f"  Distance:     {args.min_waypoint_dist}m ~ {args.max_waypoint_dist}m")
    print(f"  Max steps:    {args.max_steps}")
    print(f"  Format:       {args.storage_format}")
    print(f"  Return home:  {args.return_to_start}")
    print("=" * 60)

    # ==================== 环境初始化 ====================
    config = get_extended_config(args.config)
    config.defrost()
    config.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
    config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = args.gpu
    config.freeze()

    subtype = getattr(config.SIMULATOR.RGB_SENSOR, "SENSOR_SUBTYPE", "PINHOLE")
    if "EQUIRECT" in str(subtype).upper():
        raise ValueError("This collector requires PINHOLE projection mode")

    env = habitat.Env(config=config)
    sim = env.sim

    intrinsics = compute_intrinsics(config)
    K = np.array(intrinsics["K"], dtype=np.float32)
    T_agent_cam = get_sensor_extrinsics(config)
    print(f"Camera: {intrinsics['width']}x{intrinsics['height']}, HFOV={intrinsics['hfov']}°")

    dataset = env._dataset
    print(f"Dataset: {len(dataset.episodes)} episodes")

    scenes = {}
    for i, ep in enumerate(dataset.episodes):
        scene_name = ep.scene_id.split("/")[-1].replace(".glb", "")
        scenes.setdefault(scene_name, []).append(i)
    print(f"Scenes: {len(scenes)}")

    # ==================== 断点续采 ====================
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    existing_clips = list(output_root.glob("*/clip_*/meta.json"))
    if existing_clips:
        max_id = 0
        for mf in existing_clips:
            try:
                max_id = max(max_id, int(mf.parent.name.split("_")[1]))
            except (ValueError, IndexError):
                pass
        clip_id = max_id + 1
        print(f"Resuming from clip_{clip_id:06d} ({len(existing_clips)} existing)")
    else:
        clip_id = 1

    stats = {"successful": 0, "failed": 0, "total_frames": 0, "scenes": {}}
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers)
    io_futures: List[concurrent.futures.Future] = []

    start_time = time.time()
    scene_list = list(scenes.keys())
    random.shuffle(scene_list)
    scene_idx = 0

    # ==================== 主采集循环 ====================
    while clip_id <= args.num_clips:
        if scene_idx >= len(scene_list):
            scene_idx = 0
            random.shuffle(scene_list)

        scene_name = scene_list[scene_idx]
        episode_idx = random.choice(scenes[scene_name])
        episode = dataset.episodes[episode_idx]
        scene_idx += 1

        print(f"\nClip {clip_id}/{args.num_clips} - Scene: {scene_name}")

        try:
            env._current_episode = episode
            observations = env.reset()
            sim = env.sim
            follower = ShortestPathFollower(sim, goal_radius=0.5, return_one_hot=False)

            start_pos = sim.get_agent_state().position.copy()

            waypoints = sample_navigable_points(
                sim, args.num_waypoints,
                start_position=start_pos,
                min_distance=args.min_waypoint_dist,
                max_distance=args.max_waypoint_dist,
                require_reachable=True,
            )
            if len(waypoints) < 2:
                print("  Skip: not enough reachable waypoints")
                stats["failed"] += 1
                continue

            waypoints.insert(0, start_pos)
            patrol_path = plan_patrol_path(waypoints, return_to_start=args.return_to_start)
            print(f"  Patrol path: {len(patrol_path)} points")

            # 创建输出目录
            clip_dir = output_root / scene_name / f"clip_{clip_id:06d}"
            clip_dir.mkdir(parents=True, exist_ok=True)

            if args.storage_format == "frames":
                rgb_dir = clip_dir / "rgb"
                depth_dir = clip_dir / "depth"
                for d in DIRECTIONS:
                    (rgb_dir / d).mkdir(parents=True, exist_ok=True)
                    (depth_dir / d).mkdir(parents=True, exist_ok=True)
            else:
                chunks_dir = clip_dir / "chunks"
                chunks_dir.mkdir(parents=True, exist_ok=True)

            # 数据缓冲区
            poses = []
            trajectory_3d = []
            frame_id = 0
            chunk_id = 0
            chunk_fids = []
            chunk_rgb = {d: [] for d in DIRECTIONS}
            chunk_depth = {d: [] for d in DIRECTIONS}
            chunk_pose = {d: [] for d in DIRECTIONS}

            def flush_chunk():
                nonlocal chunk_id
                if not chunk_fids:
                    return
                fids = np.array(chunk_fids, dtype=np.int32)
                r = {d: np.stack(chunk_rgb[d]).astype(np.uint8, copy=False) for d in DIRECTIONS}
                dp = {d: np.stack(chunk_depth[d]).astype(np.float16, copy=False) for d in DIRECTIONS}
                ps = {d: np.stack(chunk_pose[d]).astype(np.float32, copy=False) for d in DIRECTIONS}
                chunk_path = chunks_dir / f"chunk_{chunk_id:05d}.npz"
                submit_io_task(executor, io_futures, args.max_pending_io,
                               save_chunk_npz, str(chunk_path), fids, r, dp, ps, args.jpg_quality)
                chunk_id += 1
                chunk_fids.clear()
                for d in DIRECTIONS:
                    chunk_rgb[d].clear()
                    chunk_depth[d].clear()
                    chunk_pose[d].clear()

            # 遍历巡逻路径
            for target_idx, target_point in enumerate(patrol_path[1:], 1):
                steps_to_target = 0
                max_steps_per_target = args.max_steps // len(patrol_path)

                while steps_to_target < max_steps_per_target:
                    agent_state = sim.get_agent_state()
                    current_pos = agent_state.position.copy()

                    if np.linalg.norm(current_pos - target_point) < 0.5:
                        break

                    multiview = capture_multiview(sim, T_agent_cam)

                    frame_poses = {}
                    for d in DIRECTIONS:
                        obs_d = multiview[d]
                        rgb = obs_d["rgb"]
                        if rgb.shape[2] == 4:
                            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
                        elif rgb.shape[2] == 3:
                            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                        depth = obs_d["depth"]
                        if depth.dtype != np.float16:
                            depth = depth.astype(np.float16)
                        pose = obs_d["pose"].astype(np.float32)

                        if args.storage_format == "frames":
                            rgb_path = rgb_dir / d / f"{frame_id:06d}.jpg"
                            submit_io_task(executor, io_futures, args.max_pending_io,
                                           cv2.imwrite, str(rgb_path), rgb,
                                           [cv2.IMWRITE_JPEG_QUALITY, args.jpg_quality])
                            depth_path = depth_dir / d / f"{frame_id:06d}.npy"
                            submit_io_task(executor, io_futures, args.max_pending_io,
                                           np.save, str(depth_path), depth)
                            frame_poses[d] = pose.tolist()
                        else:
                            chunk_rgb[d].append(rgb)
                            chunk_depth[d].append(depth)
                            chunk_pose[d].append(pose)
                            frame_poses[d] = pose.tolist()

                    if args.storage_format == "frames":
                        poses.append(frame_poses)

                    trajectory_3d.append(current_pos.copy())

                    if args.storage_format == "chunks":
                        chunk_fids.append(frame_id)

                    frame_id += 1

                    if args.storage_format == "chunks" and len(chunk_fids) >= args.chunk_size:
                        flush_chunk()

                    action = follower.get_next_action(target_point)
                    if action is None or action == HabitatSimActions.STOP:
                        break
                    observations = env.step(action)
                    steps_to_target += 1

                if frame_id >= args.max_steps:
                    print("  Reached max steps")
                    break

            MIN_FRAMES = 20
            if frame_id < MIN_FRAMES:
                print(f"  Skip: too few frames ({frame_id})")
                shutil.rmtree(clip_dir, ignore_errors=True)
                continue

            # 保存尾部 chunk
            if args.storage_format == "chunks" and chunk_fids:
                flush_chunk()

            if args.storage_format == "frames":
                with open(clip_dir / "poses.json", "w") as f:
                    json.dump(poses, f, separators=(",", ":"))

            traj_arr = np.array(trajectory_3d, dtype=np.float32)
            np.save(clip_dir / "trajectory_3d.npy", traj_arr)

            # 上帝视角轨迹图
            try:
                topdown_map, topdown_tf = generate_topdown_trajectory_map(
                    sim, traj_arr, waypoints, output_size=512, padding_meters=5.0,
                )
                submit_io_task(executor, io_futures, args.max_pending_io,
                               cv2.imwrite, str(clip_dir / "topdown_trajectory.jpg"),
                               topdown_map, [cv2.IMWRITE_JPEG_QUALITY, 90])
                with open(clip_dir / "topdown_transform.json", "w") as f:
                    json.dump(topdown_tf, f, separators=(",", ":"))
            except Exception as e:
                print(f"  Topdown map failed: {e}")

            with open(clip_dir / "intrinsics.json", "w") as f:
                json.dump(intrinsics, f, separators=(",", ":"))

            meta = {
                "scene_id": scene_name,
                "episode_id": episode.episode_id,
                "num_frames": frame_id,
                "num_waypoints": len(waypoints),
                "waypoints": [wp.tolist() for wp in waypoints],
                "return_to_start": args.return_to_start,
                "storage_format": args.storage_format,
                "data_format": {
                    "rgb": "4-direction RGB (frames: per-dir JPG folders, chunks: in NPZ)",
                    "depth": "4-direction depth (frames: per-dir NPY folders, chunks: in NPZ)",
                    "trajectory_3d": "NPY float32 [T,3] world positions",
                    "topdown_trajectory": "JPG bird's-eye trajectory map",
                    "directions": DIRECTIONS,
                },
            }
            with open(clip_dir / "meta.json", "w") as f:
                json.dump(meta, f, separators=(",", ":"))

            stats["successful"] += 1
            stats["total_frames"] += frame_id
            stats["scenes"][scene_name] = stats["scenes"].get(scene_name, 0) + 1
            print(f"  Done: {frame_id} frames")
            clip_id += 1

        except Exception as e:
            print(f"  Failed: {e}")
            stats["failed"] += 1
            if "clip_dir" in locals() and clip_dir.exists():
                shutil.rmtree(clip_dir, ignore_errors=True)
            continue

    # ==================== 收尾 ====================
    drain_io_futures(io_futures)
    env.close()

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("Collection complete!")
    print("=" * 60)
    print(f"  Success: {stats['successful']}/{args.num_clips}")
    print(f"  Failed:  {stats['failed']}")
    print(f"  Frames:  {stats['total_frames']}")
    print(f"  Time:    {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"  Scenes:  {len(stats['scenes'])}")

    with open(output_root / "collection_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved to {output_root / 'collection_stats.json'}")


if __name__ == "__main__":
    main()
