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
import itertools
import json
import random
import shutil
import time
import concurrent.futures
from collections import defaultdict
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
    p.add_argument("--output", default="data/collected/heatmap_train_data")
    p.add_argument("--num-clips", type=int, default=1000)
    p.add_argument("--num-waypoints", type=int, default=4)
    p.add_argument("--min-waypoint-dist", type=float, default=3.0)
    p.add_argument("--max-waypoint-dist", type=float, default=10.0)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--min-frames", type=int, default=30)
    p.add_argument(
        "--min-keyframe-translation",
        type=float,
        default=0.20,
        help="Do not render another panorama until the agent moved this many metres",
    )
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--max-pending-io", type=int, default=512)
    p.add_argument("--jpg-quality", type=int, default=90)
    p.add_argument("--storage-format", default="chunks", choices=["frames", "chunks"])
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--clip-id-start",
        type=int,
        default=0,
        help="Explicit first global clip id for a parallel worker (0 = serial resume mode)",
    )
    p.add_argument(
        "--stats-file",
        default="collection_stats.json",
        help="Per-worker stats filename written under --output",
    )
    p.add_argument("--worker-id", type=int, default=0)
    p.add_argument("--episode-shard-id", type=int, default=0)
    p.add_argument("--episode-shard-count", type=int, default=1)
    p.add_argument(
        "--episode-offset",
        type=int,
        default=0,
        help="Rotate the worker episode stream by this many episodes",
    )
    p.add_argument(
        "--scene-block-size",
        type=int,
        default=8,
        help="Collect this many episodes per scene before loading another scene",
    )
    p.add_argument("--return-to-start", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()

    if args.num_clips <= 0:
        raise ValueError("--num-clips must be positive")
    if args.clip_id_start < 0:
        raise ValueError("--clip-id-start must be non-negative")
    if Path(args.stats_file).name != args.stats_file:
        raise ValueError("--stats-file must be a filename, not a path")
    if args.num_waypoints < 2:
        raise ValueError("--num-waypoints must be at least 2")
    if not (0.0 < args.min_waypoint_dist < args.max_waypoint_dist):
        raise ValueError("Require 0 < min-waypoint-dist < max-waypoint-dist")
    if args.min_frames < 1 or args.min_frames > args.max_steps:
        raise ValueError("Require 1 <= min-frames <= max-steps")
    if args.min_keyframe_translation < 0:
        raise ValueError("--min-keyframe-translation must be non-negative")
    if not (0 <= args.episode_shard_id < args.episode_shard_count):
        raise ValueError("Require 0 <= episode-shard-id < episode-shard-count")
    if args.episode_offset < 0:
        raise ValueError("--episode-offset must be non-negative")
    if args.scene_block_size < 1:
        raise ValueError("--scene-block-size must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("Multi-point Patrolling Data Collection")
    print("=" * 60)
    print(f"  Output:       {args.output}")
    print(f"  Clips:        {args.num_clips}")
    print(f"  Waypoints:    {args.num_waypoints}")
    print(f"  Distance:     {args.min_waypoint_dist}m ~ {args.max_waypoint_dist}m")
    print(f"  Max steps:    {args.max_steps}")
    print(f"  Min frames:   {args.min_frames}")
    print(f"  Keyframe m:   {args.min_keyframe_translation}")
    print(f"  Format:       {args.storage_format}")
    print(f"  Return home:  {args.return_to_start}")
    print(f"  Seed:         {args.seed}")
    print(f"  Worker:       {args.worker_id}")
    print(
        f"  Episode:      shard={args.episode_shard_id}/{args.episode_shard_count}"
    )
    print(f"  Episode off:  {args.episode_offset}")
    print(f"  Scene block:  {args.scene_block_size} clips")
    print("=" * 60)

    # Check resume state before constructing Habitat.  Native simulator startup
    # is the dominant cost for a retry whose id range is already complete.
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    existing_clips = list(output_root.glob("*/clip_*/meta.json"))
    existing_ids = set()
    for meta_path in existing_clips:
        try:
            existing_ids.add(int(meta_path.parent.name.split("_")[1]))
        except (ValueError, IndexError):
            pass
    if args.clip_id_start > 0:
        requested_ids = set(
            range(args.clip_id_start, args.clip_id_start + args.num_clips)
        )
        if requested_ids.issubset(existing_ids):
            print(
                f"Requested parallel range is already complete: "
                f"clip_{args.clip_id_start:06d}.."
                f"clip_{args.clip_id_start + args.num_clips - 1:06d}"
            )
            return
    elif set(range(1, args.num_clips + 1)).issubset(existing_ids):
        print(f"Serial target already complete: {len(existing_ids)} clips")
        return

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
    if hasattr(env, "seed"):
        env.seed(args.seed)
    sim = env.sim

    intrinsics = compute_intrinsics(config)
    T_agent_cam = get_sensor_extrinsics(config)
    print(f"Camera: {intrinsics['width']}x{intrinsics['height']}, HFOV={intrinsics['hfov']}°")

    dataset = env._dataset
    print(f"Dataset: {len(dataset.episodes)} episodes")

    dataset_scenes = {
        ep.scene_id.split("/")[-1].replace(".glb", "")
        for ep in dataset.episodes
    }
    print(f"Scenes: {len(dataset_scenes)}")

    # Assign disjoint scenes to workers, then visit episodes in small same-scene
    # blocks.  Habitat scene reconfiguration is expensive under CPU/GLX; this
    # keeps start-position diversity without paying that cost for every clip.
    episodes_by_scene = defaultdict(list)
    for episode in dataset.episodes:
        episodes_by_scene[episode.scene_id].append(episode)
    all_scene_ids = sorted(episodes_by_scene)
    worker_scene_ids = all_scene_ids[
        args.episode_shard_id :: args.episode_shard_count
    ]
    worker_episodes = []
    max_scene_episodes = max(
        (len(episodes_by_scene[scene_id]) for scene_id in worker_scene_ids),
        default=0,
    )
    for block_start in range(0, max_scene_episodes, args.scene_block_size):
        for scene_id in worker_scene_ids:
            worker_episodes.extend(
                episodes_by_scene[scene_id][
                    block_start : block_start + args.scene_block_size
                ]
            )
    if not worker_episodes:
        raise ValueError(
            f"Empty scene shard {args.episode_shard_id}/{args.episode_shard_count}"
        )
    effective_episode_offset = args.episode_offset % len(worker_episodes)
    if effective_episode_offset:
        worker_episodes = (
            worker_episodes[effective_episode_offset:]
            + worker_episodes[:effective_episode_offset]
        )
    env.episode_iterator = itertools.cycle(worker_episodes)
    worker_scene_count = len(worker_scene_ids)
    print(
        f"Scene shard: episodes={len(worker_episodes)} scenes={worker_scene_count} "
        f"block_size={args.scene_block_size} offset={effective_episode_offset}"
    )

    # ==================== 断点续采 ====================
    if args.clip_id_start > 0:
        clip_id = args.clip_id_start
        clip_id_end = args.clip_id_start + args.num_clips - 1
        print(
            f"Parallel worker range: clip_{clip_id:06d}..clip_{clip_id_end:06d} "
            f"({len(existing_ids)} ids already present globally)"
        )
    elif existing_ids:
        clip_id = max(existing_ids) + 1
        clip_id_end = args.num_clips
        print(f"Resuming from clip_{clip_id:06d} ({len(existing_clips)} existing)")
    else:
        clip_id = 1
        clip_id_end = args.num_clips

    while clip_id <= clip_id_end and clip_id in existing_ids:
        print(f"Skipping existing clip id {clip_id}")
        clip_id += 1

    stats = {
        "target_total_clips": args.num_clips,
        "clip_id_start": args.clip_id_start,
        "clip_id_end": clip_id_end,
        "existing_clips": len(existing_clips),
        "successful": 0,
        "failed": 0,
        "total_frames": 0,
        "scenes": {},
        "seed": args.seed,
        "worker_id": args.worker_id,
        "episode_shard_id": args.episode_shard_id,
        "episode_shard_count": args.episode_shard_count,
        "episode_offset": effective_episode_offset,
        "scene_block_size": args.scene_block_size,
        "worker_scene_count": worker_scene_count,
    }
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers)
    io_futures: List[concurrent.futures.Future] = []

    start_time = time.time()

    # ==================== 主采集循环 ====================
    while clip_id <= clip_id_end:
        clip_dir = None
        try:
            # Habitat owns the episode iterator.  Setting _current_episode and
            # then calling reset() is unsafe: reset may silently replace it.
            # Bind every output path and metadata field to the actual episode
            # returned by reset instead.
            observations = env.reset()
            episode = env.current_episode
            scene_name = episode.scene_id.split("/")[-1].replace(".glb", "")
            print(
                f"\nWorker {args.worker_id} clip {clip_id}/{clip_id_end} - "
                f"Scene: {scene_name} - Episode: {episode.episode_id}"
            )
            sim = env.sim
            # habitat.Env.seed() does not seed PathFinder's native RNG in
            # Habitat-Sim 0.1.7.  Seed it explicitly per global clip id so a
            # resumed/extended run cannot silently reproduce old patrols.
            patrol_seed = int((args.seed * 1_000_003 + clip_id) % (2**32))
            if not hasattr(sim.pathfinder, "seed"):
                raise RuntimeError("Habitat PathFinder does not expose seed()")
            sim.pathfinder.seed(patrol_seed)
            random.seed(patrol_seed)
            np.random.seed(patrol_seed)
            print(f"  Patrol seed: {patrol_seed}")
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
            if clip_dir.exists():
                raise FileExistsError(f"Refusing to overwrite existing clip: {clip_dir}")
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
            last_saved_position = None

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

            def record_multiview_frame(current_pos):
                nonlocal frame_id
                multiview = capture_multiview(sim, T_agent_cam)

                frame_poses = {}
                for direction in DIRECTIONS:
                    obs_d = multiview[direction]
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
                        rgb_path = rgb_dir / direction / f"{frame_id:06d}.jpg"
                        submit_io_task(
                            executor,
                            io_futures,
                            args.max_pending_io,
                            cv2.imwrite,
                            str(rgb_path),
                            rgb,
                            [cv2.IMWRITE_JPEG_QUALITY, args.jpg_quality],
                        )
                        depth_path = depth_dir / direction / f"{frame_id:06d}.npy"
                        submit_io_task(
                            executor,
                            io_futures,
                            args.max_pending_io,
                            np.save,
                            str(depth_path),
                            depth,
                        )
                        frame_poses[direction] = pose.tolist()
                    else:
                        chunk_rgb[direction].append(rgb)
                        chunk_depth[direction].append(depth)
                        chunk_pose[direction].append(pose)
                        frame_poses[direction] = pose.tolist()

                if args.storage_format == "frames":
                    poses.append(frame_poses)
                else:
                    chunk_fids.append(frame_id)
                trajectory_3d.append(current_pos.copy())
                frame_id += 1

                if args.storage_format == "chunks" and len(chunk_fids) >= args.chunk_size:
                    flush_chunk()

            # 遍历巡逻路径。动作步数使用整条路线的全局预算；如果任一目标
            # 未到达，就丢弃该 clip，避免把未闭环轨迹标成 return_to_start。
            route_completed = True
            total_action_steps = 0
            for target_idx, target_point in enumerate(patrol_path[1:], 1):
                target_reached = False

                while total_action_steps < args.max_steps:
                    agent_state = sim.get_agent_state()
                    current_pos = agent_state.position.copy()

                    if np.linalg.norm(current_pos - target_point) < 0.5:
                        if (
                            last_saved_position is None
                            or np.linalg.norm(current_pos - last_saved_position)
                            >= args.min_keyframe_translation
                        ):
                            record_multiview_frame(current_pos)
                            last_saved_position = current_pos.copy()
                        target_reached = True
                        break

                    if (
                        last_saved_position is None
                        or np.linalg.norm(current_pos - last_saved_position)
                        >= args.min_keyframe_translation
                    ):
                        record_multiview_frame(current_pos)
                        last_saved_position = current_pos.copy()

                    action = follower.get_next_action(target_point)
                    if action is None or action == HabitatSimActions.STOP:
                        break
                    observations = env.step(action)
                    total_action_steps += 1

                if not target_reached:
                    route_completed = False
                    print(
                        f"  Skip: target {target_idx}/{len(patrol_path) - 1} "
                        f"not reached within {total_action_steps}/{args.max_steps} actions"
                    )
                    break

            final_return_distance = float(
                np.linalg.norm(sim.get_agent_state().position - start_pos)
            )
            if args.return_to_start and final_return_distance > 0.75:
                route_completed = False
                print(
                    f"  Skip: patrol did not close "
                    f"(return distance {final_return_distance:.3f}m)"
                )

            if not route_completed:
                # Full chunks may already be in flight.  Wait before deleting
                # the rejected clip so writers cannot recreate partial files.
                drain_io_futures(io_futures)
                shutil.rmtree(clip_dir, ignore_errors=True)
                stats["failed"] += 1
                continue

            if frame_id < args.min_frames:
                print(f"  Skip: too few frames ({frame_id})")
                drain_io_futures(io_futures)
                shutil.rmtree(clip_dir, ignore_errors=True)
                stats["failed"] += 1
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

            # A clip is not valid until every asynchronous RGB/depth/pose
            # write has completed successfully.  This prevents meta.json from
            # advertising missing or partially written chunks.
            drain_io_futures(io_futures)

            with open(clip_dir / "intrinsics.json", "w") as f:
                json.dump(intrinsics, f, separators=(",", ":"))

            meta = {
                "scene_id": scene_name,
                "episode_id": episode.episode_id,
                "num_frames": frame_id,
                "num_waypoints": len(waypoints),
                "waypoints": [wp.tolist() for wp in waypoints],
                "return_to_start": args.return_to_start,
                "route_completed": route_completed,
                "return_distance_m": final_return_distance,
                "action_steps": total_action_steps,
                "storage_format": args.storage_format,
                "seed": args.seed,
                "patrol_seed": patrol_seed,
                "worker_id": args.worker_id,
                "episode_shard_id": args.episode_shard_id,
                "episode_shard_count": args.episode_shard_count,
                "episode_offset": effective_episode_offset,
                "scene_block_size": args.scene_block_size,
                "min_keyframe_translation_m": args.min_keyframe_translation,
                "data_format": {
                    "rgb": "4-direction RGB (frames: per-dir JPG folders, chunks: in NPZ)",
                    "depth": "4-direction depth (frames: per-dir NPY folders, chunks: in NPZ)",
                    "depth_unit": "meters",
                    "pose": "camera-to-world float32 [4,4] per direction",
                    "trajectory_3d": "NPY float32 [T,3] world positions",
                    "topdown_trajectory": "JPG bird's-eye trajectory map",
                    "directions": DIRECTIONS,
                    "camera_height_m": float(config.SIMULATOR.RGB_SENSOR.POSITION[1]),
                    "hfov_deg": float(intrinsics["hfov"]),
                },
            }
            with open(clip_dir / "meta.json", "w") as f:
                json.dump(meta, f, separators=(",", ":"))

            stats["successful"] += 1
            stats["total_frames"] += frame_id
            stats["scenes"][scene_name] = stats["scenes"].get(scene_name, 0) + 1
            print(f"  Done: {frame_id} frames")
            clip_id += 1
            while clip_id <= clip_id_end and clip_id in existing_ids:
                print(f"Skipping existing clip id {clip_id}")
                clip_id += 1

        except Exception as e:
            print(f"  Failed: {e}")
            stats["failed"] += 1
            try:
                drain_io_futures(io_futures)
            except Exception as io_error:
                print(f"  Pending I/O also failed: {io_error}")
            if clip_dir is not None and clip_dir.exists():
                shutil.rmtree(clip_dir, ignore_errors=True)
            continue

    # ==================== 收尾 ====================
    drain_io_futures(io_futures)
    executor.shutdown(wait=True)
    env.close()

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("Collection complete!")
    print("=" * 60)
    print(f"  New success: {stats['successful']}")
    print(f"  Failed:  {stats['failed']}")
    print(f"  Frames:  {stats['total_frames']}")
    print(f"  Time:    {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"  Scenes:  {len(stats['scenes'])}")

    stats_path = output_root / args.stats_file
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved to {stats_path}")


if __name__ == "__main__":
    main()
