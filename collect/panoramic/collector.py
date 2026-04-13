#!/usr/bin/env python3
"""
R2R-CE 全景数据采集 (Panoramic Collection)

沿 R2R-CE episode 的 reference_path 导航，每步采集 front/right/back/left
四个方向的 RGB+Depth+Pose，外加可选的 front_down 观测（pitch=30°）。

存储使用 chunks 模式（分块 NPZ），大幅减少小文件数量。

渲染依赖 (GLX + Xvfb):
  habitat-sim 使用 GLX 渲染，无显示器时需要:
    Xvfb :99 -screen 0 1024x768x24 &
    DISPLAY=:99 python -m collect panoramic --gpu 0 ...

输出格式 (per clip):
  chunks/chunk_*.npz       — 分块 RGB(JPEG)+Depth+Pose
  trajectory_3d.npy        — [T, 3] float32 agent 世界坐标
  actions.npy              — [T, 2] float32 agent-local (dx, dy)
  discrete_actions.npy     — [T] int32 HabitatSimActions
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
    compute_2d_action,
)
from collect.common.io_utils import (
    submit_io_task,
    save_chunk_npz,
    drain_io_futures,
)
from collect.common.multiview import (
    DIRECTIONS,
    LOOKDOWN_PITCH_DEG,
    LOOKDOWN_DIRECTION,
    capture_multiview,
)


def parse_args():
    p = argparse.ArgumentParser(description="R2R-CE 全景数据采集")
    p.add_argument("--config", default="habitat_extensions/config/vlnce_collect.yaml")
    p.add_argument("--output", default="/workspace/r2r_panoramic_data")
    p.add_argument("--split", default="train",
                   choices=["train", "val_seen", "val_unseen", "test"])
    p.add_argument("--num-clips", type=int, default=5000)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--max-pending-io", type=int, default=512)
    p.add_argument("--jpg-quality", type=int, default=90)
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--lookdown-pitch", type=float, default=LOOKDOWN_PITCH_DEG,
                   help="Pitch angle for extra lookdown view. 0 to disable.")
    p.add_argument("--depth-directions", type=str, nargs="+", default=["front"],
                   help="Directions to save depth for. Use 'all' for every direction.")
    return p.parse_args()


def main():
    args = parse_args()

    lookdown_pitch = args.lookdown_pitch
    depth_directions = None if args.depth_directions == ["all"] else set(args.depth_directions)

    print("=" * 60)
    print("R2R-CE Panoramic Data Collection")
    print("=" * 60)
    print(f"  Output:    {args.output}")
    print(f"  Split:     {args.split}")
    print(f"  Clips:     {args.num_clips}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Lookdown:  {lookdown_pitch}°" if lookdown_pitch > 0 else "  Lookdown:  disabled")
    depth_str = "all" if depth_directions is None else str(sorted(depth_directions))
    print(f"  Depth for: {depth_str}")
    print("=" * 60)

    # ==================== 环境初始化 ====================
    config = get_extended_config(args.config)
    config.defrost()
    config.DATASET.SPLIT = args.split
    config.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
    config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = args.gpu
    config.freeze()

    env = habitat.Env(config=config)
    sim = env.sim
    follower = ShortestPathFollower(sim, goal_radius=0.2, return_one_hot=False)

    T_agent_cam = get_sensor_extrinsics(config)
    intrinsics = compute_intrinsics(config)

    dataset = env._dataset
    print(f"Dataset: {len(dataset.episodes)} episodes")

    episodes_by_scene = {}
    for i, ep in enumerate(dataset.episodes):
        scene = ep.scene_id.split("/")[-1].replace(".glb", "")
        episodes_by_scene.setdefault(scene, []).append(i)
    print(f"Scenes: {len(episodes_by_scene)}")

    scene_names = list(episodes_by_scene.keys())
    random.seed(42)
    random.shuffle(scene_names)
    all_indices = []
    for s in scene_names:
        eps = list(episodes_by_scene[s])
        random.shuffle(eps)
        all_indices.extend(eps)

    # ==================== 断点续采 ====================
    output_root = Path(args.output)
    split_dir = output_root / args.split
    split_dir.mkdir(parents=True, exist_ok=True)

    collected_ids = set()
    for mf in split_dir.rglob("meta.json"):
        try:
            eid = json.load(open(mf)).get("episode_id")
            if eid is not None:
                collected_ids.add(str(eid))
        except Exception:
            pass
    if collected_ids:
        print(f"Resuming: {len(collected_ids)} episodes already collected")

    existing_clips = list(split_dir.rglob("meta.json"))
    if existing_clips:
        max_id = 0
        for mf in existing_clips:
            try:
                max_id = max(max_id, int(mf.parent.name.split("_")[1]))
            except Exception:
                pass
        clip_id = max_id + 1
    else:
        clip_id = 1

    stats = {"successful": 0, "failed": 0, "total_frames": 0, "scenes": {}}
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers)
    io_futures: List[concurrent.futures.Future] = []

    start_time = time.time()
    ep_ptr = 0

    # ==================== 主采集循环 ====================
    while clip_id <= args.num_clips and ep_ptr < len(all_indices):
        episode_idx = all_indices[ep_ptr]
        ep_ptr += 1
        episode = dataset.episodes[episode_idx]

        if str(episode.episode_id) in collected_ids:
            continue

        scene_name = episode.scene_id.split("/")[-1].replace(".glb", "")
        print(f"\nClip {clip_id}/{args.num_clips}  scene={scene_name}  ep={episode.episode_id}")

        ok = True
        if episode.goals is None or len(episode.goals) == 0:
            ok = False
        if episode.reference_path is None or len(episode.reference_path) == 0:
            ok = False
        if not ok:
            print("  Skip: missing goals/reference_path")
            stats["failed"] += 1
            continue

        instruction_text = ""
        if episode.instruction is not None and hasattr(episode.instruction, "instruction_text"):
            instruction_text = episode.instruction.instruction_text or ""
        trajectory_id = getattr(episode, "trajectory_id", None) or "unknown"

        try:
            env._current_episode = episode
            observations = env.reset()
            sim = env.sim
            if not hasattr(env, "_last_scene") or env._last_scene != scene_name:
                follower = ShortestPathFollower(sim, goal_radius=0.2, return_one_hot=False)
                env._last_scene = scene_name

            clip_dir = split_dir / scene_name / f"clip_{clip_id:06d}"
            chunks_dir = clip_dir / "chunks"
            chunks_dir.mkdir(parents=True, exist_ok=True)

            all_capture_dirs = list(DIRECTIONS)
            if lookdown_pitch > 0:
                all_capture_dirs.append(LOOKDOWN_DIRECTION)

            save_depth_dirs = (
                all_capture_dirs if depth_directions is None
                else [d for d in all_capture_dirs if d in depth_directions]
            )

            trajectory_3d = []
            front_poses_4x4 = []
            discrete_actions = []
            frame_id = 0
            chunk_id = 0
            chunk_fids = []
            chunk_rgb = {d: [] for d in all_capture_dirs}
            chunk_depth = {d: [] for d in save_depth_dirs}
            chunk_pose = {d: [] for d in all_capture_dirs}

            def flush_chunk():
                nonlocal chunk_id
                if not chunk_fids:
                    return
                fids = np.array(chunk_fids, dtype=np.int32)
                r = {d: np.stack(chunk_rgb[d]).astype(np.uint8, copy=False) for d in all_capture_dirs}
                dp = {d: np.stack(chunk_depth[d]).astype(np.float16, copy=False) for d in save_depth_dirs}
                ps = {d: np.stack(chunk_pose[d]).astype(np.float32, copy=False) for d in all_capture_dirs}
                path = str(chunks_dir / f"chunk_{chunk_id:05d}.npz")
                submit_io_task(executor, io_futures, args.max_pending_io,
                               save_chunk_npz, path, fids, r, dp, ps, args.jpg_quality)
                chunk_id += 1
                chunk_fids.clear()
                for d in all_capture_dirs:
                    chunk_rgb[d].clear()
                    chunk_pose[d].clear()
                for d in save_depth_dirs:
                    chunk_depth[d].clear()

            def record_frame():
                nonlocal frame_id
                mv = capture_multiview(sim, T_agent_cam, lookdown_pitch_deg=lookdown_pitch)
                agent_state = sim.get_agent_state()
                trajectory_3d.append(agent_state.position.copy())
                front_poses_4x4.append(mv["front"]["pose"].tolist())

                for d in all_capture_dirs:
                    rgb = mv[d]["rgb"]
                    if rgb.shape[2] == 4:
                        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
                    elif rgb.shape[2] == 3:
                        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    chunk_rgb[d].append(rgb)
                    chunk_pose[d].append(mv[d]["pose"].astype(np.float32))
                    if d in chunk_depth:
                        depth = mv[d]["depth"]
                        if depth.dtype != np.float16:
                            depth = depth.astype(np.float16)
                        chunk_depth[d].append(depth)

                chunk_fids.append(frame_id)
                frame_id += 1
                if len(chunk_fids) >= args.chunk_size:
                    flush_chunk()

            record_frame()

            reference_path = episode.reference_path
            target_idx = 0
            steps_taken = 0

            while target_idx < len(reference_path) and steps_taken < args.max_steps:
                goal_point = reference_path[target_idx]
                action = follower.get_next_action(goal_point)
                if action is None or action == HabitatSimActions.STOP:
                    target_idx += 1
                    continue
                discrete_actions.append(int(action))
                observations = env.step(action)
                steps_taken += 1
                record_frame()

            discrete_actions.append(0)  # STOP

            MIN_FRAMES = 10
            if frame_id < MIN_FRAMES:
                print(f"  Skip: too few frames ({frame_id})")
                shutil.rmtree(clip_dir, ignore_errors=True)
                stats["failed"] += 1
                continue

            flush_chunk()

            # 计算 2D 连续动作
            actions_2d = []
            for i in range(len(front_poses_4x4)):
                if i == len(front_poses_4x4) - 1:
                    actions_2d.append(np.zeros(2, dtype=np.float32))
                else:
                    actions_2d.append(compute_2d_action(front_poses_4x4[i], front_poses_4x4[i + 1]))
            actions_2d = np.array(actions_2d, dtype=np.float32)
            discrete_arr = np.array(discrete_actions, dtype=np.int32)

            # 保存
            traj_3d = np.array(trajectory_3d, dtype=np.float32)
            np.save(str(clip_dir / "trajectory_3d.npy"), traj_3d)
            np.save(str(clip_dir / "actions.npy"), actions_2d)
            np.save(str(clip_dir / "discrete_actions.npy"), discrete_arr)

            with open(clip_dir / "intrinsics.json", "w") as f:
                json.dump(intrinsics, f, separators=(",", ":"))

            meta = {
                "scene_id": scene_name,
                "episode_id": episode.episode_id,
                "trajectory_id": trajectory_id,
                "instruction": instruction_text,
                "num_frames": frame_id,
                "reference_path": [list(p) for p in reference_path],
                "storage_format": "chunks",
                "lookdown_pitch_deg": lookdown_pitch if lookdown_pitch > 0 else None,
                "data_format": {
                    "chunks": "NPZ: frame_ids + rgb/pose per direction, depth only for depth_directions",
                    "trajectory_3d": "NPY float32 [T,3]",
                    "actions": "NPY float32 [T,2] agent-local (dx,dy)",
                    "discrete_actions": "NPY int32 [T] (0=STOP,1=FWD,2=LEFT,3=RIGHT)",
                    "directions": all_capture_dirs,
                    "depth_directions": save_depth_dirs,
                },
            }
            with open(clip_dir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)

            collected_ids.add(str(episode.episode_id))
            stats["successful"] += 1
            stats["total_frames"] += frame_id
            stats["scenes"][scene_name] = stats["scenes"].get(scene_name, 0) + 1
            print(f"  Done: {frame_id} frames, {len(actions_2d)} actions")
            clip_id += 1

        except Exception as e:
            print(f"  Failed: {e}")
            stats["failed"] += 1
            if "clip_dir" in locals() and Path(clip_dir).exists():
                shutil.rmtree(clip_dir, ignore_errors=True)

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
    print(f"  Time:    {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"  Scenes:  {len(stats['scenes'])}")

    with open(output_root / "collection_stats.json", "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
