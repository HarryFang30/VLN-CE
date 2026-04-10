#!/usr/bin/env python3
"""
R2R-CE 全景数据采集脚本 (Panoramic R2R-CE Collection)

合并自 collect.py (R2R-CE 导航) 和 collect_heatmap.py (全景采集 + chunks 存储),
用于同时训练热力图 (System 2) 和轨迹预测 (System 1).

采集策略:
  使用 ShortestPathFollower 沿 R2R-CE episode 的 reference_path 导航,
  每一步采集 front/right/back/left 四个方向的 RGB + Depth + Pose.

输出格式 (per clip, chunks 模式):
  chunks/chunk_*.npz
    - frame_ids: [N] int32
    - rgb_{front,right,back,left}: [N] object (JPEG bytes)
    - depth_{front,right,back,left}: [N, H, W] float16
    - pose_{front,right,back,left}: [N, 4, 4] float32
  trajectory_3d.npy     — [T, 3] float32, agent 3D 世界坐标
  actions.npy           — [T, 2] float32, agent-local (dx, dy)
  discrete_actions.npy  — [T] int32, HabitatSimActions
  intrinsics.json       — Pinhole 内参
  meta.json             — episode / instruction / 动作统计 等元数据
  topdown_trajectory.jpg + topdown_transform.json (可选)
"""
import habitat
import cv2
import numpy as np
import json
import math
import time
import random
import shutil
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import concurrent.futures

try:
    import quaternion as _quaternion
except ImportError:
    _quaternion = None

import habitat_extensions
from habitat_extensions.config.default import get_extended_config
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.sims.habitat_simulator.actions import HabitatSimActions

# ==================== 多方向采集配置 ====================

DIRECTIONS = ["front", "right", "back", "left"]
DIRECTION_YAW_OFFSETS = {
    "front": 0.0,
    "right": -np.pi / 2,
    "back": np.pi,
    "left": np.pi / 2,
}

LOOKDOWN_PITCH_DEG = 30.0
LOOKDOWN_DIRECTION = "front_down"


# ==================== IO ====================

def submit_io_task(executor, io_futures, max_pending, fn, *args):
    io_futures.append(executor.submit(fn, *args))
    if len(io_futures) >= max_pending:
        _, not_done = concurrent.futures.wait(
            io_futures, return_when=concurrent.futures.FIRST_COMPLETED,
        )
        io_futures[:] = list(not_done)


def save_chunk_npz(
    chunk_path: str,
    frame_ids: np.ndarray,
    rgb_by_dir: Dict[str, np.ndarray],
    depth_by_dir: Dict[str, np.ndarray],
    pose_by_dir: Dict[str, np.ndarray],
    jpg_quality: int = 90,
):
    chunk_dict = {"frame_ids": frame_ids}
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpg_quality]
    all_dirs = list(rgb_by_dir.keys())
    for d in all_dirs:
        jpg_list = []
        for i in range(len(rgb_by_dir[d])):
            _, buf = cv2.imencode(".jpg", rgb_by_dir[d][i], encode_params)
            jpg_list.append(buf.astype(np.uint8).ravel())
        chunk_dict[f"rgb_{d}"] = np.array(jpg_list, dtype=object)
        chunk_dict[f"depth_{d}"] = depth_by_dir[d]
        chunk_dict[f"pose_{d}"] = pose_by_dir[d]
    np.savez(chunk_path, **chunk_dict)


# ==================== 几何工具 ====================

def quaternion_to_rotation_matrix(q) -> np.ndarray:
    if hasattr(q, "scalar") and hasattr(q, "vector"):
        w, x, y, z = q.scalar, q.vector.x, q.vector.y, q.vector.z
    elif hasattr(q, "w") and hasattr(q, "x"):
        w, x, y, z = q.w, q.x, q.y, q.z
    elif hasattr(q, "__getitem__"):
        w, x, y, z = q[0], q[1], q[2], q[3]
    else:
        raise ValueError(f"Unknown quaternion type: {type(q)}")
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


def compute_camera_pose(agent_state, T_agent_cam: np.ndarray) -> np.ndarray:
    R_agent = quaternion_to_rotation_matrix(agent_state.rotation)
    T_w_agent = np.eye(4, dtype=np.float32)
    T_w_agent[:3, :3] = R_agent
    T_w_agent[:3, 3] = agent_state.position
    return T_w_agent @ T_agent_cam


def get_sensor_extrinsics(config) -> np.ndarray:
    sensor_cfg = config.SIMULATOR.RGB_SENSOR
    sensor_position = np.array(sensor_cfg.POSITION, dtype=np.float32)
    if hasattr(sensor_cfg, "ORIENTATION"):
        orientation = sensor_cfg.ORIENTATION
        if isinstance(orientation, (list, tuple)) and len(orientation) == 3:
            roll, pitch, yaw = orientation
            cy, sy = np.cos(yaw), np.sin(yaw)
            cp, sp = np.cos(pitch), np.sin(pitch)
            cr, sr = np.cos(roll), np.sin(roll)
            sensor_rotation = np.array([
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ], dtype=np.float32)
        else:
            sensor_rotation = np.eye(3, dtype=np.float32)
    else:
        sensor_rotation = np.eye(3, dtype=np.float32)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = sensor_rotation
    T[:3, 3] = sensor_position
    return T


def compute_intrinsics(config) -> Dict:
    rgb_cfg = config.SIMULATOR.RGB_SENSOR
    width = int(rgb_cfg.WIDTH)
    height = int(rgb_cfg.HEIGHT)
    hfov_deg = float(getattr(rgb_cfg, "HFOV", 90.0))
    hfov_rad = math.radians(hfov_deg)
    fx = width / (2.0 * math.tan(hfov_rad / 2.0))
    fy = fx
    cx, cy = width / 2.0, height / 2.0
    vfov_deg = math.degrees(2.0 * math.atan(height / (2.0 * fy)))
    return {
        "projection": "pinhole",
        "width": width, "height": height,
        "hfov": hfov_deg, "vfov": vfov_deg,
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "K": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
    }


# ==================== 全景采集 ====================

def _ensure_np_quaternion(q):
    if isinstance(q, np.quaternion):
        return q
    if hasattr(q, "scalar") and hasattr(q, "vector"):
        return np.quaternion(q.scalar, q.vector.x, q.vector.y, q.vector.z)
    if hasattr(q, "w"):
        return np.quaternion(q.w, q.x, q.y, q.z)
    raise ValueError(f"Cannot convert {type(q)} to np.quaternion")


def capture_multiview(sim, T_agent_cam: np.ndarray,
                      lookdown_pitch_deg: float = 0.0) -> Dict:
    """Capture 4-direction panoramic views + optional lookdown front view.

    Args:
        lookdown_pitch_deg: if > 0, also capture a downward-tilted front view
            stored under the key ``LOOKDOWN_DIRECTION``.
    """
    agent_state = sim.get_agent_state()
    orig_pos = agent_state.position.copy()
    orig_rot = _ensure_np_quaternion(agent_state.rotation)

    results = {}
    for d in DIRECTIONS:
        yaw = DIRECTION_YAW_OFFSETS[d]
        if abs(yaw) < 1e-6:
            rot = orig_rot
        else:
            q_yaw = np.quaternion(np.cos(yaw / 2), 0, np.sin(yaw / 2), 0)
            rot = orig_rot * q_yaw
        sim.set_agent_state(orig_pos, rot)
        obs = sim.get_sensor_observations()
        fake_state = SimpleNamespace(position=orig_pos, rotation=rot)
        pose = compute_camera_pose(fake_state, T_agent_cam)
        results[d] = {"rgb": obs["rgb"].copy(), "depth": obs["depth"].copy(), "pose": pose}

    if lookdown_pitch_deg > 0:
        pitch_rad = np.radians(lookdown_pitch_deg)
        q_pitch = np.quaternion(np.cos(pitch_rad / 2), np.sin(pitch_rad / 2), 0, 0)
        rot_down = orig_rot * q_pitch
        sim.set_agent_state(orig_pos, rot_down)
        obs = sim.get_sensor_observations()
        fake_state = SimpleNamespace(position=orig_pos, rotation=rot_down)
        pose = compute_camera_pose(fake_state, T_agent_cam)
        results[LOOKDOWN_DIRECTION] = {
            "rgb": obs["rgb"].copy(),
            "depth": obs["depth"].copy(),
            "pose": pose,
        }

    sim.set_agent_state(orig_pos, orig_rot)
    return results


# ==================== 动作计算 ====================

def compute_2d_action(pose_before, pose_after) -> np.ndarray:
    T_rel = np.linalg.inv(np.asarray(pose_before, dtype=np.float32)) @ np.asarray(pose_after, dtype=np.float32)
    dx = T_rel[0, 3]
    dy = -T_rel[2, 3]
    return np.array([dx, dy], dtype=np.float32)


# ==================== 主程序 ====================

def parse_args():
    p = argparse.ArgumentParser(description="R2R-CE 全景数据采集")
    p.add_argument("--config", default="habitat_extensions/config/vlnce_collect.yaml")
    p.add_argument("--output", default="/workspace/r2r_panoramic_data")
    p.add_argument("--split", default="train", choices=["train", "val_seen", "val_unseen", "test"])
    p.add_argument("--num-clips", type=int, default=5000)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--max-pending-io", type=int, default=512)
    p.add_argument("--jpg-quality", type=int, default=90)
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--lookdown-pitch", type=float, default=LOOKDOWN_PITCH_DEG,
                   help="Pitch angle (degrees) for the extra lookdown observation. "
                        "Set to 0 to disable lookdown capture.")
    return p.parse_args()


def main():
    args = parse_args()

    lookdown_pitch = args.lookdown_pitch

    print("=" * 60)
    print("R2R-CE Panoramic Data Collection")
    print("=" * 60)
    print(f"  Output:    {args.output}")
    print(f"  Split:     {args.split}")
    print(f"  Clips:     {args.num_clips}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Lookdown:  {lookdown_pitch}°" if lookdown_pitch > 0 else "  Lookdown:  disabled")
    print(f"  Format:    chunks (4-view panoramic" +
          (f" + lookdown {lookdown_pitch}°)" if lookdown_pitch > 0 else ")"))
    print("=" * 60)

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

    all_indices = []
    scene_names = list(episodes_by_scene.keys())
    random.seed(42)
    random.shuffle(scene_names)
    for s in scene_names:
        eps = list(episodes_by_scene[s])
        random.shuffle(eps)
        all_indices.extend(eps)

    output_root = Path(args.output)
    split_dir = output_root / args.split
    split_dir.mkdir(parents=True, exist_ok=True)

    collected_ids = set()
    for mf in split_dir.rglob("meta.json"):
        try:
            m = json.load(open(mf))
            eid = m.get("episode_id")
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

    while clip_id <= args.num_clips and ep_ptr < len(all_indices):
        episode_idx = all_indices[ep_ptr]
        ep_ptr += 1
        episode = dataset.episodes[episode_idx]

        if str(episode.episode_id) in collected_ids:
            continue

        scene_name = episode.scene_id.split("/")[-1].replace(".glb", "")
        print(f"\nClip {clip_id}/{args.num_clips}  scene={scene_name}  ep={episode.episode_id}")

        # Validate episode
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

            # Storage buffers — all captured directions
            all_capture_dirs = list(DIRECTIONS)
            if lookdown_pitch > 0:
                all_capture_dirs.append(LOOKDOWN_DIRECTION)

            trajectory_3d = []
            front_poses_4x4 = []
            discrete_actions = []
            frame_id = 0
            chunk_id = 0
            chunk_fids = []
            chunk_rgb = {d: [] for d in all_capture_dirs}
            chunk_depth = {d: [] for d in all_capture_dirs}
            chunk_pose = {d: [] for d in all_capture_dirs}

            def flush_chunk():
                nonlocal chunk_id
                if not chunk_fids:
                    return
                fids = np.array(chunk_fids, dtype=np.int32)
                r = {d: np.stack(chunk_rgb[d]).astype(np.uint8, copy=False) for d in all_capture_dirs}
                dp = {d: np.stack(chunk_depth[d]).astype(np.float16, copy=False) for d in all_capture_dirs}
                ps = {d: np.stack(chunk_pose[d]).astype(np.float32, copy=False) for d in all_capture_dirs}
                path = str(chunks_dir / f"chunk_{chunk_id:05d}.npz")
                submit_io_task(executor, io_futures, args.max_pending_io,
                               save_chunk_npz, path, fids, r, dp, ps, args.jpg_quality)
                chunk_id += 1
                chunk_fids.clear()
                for d in all_capture_dirs:
                    chunk_rgb[d].clear()
                    chunk_depth[d].clear()
                    chunk_pose[d].clear()

            def record_frame():
                nonlocal frame_id
                mv = capture_multiview(sim, T_agent_cam,
                                       lookdown_pitch_deg=lookdown_pitch)
                agent_state = sim.get_agent_state()
                trajectory_3d.append(agent_state.position.copy())
                front_poses_4x4.append(mv["front"]["pose"].tolist())

                for d in all_capture_dirs:
                    rgb = mv[d]["rgb"]
                    if rgb.shape[2] == 4:
                        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
                    elif rgb.shape[2] == 3:
                        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    depth = mv[d]["depth"]
                    if depth.dtype != np.float16:
                        depth = depth.astype(np.float16)
                    chunk_rgb[d].append(rgb)
                    chunk_depth[d].append(depth)
                    chunk_pose[d].append(mv[d]["pose"].astype(np.float32))

                chunk_fids.append(frame_id)
                frame_id += 1

                if len(chunk_fids) >= args.chunk_size:
                    flush_chunk()

            # Record initial frame
            record_frame()

            # Navigate along reference path
            reference_path = episode.reference_path
            target_idx = 0
            steps_taken = 0
            max_total = args.max_steps

            while target_idx < len(reference_path) and steps_taken < max_total:
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
                shutil.rmtree(clip_dir)
                stats["failed"] += 1
                continue

            # Flush remaining chunk
            flush_chunk()

            # Compute 2D continuous actions from front-view poses
            actions_2d = []
            for i in range(len(front_poses_4x4)):
                if i == len(front_poses_4x4) - 1:
                    actions_2d.append(np.zeros(2, dtype=np.float32))
                else:
                    actions_2d.append(compute_2d_action(front_poses_4x4[i], front_poses_4x4[i + 1]))
            actions_2d = np.array(actions_2d, dtype=np.float32)
            discrete_arr = np.array(discrete_actions, dtype=np.int32)

            # Save trajectory & actions
            traj_3d = np.array(trajectory_3d, dtype=np.float32)
            np.save(str(clip_dir / "trajectory_3d.npy"), traj_3d)
            np.save(str(clip_dir / "actions.npy"), actions_2d)
            np.save(str(clip_dir / "discrete_actions.npy"), discrete_arr)

            # Save intrinsics
            with open(clip_dir / "intrinsics.json", "w") as f:
                json.dump(intrinsics, f, separators=(",", ":"))

            # Save meta
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
                    "chunks": "NPZ: frame_ids + rgb/depth/pose for each direction",
                    "trajectory_3d": "NPY float32 [T,3]",
                    "actions": "NPY float32 [T,2] agent-local (dx,dy)",
                    "discrete_actions": "NPY int32 [T] (0=STOP,1=FWD,2=LEFT,3=RIGHT)",
                    "directions": all_capture_dirs,
                },
            }
            with open(clip_dir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)

            collected_ids.add(str(episode.episode_id))
            stats["successful"] += 1
            stats["total_frames"] += frame_id
            stats["scenes"][scene_name] = stats["scenes"].get(scene_name, 0) + 1

            print(f"  Done: {frame_id} frames, {len(actions_2d)} actions, instr={len(instruction_text)} chars")
            clip_id += 1

        except Exception as e:
            print(f"  Failed: {e}")
            stats["failed"] += 1
            if "clip_dir" in locals() and Path(clip_dir).exists():
                try:
                    shutil.rmtree(clip_dir)
                except Exception:
                    pass

    # Wait for IO
    if io_futures:
        print(f"\nWaiting for {len(io_futures)} IO tasks...")
        concurrent.futures.wait(io_futures)

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
