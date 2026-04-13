#!/usr/bin/env python3
"""
R2R-CE 基础数据采集（单视角 + 动作记录）

沿 R2R-CE episode 的 reference_path 导航，每步采集 RGB + Depth，
并记录离散动作 (discrete_actions.npy) 和 2D 连续动作 (actions.npy)。

输出格式 (per clip):
  rgb/          — JPEG 帧序列
  depth/        — NPY float16 深度图
  poses.json    — [T] 4x4 相机位姿
  intrinsics.json
  actions.npy   — [T, 2] float32 agent-local (dx, dy)
  discrete_actions.npy — [T] int32 HabitatSimActions
  meta.json
"""
import argparse
import json
import math
import random
import shutil
import time
import concurrent.futures
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import habitat
import habitat_extensions  # noqa: F401  (注册扩展配置的副作用)
from habitat_extensions.config.default import get_extended_config
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.sims.habitat_simulator.actions import HabitatSimActions

from collect.common.geometry import (
    get_sensor_extrinsics,
    compute_camera_pose,
    compute_intrinsics,
    compute_2d_action,
    discrete_action_to_name,
)
from collect.common.io_utils import setup_logging, drain_io_futures
from collect.r2r.utils import (
    match_keyframes_to_trajectory,
    compute_adaptive_min_valid_ratio,
)


def parse_args():
    p = argparse.ArgumentParser(description="R2R-CE 数据采集（单视角 + 动作）")
    p.add_argument("--config", default="habitat_extensions/config/vlnce_collect.yaml")
    p.add_argument("--output", default="/root/autodl-tmp/r2r_train_data")
    p.add_argument("--split", default="train",
                   choices=["train", "val", "val_seen", "val_unseen", "test"])
    p.add_argument("--num-clips", type=int, default=1000)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"collect_{args.split}_{timestamp}.log"
    setup_logging(log_file)

    print(f"Logging to: {log_file}")
    print(f"Starting data collection (WITH ACTIONS)")
    print(f"   Output: {args.output}")
    print(f"   Split: {args.split}")
    print(f"   Clips: {args.num_clips}")
    print(f"   Max steps per direction: {args.max_steps}")

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
    print("Environment created!")

    T_agent_cam = get_sensor_extrinsics(config)

    depth_cfg = getattr(config.SIMULATOR, "DEPTH_SENSOR", None)
    if depth_cfg is None:
        depth_normalize, depth_min, depth_max = False, 0.0, 10.0
    else:
        depth_normalize = getattr(depth_cfg, "NORMALIZE_DEPTH", False)
        depth_min = float(getattr(depth_cfg, "MIN_DEPTH", 0.0))
        depth_max = float(getattr(depth_cfg, "MAX_DEPTH", 10.0))
    print(f"Depth config: normalize={depth_normalize}, range=[{depth_min}, {depth_max}]")

    # ==================== Episode 索引（按场景分组以减少切换） ====================
    dataset = env._dataset
    print(f"Dataset: {len(dataset.episodes)} episodes")

    episodes_by_scene = {}
    for i, ep in enumerate(dataset.episodes):
        scene_name = ep.scene_id.split("/")[-1].replace(".glb", "")
        episodes_by_scene.setdefault(scene_name, []).append(i)
    print(f"Scenes: {len(episodes_by_scene)}")

    scene_names = list(episodes_by_scene.keys())
    random.seed(42)
    random.shuffle(scene_names)

    all_episode_indices = []
    for scene in scene_names:
        eps = list(episodes_by_scene[scene])
        random.shuffle(eps)
        all_episode_indices.extend(eps)

    # ==================== 断点续采 ====================
    split_dir = output_root / args.split
    progress_file = output_root / "progress.json"

    start_clip = 1
    start_episode_attempt = None
    if progress_file.exists():
        with open(progress_file) as f:
            progress = json.load(f)
            start_clip = progress.get("next_clip_to_try", 1)
            start_episode_attempt = progress.get("next_episode_attempt")
        print(f"Resuming from clip {start_clip}")

    collected_episode_ids = set()
    if split_dir.exists():
        for meta_file in split_dir.rglob("meta.json"):
            try:
                with open(meta_file) as f:
                    ep_id = json.load(f).get("episode_id")
                    if ep_id:
                        collected_episode_ids.add(str(ep_id))
            except Exception:
                pass
        if collected_episode_ids:
            print(f"Found {len(collected_episode_ids)} already collected episodes")

    # ==================== 统计 ====================
    stats = {
        "successful": 0,
        "failed": 0,
        "failed_clips": [],
        "scenes": {},
        "total_frames": 0,
        "total_actions": 0,
        "action_distribution": {"STOP": 0, "MOVE_FORWARD": 0, "TURN_LEFT": 0, "TURN_RIGHT": 0},
        "missing_fields": {
            "goals_missing": 0, "goals_empty": 0,
            "reference_path_missing": 0, "reference_path_empty": 0,
            "instruction_missing": 0, "instruction_text_missing": 0,
            "trajectory_id_missing": 0,
        },
    }
    collected = []
    start_time = time.time()

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers)
    io_futures: list = []

    clip_id = start_clip
    episode_attempt = start_episode_attempt if start_episode_attempt is not None else clip_id - 1

    # ==================== 主采集循环 ====================
    while clip_id <= args.num_clips:
        if episode_attempt >= len(all_episode_indices):
            print("\nWarning: Ran out of prepared episodes")
            break

        print(f"\nCollecting clip {clip_id}/{args.num_clips}")

        try:
            episode_idx = all_episode_indices[episode_attempt]
            episode_attempt += 1
            episode = dataset.episodes[episode_idx]

            if str(episode.episode_id) in collected_episode_ids:
                print(f"  Skipping already collected episode {episode.episode_id}")
                continue

            current_scene = episode.scene_id.split("/")[-1].replace(".glb", "")

            if not hasattr(env, "_last_scene") or env._last_scene != current_scene:
                print(f"  Loading scene: {current_scene}")

            env._current_episode = episode
            observations = env.reset()

            if not hasattr(env, "_last_scene") or env._last_scene != current_scene:
                sim = env.sim
                follower = ShortestPathFollower(sim, goal_radius=0.2, return_one_hot=False)
                env._last_scene = current_scene

            missing = [k for k in ("rgb", "depth") if k not in observations]
            if missing:
                raise RuntimeError(f"Missing observations: {missing}")

            clip_dir = output_root / args.split / current_scene / f"clip_{clip_id:06d}"
            rgb_dir = clip_dir / "rgb"
            depth_dir = clip_dir / "depth"
            rgb_dir.mkdir(parents=True, exist_ok=True)
            depth_dir.mkdir(parents=True, exist_ok=True)

            # IO 健康检查
            test_file = rgb_dir / ".io_health_check"
            try:
                with open(test_file, "wb") as f:
                    f.write(b"ok")
                with open(test_file, "rb") as f:
                    if f.read() != b"ok":
                        raise IOError("I/O verification mismatch")
                test_file.unlink()
            except Exception as io_err:
                print(f"  I/O health check failed: {io_err}")
                shutil.rmtree(clip_dir, ignore_errors=True)
                stats["failed"] += 1
                stats["failed_clips"].append({
                    "clip_id": clip_id, "episode_id": episode.episode_id,
                    "error": str(io_err), "stage": "io_check",
                })
                continue

            # Episode 字段验证
            validation_errors = []
            if episode.goals is None:
                validation_errors.append("goals is None")
                stats["missing_fields"]["goals_missing"] += 1
            elif len(episode.goals) == 0:
                validation_errors.append("goals is empty")
                stats["missing_fields"]["goals_empty"] += 1
            elif not hasattr(episode.goals[0], "position"):
                validation_errors.append("goals[0] missing position")

            if episode.reference_path is None:
                validation_errors.append("reference_path is None")
                stats["missing_fields"]["reference_path_missing"] += 1
            elif len(episode.reference_path) == 0:
                validation_errors.append("reference_path is empty")
                stats["missing_fields"]["reference_path_empty"] += 1

            if episode.instruction is None:
                validation_errors.append("instruction is None")
                stats["missing_fields"]["instruction_missing"] += 1
            elif not hasattr(episode.instruction, "instruction_text"):
                validation_errors.append("instruction missing instruction_text")
                stats["missing_fields"]["instruction_text_missing"] += 1

            if validation_errors:
                print(f"  Episode validation failed: {', '.join(validation_errors)}")
                stats["failed"] += 1
                stats["failed_clips"].append({
                    "clip_id": clip_id, "episode_id": episode.episode_id,
                    "error": f"Invalid: {', '.join(validation_errors)}",
                    "stage": "episode_validation",
                })
                continue

            reference_path = episode.reference_path
            if episode.instruction is None or episode.instruction.instruction_text is None:
                instruction_text = ""
                stats["missing_fields"]["instruction_text_missing"] += 1
            else:
                instruction_text = episode.instruction.instruction_text

            trajectory_id = getattr(episode, "trajectory_id", None)
            if trajectory_id is None:
                trajectory_id = "unknown"
                stats["missing_fields"]["trajectory_id_missing"] += 1

            # ==================== 导航采集 ====================
            poses = []
            trajectory_positions = []
            discrete_actions = []
            frame_id = 0

            # 记录起始帧
            def _save_frame(obs, fid):
                rgb = obs["rgb"]
                if rgb.shape[2] == 4:
                    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
                elif rgb.shape[2] == 3:
                    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                rgb_path = rgb_dir / f"{fid:06d}.jpg"
                io_futures.append(executor.submit(
                    cv2.imwrite, str(rgb_path), rgb.copy(),
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                ))
                depth = obs["depth"]
                depth_path = depth_dir / f"{fid:06d}.npy"
                io_futures.append(executor.submit(
                    np.save, str(depth_path), depth.copy().astype(np.float16),
                ))

            _save_frame(observations, frame_id)
            agent_state = sim.get_agent_state()
            trajectory_positions.append(agent_state.position.copy())
            poses.append(compute_camera_pose(agent_state, T_agent_cam).tolist())
            frame_id += 1

            target_idx = 0
            steps_taken = 0
            max_steps_total = args.max_steps * max(1, len(reference_path))
            clip_failed = False

            while not clip_failed and target_idx < len(reference_path):
                goal_point = reference_path[target_idx]
                action = follower.get_next_action(goal_point)
                if action is None:
                    action = HabitatSimActions.STOP
                if action == HabitatSimActions.STOP:
                    target_idx += 1
                    continue

                discrete_actions.append(int(action))
                observations = env.step(action)
                steps_taken += 1

                _save_frame(observations, frame_id)
                agent_state = sim.get_agent_state()
                trajectory_positions.append(agent_state.position.copy())
                poses.append(compute_camera_pose(agent_state, T_agent_cam).tolist())
                frame_id += 1

                if steps_taken >= max_steps_total:
                    clip_failed = True

            discrete_actions.append(0)  # STOP

            start_frame = 0
            end_frame = max(0, frame_id - 1)
            num_frames = end_frame - start_frame + 1

            if clip_failed:
                print(f"  Skipping: exceeded step limit ({steps_taken})")
                shutil.rmtree(clip_dir, ignore_errors=True)
                stats["failed"] += 1
                stats["failed_clips"].append({
                    "clip_id": clip_id, "episode_id": episode.episode_id,
                    "error": f"Exceeded steps ({steps_taken})", "stage": "trajectory",
                })
                continue

            MIN_FRAMES = 5
            if num_frames < MIN_FRAMES:
                print(f"  Skipping: too short ({num_frames} frames)")
                shutil.rmtree(clip_dir, ignore_errors=True)
                stats["failed"] += 1
                stats["failed_clips"].append({
                    "clip_id": clip_id, "episode_id": episode.episode_id,
                    "error": f"Too short ({num_frames} frames)", "stage": "trajectory_length",
                })
                continue

            # ==================== 计算动作 ====================
            actions_2d = []
            for i in range(len(poses)):
                if i == len(poses) - 1:
                    actions_2d.append(np.zeros(2, dtype=np.float32))
                else:
                    actions_2d.append(compute_2d_action(poses[i], poses[i + 1]))
            actions_2d = np.array(actions_2d, dtype=np.float32)
            discrete_arr = np.array(discrete_actions, dtype=np.int32)

            for act in discrete_actions:
                name = discrete_action_to_name(act)
                if name in stats["action_distribution"]:
                    stats["action_distribution"][name] += 1

            action_stats = {
                "dx": {"min": float(actions_2d[:, 0].min()), "max": float(actions_2d[:, 0].max()),
                        "mean": float(actions_2d[:, 0].mean())},
                "dy": {"min": float(actions_2d[:, 1].min()), "max": float(actions_2d[:, 1].max()),
                        "mean": float(actions_2d[:, 1].mean())},
            }

            # 关键帧匹配
            positions = trajectory_positions[start_frame:end_frame + 1]
            kf_indices, kf_distances = match_keyframes_to_trajectory(positions, reference_path)
            kf_indices = [idx + start_frame for idx in kf_indices]

            # 内参
            intrinsics_data = compute_intrinsics(config)
            h_obs, w_obs = observations["rgb"].shape[:2]
            if w_obs != intrinsics_data["width"] or h_obs != intrinsics_data["height"]:
                hfov_rad = math.radians(intrinsics_data["hfov"])
                fx = w_obs / (2.0 * math.tan(hfov_rad / 2.0))
                fy = fx
                cx, cy = w_obs / 2.0, h_obs / 2.0
                vfov_deg = math.degrees(2.0 * math.atan(h_obs / (2.0 * fy)))
                intrinsics_data.update({
                    "width": w_obs, "height": h_obs,
                    "fx": fx, "fy": fy, "cx": cx, "cy": cy, "vfov": vfov_deg,
                    "K": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                })

            # ==================== 保存数据 ====================
            with open(clip_dir / "poses.json", "w") as f:
                json.dump(poses, f, indent=2)
            with open(clip_dir / "intrinsics.json", "w") as f:
                json.dump({k: v for k, v in intrinsics_data.items()
                           if k != "K" or True}, f, indent=2)

            np.save(clip_dir / "actions.npy", actions_2d)
            np.save(clip_dir / "discrete_actions.npy", discrete_arr)

            MIN_VALID_RATIO, quality_tier = compute_adaptive_min_valid_ratio(kf_distances)
            meta = {
                "episode_id": episode.episode_id,
                "trajectory_id": trajectory_id,
                "scene_id": current_scene,
                "instruction": instruction_text,
                "sampling_strategy": "walk_to_goal",
                "num_frames": frame_id,
                "trajectory": {"start_frame": start_frame, "end_frame": end_frame,
                                "num_frames": num_frames},
                "reference_path": reference_path,
                "keyframe_indices": kf_indices,
                "keyframe_distances": kf_distances,
                "max_keyframe_distance": float(np.max(kf_distances)),
                "mean_keyframe_distance": float(np.mean(kf_distances)),
                "actions": {
                    "num_actions": len(actions_2d),
                    "action_dim": 2,
                    "action_semantic": "action[i] = from frame[i] to frame[i+1]",
                    "action_format": "(dx, dy) - agent-local 2D displacement",
                    "discrete_action_format": "HabitatSimActions (0=STOP,1=FWD,2=LEFT,3=RIGHT)",
                    "stats": action_stats,
                    "files": {"continuous": "actions.npy", "discrete": "discrete_actions.npy"},
                },
                "quality_control": {
                    "min_valid_ratio_used": float(MIN_VALID_RATIO),
                    "avg_keyframe_distance": float(np.mean(kf_distances)),
                    "quality_tier": quality_tier,
                },
            }
            with open(clip_dir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)

            # 更新统计
            stats["successful"] += 1
            stats["total_frames"] += frame_id
            stats["total_actions"] += len(actions_2d)
            stats["scenes"][current_scene] = stats["scenes"].get(current_scene, 0) + 1
            collected_episode_ids.add(str(episode.episode_id))

            if len(io_futures) > 100:
                concurrent.futures.wait(io_futures)
                io_futures.clear()

            collected.append(meta)
            print(f"  Clip {clip_id} done: {frame_id} frames, {len(actions_2d)} actions")
            clip_id += 1

        except Exception as e:
            print(f"  Clip {clip_id} failed: {e}")
            if "clip_dir" in locals() and clip_dir.exists():
                shutil.rmtree(clip_dir, ignore_errors=True)
            stats["failed"] += 1
            info = {"clip_id": clip_id, "error": str(e)}
            if "episode" in locals():
                info["episode_id"] = episode.episode_id
            stats["failed_clips"].append(info)
            continue

        finally:
            with open(progress_file, "w") as f:
                json.dump({"next_clip_to_try": clip_id,
                           "next_episode_attempt": episode_attempt}, f)

    # ==================== 收尾 ====================
    drain_io_futures(io_futures)
    env.close()

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("Data collection completed (WITH ACTIONS)!")
    print("=" * 60)
    print(f"  Successful: {stats['successful']}/{args.num_clips}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Total frames: {stats['total_frames']}")
    print(f"  Total actions: {stats['total_actions']}")
    print(f"  Time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"  Scenes: {len(stats['scenes'])}")

    if stats["total_actions"] > 0:
        print("\nAction distribution:")
        for name, count in stats["action_distribution"].items():
            pct = count / stats["total_actions"] * 100
            print(f"  {name}: {count} ({pct:.1f}%)")

    with open(output_root / "collection_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    index_path = output_root / f"{args.split}_index.json"
    with open(index_path, "w") as f:
        json.dump(collected, f, indent=2)
    print(f"\nStats saved to {output_root / 'collection_stats.json'}")
    print(f"Index saved to {index_path}")


if __name__ == "__main__":
    main()
