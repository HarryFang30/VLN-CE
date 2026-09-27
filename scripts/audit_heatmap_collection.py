#!/usr/bin/env python3
"""Validate four-view heatmap patrol clips before large-scale collection."""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


DIRECTIONS = ("front", "right", "back", "left")
EXPECTED_ANGLES = {
    "front": 0.0,
    "right": math.pi / 2.0,
    "back": math.pi,
    "left": math.pi / 2.0,
}


def rotation_angle(matrix: np.ndarray) -> float:
    cosine = (float(np.trace(matrix)) - 1.0) / 2.0
    return float(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--max-clips", type=int, default=0)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    meta_paths = sorted(root.glob("*/clip_*/meta.json"))
    if args.max_clips > 0:
        meta_paths = meta_paths[: args.max_clips]
    if not meta_paths:
        raise SystemExit(f"No clips found under {root}")

    errors = []
    warnings = []
    scenes = set()
    total_frames = 0
    depth_valid = 0
    depth_total = 0
    depth_min = float("inf")
    depth_max = 0.0
    max_center_error = 0.0
    max_rotation_error_deg = 0.0
    max_det_error = 0.0
    duplicate_pose_steps = 0
    translation_stationary_steps = 0
    transition_steps = 0
    return_distances = []
    frame_counts = []

    for meta_path in meta_paths:
        clip = meta_path.parent
        try:
            meta = json.loads(meta_path.read_text())
            num_frames = int(meta["num_frames"])
            scene_id = str(meta["scene_id"])
            scenes.add(scene_id)
            frame_counts.append(num_frames)
            total_frames += num_frames

            intrinsics = json.loads((clip / "intrinsics.json").read_text())
            width = int(intrinsics["width"])
            height = int(intrinsics["height"])
            K = np.asarray(intrinsics["K"], dtype=np.float64)
            if K.shape != (3, 3) or width <= 0 or height <= 0:
                errors.append(f"{clip}: invalid intrinsics")

            chunks = sorted((clip / "chunks").glob("chunk_*.npz"))
            if not chunks:
                errors.append(f"{clip}: no chunks")
                continue

            all_frame_ids = []
            front_poses = []
            decoded_rgb = False
            for chunk_path in chunks:
                with np.load(chunk_path, allow_pickle=True) as data:
                    required = {"frame_ids"}
                    for direction in DIRECTIONS:
                        required.update(
                            {
                                f"rgb_{direction}",
                                f"depth_{direction}",
                                f"pose_{direction}",
                            }
                        )
                    missing = sorted(required - set(data.files))
                    if missing:
                        errors.append(f"{chunk_path}: missing {missing}")
                        continue

                    frame_ids = np.asarray(data["frame_ids"], dtype=np.int64)
                    n = len(frame_ids)
                    all_frame_ids.extend(frame_ids.tolist())

                    poses = {}
                    for direction in DIRECTIONS:
                        rgb = data[f"rgb_{direction}"]
                        depth = np.asarray(data[f"depth_{direction}"])
                        pose = np.asarray(data[f"pose_{direction}"], dtype=np.float64)
                        poses[direction] = pose
                        if len(rgb) != n or depth.shape[0] != n or pose.shape != (n, 4, 4):
                            errors.append(
                                f"{chunk_path}: length/shape mismatch for {direction}"
                            )
                            continue

                        finite_depth = np.isfinite(depth)
                        valid_depth = finite_depth & (depth > 0)
                        depth_valid += int(valid_depth.sum())
                        depth_total += int(depth.size)
                        if valid_depth.any():
                            values = depth[valid_depth]
                            depth_min = min(depth_min, float(values.min()))
                            depth_max = max(depth_max, float(values.max()))

                        rotations = pose[:, :3, :3]
                        determinants = np.linalg.det(rotations)
                        max_det_error = max(
                            max_det_error,
                            float(np.max(np.abs(determinants - 1.0))),
                        )

                        if not decoded_rgb and n:
                            image = cv2.imdecode(
                                np.asarray(rgb[0], dtype=np.uint8), cv2.IMREAD_COLOR
                            )
                            if image is None or image.shape[:2] != (height, width):
                                errors.append(f"{chunk_path}: RGB decode/shape failure")
                            decoded_rgb = True

                    if any(direction not in poses for direction in DIRECTIONS):
                        continue
                    front = poses["front"]
                    front_poses.append(front)
                    front_centers = front[:, :3, 3]
                    for direction in DIRECTIONS:
                        current = poses[direction]
                        center_error = np.linalg.norm(
                            current[:, :3, 3] - front_centers, axis=1
                        )
                        max_center_error = max(
                            max_center_error, float(center_error.max(initial=0.0))
                        )
                        for index in range(n):
                            relative = front[index, :3, :3].T @ current[index, :3, :3]
                            angle = rotation_angle(relative)
                            error = abs(angle - EXPECTED_ANGLES[direction])
                            max_rotation_error_deg = max(
                                max_rotation_error_deg, math.degrees(error)
                            )

            expected_ids = list(range(num_frames))
            if all_frame_ids != expected_ids:
                errors.append(
                    f"{clip}: frame ids are incomplete/non-contiguous "
                    f"({len(all_frame_ids)} vs {num_frames})"
                )

            if front_poses:
                poses = np.concatenate(front_poses, axis=0)
                if len(poses) != num_frames:
                    errors.append(f"{clip}: pose count {len(poses)} != {num_frames}")
                if len(poses) > 1:
                    for previous, current in zip(poses[:-1], poses[1:]):
                        transition_steps += 1
                        translation = float(
                            np.linalg.norm(current[:3, 3] - previous[:3, 3])
                        )
                        angle = rotation_angle(previous[:3, :3].T @ current[:3, :3])
                        if translation < 1e-4:
                            translation_stationary_steps += 1
                        if translation < 1e-4 and angle < 1e-4:
                            duplicate_pose_steps += 1
                    return_distances.append(
                        float(np.linalg.norm(poses[-1, :3, 3] - poses[0, :3, 3]))
                    )

            trajectory = np.load(clip / "trajectory_3d.npy")
            if trajectory.shape != (num_frames, 3):
                errors.append(f"{clip}: invalid trajectory shape {trajectory.shape}")
            if meta.get("data_format", {}).get("depth_unit") != "meters":
                warnings.append(f"{clip}: missing depth_unit=meters metadata")
        except Exception as error:
            errors.append(f"{clip}: {type(error).__name__}: {error}")

    valid_ratio = depth_valid / max(depth_total, 1)
    duplicate_ratio = duplicate_pose_steps / max(transition_steps, 1)
    stationary_ratio = translation_stationary_steps / max(transition_steps, 1)
    summary = {
        "root": str(root),
        "clips_checked": len(meta_paths),
        "scenes": len(scenes),
        "total_frames": total_frames,
        "frames_min": min(frame_counts),
        "frames_mean": float(np.mean(frame_counts)),
        "frames_max": max(frame_counts),
        "depth_valid_ratio": valid_ratio,
        "depth_min_m": None if depth_min == float("inf") else depth_min,
        "depth_max_m": depth_max,
        "max_four_view_center_error_m": max_center_error,
        "max_four_view_rotation_error_deg": max_rotation_error_deg,
        "max_rotation_determinant_error": max_det_error,
        "translation_stationary_step_ratio": stationary_ratio,
        "exact_duplicate_pose_step_ratio": duplicate_ratio,
        "return_distance_mean_m": float(np.mean(return_distances)),
        "return_distance_max_m": float(np.max(return_distances)),
        "warnings": warnings[:100],
        "errors": errors[:100],
        "passed": not errors
        and valid_ratio >= 0.90
        and max_center_error <= 1e-3
        and max_rotation_error_deg <= 0.1
        and max_det_error <= 1e-3,
    }
    output_path = root / "audit_summary.json"
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Audit saved to {output_path}")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
