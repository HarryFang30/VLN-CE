#!/usr/bin/env python3
"""Audit patrol clips through HeatmapVLN's real heatmap-label data path."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


PANORAMIC_DIRECTIONS = ("front", "right", "back", "left")
DISTANCE_BINS = (0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, math.inf)
GAP_BINS = (0, 5, 10, 20, 40, 80, 160, sys.maxsize)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure real HeatmapVLN label visibility on collected clips."
    )
    parser.add_argument("root", type=Path, help="Collected dataset root")
    parser.add_argument(
        "--training-repo",
        type=Path,
        default=Path("/mnt/afs/lixiaoou/intern/fjl/HeatmapVLN"),
    )
    parser.add_argument("--sample-stride", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--min-history", type=int, default=5)
    parser.add_argument("--num-history", type=int, default=8)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def bin_label(value: float, edges: tuple[float, ...]) -> str:
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo <= value < hi:
            hi_label = "inf" if math.isinf(float(hi)) or hi == sys.maxsize else f"{hi:g}"
            return f"[{lo:g},{hi_label})"
    return "overflow"


def ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    training_repo = args.training_repo.expanduser().resolve()
    output = args.output or (root / "training_label_audit.json")

    if not root.exists():
        raise FileNotFoundError(root)
    if not (training_repo / "src" / "data" / "sliding_window_dataset.py").exists():
        raise FileNotFoundError(f"HeatmapVLN training repo not found: {training_repo}")

    sys.path.insert(0, str(training_repo))
    from src.data.heatmap_geometry import compute_history_heatmap
    from src.data.sliding_window_dataset import VLNSlidingWindowDataset
    from src.data.trajectory_utils import compute_history_rel_poses

    dataset = VLNSlidingWindowDataset(
        root=str(root),
        split="all",
        min_history=args.min_history,
        num_history_sample=args.num_history,
        image_size=(256, 256),
        hm_size=(64, 64),
        load_depth=True,
        cache_poses=True,
        sample_stride=max(1, args.sample_stride),
        enable_augmentation=False,
        clip_level_sampling=False,
        load_history_frames=False,
    )

    selected = np.arange(len(dataset.sample_index), dtype=np.int64)
    if args.max_samples > 0 and len(selected) > args.max_samples:
        selected = np.linspace(0, len(selected) - 1, args.max_samples, dtype=np.int64)

    histories_total = 0
    histories_visible = 0
    histories_in_fov = 0
    samples_with_visible = 0
    view_visible = np.zeros(4, dtype=np.int64)
    view_in_fov = np.zeros(4, dtype=np.int64)
    visible_view_histogram = np.zeros(5, dtype=np.int64)
    distance_stats: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    gap_stats: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    positive_peaks: list[float] = []
    failures: list[str] = []
    scene_names: set[str] = set()
    clip_names: set[str] = set()

    for selected_idx in selected.tolist():
        clip_idx, current_t = dataset.sample_index[int(selected_idx)]
        clip_dir = dataset.clips[clip_idx]
        scene_names.add(clip_dir.parent.name)
        clip_names.add(f"{clip_dir.parent.name}/{clip_dir.name}")
        try:
            history_indices = dataset._sample_history_indices(
                0, int(current_t), args.num_history
            )
            poses = dataset._load_poses(clip_idx)
            history_poses = [poses[int(i)] for i in history_indices]
            current_pose = poses[int(current_t)]
            img_size, intrinsics = dataset._load_intrinsics(clip_idx, clip_dir)

            heatmaps, visibility = dataset._compute_per_history_multiview_heatmaps(
                clip_idx=clip_idx,
                clip_dir=clip_dir,
                history_poses=history_poses,
                current_t=int(current_t),
                img_size=img_size,
                K=intrinsics,
                hm_size=(64, 64),
            )
            heatmaps_np = heatmaps.numpy()
            visibility_np = visibility.numpy().astype(bool)
            if heatmaps_np.shape != (len(history_poses), 4, 64, 64):
                raise ValueError(f"unexpected heatmap shape {heatmaps_np.shape}")
            if visibility_np.shape != (len(history_poses), 4):
                raise ValueError(f"unexpected visibility shape {visibility_np.shape}")

            current_view_poses = {
                direction: np.asarray(
                    dataset._get_chunk_frame_array(
                        clip_idx, int(current_t), "pose", direction=direction
                    ),
                    dtype=np.float32,
                )
                for direction in PANORAMIC_DIRECTIONS
            }
            geometry_visibility = np.zeros_like(visibility_np)
            for hist_idx, history_pose in enumerate(history_poses):
                for view_idx, direction in enumerate(PANORAMIC_DIRECTIONS):
                    _, geometry_count = compute_history_heatmap(
                        history_poses=[history_pose],
                        current_pose=current_view_poses[direction],
                        current_depth=None,
                        hm_size=(64, 64),
                        img_size=img_size,
                        K=intrinsics,
                        depth_normalize=False,
                    )
                    geometry_visibility[hist_idx, view_idx] = geometry_count > 0

            relative_poses = compute_history_rel_poses(history_poses, current_pose)
            distances = np.linalg.norm(relative_poses[:, :2], axis=1)
            visible_any = visibility_np.any(axis=1)
            in_fov_any = geometry_visibility.any(axis=1)

            histories_total += len(history_poses)
            histories_visible += int(visible_any.sum())
            histories_in_fov += int(in_fov_any.sum())
            samples_with_visible += int(bool(visible_any.any()))
            view_visible += visibility_np.sum(axis=0)
            view_in_fov += geometry_visibility.sum(axis=0)
            for count in visibility_np.sum(axis=1).astype(int).tolist():
                visible_view_histogram[count] += 1

            positive_values = heatmaps_np[visibility_np]
            if positive_values.size:
                positive_peaks.extend(
                    positive_values.reshape(-1, 64 * 64).max(axis=1).astype(float).tolist()
                )

            for distance, gap, visible, in_fov in zip(
                distances.tolist(),
                (int(current_t) - history_indices).tolist(),
                visible_any.tolist(),
                in_fov_any.tolist(),
            ):
                distance_bucket = distance_stats[bin_label(float(distance), DISTANCE_BINS)]
                distance_bucket[0] += 1
                distance_bucket[1] += int(visible)
                distance_bucket[2] += int(in_fov)
                gap_bucket = gap_stats[bin_label(float(gap), GAP_BINS)]
                gap_bucket[0] += 1
                gap_bucket[1] += int(visible)
                gap_bucket[2] += int(in_fov)
        except Exception as exc:  # Keep enough context to diagnose a corrupt sample.
            failures.append(
                f"sample={selected_idx} clip={clip_dir} current_t={current_t}: "
                f"{type(exc).__name__}: {exc}"
            )

    samples_checked = len(selected)
    samples_succeeded = samples_checked - len(failures)
    visible_ratio = ratio(histories_visible, histories_total)
    in_fov_ratio = ratio(histories_in_fov, histories_total)
    sample_positive_ratio = ratio(samples_with_visible, samples_succeeded)
    occlusion_survival = ratio(histories_visible, histories_in_fov)

    def serialize_buckets(stats: dict[str, list[int]], edges: tuple[float, ...]) -> dict:
        result = {}
        for lo, hi in zip(edges[:-1], edges[1:]):
            label = bin_label(float(lo), edges)
            total, visible, in_fov = stats.get(label, [0, 0, 0])
            result[label] = {
                "histories": total,
                "visible": visible,
                "in_fov": in_fov,
                "visible_ratio": ratio(visible, total),
                "in_fov_ratio": ratio(in_fov, total),
            }
        return result

    warnings = []
    if visible_ratio < 0.15:
        warnings.append(
            "Per-history visible-positive ratio is below 15%; heatmap supervision is sparse."
        )
    if sample_positive_ratio < 0.70:
        warnings.append(
            "Fewer than 70% of current panoramas contain any visible sampled history point."
        )
    if occlusion_survival < 0.20:
        warnings.append(
            "Most geometrically projectable history points are rejected by the depth test."
        )

    summary = {
        "root": str(root),
        "training_repo": str(training_repo),
        "clips_in_dataset": len(dataset.clips),
        "clips_checked": len(clip_names),
        "scenes_checked": len(scene_names),
        "sample_stride": max(1, args.sample_stride),
        "samples_indexed": len(dataset.sample_index),
        "samples_checked": samples_checked,
        "samples_succeeded": samples_succeeded,
        "sample_failure_count": len(failures),
        "sample_any_visible_ratio": sample_positive_ratio,
        "histories_total": histories_total,
        "history_in_any_view_fov_ratio": in_fov_ratio,
        "history_depth_visible_ratio": visible_ratio,
        "depth_visibility_given_in_fov_ratio": occlusion_survival,
        "per_view_in_fov_ratio": {
            direction: ratio(int(view_in_fov[idx]), histories_total)
            for idx, direction in enumerate(PANORAMIC_DIRECTIONS)
        },
        "per_view_depth_visible_ratio": {
            direction: ratio(int(view_visible[idx]), histories_total)
            for idx, direction in enumerate(PANORAMIC_DIRECTIONS)
        },
        "visible_view_count_histogram": {
            str(index): int(count)
            for index, count in enumerate(visible_view_histogram.tolist())
        },
        "positive_heatmap_peak": {
            "min": float(np.min(positive_peaks)) if positive_peaks else 0.0,
            "mean": float(np.mean(positive_peaks)) if positive_peaks else 0.0,
            "max": float(np.max(positive_peaks)) if positive_peaks else 0.0,
        },
        "by_distance_m": serialize_buckets(distance_stats, DISTANCE_BINS),
        "by_temporal_gap_frames": serialize_buckets(gap_stats, GAP_BINS),
        "warnings": warnings,
        "failure_examples": failures[:20],
        "training_loader_passed": len(failures) == 0 and samples_succeeded > 0,
        "label_density_usable": (
            visible_ratio >= 0.15
            and sample_positive_ratio >= 0.70
            and occlusion_survival >= 0.20
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Training-label audit saved to {output}")
    return 0 if summary["training_loader_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
