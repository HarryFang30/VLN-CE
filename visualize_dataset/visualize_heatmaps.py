#!/usr/bin/env python3
"""
Visualize navigation goal heatmaps overlaid on RGB images.

This script loads heatmaps (both single and split formats) and overlays them
on corresponding RGB images to visualize the navigation goal locations.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np


def load_clip_data(clip_path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], dict, dict, List[str]]:
    """
    Load all necessary data from a clip directory.
    Supports both old format (single heatmaps.npy) and new format (split history/future).

    Args:
        clip_path: Path to the clip directory

    Returns:
        Tuple of (heatmaps, heatmaps_history, heatmaps_future, meta, intrinsics, rgb_paths)
        - For old format: (heatmaps, None, None, ...)
        - For new format: (None, heatmaps_history, heatmaps_future, ...)
    """
    # Load metadata first to check format
    meta_path = clip_path / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    # Check if this is split format
    heatmap_type = meta.get('heatmap_type', '')
    is_split = 'split' in heatmap_type.lower()

    heatmaps = None
    heatmaps_history = None
    heatmaps_future = None

    if is_split:
        # Load split heatmaps
        heatmaps_history_path = clip_path / "heatmaps_history.npy"
        heatmaps_future_path = clip_path / "heatmaps_future.npy"

        if not heatmaps_history_path.exists():
            raise FileNotFoundError(f"History heatmaps not found: {heatmaps_history_path}")
        if not heatmaps_future_path.exists():
            raise FileNotFoundError(f"Future heatmaps not found: {heatmaps_future_path}")

        heatmaps_history = np.load(heatmaps_history_path)
        heatmaps_future = np.load(heatmaps_future_path)
    else:
        # Load single heatmap file (old format)
        heatmaps_path = clip_path / "heatmaps.npy"
        if not heatmaps_path.exists():
            raise FileNotFoundError(f"Heatmaps not found: {heatmaps_path}")
        heatmaps = np.load(heatmaps_path)

    # Load camera intrinsics
    intrinsics_path = clip_path / "intrinsics.json"
    if not intrinsics_path.exists():
        raise FileNotFoundError(f"Intrinsics not found: {intrinsics_path}")
    with open(intrinsics_path, 'r') as f:
        intrinsics = json.load(f)

    # Get RGB image paths
    rgb_dir = clip_path / "rgb"
    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")

    # Get all RGB images sorted by filename
    rgb_paths = sorted(list(rgb_dir.glob("*.png")))
    if not rgb_paths:
        raise FileNotFoundError(f"No RGB images found in: {rgb_dir}")

    return heatmaps, heatmaps_history, heatmaps_future, meta, intrinsics, rgb_paths


def resize_heatmap(heatmap: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize heatmap to target image size.

    Args:
        heatmap: Original heatmap (typically 64x64)
        target_size: Target size as (width, height)

    Returns:
        Resized heatmap
    """
    return cv2.resize(heatmap, target_size, interpolation=cv2.INTER_LINEAR)


def apply_colormap(heatmap: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Apply color mapping to heatmap.

    Args:
        heatmap: Normalized heatmap (values 0-1)
        colormap: OpenCV colormap constant

    Returns:
        Colored heatmap as BGR image
    """
    # Normalize to 0-255 range
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)

    # Apply colormap
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)

    return colored_heatmap


def overlay_heatmap(rgb_image: np.ndarray, heatmap: np.ndarray,
                   alpha: float = 0.5, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Overlay heatmap on RGB image.

    Args:
        rgb_image: RGB image (H, W, 3)
        heatmap: Heatmap array (64, 64) or matching RGB size
        alpha: Transparency of heatmap overlay (0-1)
        colormap: OpenCV colormap to use

    Returns:
        Overlaid image
    """
    # Get image dimensions
    h, w = rgb_image.shape[:2]

    # Resize heatmap if needed
    if heatmap.shape != (h, w):
        heatmap_resized = resize_heatmap(heatmap, (w, h))
    else:
        heatmap_resized = heatmap.copy()

    # Heatmaps are already normalized to 0-1 range in the dataset
    # No need to normalize again - just use them directly

    # Apply colormap to the heatmap (already in 0-1 range)
    colored_heatmap = apply_colormap(heatmap_resized, colormap)

    # Create a mask for significant heatmap values only
    # Use a threshold to show only meaningful hotspots
    threshold = 0.05  # Show only values above 5% (helps reduce noise)
    mask = heatmap_resized > threshold
    mask_3d = np.stack([mask, mask, mask], axis=-1)  # Expand to 3 channels

    # Blend entire images first
    blended = cv2.addWeighted(rgb_image, 1 - alpha, colored_heatmap, alpha, 0)

    # Use mask to selectively apply blended regions
    overlaid = np.where(mask_3d, blended, rgb_image)

    return overlaid


def overlay_dual_heatmaps(rgb_image: np.ndarray,
                         heatmap_history: np.ndarray,
                         heatmap_future: np.ndarray,
                         alpha: float = 0.5,
                         threshold: float = 0.05) -> np.ndarray:
    """
    Overlay both history and future heatmaps on RGB image with different colors.

    Args:
        rgb_image: RGB image (H, W, 3)
        heatmap_history: History heatmap (64, 64)
        heatmap_future: Future heatmap (64, 64)
        alpha: Transparency of heatmap overlay (0-1)
        threshold: Minimum value to display

    Returns:
        Overlaid image with history (cyan/blue) and future (red/hot)
    """
    h, w = rgb_image.shape[:2]

    # Resize both heatmaps
    if heatmap_history.shape != (h, w):
        heatmap_history_resized = resize_heatmap(heatmap_history, (w, h))
    else:
        heatmap_history_resized = heatmap_history.copy()

    if heatmap_future.shape != (h, w):
        heatmap_future_resized = resize_heatmap(heatmap_future, (w, h))
    else:
        heatmap_future_resized = heatmap_future.copy()

    # Apply colormaps (cyan for history, hot for future)
    colored_history = apply_colormap(heatmap_history_resized, cv2.COLORMAP_WINTER)
    colored_future = apply_colormap(heatmap_future_resized, cv2.COLORMAP_HOT)

    # Create masks
    mask_history = heatmap_history_resized > threshold
    mask_future = heatmap_future_resized > threshold

    # Start with RGB image
    result = rgb_image.copy()

    # Overlay future first (in background)
    if mask_future.any():
        mask_future_3d = np.stack([mask_future, mask_future, mask_future], axis=-1)
        blended_future = cv2.addWeighted(result, 1 - alpha, colored_future, alpha, 0)
        result = np.where(mask_future_3d, blended_future, result)

    # Overlay history on top
    if mask_history.any():
        mask_history_3d = np.stack([mask_history, mask_history, mask_history], axis=-1)
        blended_history = cv2.addWeighted(result, 1 - alpha, colored_history, alpha, 0)
        result = np.where(mask_history_3d, blended_history, result)

    return result


def create_visualization_grid(images: List[np.ndarray], max_cols: int = 3) -> np.ndarray:
    """
    Create a grid of images for display.

    Args:
        images: List of images to arrange in grid
        max_cols: Maximum number of columns

    Returns:
        Grid image
    """
    if not images:
        return None

    n_images = len(images)
    n_cols = min(n_images, max_cols)
    n_rows = (n_images + n_cols - 1) // n_cols

    # Get image dimensions (assuming all same size)
    h, w = images[0].shape[:2]

    # Create blank grid
    grid = np.zeros((h * n_rows, w * n_cols, 3), dtype=np.uint8)

    # Fill grid
    for idx, img in enumerate(images):
        row = idx // n_cols
        col = idx % n_cols
        grid[row*h:(row+1)*h, col*w:(col+1)*w] = img

    return grid


def visualize_clip(clip_path: Path, output_dir: Path, alpha: float = 0.5,
                  colormap_name: str = 'JET', save_individual: bool = False,
                  mode: str = 'both', max_frames: int = None) -> None:
    """
    Visualize heatmaps for a clip. Supports both old and new (split) formats.

    Args:
        clip_path: Path to clip directory
        output_dir: Output directory for visualizations
        alpha: Heatmap transparency
        colormap_name: Name of OpenCV colormap (for single heatmaps)
        save_individual: Whether to save individual overlaid images
        mode: 'history', 'future', or 'both' for split format
        max_frames: Maximum frames to visualize (default: all, or 20 for grid)
    """
    # Load data
    print(f"Loading data from: {clip_path}")
    heatmaps, heatmaps_history, heatmaps_future, meta, intrinsics, rgb_paths = load_clip_data(clip_path)

    num_frames = len(rgb_paths)
    is_split = heatmaps_history is not None

    # Determine format and setup
    if is_split:
        num_heatmaps = len(heatmaps_history)
        heatmap_type = meta.get('heatmap_type', 'split')
        valid_info = meta.get('valid_heatmaps', {})
        print(f"Found split heatmaps: {num_heatmaps} frames")
        print(f"Valid - history: {valid_info.get('history', 'N/A')}, future: {valid_info.get('future', 'N/A')}")
    else:
        num_heatmaps = len(heatmaps)
        heatmap_type = meta.get('heatmap_type', 'single')
        print(f"Found {num_heatmaps} heatmaps")

    print(f"Heatmap type: {heatmap_type}")
    print(f"RGB frames: {num_frames}")
    print(f"Instruction: {meta.get('instruction', 'N/A')}")

    # Get colormap for single mode
    colormap_mapping = {
        'JET': cv2.COLORMAP_JET,
        'HOT': cv2.COLORMAP_HOT,
        'VIRIDIS': cv2.COLORMAP_VIRIDIS,
        'PLASMA': cv2.COLORMAP_PLASMA,
        'INFERNO': cv2.COLORMAP_INFERNO,
        'MAGMA': cv2.COLORMAP_MAGMA,
    }
    colormap = colormap_mapping.get(colormap_name.upper(), cv2.COLORMAP_JET)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine frames to process
    if max_frames is None:
        max_frames = min(num_heatmaps, num_frames) if not save_individual else 20

    total_to_process = min(max_frames, num_heatmaps, num_frames)
    frame_indices = np.linspace(0, min(num_heatmaps, num_frames)-1, total_to_process, dtype=int)

    # Process frames
    if is_split and mode == 'both':
        # Generate separate visualizations for history and future
        overlaid_history = []
        overlaid_future = []

        for idx in frame_indices:
            i = int(idx)
            print(f"Processing frame {i}/{num_frames}")

            # Load corresponding RGB image
            rgb_image = cv2.imread(str(rgb_paths[i]))
            if rgb_image is None:
                print(f"Error loading image: {rgb_paths[i]}")
                continue

            # Create history overlay
            overlay_hist = overlay_heatmap(rgb_image, heatmaps_history[i], alpha, cv2.COLORMAP_WINTER)
            text_hist = f"Frame {i} - History"
            cv2.putText(overlay_hist, text_hist, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(overlay_hist, text_hist, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 0, 0), 1, cv2.LINE_AA)
            overlaid_history.append(overlay_hist)

            # Create future overlay
            overlay_fut = overlay_heatmap(rgb_image, heatmaps_future[i], alpha, cv2.COLORMAP_HOT)
            text_fut = f"Frame {i} - Future"
            cv2.putText(overlay_fut, text_fut, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(overlay_fut, text_fut, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 0, 0), 1, cv2.LINE_AA)
            overlaid_future.append(overlay_fut)

            # Save individual images if requested
            if save_individual:
                output_path_hist = output_dir / f"frame_{i:06d}_history.png"
                output_path_fut = output_dir / f"frame_{i:06d}_future.png"
                cv2.imwrite(str(output_path_hist), overlay_hist)
                cv2.imwrite(str(output_path_fut), overlay_fut)

        # Save separate grids
        if overlaid_history:
            grid_hist = create_visualization_grid(overlaid_history, max_cols=3)
            grid_path_hist = output_dir / "heatmaps_grid_history.png"
            cv2.imwrite(str(grid_path_hist), grid_hist)
            print(f"\nSaved history grid: {grid_path_hist}")

        if overlaid_future:
            grid_fut = create_visualization_grid(overlaid_future, max_cols=3)
            grid_path_fut = output_dir / "heatmaps_grid_future.png"
            cv2.imwrite(str(grid_path_fut), grid_fut)
            print(f"Saved future grid: {grid_path_fut}")
            print(f"Total frames processed: {len(overlaid_future)}")

    else:
        # Single mode visualization
        overlaid_images = []
        for idx in frame_indices:
            i = int(idx)
            print(f"Processing frame {i}/{num_frames}")

            # Load corresponding RGB image
            rgb_image = cv2.imread(str(rgb_paths[i]))
            if rgb_image is None:
                print(f"Error loading image: {rgb_paths[i]}")
                continue

            # Create overlay based on format and mode
            if is_split:
                if mode == 'history':
                    overlaid = overlay_heatmap(rgb_image, heatmaps_history[i], alpha, cv2.COLORMAP_WINTER)
                    text = f"Frame {i} - History"
                else:  # future
                    overlaid = overlay_heatmap(rgb_image, heatmaps_future[i], alpha, cv2.COLORMAP_HOT)
                    text = f"Frame {i} - Future"
            else:
                overlaid = overlay_heatmap(rgb_image, heatmaps[i], alpha, colormap)
                text = f"Frame {i}"

            # Add text annotation
            cv2.putText(overlaid, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(overlaid, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 0, 0), 1, cv2.LINE_AA)

            overlaid_images.append(overlaid)

            # Save individual image if requested
            if save_individual:
                suffix = f"_{mode}" if is_split else ""
                output_path = output_dir / f"frame_{i:06d}{suffix}.png"
                cv2.imwrite(str(output_path), overlaid)

        # Create and save grid visualization
        if overlaid_images:
            grid = create_visualization_grid(overlaid_images, max_cols=3)
            suffix = f"_{mode}" if is_split else ""
            grid_path = output_dir / f"heatmaps_grid{suffix}.png"
            cv2.imwrite(str(grid_path), grid)
            print(f"\nSaved grid visualization: {grid_path}")
            print(f"Total frames processed: {len(overlaid_images)}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize navigation goal heatmaps overlaid on RGB images"
    )
    parser.add_argument(
        '--clip_path', type=str, required=True,
        help='Path to clip directory containing heatmaps.npy and rgb/ folder'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./visualizations',
        help='Output directory for visualizations (default: ./visualizations)'
    )
    parser.add_argument(
        '--alpha', type=float, default=0.5,
        help='Heatmap transparency (0-1, default: 0.5)'
    )
    parser.add_argument(
        '--colormap', type=str, default='JET',
        choices=['JET', 'HOT', 'VIRIDIS', 'PLASMA', 'INFERNO', 'MAGMA'],
        help='OpenCV colormap to use (default: JET)'
    )
    parser.add_argument(
        '--save_individual', action='store_true',
        help='Save each overlaid image individually'
    )
    parser.add_argument(
        '--mode', type=str, default='both',
        choices=['history', 'future', 'both'],
        help='For split heatmaps: history, future, or both (default: both)'
    )
    parser.add_argument(
        '--max_frames', type=int, default=None,
        help='Maximum number of frames to visualize in grid (default: 20)'
    )

    args = parser.parse_args()

    # Convert paths
    clip_path = Path(args.clip_path)
    output_dir = Path(args.output_dir)

    # Validate clip path
    if not clip_path.exists():
        print(f"Error: Clip path does not exist: {clip_path}")
        return 1

    # Run visualization
    try:
        visualize_clip(
            clip_path=clip_path,
            output_dir=output_dir,
            alpha=args.alpha,
            colormap_name=args.colormap,
            save_individual=args.save_individual,
            mode=args.mode,
            max_frames=args.max_frames
        )
        print("\n✓ Visualization complete!")
        return 0
    except Exception as e:
        print(f"\n✗ Error during visualization: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
