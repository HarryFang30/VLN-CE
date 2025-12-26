#!/usr/bin/env python3
"""
Batch visualize multiple clips from the dataset.
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import List
import sys


def get_all_clips(train_dir: Path, max_clips: int = None) -> List[Path]:
    """
    Get all clip directories from the training set.

    Args:
        train_dir: Path to train directory
        max_clips: Maximum number of clips to process (None for all)

    Returns:
        List of clip directory paths
    """
    clips = []

    # Iterate through scene directories
    for scene_dir in sorted(train_dir.iterdir()):
        if not scene_dir.is_dir():
            continue

        # Iterate through clip directories
        for clip_dir in sorted(scene_dir.iterdir()):
            if not clip_dir.is_dir() or not clip_dir.name.startswith('clip_'):
                continue

            # Check if heatmaps exist (either old or new format)
            has_heatmaps = (clip_dir / 'heatmaps.npy').exists() or \
                          ((clip_dir / 'heatmaps_history.npy').exists() and \
                           (clip_dir / 'heatmaps_future.npy').exists())

            if has_heatmaps:
                clips.append(clip_dir)

                if max_clips and len(clips) >= max_clips:
                    return clips

    return clips


def visualize_clip_wrapper(clip_path: Path, output_base: Path,
                          alpha: float, colormap: str) -> bool:
    """
    Wrapper to call visualize_heatmaps.py for a single clip.

    Args:
        clip_path: Path to clip directory
        output_base: Base output directory
        alpha: Heatmap transparency
        colormap: Colormap name

    Returns:
        True if successful, False otherwise
    """
    # Create output directory name from clip path
    scene_name = clip_path.parent.name
    clip_name = clip_path.name
    output_dir = output_base / scene_name / clip_name

    # Build command
    heatmaps_script = (Path(__file__).resolve().parent / 'visualize_heatmaps.py')
    cmd = [
        sys.executable,  # Use same Python interpreter
        str(heatmaps_script),
        '--clip_path', str(clip_path),
        '--output_dir', str(output_dir),
        '--alpha', str(alpha),
        '--colormap', colormap
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout per clip
        )

        if result.returncode == 0:
            return True
        else:
            print(f"  Error: {result.stderr.strip()[:200]}")
            return False

    except subprocess.TimeoutExpired:
        print(f"  Timeout after 60 seconds")
        return False
    except Exception as e:
        print(f"  Exception: {e}")
        return False


def create_index_html(output_dir: Path, clips_info: List[dict]) -> None:
    """
    Create an HTML index page to browse all visualizations.

    Args:
        output_dir: Output directory
        clips_info: List of dicts with clip information
    """
    html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Heatmap Visualizations</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
        }
        .summary {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .clip {
            background: white;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .clip h3 {
            margin-top: 0;
            color: #2c3e50;
        }
        .instruction {
            font-style: italic;
            color: #555;
            margin: 10px 0;
            padding: 10px;
            background: #f8f9fa;
            border-left: 3px solid #007bff;
        }
        .meta {
            color: #666;
            font-size: 0.9em;
            margin: 5px 0;
        }
        .success {
            color: green;
        }
        .failed {
            color: red;
        }
        img {
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin-top: 10px;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }
        .stat-box {
            background: #e9ecef;
            padding: 10px;
            border-radius: 4px;
        }
        .stat-label {
            font-weight: bold;
            color: #495057;
        }
    </style>
</head>
<body>
    <h1>🗺️ Heatmap Visualizations</h1>
"""

    # Add summary
    total = len(clips_info)
    successful = sum(1 for c in clips_info if c['success'])
    failed = total - successful

    html_content += f"""
    <div class="summary">
        <h2>Summary</h2>
        <div class="stats">
            <div class="stat-box">
                <div class="stat-label">Total Clips</div>
                <div>{total}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Successful</div>
                <div class="success">{successful}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Failed</div>
                <div class="failed">{failed}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Success Rate</div>
                <div>{successful/total*100:.1f}%</div>
            </div>
        </div>
    </div>
"""

    # Add each clip
    for i, clip in enumerate(clips_info, 1):
        status = "✓ Success" if clip['success'] else "✗ Failed"
        status_class = "success" if clip['success'] else "failed"

        html_content += f"""
    <div class="clip">
        <h3>#{i}: {clip['scene_name']} / {clip['clip_name']}</h3>
        <div class="meta"><strong>Status:</strong> <span class="{status_class}">{status}</span></div>
        <div class="meta"><strong>Path:</strong> {clip['path']}</div>
"""

        if clip.get('instruction'):
            html_content += f"""
        <div class="instruction">📝 {clip['instruction']}</div>
"""

        if clip.get('num_heatmaps'):
            html_content += f"""
        <div class="meta"><strong>Heatmaps:</strong> {clip['num_heatmaps']}</div>
"""

        if clip['success'] and clip.get('image_path'):
            # Make relative path
            rel_path = Path(clip['image_path']).relative_to(output_dir)
            html_content += f"""
        <img src="{rel_path.as_posix()}" alt="Heatmap visualization">
"""

        html_content += """
    </div>
"""

    html_content += """
</body>
</html>
"""

    # Write HTML file
    index_path = output_dir / 'index.html'
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\n📄 Created index page: {index_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch visualize multiple clips from the dataset"
    )
    parser.add_argument(
        '--train_dir', type=str, default='./train',
        help='Path to train directory (default: ./train)'
    )
    parser.add_argument(
        '--num_clips', type=int, default=100,
        help='Number of clips to visualize (default: 100)'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./batch_visualizations',
        help='Output directory (default: ./batch_visualizations)'
    )
    parser.add_argument(
        '--alpha', type=float, default=0.5,
        help='Heatmap transparency (default: 0.5)'
    )
    parser.add_argument(
        '--colormap', type=str, default='JET',
        choices=['JET', 'HOT', 'VIRIDIS', 'PLASMA', 'INFERNO', 'MAGMA'],
        help='Colormap to use (default: JET)'
    )

    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    output_dir = Path(args.output_dir)

    if not train_dir.exists():
        print(f"Error: Train directory not found: {train_dir}")
        return 1

    # Get clips
    print(f"🔍 Scanning for clips in {train_dir}...")
    clips = get_all_clips(train_dir, max_clips=args.num_clips)
    print(f"Found {len(clips)} clips to process\n")

    if not clips:
        print("No clips found!")
        return 1

    # Process each clip
    clips_info = []
    successful = 0
    failed = 0

    for i, clip_path in enumerate(clips, 1):
        scene_name = clip_path.parent.name
        clip_name = clip_path.name

        print(f"[{i}/{len(clips)}] Processing {scene_name}/{clip_name}...")

        # Load metadata for instruction
        instruction = None
        num_heatmaps = None
        try:
            meta_path = clip_path / 'meta.json'
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                    instruction = meta.get('instruction', '')
                    num_heatmaps = meta.get('num_heatmaps', 0)
        except:
            pass

        # Visualize
        success = visualize_clip_wrapper(
            clip_path, output_dir, args.alpha, args.colormap
        )

        if success:
            successful += 1
            print(f"  ✓ Success")
            image_path = output_dir / scene_name / clip_name / 'heatmaps_grid.png'
        else:
            failed += 1
            print(f"  ✗ Failed")
            image_path = None

        # Record info
        clips_info.append({
            'scene_name': scene_name,
            'clip_name': clip_name,
            'path': str(clip_path.relative_to(train_dir.parent)),
            'instruction': instruction,
            'num_heatmaps': num_heatmaps,
            'success': success,
            'image_path': str(image_path) if image_path else None
        })

    # Create index HTML
    create_index_html(output_dir, clips_info)

    # Print summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    print(f"Total clips processed: {len(clips)}")
    print(f"Successful: {successful} ({successful/len(clips)*100:.1f}%)")
    print(f"Failed: {failed} ({failed/len(clips)*100:.1f}%)")
    print(f"\nOutput directory: {output_dir}")
    print(f"Index page: {output_dir}/index.html")
    print("="*60)

    return 0


if __name__ == '__main__':
    exit(main())
