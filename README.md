# VLN-CE Data Collection Toolkit

A high-quality data collection pipeline for Vision-Language Navigation in Continuous Environments (VLN-CE), built on Habitat-Sim and Matterport3D.

## Overview

This toolkit provides scripts for collecting expert navigation trajectories with corresponding visual observations, camera poses, and action labels. The collected data is designed for training navigation models with **history-aware heatmap generation**.

### Key Features

- **Pinhole Camera Model**: Standard perspective projection (640×480 RGB, 640×480 Depth)
- **Complete Trajectory Data**: RGB, Depth, Camera Poses, Actions (continuous + discrete)
- **Dynamic Heatmap Generation**: History heatmaps computed during training (not pre-saved)
- **Resume Support**: Automatic checkpoint-based resumption for interrupted collection
- **Quality Control**: Multi-layer validation with configurable thresholds

## Repository Structure

```
VLN-CE/
├── collect_heatmap.py       # Main data collection script
├── visualize_clips.py       # Visualization tool for collected data
├── analyze.py               # Data quality analysis
├── habitat_extensions/      # Habitat-Sim extensions
│   └── config/
│       └── vlnce_collect.yaml
├── data/                    # Symlinks to external datasets
│   ├── datasets/
│   │   └── R2R_VLNCE_v1-3_preprocessed/
│   └── scene_datasets/
│       └── mp3d/
└── README.md
```

## Quick Start

### 1. Environment Setup

```bash
# Ensure Habitat-Sim is installed
pip install habitat-sim habitat-lab

# Create symlinks to external data
mkdir -p data/datasets data/scene_datasets
ln -snf /path/to/R2R_VLNCE_v1-3_preprocessed data/datasets/R2R_VLNCE_v1-3_preprocessed
ln -snf /path/to/mp3d data/scene_datasets/mp3d

# Verify setup
test -f data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz && echo "Dataset OK"
test -d data/scene_datasets/mp3d && echo "MP3D OK"
```

### 2. Run Data Collection

```bash
python collect_heatmap.py \
  --config habitat_extensions/config/vlnce_collect.yaml \
  --output /path/to/output_dataset \
  --split train \
  --num-clips 2000 \
  --max-steps 100 \
  --gpu 0
```

### 3. Background Execution (Recommended)

```bash
# Using tmux
tmux new-session -d -s collect "python collect_heatmap.py \
  --config habitat_extensions/config/vlnce_collect.yaml \
  --output /path/to/output_dataset \
  --split train \
  --num-clips 2000 \
  --gpu 0"

# Monitor progress
tmux attach -t collect

# Or using nohup
nohup python collect_heatmap.py --output /path/to/output --split train --num-clips 2000 --gpu 0 \
  > collection.log 2>&1 &
```

## Output Data Format

```
/path/to/output_dataset/
├── <scene_name>/
│   └── clip_XXXXXX/
│       ├── rgb/                    # RGB images (JPEG, 640×480)
│       │   ├── 000000.jpg
│       │   └── ...
│       ├── depth/                  # Depth maps (float16 NPY, 640×480)
│       │   ├── 000000.npy
│       │   └── ...
│       ├── poses.json              # Camera poses [T, 4, 4] (T_world_camera)
│       ├── intrinsics.json         # Camera intrinsics (Pinhole)
│       ├── actions.npy             # Continuous actions [T, 2] (dx, dy)
│       ├── discrete_actions.npy    # Discrete actions [T] (0-3)
│       ├── meta.json               # Episode metadata
│       ├── topdown_trajectory.jpg  # Top-down visualization
│       ├── topdown_transform.json  # Top-down map transform
│       └── trajectory_3d.npy       # 3D trajectory points
├── train/                          # Symlink for train split
├── val/                            # Symlink for val split
└── split_info.json                 # Train/val scene split info
```

### Data Fields

| File | Format | Description |
|------|--------|-------------|
| `rgb/*.jpg` | JPEG (quality=95) | First-person RGB observations |
| `depth/*.npy` | float16, normalized [0,1] | Depth maps (max_depth=10m) |
| `poses.json` | List[4×4 matrix] | Camera-to-world transforms |
| `intrinsics.json` | {fx, fy, cx, cy, width, height} | Pinhole camera parameters |
| `actions.npy` | float32 [T, 2] | Agent-local displacement (dx, dy) |
| `discrete_actions.npy` | int32 [T] | Action type (0=STOP, 1=FWD, 2=LEFT, 3=RIGHT) |

### Action Semantics

```
action[i] = movement from frame[i] to frame[i+1]
```

- **Continuous action** `(dx, dy)`: Agent-local 2D displacement
  - `dx`: lateral movement (right = positive)
  - `dy`: forward movement (forward = positive)
- **Discrete action**: Habitat action index
  - `0`: STOP, `1`: MOVE_FORWARD, `2`: TURN_LEFT, `3`: TURN_RIGHT

## Visualization

Visualize collected clips with dynamically generated heatmaps:

```bash
python visualize_clips.py \
  --input /path/to/output_dataset \
  --output visualize \
  --random 10 \
  --seed 42 \
  --interval 3
```

### Output

Each visualization page contains:
- **RGB**: Current frame observation
- **Depth**: Depth map (colorized)
- **Heatmap**: History camera positions projected to current view
- **Top-down**: Bird's-eye trajectory (red=history, green=current)

---

## History Heatmap Generation

The heatmap represents "where I have been" from the current camera's perspective. It is computed dynamically during training, not pre-saved.

### Algorithm Pipeline

```
History Poses → Coordinate Transform → Pinhole Projection → Occlusion Check 
    → Adaptive Sigma → Distance Decay → Max Merge → Heatmap [0,1]
```

### 1. Pinhole Projection

Project historical camera centers to the current image plane:

```python
def project_point_pinhole(p_cam, K, width, height):
    """
    Project a 3D point (in camera coordinates) to image plane.
    
    Args:
        p_cam: [4,] homogeneous coordinates in camera frame
        K: [3,3] intrinsic matrix
        width, height: image dimensions
    
    Returns:
        (u, v, z_depth) or None if behind camera or out of bounds
    """
    x, y, z = p_cam[:3]
    if z <= 0.1:
        return None
    
    u = K[0,0] * x / z + K[0,2]
    v = K[1,1] * y / z + K[1,2]
    
    if not (0 <= u < width and 0 <= v < height):
        return None
    
    return (u, v, z)
```

### 2. Occlusion Detection

Filter occluded history points using depth buffer:

```python
def is_occluded(u, v, z_depth, depth_image, tolerance=0.5):
    """
    Check if the projected point is occluded by scene geometry.
    
    Principle:
    - observed_depth: depth at pixel (u, v) in current frame
    - z_depth: actual distance to history camera center
    - If observed_depth < z_depth - tolerance → occluded
    """
    observed_depth = depth_image[int(v), int(u)] * 10.0  # denormalize
    return observed_depth > 0 and observed_depth < z_depth - tolerance
```

**Tolerance Parameter** (`occlusion_tolerance`):
- Default: 0.5m
- Too small → visible points incorrectly marked as occluded
- Too large → occluded points incorrectly shown

### 3. Adaptive Sigma (Perspective Scaling)

Gaussian kernel size scales with depth (near = large, far = small):

```python
def compute_adaptive_sigma(z_depth, fx, hm_width=64, img_width=640,
                           object_size=0.5, min_sigma=1.5, max_sigma=6.0):
    """
    Compute adaptive Gaussian sigma based on projected size.
    
    Principle:
    - projected_size = object_size * focal_length / depth
    - sigma = projected_size / 3 (3-sigma rule)
    """
    scale = img_width / hm_width
    fx_hm = fx / scale
    projected_size = object_size * fx_hm / max(z_depth, 0.1)
    sigma = np.clip(projected_size / 3.0, min_sigma, max_sigma)
    return sigma
```

| Distance (m) | Projected Size (px) | Sigma (px) |
|--------------|---------------------|------------|
| 1.0 | 16.0 | 5.3 |
| 3.0 | 5.3 | 1.8 |
| 5.0 | 3.2 | 1.5 (min) |
| 10.0 | 1.6 | 1.5 (min) |

### 4. Distance Decay

Closer history frames appear brighter:

```python
def compute_peak_value(distance, decay_ref=5.0, min_peak=0.3):
    """
    Compute distance-decayed peak value.
    
    Formula: peak = min_peak + (1 - min_peak) / (1 + distance / decay_ref)
    """
    decay = 1.0 / (1.0 + distance / decay_ref)
    return min_peak + (1.0 - min_peak) * decay
```

| Distance (m) | Peak Value |
|--------------|------------|
| 0.0 | 1.00 |
| 2.5 | 0.77 |
| 5.0 | 0.65 |
| 10.0 | 0.53 |
| 15.0 | 0.48 |

### 5. Max Merge (Overlap Handling)

Use element-wise maximum instead of addition for overlapping Gaussians:

```python
# Max merge: stable [0,1] range, no saturation
np.maximum(heatmap_roi, gaussian_blob, out=heatmap_roi)

# vs. Addition (deprecated): unbounded, requires normalization
# np.add(heatmap_roi, gaussian_blob, out=heatmap_roi)
```

**Advantages of Max Merge:**
- Stable value range [0, 1]
- No saturation in high-overlap regions
- Clear semantics: brightness = proximity of closest history frame

### 6. Complete Algorithm

```python
def compute_history_heatmap(
    history_poses: List[np.ndarray],    # [N, 4, 4] T_world_camera
    current_pose: np.ndarray,           # [4, 4] T_world_camera
    current_depth: np.ndarray,          # [H, W] normalized depth
    K: np.ndarray,                       # [3, 3] intrinsic matrix
    hm_size: Tuple[int, int] = (64, 64),
    img_size: Tuple[int, int] = (640, 480),
    occlusion_tolerance: float = 0.5,
    max_visible_distance: float = 15.0,
    use_distance_decay: bool = True,
    distance_decay_ref: float = 5.0,
    min_peak_value: float = 0.3,
) -> np.ndarray:
    """Generate history heatmap using Pinhole projection."""
    
    heatmap = np.zeros(hm_size, dtype=np.float32)
    T_inv = np.linalg.inv(current_pose)
    
    for pose in history_poses:
        # Transform to current camera frame
        p_world = np.array([*pose[:3, 3], 1.0])
        p_cam = T_inv @ p_world
        distance = np.linalg.norm(p_cam[:3])
        
        # Distance filter
        if distance < 0.01 or distance > max_visible_distance:
            continue
        
        # Project to image
        proj = project_point_pinhole(p_cam, K, *img_size)
        if proj is None:
            continue
        u, v, z = proj
        
        # Occlusion check
        if is_occluded(u, v, z, current_depth, occlusion_tolerance):
            continue
        
        # Adaptive sigma
        sigma = compute_adaptive_sigma(z, K[0,0], hm_size[1], img_size[0])
        
        # Distance decay
        peak = compute_peak_value(distance, distance_decay_ref, min_peak_value) \
               if use_distance_decay else 1.0
        
        # Draw Gaussian (max merge)
        u_hm = u * hm_size[1] / img_size[0]
        v_hm = v * hm_size[0] / img_size[1]
        draw_gaussian_max(heatmap, (u_hm, v_hm), sigma, peak)
    
    return heatmap  # [0, 1], no normalization needed
```

### 7. Configuration Parameters

```yaml
# HeatmapVLN/configs/train_heatmap_config.yaml
data:
  sliding_window:
    num_history_sample: 32        # Number of history frames to sample
    occlusion_tolerance: 0.5      # Occlusion detection margin (m)
    max_visible_distance: 15.0    # Maximum projection distance (m)
    min_sigma: 1.5                # Minimum Gaussian sigma (heatmap px)
    max_sigma: 6.0                # Maximum Gaussian sigma (heatmap px)
    object_size_3d: 0.5           # Virtual object size for projection (m)
    use_max_merge: true           # Use max instead of add
    use_distance_decay: true      # Enable distance-based peak decay
    distance_decay_ref: 5.0       # Distance decay reference (m)
    min_peak_value: 0.3           # Minimum peak value at max distance
```

---

## Collection Statistics

| Configuration | Episodes | Clips | Time | Disk |
|---------------|----------|-------|------|------|
| Test | 5,000 | ~1,000 | 2.8h | 8GB |
| Full | 10,819 | ~2,100 | 6h | 17GB |

**R2R Train Set**: 10,819 episodes, ~20% success rate → ~2,100 valid clips

## Monitoring & Troubleshooting

```bash
# Check progress
find /path/to/output -name "meta.json" | wc -l

# View latest logs
tmux attach -t collect

# Resume interrupted collection (automatic)
python collect_heatmap.py --output /path/to/output --split train --num-clips 2000 --gpu 0

# Check disk usage
du -sh /path/to/output
```

## License

This project builds upon VLN-CE and Habitat-Lab. Please refer to their respective licenses.

## Citation

If you use this toolkit, please cite:

```bibtex
@inproceedings{krantz2020navgraph,
  title={Beyond the Nav-Graph: Vision-and-Language Navigation in Continuous Environments},
  author={Krantz, Jacob and Wijmans, Erik and Majumdar, Arjun and Batra, Dhruv and Lee, Stefan},
  booktitle={ECCV},
  year={2020}
}
```
