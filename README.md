# VLN-CE 高质量数据采集工具

## 核心算法

**双向导航 + 多帧融合 + 四层过滤 → 97.2%高质量数据**

1. Agent执行 起点→终点(Forward) + 终点→起点(Backward)
2. 每个参考点从所有帧中选择最佳视角生成热力图
3. 四层过滤：总帧≥5、双向≥5、有效率≥70%、完整性验证

## collect.py 的用处（你会得到什么）

`collect.py` 是本仓库的**数据采集入口**，用于在 Habitat-Sim / Matterport3D 场景中，根据 VLN-CE（R2R）数据集 episode：

- **自动走最短路径**从起点前往目标（walk-to-goal）
- **逐帧保存观测**：RGB（png）+ Depth（npy）
- **保存几何与动作监督**：
  - `poses.json`：每帧 4×4 位姿
  - `intrinsics.json`：相机内参（此版本为 equirectangular）
  - `discrete_actions.npy`：离散动作（0=STOP, 1=FORWARD, 2=LEFT, 3=RIGHT）
  - `actions.npy`：2D 连续动作 (dx, dy)，由相邻位姿推导
- **断点续采**：自动写入 `progress.json`，下次运行可继续
- **统计与索引**：`collection_stats.json` + `<split>_index.json`

> 备注：当前 `collect.py` 会提示“热力图将在训练时动态计算”，因此**采集输出不再包含 `heatmaps.npy/mask.npy`**（与旧版 README 不同）。

## 快速开始（推荐命令行方式）

### 1) 准备外部数据（软链接/路径）

默认配置（`habitat_extensions/config/vlnce_collect.yaml`）期望以下路径存在：

- `data/datasets/R2R_VLNCE_v1-3_preprocessed/...`：R2R VLN-CE 预处理数据
- `data/scene_datasets/mp3d/...`：Matterport3D 场景（`.glb` 等）
- `data/connectivity_graphs.pkl`：连通图（本仓库默认已放在 `data/` 下；若你放在外部，也可以软链接）

如果你的数据在外部磁盘/共享目录，最稳妥做法是通过软链接把它们“挂载”到仓库期望的位置：

```bash
cd /path/to/VLN-CE

mkdir -p data/datasets data/scene_datasets

# 1) R2R VLN-CE 数据集（把 /your/path/... 换成你的实际位置）
ln -snf /your/path/R2R_VLNCE_v1-3_preprocessed data/datasets/R2R_VLNCE_v1-3_preprocessed

# 2) Matterport3D 场景
ln -snf /your/path/mp3d data/scene_datasets/mp3d

# 3)（可选）连通图
# ln -snf /your/path/connectivity_graphs.pkl data/connectivity_graphs.pkl

# 快速校验（至少保证 train split 存在）
test -f data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz && echo "✅ dataset ok"
test -d data/scene_datasets/mp3d && echo "✅ mp3d ok"
```

### 2) 运行采集

`collect.py` 已提供 CLI 参数，不需要再去改脚本内部行号/常量：

```bash
python collect.py \
  --config habitat_extensions/config/vlnce_collect.yaml \
  --output /path/to/output_dataset \
  --split train \
  --num-clips 1000 \
  --max-steps 100 \
  --num-workers 16 \
  --gpu 0
```

常见运行方式：

- **后台运行**：

```bash
nohup python collect.py --config habitat_extensions/config/vlnce_collect.yaml --output /path/to/output_dataset --split train --num-clips 1000 --gpu 0 \
  > collection.log 2>&1 &
echo $! > collection.pid
tail -f collection.log
```

- **CPU 跑（极慢）**：把 `--gpu -1`，或在配置里设置 `SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID: -1`

## vlnce_collect.yaml 配置说明（重点）

配置文件：`habitat_extensions/config/vlnce_collect.yaml`

- **ENVIRONMENT.MAX_EPISODE_STEPS**：每个 episode 最大步数上限（防止卡死）
- **SIMULATOR.FORWARD_STEP_SIZE / TURN_ANGLE**：离散动作的步长与转角（影响轨迹密度）
- **SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID**：GPU ID（`collect.py` 会用 `--gpu` 覆盖这个值）
- **SIMULATOR.RGB_SENSOR / DEPTH_SENSOR**：
  - 当前使用 **EQUIRECTANGULAR**（等距柱状全景）传感器，分辨率 `640×480`
  - Depth 提供 `MIN_DEPTH/MAX_DEPTH`，`collect.py` 会做一次 sanity check 决定是否需要 normalize
- **TASK.SUCCESS_DISTANCE / SPL.SUCCESS_DISTANCE**：成功阈值（米）
- **DATASET.DATA_PATH / DATASET.SCENES_DIR**（最重要）：
  - `DATA_PATH: data/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz`
  - `SCENES_DIR: data/scene_datasets/`
  - 如果你没使用软链接、或目录结构不同，就需要改这两项

## 输出数据结构（collect.py 实际产物）

```text
/path/to/output_dataset/
├── progress.json                  # 断点续采用
├── collection_stats.json          # 统计（成功/失败/动作分布等）
├── train_index.json               # 采集到的 clip 元数据索引
└── train/
    └── <scene_name>/
        └── clip_000001/
            ├── rgb/               # 逐帧 RGB（png）
            ├── depth/             # 逐帧 Depth（npy）
            ├── poses.json         # [T,4,4] 位姿序列（json 存储）
            ├── intrinsics.json    # 相机内参（equirectangular）
            ├── actions.npy        # [T,2] 连续动作 (dx,dy)
            ├── discrete_actions.npy # [T] 离散动作（HabitatSimActions）
            └── meta.json          # 指令、reference_path、关键帧匹配、动作统计等
```

## 关键参数

| 参数 | 文件 | 行号 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--config` | collect.py | - | habitat_extensions/config/vlnce_collect.yaml | Habitat 配置文件 |
| `--output` | collect.py | - | /root/autodl-tmp/dataset_with_actions | 输出根目录 |
| `--split` | collect.py | - | train | 数据划分 |
| `--num-clips` | collect.py | - | 1000 | 目标 clip 数 |
| `--max-steps` | collect.py | - | 100 | 每个方向最大步数（上限） |
| `--num-workers` | collect.py | - | 16 | I/O 线程数 |
| `--gpu` | collect.py | - | 0 | GPU 设备 ID（会覆盖 yaml） |
| `DATA_PATH` | vlnce_collect.yaml | - | data/datasets/R2R_VLNCE_v1-3_preprocessed/... | R2R 数据集 |
| `SCENES_DIR` | vlnce_collect.yaml | - | data/scene_datasets/ | 场景数据 |

## 数据质量指标

> 以运行结束时生成的 `collection_stats.json` 为准（成功/失败、动作分布、场景分布、缺失字段统计等都会写入）。

## 采集规模参考

| 配置 | episodes | clips | 耗时 | 磁盘 |
|------|---------|-------|------|------|
| 测试 | 5000 | ~1000 | 2.8h | 8GB |
| 完整 | 10819 | ~2156 | 6h | 17GB |

**R2R训练集总规模**: 10,819 episodes，基于19.9%成功率最多可采2156条

## 监控与故障处理

**查看进度**:
```bash
# 查看进程
ps aux | grep collect.py

# 查看已采集数量（把 OUTPUT 改成你的 --output）
OUTPUT=/path/to/output_dataset
ls -d "$OUTPUT"/train/*/clip_* 2>/dev/null | wc -l

# 查看最新日志
tail -20 collection.log
```

**停止采集**:
```bash
kill $(cat collection.pid)
```

**采集中断**（支持断点续传）:
```bash
# 直接用同样的命令重跑即可：只要 --output 下存在 progress.json，就会从上次位置继续
python collect.py --config habitat_extensions/config/vlnce_collect.yaml --output /path/to/output_dataset --split train --num-clips 1000 --gpu 0
```

**检查磁盘**:
```bash
df -h .  # 每个clip约8MB
```

## 核心文件

- `collect.py` - 数据采集（包含所有质量控制）
- `analyze.py` - 质量分析（基础+详细+验证 三合一）
- `habitat_extensions/` - VLN-CE任务扩展（必需）
  - `config/vlnce_collect.yaml` - 采集用配置（本 README 重点）
  - `config/vlnce_task.yaml` - 训练/评估用配置（包含更多 task sensors/measurements）
  - `config/default.py` - 配置加载器
- `README.md` - 本文档

## 默认路径结构（建议按此组织/软链接）

```
VLN-CE/
├── collect.py
├── analyze.py
├── habitat_extensions/
│   └── config/
│       ├── vlnce_collect.yaml       # 采集配置
│       └── vlnce_task.yaml          # 训练/评估配置
├── data/
│   ├── datasets/
│   │   └── R2R_VLNCE_v1-3_preprocessed/
│   │       └── train/
│   │           ├── train.json.gz
│   │           └── train_gt.json.gz
│   └── scene_datasets/
│       └── mp3d/                     # Matterport3D场景
└── /path/to/output_dataset/          # 采集输出目录（由 --output 指定）
    └── train/
```

## 完整工作流

```bash
# 1) 软链接外部数据（按需）
mkdir -p data/datasets data/scene_datasets
ln -snf /your/path/R2R_VLNCE_v1-3_preprocessed data/datasets/R2R_VLNCE_v1-3_preprocessed
ln -snf /your/path/mp3d data/scene_datasets/mp3d

# 2) 运行采集
python collect.py --config habitat_extensions/config/vlnce_collect.yaml --output /path/to/output_dataset --split train --num-clips 1000 --gpu 0

# 3) 完成后分析（可选）
python analyze.py
```

---

## 数据采集流程与动作语义（重要）

### 采集策略

`collect.py` 的核心采集流程：

1. **环境初始化**
   - 加载 Habitat-Sim 环境和 VLN-CE 数据集
   - 初始化 `ShortestPathFollower` 用于生成最优路径
   - 配置 RGB 和 Depth 传感器（Equirectangular 全景图）

2. **轨迹采集**
   - 对每个 episode，从起点出发，使用 `ShortestPathFollower` 导航到目标
   - 每执行一个动作前记录当前帧和动作
   - 动作执行后记录下一帧
   - 直到到达目标或超过最大步数

3. **动作记录**
   - **离散动作**：直接从 `ShortestPathFollower.get_next_action()` 获取
   - **连续动作**：从相邻帧的位姿矩阵差分计算 2D 位移

### 动作数据语义（核心）

#### 动作索引约定

**重要**：所有动作数据遵循统一的时序语义：

```
action[i] = 从 frame[i] 到 frame[i+1] 的动作
```

这意味着：
- 给定当前帧 `frame[i]` 和历史观测
- 模型应该预测 `action[i]`
- 执行 `action[i]` 后到达 `frame[i+1]`

#### 1. 连续动作 (`actions.npy`)

**文件格式**：
- 形状：`[T, 2]` float32 数组
- T = clip 的总帧数

**数据内容**：
- `action[i] = [dx, dy]`：从 `frame[i]` 到 `frame[i+1]` 的 agent-local 2D 位移
  - `dx`：X 方向位移（agent 坐标系，右为正）
  - `dy`：Z 方向位移（agent 坐标系，前为正）

**计算方式**：
```python
def compute_2d_action_from_poses(pose_before, pose_after):
    """从相邻位姿计算 2D 连续动作"""
    # 计算相对变换
    T_rel = inv(pose_before) @ pose_after
    
    # 提取平移分量（在 before 帧的局部坐标系下）
    dx = T_rel[0, 3]  # X 方向（右/左）
    dz = T_rel[2, 3]  # Z 方向（后/前）
    
    # Habitat 坐标系：-Z 是前方，所以 dy = -dz
    dy = -dz
    
    return np.array([dx, dy], dtype=np.float32)
```

**坐标系说明**：
- Habitat Agent-local 坐标系：
  - X 轴：向右（正方向）
  - Y 轴：向上（正方向）
  - Z 轴：向后（-Z 是前方）
- 返回的 `(dx, dy)` 中：
  - `dx`：左右移动（右为正）
  - `dy`：前后移动（前为正）

**特殊情况**：
- `action[T-1] = [0.0, 0.0]`：最后一帧没有后续帧，动作为零向量

#### 2. 离散动作 (`discrete_actions.npy`)

**文件格式**：
- 形状：`[T]` int32 数组

**数据内容**：
- `discrete_action[i]`：从 `frame[i]` 到 `frame[i+1]` 的离散动作类型
- 动作枚举（来自 `HabitatSimActions`）：
  - `0` = **STOP**（停止，到达目标）
  - `1` = **MOVE_FORWARD**（前进）
  - `2` = **TURN_LEFT**（左转）
  - `3` = **TURN_RIGHT**（右转）

**数据来源**：
- 直接来自 `ShortestPathFollower.get_next_action(waypoint)`
- 保证是从当前位置到目标的最优动作序列

**特殊情况**：
- `discrete_action[T-1] = 0` (STOP)：最后一帧标记为停止

#### 3. 位姿数据 (`poses.json`)

**文件格式**：
- 形状：`[T, 4, 4]` 位姿矩阵列表（JSON 格式）

**数据内容**：
- `pose[i]`：第 i 帧的相机到世界坐标系的变换矩阵 `T_world_camera`
- 4×4 齐次变换矩阵：
  ```
  [R | t]
  [0 | 1]
  ```
  - `R`：3×3 旋转矩阵
  - `t`：3×1 平移向量（世界坐标系中的相机位置）

**用途**：
- 用于计算连续动作（相邻帧位姿差分）
- 用于计算热力图（历史帧在当前帧中的投影位置）

### 训练时的使用方式

#### 滑动窗口数据集

训练代码（`VLNSlidingWindowDataset`）将每个 clip 扩展为多个训练样本：

```python
# 对于 T 帧的 clip，生成 T - min_history 个样本
for current_t in range(min_history, T):
    sample = {
        "history_frames": frames[0:current_t],      # 历史观测
        "current_frame": frames[current_t],          # 当前帧
        "action": actions[current_t],                # 要预测的动作
        "discrete_action": discrete_actions[current_t],
        "instruction": instruction,
        ...
    }
```

#### 动作预测的监督学习

**任务定义**：
- 输入：历史帧 + 当前帧 + 指令
- 输出：
  1. **连续动作** `(dx, dy)`：预测下一步的 2D 位移
  2. **STOP 预测**：二分类，判断是否应该停止

**为什么分离连续动作和离散动作？**

1. **连续动作** (`dx`, `dy`) 适合用扩散模型学习：
   - 可以捕捉运动的多模态性（同一个指令可能有多种合理的路径）
   - 扩散模型生成平滑、自然的运动轨迹
   - 训练使用 `DiffusionActionHead`

2. **STOP 动作** 是稀疏的离散事件：
   - 在整个数据集中只占约 0.6% 的动作
   - 很难用连续回归准确预测这种离散的、稀有的事件
   - 使用单独的二分类头 `StopPredictionHead`，并采用 Focal Loss 处理类别不平衡

3. **混合架构在 VLN 任务中效果更好**：
   - 扩散模型：负责预测"怎么走"（连续位移）
   - 分类头：负责预测"何时停"（到达目标）

### 数据质量控制

采集脚本内置多层质量检查：

1. **Episode 验证**：
   - 检查 goals、reference_path、instruction 等必需字段
   - 跳过缺失关键数据的 episode

2. **轨迹长度**：
   - 最小帧数：5 帧
   - 最大步数：由 `--max-steps` 参数控制

3. **关键帧匹配**：
   - 将采集的轨迹与 VLN-CE 的 reference_path 对齐
   - 计算平均偏差，自动调整质量阈值
   - 偏差 < 0.3m：excellent，0.3-0.8m：high，0.8-1.5m：medium

4. **I/O 健康检查**：
   - 每个 clip 开始前验证磁盘读写
   - 允许 5% 的帧保存失败（增强鲁棒性）

5. **动作统计**：
   - 自动计算每个 clip 的动作范围 `(dx_min, dx_max, dy_min, dy_max)`
   - 记录在 `meta.json` 中
   - 训练时用于动作归一化

### meta.json 结构说明

每个 clip 的 `meta.json` 包含完整的元数据：

```json
{
  "episode_id": "...",
  "trajectory_id": "...",
  "scene_id": "...",
  "instruction": "...",
  "num_frames": 42,
  
  "actions": {
    "num_actions": 42,
    "action_dim": 2,
    "action_semantic": "action[i] = from frame[i] to frame[i+1]",
    "action_format": "(dx, dy) - agent-local 2D displacement",
    "discrete_action_format": "HabitatSimActions (0=STOP, 1=FORWARD, 2=LEFT, 3=RIGHT)",
    "last_action_note": "action[T-1] = (0,0) / STOP since no next frame",
    "stats": {
      "dx": {"min": -0.05, "max": 0.03, "mean": 0.001},
      "dy": {"min": 0.0, "max": 0.25, "mean": 0.18}
    },
    "files": {
      "continuous": "actions.npy",
      "discrete": "discrete_actions.npy"
    }
  },
  
  "reference_path": [[x1, y1, z1], [x2, y2, z2], ...],
  "keyframe_indices": [0, 8, 15, 23, 41],
  "keyframe_distances": [0.15, 0.23, 0.18, 0.12, 0.08],
  
  "quality_control": {
    "min_valid_ratio_used": 0.70,
    "avg_keyframe_distance": 0.152,
    "quality_tier": "high"
  }
}
```

### 常见问题

**Q: 为什么最后一帧的动作是 (0, 0) / STOP？**

A: 因为动作定义为"从当前帧到下一帧"，最后一帧没有后续帧，所以动作无效。训练时会通过 `action_valid=0` 标记忽略最后一帧的动作损失。

**Q: 动作的坐标系是什么？**

A: Agent-local 坐标系（以当前帧的 agent 朝向为参考）。`dx` 是左右移动，`dy` 是前后移动。这样模型学习的是相对运动，而不是全局坐标，更符合第一人称导航的直觉。

**Q: 离散动作和连续动作是否一致？**

A: 是的。离散动作（如 MOVE_FORWARD）在执行时会产生对应的连续位移（如 `dy ≈ 0.25`）。两者都是从相同的 Habitat 仿真中采集的，保证了一致性。

**Q: 如何使用这些动作数据训练模型？**

A: 参考 `/root/VLN/Project` 中的训练代码：
- 使用 `VLNSlidingWindowDataset` 加载数据
- 使用 `DiffusionActionHead` 预测连续动作
- 使用 `StopPredictionHead` 预测 STOP 动作
- 详见 `/root/VLN/Project/README.md`
