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
