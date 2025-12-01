# VLN-CE 高质量数据采集工具

## 核心算法

**双向导航 + 多帧融合 + 四层过滤 → 97.2%高质量数据**

1. Agent执行 起点→终点(Forward) + 终点→起点(Backward)
2. 每个参考点从所有帧中选择最佳视角生成热力图
3. 四层过滤：总帧≥5、双向≥5、有效率≥70%、完整性验证

## 快速部署（服务器）

### 1. 上传
```bash
# 本地打包上传
scp vlnce-data-collection.tar.gz user@server:/path/to/VLN-CE/
```

### 2. 解压配置
```bash
# 服务器端
ssh user@server
cd /path/to/VLN-CE
tar -xzf vlnce-data-collection.tar.gz

# 激活VLN-CE环境
conda activate habitat  # 或你的环境名
```

**⚠️ 重要说明**:
- 脚本依赖 `habitat_extensions` 目录（已包含在压缩包中）
- 确保你在VLN-CE项目根目录下解压（与data/目录同级）
- 如果你的VLN-CE已有habitat_extensions，解压会覆盖config/vlnce_task.yaml

### 3. 修改配置

**编辑 collect.py**:
```python
# Line 325: 目标clips数
NUM_CLIPS = 1000           # 测试：1000条（2.8h）
NUM_CLIPS = 10000          # 完整：2156条（6h，用尽所有episodes）

# Line 377: 准备episodes数
for _ in range(NUM_CLIPS * 5):        # 1000条配置
for _ in range(len(all_episodes)):    # 完整数据集配置
```

**数据集路径配置** - 编辑 `habitat_extensions/config/vlnce_task.yaml`:
```yaml
# Line 50: R2R数据集路径
DATA_PATH: data/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz

# Line 51: 场景数据路径（Matterport3D）
SCENES_DIR: data/scene_datasets/
```

**如果你的路径不是默认值，必须修改这两行！**

**GPU配置**（可选）:
- 默认使用GPU 0 (collect.py:339硬编码)
- 如需指定其他GPU：修改 `habitat_extensions/config/vlnce_task.yaml` Line 10
  ```yaml
  GPU_DEVICE_ID: 0  # 改为你想用的GPU ID，-1表示CPU
  ```

### 4. 运行
```bash
# 后台运行
nohup python collect.py > collection.log 2>&1 &
echo $! > collection.pid

# 监控进度
tail -f collection.log
# 或
watch -n 30 "ls -d dataset_train/train/*/clip_* 2>/dev/null | wc -l"
```

### 5. 采集完成后
```bash
# 质量分析（一键完成：基础+详细+验证）
python analyze.py

# 打包下载
cd dataset_train
tar -czf train_data.tar.gz train/
# 本地: scp user@server:/path/to/dataset_train/train_data.tar.gz ./
```

## 输出数据结构

```
dataset_train/
├── train/
│   ├── GdvgFV5R1Z5/clip_000001/
│   │   ├── rgb/               # RGB图像序列
│   │   ├── heatmaps.npy       # 热力图 (K, 64, 64)
│   │   ├── mask.npy           # 有效性掩码 (K,)
│   │   ├── poses.json         # 相机pose序列
│   │   ├── intrinsics.json    # 相机内参
│   │   └── meta.json          # 元数据（指令、路径等）
│   └── ...
└── collection_stats.json
```

## 关键参数

| 参数 | 文件 | 行号 | 默认值 | 说明 |
|------|------|------|--------|------|
| `NUM_CLIPS` | collect.py | 325 | 1000 | 目标clips数 |
| `MIN_VALID_RATIO` | collect.py | 683 | 0.7 | 最小有效率 |
| `MIN_FRAMES` | collect.py | 540 | 5 | 单向最小帧数 |
| `MAX_FRAMES` | collect.py | 459 | 50 | 单向最大帧数 |
| `DATA_PATH` | vlnce_task.yaml | 50 | data/datasets/R2R_VLNCE_v1-3_preprocessed/... | R2R数据集 |
| `SCENES_DIR` | vlnce_task.yaml | 51 | data/scene_datasets/ | 场景数据 |

## 数据质量指标

| 指标 | 数值 | 说明 |
|------|------|------|
| 有效率 | 97.2% | 热力图有效比例 |
| 标准差 | 7.1% | 质量稳定性 |
| 无效clips | 0个 | 完全无效的clips |
| 成功率 | 19.9% | 采集成功率（严格过滤） |
| 采集速度 | ~6 clips/分钟 | 实测速度 |

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

# 查看已采集数量
ls -d dataset_train/train/*/clip_* 2>/dev/null | wc -l

# 查看最新日志
tail -20 collection.log
```

**停止采集**:
```bash
kill $(cat collection.pid)
```

**采集中断**（支持断点续传）:
```bash
nohup python collect.py > collection_resume.log 2>&1 &
```

**检查磁盘**:
```bash
df -h .  # 每个clip约8MB
```

## 核心文件

- `collect.py` - 数据采集（包含所有质量控制）
- `analyze.py` - 质量分析（基础+详细+验证 三合一）
- `habitat_extensions/` - VLN-CE任务扩展（必需）
  - `config/vlnce_task.yaml` - 数据集路径配置
  - `config/default.py` - 配置加载器
- `README.md` - 本文档

## 默认路径结构

```
VLN-CE/
├── collect.py
├── analyze.py
├── habitat_extensions/
│   └── config/
│       └── vlnce_task.yaml          # 数据集路径配置
├── data/
│   ├── datasets/
│   │   └── R2R_VLNCE_v1-3_preprocessed/
│   │       └── train/
│   │           ├── train.json.gz
│   │           └── train_gt.json.gz
│   └── scene_datasets/
│       └── mp3d/                     # Matterport3D场景
└── dataset_train/                    # 采集输出目录
    └── train/
```

## 完整工作流

```bash
# 1. 上传部署
scp vlnce-data-collection.tar.gz server:/path/
ssh server
tar -xzf vlnce-data-collection.tar.gz

# 2. 配置
vim collect.py  # 修改Line 325, 377
# 如需修改数据集路径：
vim habitat_extensions/config/vlnce_task.yaml

# 3. 运行
conda activate habitat
nohup python collect.py > collection.log 2>&1 &

# 4. 监控
tail -f collection.log

# 5. 完成后分析
python analyze.py

# 6. 下载
cd dataset_train && tar -czf train_data.tar.gz train/
```
