#!/usr/bin/env bash
# ============================================================
# VLN-CE Heatmap 采集 — val_unseen 划分专用启动器
# 固定使用 habitat_extensions/config/vlnce_collect_val_unseen.yaml
# 其余逻辑与 run_collect_heatmap.sh 相同（conda / EGL / 日志 tee）。
# ============================================================
#
# Usage:
#   ./run_collect_heatmap_val_unseen.sh \
#     [OUTPUT_DIR] [NUM_CLIPS] [NUM_WAYPOINTS] [STORAGE_FORMAT] [HABITAT_GPU]
#
# Defaults (相对仓库根):
#   OUTPUT_DIR       -> <repo>/data/collected/heatmap_val_unseen_data
#   NUM_CLIPS        -> 50
#   NUM_WAYPOINTS    -> 4
#   STORAGE_FORMAT   -> chunks
#   HABITAT_GPU      -> 0
#
# 与下列等价（在仓库根、且已 conda activate 或设置 VLNCE_CONDA_ENV）:
#   export HEATMAP_CONFIG=habitat_extensions/config/vlnce_collect_val_unseen.yaml
#   export HEATMAP_NUM_WAYPOINTS=4
#   export HEATMAP_STORAGE_FORMAT=chunks
#   ./run_collect_heatmap.sh /path/to/out 50 0
#
# Example (intern 路径):
#   conda activate dataset_collect
#   ./run_collect_heatmap_val_unseen.sh \
#     /home/intern/zhr/fjl/data/val_heatmap \
#     50 \
#     4 \
#     chunks \
#     0
#
set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
PROJECT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"

DEFAULT_OUTPUT="${PROJECT_DIR}/data/collected/heatmap_val_unseen_data"
OUTPUT="${1:-$DEFAULT_OUTPUT}"
NUM_CLIPS="${2:-50}"
NUM_WAYPOINTS="${3:-4}"
STORAGE_FORMAT="${4:-chunks}"
HABITAT_GPU="${5:-0}"

export HEATMAP_CONFIG="${HEATMAP_CONFIG:-${PROJECT_DIR}/habitat_extensions/config/vlnce_collect_val_unseen.yaml}"
export HEATMAP_NUM_WAYPOINTS="$NUM_WAYPOINTS"
export HEATMAP_STORAGE_FORMAT="$STORAGE_FORMAT"

exec "${PROJECT_DIR}/run_collect_heatmap.sh" "$OUTPUT" "$NUM_CLIPS" "$HABITAT_GPU"
