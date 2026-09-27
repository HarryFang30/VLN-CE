#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
PROJECT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
ALLOWED_ROOT="/mnt/afs/lixiaoou/intern/fjl"

OUTPUT="${1:-${ALLOWED_ROOT}/data/heatmap_randomwalk_val_unseen_v1}"
NUM_CLIPS="${2:-50}"
NUM_WAYPOINTS="${3:-4}"
STORAGE_FORMAT="${4:-chunks}"
HABITAT_GPU="${5:-0}"

export HEATMAP_CONFIG="${HEATMAP_CONFIG:-${PROJECT_DIR}/habitat_extensions/config/vlnce_collect_val_unseen.yaml}"

exec "${PROJECT_DIR}/run_collect_heatmap.sh" \
  "$OUTPUT" \
  "$NUM_CLIPS" \
  "$NUM_WAYPOINTS" \
  "$STORAGE_FORMAT" \
  "$HABITAT_GPU"
