#!/usr/bin/env bash
set -euo pipefail

# Heatmap patrol collection on finn_cci_c500.
# Rendering is CPU/GLX (Mesa llvmpipe); the MetaX accelerators are not used.

ALLOWED_ROOT="/mnt/afs/lixiaoou/intern/fjl"
SCRIPT_PATH="${BASH_SOURCE[0]}"
PROJECT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
CONDA_SH="${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
CONDA_ENV="${VLNCE_CONDA_ENV:-${ALLOWED_ROOT}/envs/vlnce}"

DEFAULT_OUTPUT="${ALLOWED_ROOT}/data/heatmap_randomwalk_pilot_v1"
OUTPUT="${1:-$DEFAULT_OUTPUT}"
NUM_CLIPS="${2:-20}"
NUM_WAYPOINTS="${3:-4}"
STORAGE_FORMAT="${4:-chunks}"
HABITAT_GPU="${5:-0}"

HEATMAP_CONFIG="${HEATMAP_CONFIG:-${PROJECT_DIR}/habitat_extensions/config/vlnce_collect.yaml}"
HEATMAP_SEED="${HEATMAP_SEED:-42}"
HEATMAP_MIN_WAYPOINT_DIST="${HEATMAP_MIN_WAYPOINT_DIST:-2.0}"
HEATMAP_MAX_WAYPOINT_DIST="${HEATMAP_MAX_WAYPOINT_DIST:-10.0}"
HEATMAP_MAX_STEPS="${HEATMAP_MAX_STEPS:-500}"
HEATMAP_MIN_FRAMES="${HEATMAP_MIN_FRAMES:-30}"
HEATMAP_MIN_KEYFRAME_TRANSLATION="${HEATMAP_MIN_KEYFRAME_TRANSLATION:-0.20}"
VLNCE_DISPLAY="${VLNCE_DISPLAY:-localhost:200.0}"
HEATMAP_WORKER_ID="${HEATMAP_WORKER_ID:-0}"
HEATMAP_CLIP_ID_START="${HEATMAP_CLIP_ID_START:-0}"
HEATMAP_STATS_FILE="${HEATMAP_STATS_FILE:-collection_stats.json}"
HEATMAP_IO_WORKERS="${HEATMAP_IO_WORKERS:-4}"
HEATMAP_MAX_PENDING_IO="${HEATMAP_MAX_PENDING_IO:-64}"
HEATMAP_RENDER_THREADS="${HEATMAP_RENDER_THREADS:-4}"
HEATMAP_EPISODE_SHARD_ID="${HEATMAP_EPISODE_SHARD_ID:-0}"
HEATMAP_EPISODE_SHARD_COUNT="${HEATMAP_EPISODE_SHARD_COUNT:-1}"
HEATMAP_EPISODE_OFFSET="${HEATMAP_EPISODE_OFFSET:-0}"
HEATMAP_SCENE_BLOCK_SIZE="${HEATMAP_SCENE_BLOCK_SIZE:-8}"

canonical_path() {
  realpath -m "$1"
}

assert_under_allowed_root() {
  local label="$1"
  local resolved
  resolved="$(canonical_path "$2")"
  case "$resolved" in
    "$ALLOWED_ROOT"|"$ALLOWED_ROOT"/*) ;;
    *)
      echo "[ERROR] ${label} escapes allowed root ${ALLOWED_ROOT}: ${resolved}" >&2
      exit 1
      ;;
  esac
}

assert_under_allowed_root "project" "$PROJECT_DIR"
assert_under_allowed_root "output" "$OUTPUT"
assert_under_allowed_root "config" "$HEATMAP_CONFIG"
assert_under_allowed_root "conda env" "$CONDA_ENV"

if [[ "$HABITAT_GPU" != "0" ]]; then
  echo "[ERROR] CPU/GLX Habitat rendering requires logical device 0, got ${HABITAT_GPU}." >&2
  exit 1
fi
if [[ ! -f "$CONDA_SH" ]]; then
  echo "[ERROR] conda.sh not found: $CONDA_SH" >&2
  exit 1
fi
if [[ ! -f "$HEATMAP_CONFIG" ]]; then
  echo "[ERROR] collection config not found: $HEATMAP_CONFIG" >&2
  exit 1
fi

OUTPUT="$(canonical_path "$OUTPUT")"
LOG_DIR="${PROJECT_DIR}/logs"
LOG_FILE="${LOG_DIR}/collect_heatmap_w${HEATMAP_WORKER_ID}_${NUM_CLIPS}_wp${NUM_WAYPOINTS}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$LOG_DIR" "$OUTPUT"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo "VLN-CE Heatmap Patrol Collection (CPU/GLX)"
echo "============================================================"
echo "Project dir:       $PROJECT_DIR"
echo "Output:            $OUTPUT"
echo "Config:            $HEATMAP_CONFIG"
echo "Num clips target:  $NUM_CLIPS"
echo "Num waypoints:     $NUM_WAYPOINTS"
echo "Waypoint distance: ${HEATMAP_MIN_WAYPOINT_DIST}m ~ ${HEATMAP_MAX_WAYPOINT_DIST}m"
echo "Max steps:         $HEATMAP_MAX_STEPS"
echo "Min frames:        $HEATMAP_MIN_FRAMES"
echo "Keyframe move:     $HEATMAP_MIN_KEYFRAME_TRANSLATION m"
echo "Storage format:    $STORAGE_FORMAT"
echo "Seed:              $HEATMAP_SEED"
echo "Worker id:         $HEATMAP_WORKER_ID"
echo "Clip id start:     $HEATMAP_CLIP_ID_START"
echo "Stats file:        $HEATMAP_STATS_FILE"
echo "Episode shard:     $HEATMAP_EPISODE_SHARD_ID/$HEATMAP_EPISODE_SHARD_COUNT"
echo "Episode offset:    $HEATMAP_EPISODE_OFFSET"
echo "Scene block size:  $HEATMAP_SCENE_BLOCK_SIZE"
echo "Display:           $VLNCE_DISPLAY"
echo "Log file:          $LOG_FILE"
echo "============================================================"

# shellcheck source=/dev/null
source "$CONDA_SH"
conda activate "$CONDA_ENV"

if [[ "$(command -v python)" != "${CONDA_ENV}/bin/python" ]]; then
  echo "[ERROR] wrong Python after conda activate: $(command -v python)" >&2
  exit 1
fi

# Habitat-Sim 0.1.7 uses a GLX context on this host.  Exposing one logical
# device avoids its multi-GPU GLX guard; glxinfo confirms the renderer itself
# is Mesa llvmpipe, not a MetaX/NVIDIA accelerator.
export DISPLAY="$VLNCE_DISPLAY"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export LP_NUM_THREADS="$HEATMAP_RENDER_THREADS"
export OMP_NUM_THREADS="$HEATMAP_RENDER_THREADS"
# Habitat-Sim 0.1.7 can occasionally abort inside its native GLX renderer.
# Do not leave multi-gigabyte core files in the collection tree; the parallel
# launcher retries the worker with the same id range instead.
ulimit -c 0
unset WAYLAND_DISPLAY EGL_PLATFORM __EGL_VENDOR_LIBRARY_FILENAMES
unset LIBGL_ALWAYS_INDIRECT MESA_LOADER_DRIVER_OVERRIDE LIBGL_DRIVERS_PATH

echo "[INFO] Conda env:      ${CONDA_DEFAULT_ENV:-}"
echo "[INFO] Python:         $(command -v python)"
echo "[INFO] Python version: $(python --version)"
echo "[INFO] OpenGL renderer:"
if ! glxinfo -B 2>&1 | grep -E "OpenGL (vendor|renderer) string|direct rendering"; then
  echo "[ERROR] GLX display is not usable: DISPLAY=$DISPLAY" >&2
  exit 1
fi

cd "$PROJECT_DIR"

COLLECT_ARGS=(
  -m collect heatmap
  --config "$HEATMAP_CONFIG"
  --output "$OUTPUT"
  --num-clips "$NUM_CLIPS"
  --num-waypoints "$NUM_WAYPOINTS"
  --min-waypoint-dist "$HEATMAP_MIN_WAYPOINT_DIST"
  --max-waypoint-dist "$HEATMAP_MAX_WAYPOINT_DIST"
  --max-steps "$HEATMAP_MAX_STEPS"
  --min-frames "$HEATMAP_MIN_FRAMES"
  --min-keyframe-translation "$HEATMAP_MIN_KEYFRAME_TRANSLATION"
  --num-workers "$HEATMAP_IO_WORKERS"
  --max-pending-io "$HEATMAP_MAX_PENDING_IO"
  --storage-format "$STORAGE_FORMAT"
  --seed "$HEATMAP_SEED"
  --worker-id "$HEATMAP_WORKER_ID"
  --episode-shard-id "$HEATMAP_EPISODE_SHARD_ID"
  --episode-shard-count "$HEATMAP_EPISODE_SHARD_COUNT"
  --episode-offset "$HEATMAP_EPISODE_OFFSET"
  --scene-block-size "$HEATMAP_SCENE_BLOCK_SIZE"
  --stats-file "$HEATMAP_STATS_FILE"
  --gpu "$HABITAT_GPU"
)
if [[ "$HEATMAP_CLIP_ID_START" -gt 0 ]]; then
  COLLECT_ARGS+=(--clip-id-start "$HEATMAP_CLIP_ID_START")
fi
python "${COLLECT_ARGS[@]}"

echo "============================================================"
echo "[DONE] Heatmap collection finished"
echo "Log saved to: $LOG_FILE"
echo "Output saved to: $OUTPUT"
echo "============================================================"
