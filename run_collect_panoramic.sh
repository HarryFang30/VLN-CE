#!/usr/bin/env bash
# ============================================================
# VLN-CE R2R panoramic collection launcher
# ============================================================
#
# Usage (from repo root):
#   ./run_collect_panoramic.sh [OUTPUT_DIR] [SPLIT] [NUM_CLIPS] [HABITAT_GPU]
#
# Example:
#   ./run_collect_panoramic.sh /path/to/r2r_panoramic_data_v2 train 5000 0
#
# Optional environment:
#   CONDA_BASE, VLNCE_CONDA_ENV, VLNCE_NV_EGL_FIX, CUDA_VISIBLE_DEVICES
#
set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
PROJECT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"

OUTPUT="${1:-${PROJECT_DIR}/data/collected/r2r_panoramic_data}"
SPLIT="${2:-train}"
NUM_CLIPS="${3:-1000}"
HABITAT_GPU="${4:-0}"

LOG_DIR="${PROJECT_DIR}/logs"
LOG_FILE="${LOG_DIR}/collect_panoramic_${SPLIT}_${NUM_CLIPS}_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo "VLN-CE Panoramic Collection Launcher"
echo "============================================================"
echo "Project dir:       $PROJECT_DIR"
echo "Output:            $OUTPUT"
echo "Split:             $SPLIT"
echo "Num clips:         $NUM_CLIPS"
echo "Habitat gpu:       $HABITAT_GPU"
echo "Log file:          $LOG_FILE"
echo "============================================================"

CONDA_BASE="${CONDA_BASE:-}"
if [ -z "$CONDA_BASE" ]; then
  if [ -d "$HOME/miniconda3" ]; then
    CONDA_BASE="$HOME/miniconda3"
  elif [ -d "/home/intern/suyihan/miniconda3" ]; then
    CONDA_BASE="/home/intern/suyihan/miniconda3"
  elif [ -d "$HOME/Miniforge3" ]; then
    CONDA_BASE="$HOME/Miniforge3"
  fi
fi

if [ -n "${VLNCE_CONDA_ENV:-}" ]; then
  if [ -z "$CONDA_BASE" ] || [ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    echo "[ERROR] VLNCE_CONDA_ENV is set but conda.sh not found."
    exit 1
  fi
  # shellcheck source=/dev/null
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "$VLNCE_CONDA_ENV"
  echo "[INFO] Conda env: ${CONDA_DEFAULT_ENV:-}"
fi

if [ -n "${VLNCE_NV_EGL_FIX:-}" ]; then
  NV_EGL_LIB="${VLNCE_NV_EGL_FIX}/local-lib"
  NV_RUNFILE="$(find "$VLNCE_NV_EGL_FIX" -maxdepth 1 -type d -name 'NVIDIA-Linux-x86_64-*' 2>/dev/null | head -1 || true)"
  NV_EGL_VENDOR="${VLNCE_NV_EGL_FIX}/10_nvidia_local.json"
  unset DISPLAY WAYLAND_DISPLAY EGL_PLATFORM LIBGL_ALWAYS_INDIRECT MESA_LOADER_DRIVER_OVERRIDE LIBGL_DRIVERS_PATH || true
  export LD_LIBRARY_PATH="${NV_EGL_LIB}:${NV_RUNFILE}:${LD_LIBRARY_PATH:-}"
  export __EGL_VENDOR_LIBRARY_FILENAMES="$NV_EGL_VENDOR"
else
  export DISPLAY="${DISPLAY:-:99}"
fi

cd "$PROJECT_DIR"

python -m collect panoramic \
  --output "$OUTPUT" \
  --split "$SPLIT" \
  --num-clips "$NUM_CLIPS" \
  --gpu "$HABITAT_GPU" \
  --depth-directions front front_down

echo "============================================================"
echo "[DONE] Panoramic collection finished"
echo "Log: $LOG_FILE"
echo "Output: $OUTPUT"
echo "============================================================"
