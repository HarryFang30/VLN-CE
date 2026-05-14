#!/usr/bin/env bash
# ============================================================
# VLN-CE Heatmap (multi-waypoint patrol) collection launcher
# Same layout style as run_collect_panoramic (conda / EGL optional).
# ============================================================
#
# Usage (from anywhere):
#   ./run_collect_heatmap.sh [OUTPUT_DIR] [NUM_CLIPS] [HABITAT_GPU]
#
# Defaults (relative to this repo root):
#   OUTPUT_DIR  -> <repo>/data/collected/heatmap_train_data
#   NUM_CLIPS   -> 1000
#   HABITAT_GPU -> 0
#
# Optional environment (server-specific, not hardcoded):
#   CONDA_BASE         default: $HOME/miniconda3 (also tries Miniforge3)
#   VLNCE_CONDA_ENV    if set, conda activate this env after sourcing conda.sh
#   VLNCE_NV_EGL_FIX   if set to a directory, enable NVIDIA EGL overlay (see below)
#   HEATMAP_CONFIG     default: habitat_extensions/config/vlnce_collect.yaml
#   HEATMAP_NUM_WAYPOINTS   if set, passed as --num-waypoints (int)
#   HEATMAP_STORAGE_FORMAT  if set, passed as --storage-format (e.g. chunks)
#   CUDA_VISIBLE_DEVICES  passed through when set (e.g. bind physical GPU)
#
# NVIDIA EGL overlay (optional, same idea as panoramic launcher on fixed-driver hosts):
#   export VLNCE_NV_EGL_FIX=/path/to/nv-egl-fix
# Expect under that directory: local-lib/, NVIDIA-Linux-.../, 10_nvidia_local.json
#
# Example:
#   export VLNCE_CONDA_ENV=dataset_collect
#   export CONDA_BASE=/opt/conda
#   ./run_collect_heatmap.sh ./data/collected/heatmap_train_data 5000 0
#
# val_unseen 划分（5 个位置参数：含 waypoints / storage）请用:
#   ./run_collect_heatmap_val_unseen.sh [OUTPUT] [NUM_CLIPS] [NUM_WAYPOINTS] [STORAGE_FORMAT] [GPU]
#
set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
PROJECT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"

# --- positional args (heatmap has no --split; split comes from HEATMAP_CONFIG / yaml) ---
DEFAULT_OUTPUT="${PROJECT_DIR}/data/collected/heatmap_train_data"
OUTPUT="${1:-$DEFAULT_OUTPUT}"
NUM_CLIPS="${2:-1000}"
HABITAT_GPU="${3:-0}"

LOG_DIR="${PROJECT_DIR}/logs"
LOG_FILE="${LOG_DIR}/collect_heatmap_${NUM_CLIPS}_$(date +%Y%m%d_%H%M%S).log"

HEATMAP_CONFIG="${HEATMAP_CONFIG:-habitat_extensions/config/vlnce_collect.yaml}"

mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo "VLN-CE Heatmap Collection Launcher"
echo "============================================================"
echo "Project dir:       $PROJECT_DIR"
echo "Output:            $OUTPUT"
echo "Num clips:         $NUM_CLIPS"
echo "Habitat gpu:       $HABITAT_GPU"
echo "Config:            $HEATMAP_CONFIG"
if [ -n "${HEATMAP_NUM_WAYPOINTS:-}" ]; then
  echo "Num waypoints:     $HEATMAP_NUM_WAYPOINTS (via HEATMAP_NUM_WAYPOINTS)"
fi
if [ -n "${HEATMAP_STORAGE_FORMAT:-}" ]; then
  echo "Storage format:    $HEATMAP_STORAGE_FORMAT (via HEATMAP_STORAGE_FORMAT)"
fi
echo "Log file:          $LOG_FILE"
echo "============================================================"

# ------------------------------------------------------------
# 1. Optional conda (only if VLNCE_CONDA_ENV is set)
# ------------------------------------------------------------
CONDA_BASE="${CONDA_BASE:-}"
if [ -z "$CONDA_BASE" ]; then
  if [ -d "$HOME/miniconda3" ]; then
    CONDA_BASE="$HOME/miniconda3"
  elif [ -d "$HOME/Miniforge3" ]; then
    CONDA_BASE="$HOME/Miniforge3"
  elif [ -d "$HOME/anaconda3" ]; then
    CONDA_BASE="$HOME/anaconda3"
  fi
fi

if [ -n "${VLNCE_CONDA_ENV:-}" ]; then
  if [ -z "$CONDA_BASE" ] || [ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    echo "[ERROR] VLNCE_CONDA_ENV is set but conda.sh not found."
    echo "        Set CONDA_BASE to your Miniconda/Anaconda root, e.g."
    echo "        export CONDA_BASE=/path/to/miniconda3"
    exit 1
  fi
  # shellcheck source=/dev/null
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "$VLNCE_CONDA_ENV"
  echo "[INFO] Conda env:      ${CONDA_DEFAULT_ENV:-}"
  echo "[INFO] Python:         $(command -v python)"
  echo "[INFO] Python version: $(python --version)"
else
  echo "[INFO] Skipping conda activate (set VLNCE_CONDA_ENV to enable)."
  echo "[INFO] Python:         $(command -v python || echo 'python not found')"
fi

# ------------------------------------------------------------
# 2. Optional headless NVIDIA EGL (VLNCE_NV_EGL_FIX)
# ------------------------------------------------------------
if [ -n "${VLNCE_NV_EGL_FIX:-}" ]; then
  NV_EGL_LIB="${VLNCE_NV_EGL_LIB:-${VLNCE_NV_EGL_FIX}/local-lib}"
  NV_RUNFILE="${VLNCE_NV_RUNFILE:-}"
  if [ -z "$NV_RUNFILE" ]; then
    # First matching extracted driver dir under the fix root
    NV_RUNFILE="$(find "$VLNCE_NV_EGL_FIX" -maxdepth 1 -type d -name 'NVIDIA-Linux-x86_64-*' 2>/dev/null | head -1 || true)"
  fi
  NV_EGL_VENDOR="${VLNCE_NV_EGL_VENDOR:-${VLNCE_NV_EGL_FIX}/10_nvidia_local.json}"

  unset DISPLAY || true
  unset WAYLAND_DISPLAY || true
  unset EGL_PLATFORM || true
  unset LIBGL_ALWAYS_INDIRECT || true
  unset MESA_LOADER_DRIVER_OVERRIDE || true
  unset LIBGL_DRIVERS_PATH || true

  export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

  if [ ! -d "$NV_EGL_LIB" ]; then
    echo "[ERROR] Missing NVIDIA EGL local lib dir: $NV_EGL_LIB"
    exit 1
  fi
  if [ -z "$NV_RUNFILE" ] || [ ! -d "$NV_RUNFILE" ]; then
    echo "[ERROR] Missing NVIDIA runfile dir under VLNCE_NV_EGL_FIX (set VLNCE_NV_RUNFILE)."
    exit 1
  fi
  if [ ! -f "$NV_EGL_VENDOR" ]; then
    echo "[ERROR] Missing NVIDIA EGL vendor file: $NV_EGL_VENDOR"
    exit 1
  fi
  if [ ! -e "$NV_EGL_LIB/libEGL_nvidia.so.0" ]; then
    echo "[ERROR] Missing local libEGL_nvidia.so.0 in $NV_EGL_LIB"
    exit 1
  fi

  export LD_LIBRARY_PATH="${NV_EGL_LIB}:${NV_RUNFILE}:${LD_LIBRARY_PATH:-}"
  export __EGL_VENDOR_LIBRARY_FILENAMES="$NV_EGL_VENDOR"

  echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo "[INFO] EGL vendor file: ${__EGL_VENDOR_LIBRARY_FILENAMES}"
  echo "[INFO] NVIDIA EGL lib:  ${NV_EGL_LIB}"
  echo "[INFO] NVIDIA runfile:  ${NV_RUNFILE}"
else
  echo "[INFO] VLNCE_NV_EGL_FIX unset — using system GL/EGL (set it for fixed-driver EGL setups)."
fi

echo "============================================================"
echo "[CHECK] NVIDIA driver (if nvidia-smi available)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo "(nvidia-smi not available or not NVIDIA host)"

echo "============================================================"
echo "[CHECK] OpenCV import"
python - <<'PY'
import cv2
print("cv2 OK:", cv2.__version__)
print("cv2 file:", cv2.__file__)
PY

echo "============================================================"
echo "[RUN] Start heatmap collection"
echo "============================================================"

cd "$PROJECT_DIR"

HEATMAP_PY_ARGS=()
if [ -n "${HEATMAP_NUM_WAYPOINTS:-}" ]; then
  HEATMAP_PY_ARGS+=(--num-waypoints "${HEATMAP_NUM_WAYPOINTS}")
fi
if [ -n "${HEATMAP_STORAGE_FORMAT:-}" ]; then
  HEATMAP_PY_ARGS+=(--storage-format "${HEATMAP_STORAGE_FORMAT}")
fi

python -m collect heatmap \
  --config "$HEATMAP_CONFIG" \
  --output "$OUTPUT" \
  --num-clips "$NUM_CLIPS" \
  --gpu "$HABITAT_GPU" \
  "${HEATMAP_PY_ARGS[@]}"

echo "============================================================"
echo "[DONE] Heatmap collection finished"
echo "Log saved to: $LOG_FILE"
echo "Output saved to: $OUTPUT"
echo "============================================================"
