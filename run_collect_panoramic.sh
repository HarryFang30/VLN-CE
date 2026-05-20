#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# VLN-CE Panoramic Data Collection Launcher
# Fully self-contained runtime wrapper for this server.
# ============================================================

PROJECT_DIR="/home/intern/zhr/fjl/habitat/VLN-CE"
CONDA_BASE="/home/intern/suyihan/miniconda3"
CONDA_ENV="dataset_collect"

NV_EGL_FIX="/home/intern/nv-egl-fix"
NV_EGL_LIB="${NV_EGL_FIX}/local-lib"
NV_RUNFILE="${NV_EGL_FIX}/NVIDIA-Linux-x86_64-550.90.07"
NV_EGL_VENDOR="${NV_EGL_FIX}/10_nvidia_local.json"

OUTPUT="${1:-/home/intern/zhr/fjl/r2r_panoramic_data_v2}"
SPLIT="${2:-train}"
NUM_CLIPS="${3:-100}"
HABITAT_GPU="${4:-0}"

LOG_DIR="${PROJECT_DIR}/logs"
LOG_FILE="${LOG_DIR}/collect_${SPLIT}_${NUM_CLIPS}_$(date +%Y%m%d_%H%M%S).log"

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

# ------------------------------------------------------------
# 1. Activate conda environment
# ------------------------------------------------------------
if [ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    echo "[ERROR] Cannot find conda.sh: ${CONDA_BASE}/etc/profile.d/conda.sh"
    exit 1
fi

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "[INFO] Conda env:      ${CONDA_DEFAULT_ENV}"
echo "[INFO] Python:         $(which python)"
echo "[INFO] Python version: $(python --version)"

# ------------------------------------------------------------
# 2. Force Habitat-Sim to use headless NVIDIA EGL
# ------------------------------------------------------------
unset DISPLAY
unset WAYLAND_DISPLAY
unset EGL_PLATFORM

# Avoid accidental Mesa / indirect GL pollution from login shells
unset LIBGL_ALWAYS_INDIRECT || true
unset MESA_LOADER_DRIVER_OVERRIDE || true
unset LIBGL_DRIVERS_PATH || true

# Physical GPU selection.
# If using physical GPU 0: CUDA_VISIBLE_DEVICES=0 and --gpu 0.
# If using physical GPU 3: CUDA_VISIBLE_DEVICES=3 and still --gpu 0.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Local NVIDIA EGL userspace libraries matching server driver 550.90.07
if [ ! -d "$NV_EGL_LIB" ]; then
    echo "[ERROR] Missing NVIDIA EGL local lib dir: $NV_EGL_LIB"
    exit 1
fi

if [ ! -d "$NV_RUNFILE" ]; then
    echo "[ERROR] Missing NVIDIA runfile dir: $NV_RUNFILE"
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

echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] EGL vendor file: ${__EGL_VENDOR_LIBRARY_FILENAMES}"
echo "[INFO] NVIDIA EGL lib:  ${NV_EGL_LIB}"
echo "[INFO] NVIDIA runfile:  ${NV_RUNFILE}"
echo "[INFO] LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"

echo "============================================================"
echo "[CHECK] NVIDIA driver"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true

echo "============================================================"
echo "[CHECK] Local NVIDIA EGL dependency"
if ldd "$NV_EGL_LIB/libEGL_nvidia.so.0" | grep -q "not found"; then
    ldd "$NV_EGL_LIB/libEGL_nvidia.so.0"
    echo "[ERROR] Local NVIDIA EGL dependency has missing libraries."
    exit 1
fi
ldd "$NV_EGL_LIB/libEGL_nvidia.so.0" | grep -E "nvidia|EGL|GL" || true

echo "============================================================"
echo "[CHECK] OpenCV import"
python - <<'PY'
import cv2
print("cv2 OK:", cv2.__version__)
print("cv2 file:", cv2.__file__)
PY

echo "============================================================"
echo "[RUN] Start collection"
echo "============================================================"

cd "$PROJECT_DIR"

python -m collect panoramic \
  --output "$OUTPUT" \
  --split "$SPLIT" \
  --num-clips "$NUM_CLIPS" \
  --gpu "$HABITAT_GPU" \
  --depth-directions front front_down

echo "============================================================"
echo "[DONE] Collection finished"
echo "Log saved to: $LOG_FILE"
echo "Output saved to: $OUTPUT"
echo "============================================================"
