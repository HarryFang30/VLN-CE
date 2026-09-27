#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# R2R-CE panoramic expert re-collection on the MetaX C500 node.
#
# Canonical copy lives in HeatmapVLN/scripts/; the deployed copy runs from
# <root>/habitat/VLN-CE (python -m collect must import from that tree).
#
# No NVIDIA EGL exists on this cluster.  Rendering uses the certified
# headless stack from the 8-GPU eval: per-worker Xvfb from the x11 bundle
# with Mesa llvmpipe software GLX.  Workers shard episodes by stable hash
# (--episode-modulo/remainder) and own disjoint clip-id blocks, so they can
# share one output root without collisions.  Re-running the same command
# resumes: already-collected episodes are skipped per shard.
#
# Built for website cluster submission into a blank container that mounts
# /mnt/afs: parameters arrive as environment variables (OUTPUT_ROOT, SPLIT,
# TOTAL_CLIPS, NUM_WORKERS, BASE_DISPLAY), everything runs from absolute
# /mnt/afs paths, and the vlnce python is invoked directly — no conda
# activation, no tmux, no pre-existing displays assumed.  Positional
# arguments override the environment for dev-machine smokes.
# ============================================================

ROOT="/mnt/afs/liwenhao/agent/370910109"
PROJECT_DIR="${ROOT}/habitat/VLN-CE"
VLNCE_PYTHON="${ROOT}/envs/vlnce/bin/python"
X11_BUNDLE="${ROOT}/tools/x11_headless_bundle_ubuntu22_20260801_v4"
XVFB_BIN="${X11_BUNDLE}/bin/Xvfb"
XDPYINFO_BIN="${X11_BUNDLE}/bin/xdpyinfo"
GLXINFO_BIN="${X11_BUNDLE}/bin/glxinfo"
X11_DRI_PATH="${X11_BUNDLE}/dri"
X11_FONT_PATH="${X11_BUNDLE}/share/fonts/misc"
X11_XKB_PATH="${X11_BUNDLE}/share/X11/xkb"

OUTPUT="${1:-${OUTPUT_ROOT:-${ROOT}/r2r_panoramic_data_v2}}"
SPLIT="${2:-${SPLIT:-train}}"
TOTAL_CLIPS="${3:-${TOTAL_CLIPS:-5000}}"
NUM_WORKERS="${4:-${NUM_WORKERS:-8}}"
BASE_DISPLAY="${5:-${BASE_DISPLAY:-230}}"

CLIP_ID_BLOCK=100000
LP_THREADS="${COLLECT_LP_NUM_THREADS:-8}"
MAX_STEPS="${COLLECT_MAX_STEPS:-300}"
IO_WORKERS="${COLLECT_IO_WORKERS:-8}"

GL_ENV_UNSET_ARGS=(
  -u DISPLAY -u WAYLAND_DISPLAY -u EGL_PLATFORM
  -u __EGL_VENDOR_LIBRARY_FILENAMES -u __GLX_VENDOR_LIBRARY_NAME
  -u LIBGL_ALWAYS_INDIRECT -u MESA_LOADER_DRIVER_OVERRIDE -u LIBGL_DRIVERS_PATH
)

if [[ "$TOTAL_CLIPS" -le 0 || "$NUM_WORKERS" -le 0 ]]; then
  echo "[ERROR] TOTAL_CLIPS and NUM_WORKERS must be positive" >&2
  exit 1
fi
if [[ "$NUM_WORKERS" -gt 16 ]]; then
  echo "[ERROR] More than 16 llvmpipe workers oversubscribes this node" >&2
  exit 1
fi
if [[ "$NUM_WORKERS" -gt "$TOTAL_CLIPS" ]]; then
  NUM_WORKERS="$TOTAL_CLIPS"
fi
for path in "$PROJECT_DIR/collect/panoramic/collector.py" \
    "$XVFB_BIN" "$XDPYINFO_BIN" "$GLXINFO_BIN" "$X11_DRI_PATH/swrast_dri.so" \
    "$X11_XKB_PATH" "$X11_FONT_PATH" "$VLNCE_PYTHON"; do
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] Missing required path: $path" >&2
    exit 1
  fi
done
if ! grep -q -- "--episode-modulo" "$PROJECT_DIR/collect/panoramic/collector.py"; then
  echo "[ERROR] collector.py lacks the sharding patch (--episode-modulo)" >&2
  exit 1
fi

mkdir -p "$OUTPUT" "${PROJECT_DIR}/logs"
exec 9>"${OUTPUT}/.parallel_collection.lock"
if ! flock -n 9; then
  echo "[ERROR] Another parallel collector is already using $OUTPUT" >&2
  exit 1
fi
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LAUNCH_LOG="${PROJECT_DIR}/logs/collect_pano_${SPLIT}_${TOTAL_CLIPS}c_${NUM_WORKERS}w_${RUN_STAMP}.log"
XVFB_RUNTIME="${PROJECT_DIR}/logs/xvfb_pano_${RUN_STAMP}"
exec > >(tee -a "$LAUNCH_LOG") 2>&1

echo "============================================================"
echo "R2R-CE Panoramic Collection (MetaX C500, Xvfb + llvmpipe)"
echo "============================================================"
echo "Output:        $OUTPUT"
echo "Split:         $SPLIT"
echo "Total clips:   $TOTAL_CLIPS"
echo "Workers:       $NUM_WORKERS"
echo "Displays:      :${BASE_DISPLAY}..:$((BASE_DISPLAY + NUM_WORKERS - 1))"
echo "LP threads:    $LP_THREADS per worker"
echo "Launcher log:  $LAUNCH_LOG"
echo "============================================================"

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
X11_TOOL_LD_LIBRARY_PATH="${X11_BUNDLE}/lib:${LD_LIBRARY_PATH}"

declare -a XVFB_PIDS=()
cleanup() {
  for pid in "${XVFB_PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${WORKER_PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM EXIT

# ------------------------------------------------------------
# 1. One isolated Xvfb per worker, from the certified bundle.
# ------------------------------------------------------------
for ((worker = 0; worker < NUM_WORKERS; worker++)); do
  display_num=$((BASE_DISPLAY + worker))
  display_addr="localhost:${display_num}.0"
  if env "${GL_ENV_UNSET_ARGS[@]}" LD_LIBRARY_PATH="$X11_TOOL_LD_LIBRARY_PATH" \
      DISPLAY="$display_addr" timeout 5 "$XDPYINFO_BIN" >/dev/null 2>&1; then
    echo "[ERROR] DISPLAY $display_addr is already active; choose another BASE_DISPLAY" >&2
    exit 1
  fi
  xvfb_dir="${XVFB_RUNTIME}/display_${display_num}"
  mkdir -p "${xvfb_dir}/.xkb-cache"
  (
    cd "$xvfb_dir"
    exec 9<"${xvfb_dir}/.xkb-cache"
    exec env "${GL_ENV_UNSET_ARGS[@]}" \
      PATH="${X11_BUNDLE}/bin:$PATH" \
      LD_LIBRARY_PATH="$X11_TOOL_LD_LIBRARY_PATH" \
      LIBGL_DRIVERS_PATH="$X11_DRI_PATH" \
      LIBGL_ALWAYS_SOFTWARE=1 \
      GALLIUM_DRIVER=llvmpipe \
      MESA_LOADER_DRIVER_OVERRIDE=swrast \
      LP_NUM_THREADS="$LP_THREADS" \
      "$XVFB_BIN" ":${display_num}" \
      -screen 0 1024x768x24 -nolock -nolisten unix -listen tcp +iglx -ac \
      -fp "$X11_FONT_PATH" -xkbdir "$X11_XKB_PATH"
  ) >"${xvfb_dir}/xvfb.log" 2>&1 &
  XVFB_PIDS+=("$!")

  ready=0
  for _ in $(seq 1 60); do
    if ! kill -0 "${XVFB_PIDS[-1]}" 2>/dev/null; then
      echo "[ERROR] Xvfb :${display_num} exited during startup" >&2
      tail -50 "${xvfb_dir}/xvfb.log" >&2 || true
      exit 1
    fi
    if env "${GL_ENV_UNSET_ARGS[@]}" LD_LIBRARY_PATH="$X11_TOOL_LD_LIBRARY_PATH" \
        DISPLAY="$display_addr" timeout 5 "$XDPYINFO_BIN" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 1
  done
  if [[ "$ready" != 1 ]]; then
    echo "[ERROR] Xvfb :${display_num} did not become ready" >&2
    exit 1
  fi
  renderer="$(env "${GL_ENV_UNSET_ARGS[@]}" \
    LD_LIBRARY_PATH="$X11_TOOL_LD_LIBRARY_PATH" \
    LIBGL_DRIVERS_PATH="$X11_DRI_PATH" \
    DISPLAY="$display_addr" \
    LIBGL_ALWAYS_SOFTWARE=1 \
    GALLIUM_DRIVER=llvmpipe \
    MESA_LOADER_DRIVER_OVERRIDE=swrast \
    timeout 120 "$GLXINFO_BIN" -B 2>/dev/null | \
    grep -F 'OpenGL renderer string:' | head -1 || true)"
  if [[ "${renderer,,}" != *llvmpipe* ]]; then
    echo "[ERROR] DISPLAY $display_addr is not llvmpipe: ${renderer:-no renderer}" >&2
    exit 1
  fi
  echo "[XVFB] :${display_num} ready (${renderer#*: })"
done

# ------------------------------------------------------------
# 2. Sharded collection workers.
# ------------------------------------------------------------
# Blank cluster containers only mount /mnt/afs: call the env python by
# absolute path instead of depending on /opt/conda + activation.
echo "[INFO] Python: $VLNCE_PYTHON ($("$VLNCE_PYTHON" --version 2>&1))"

BASE_COUNT=$((TOTAL_CLIPS / NUM_WORKERS))
REMAINDER=$((TOTAL_CLIPS % NUM_WORKERS))
declare -a WORKER_PIDS=()

run_worker() {
  local worker="$1"
  local target="$2"
  local display_addr="$3"
  local id_start=$((worker * CLIP_ID_BLOCK + 1))
  local id_end=$((worker * CLIP_ID_BLOCK + target))
  local worker_log="${PROJECT_DIR}/logs/collect_pano_w${worker}_${RUN_STAMP}.log"

  echo "[LAUNCH] worker=$worker shard=${worker}/${NUM_WORKERS} target=$target " \
       "ids=${id_start}..${id_end} display=$display_addr log=$worker_log"
  (
    cd "$PROJECT_DIR"
    export DISPLAY="$display_addr"
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES=0
    export LP_NUM_THREADS="$LP_THREADS"
    export OMP_NUM_THREADS="$LP_THREADS"
    ulimit -c 0
    # Blank cluster containers ship no system GL at all (magnum dlopens
    # libOpenGL.so.0), so the client uses the bundle's Mesa/GLVND stack with
    # forced llvmpipe software rendering — the same certified client setup as
    # the 8-GPU eval.  This path is worker-local; never inject it globally.
    export LD_LIBRARY_PATH="${X11_BUNDLE}/mesa_lib:${LD_LIBRARY_PATH:-}"
    export LIBGL_DRIVERS_PATH="$X11_DRI_PATH"
    export LIBGL_ALWAYS_SOFTWARE=1
    export GALLIUM_DRIVER=llvmpipe
    export MESA_LOADER_DRIVER_OVERRIDE=swrast
    unset WAYLAND_DISPLAY EGL_PLATFORM __EGL_VENDOR_LIBRARY_FILENAMES
    unset __GLX_VENDOR_LIBRARY_NAME LIBGL_ALWAYS_INDIRECT
    exec "$VLNCE_PYTHON" -m collect panoramic \
      --output "$OUTPUT" \
      --split "$SPLIT" \
      --num-clips "$id_end" \
      --max-steps "$MAX_STEPS" \
      --num-workers "$IO_WORKERS" \
      --gpu 0 \
      --depth-directions front front_down \
      --episode-modulo "$NUM_WORKERS" \
      --episode-remainder "$worker" \
      --clip-id-start "$id_start" \
      --clip-id-end "$id_end"
  ) >"$worker_log" 2>&1
}

for ((worker = 0; worker < NUM_WORKERS; worker++)); do
  target="$BASE_COUNT"
  if ((worker < REMAINDER)); then
    target=$((target + 1))
  fi
  display_addr="localhost:$((BASE_DISPLAY + worker)).0"
  run_worker "$worker" "$target" "$display_addr" &
  WORKER_PIDS+=("$!")
done

STATUS=0
for index in "${!WORKER_PIDS[@]}"; do
  if wait "${WORKER_PIDS[$index]}"; then
    echo "[DONE] worker=$index"
  else
    echo "[FAILED] worker=$index (see logs/collect_pano_w${index}_${RUN_STAMP}.log)" >&2
    STATUS=1
  fi
done

TOTAL_COLLECTED="$(find "$OUTPUT/$SPLIT" -mindepth 2 -maxdepth 2 -type d -name 'clip_*' 2>/dev/null | wc -l)"
echo "============================================================"
echo "[SUMMARY] clips now in $OUTPUT/$SPLIT: $TOTAL_COLLECTED"
echo "[SUMMARY] launcher log: $LAUNCH_LOG"
echo "============================================================"
exit "$STATUS"
