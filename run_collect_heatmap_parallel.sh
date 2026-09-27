#!/usr/bin/env bash
set -euo pipefail

# Parallel CPU/GLX collection.  Workers share one dataset root but own disjoint
# global clip-id ranges, displays, seeds, logs, and stats files.

ALLOWED_ROOT="/mnt/afs/lixiaoou/intern/fjl"
SCRIPT_PATH="${BASH_SOURCE[0]}"
PROJECT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"

OUTPUT="${1:-${ALLOWED_ROOT}/data/heatmap_randomwalk_pilot_v1}"
NUM_NEW_CLIPS="${2:-20}"
NUM_WORKERS="${3:-4}"
NUM_WAYPOINTS="${4:-4}"
STORAGE_FORMAT="${5:-chunks}"
BASE_DISPLAY="${6:-200}"

if [[ "$NUM_NEW_CLIPS" -le 0 || "$NUM_WORKERS" -le 0 ]]; then
  echo "[ERROR] NUM_NEW_CLIPS and NUM_WORKERS must be positive" >&2
  exit 1
fi
if [[ "$NUM_WORKERS" -gt 16 ]]; then
  echo "[ERROR] At most 16 pre-existing Xvfb displays (:200..:215) are available" >&2
  exit 1
fi
if [[ "$NUM_WORKERS" -gt "$NUM_NEW_CLIPS" ]]; then
  NUM_WORKERS="$NUM_NEW_CLIPS"
fi

OUTPUT="$(realpath -m "$OUTPUT")"
case "$OUTPUT" in
  "$ALLOWED_ROOT"|"$ALLOWED_ROOT"/*) ;;
  *)
    echo "[ERROR] Output escapes allowed root ${ALLOWED_ROOT}: ${OUTPUT}" >&2
    exit 1
    ;;
esac

mkdir -p "$OUTPUT" "${PROJECT_DIR}/logs"
exec 9>"${OUTPUT}/.parallel_collection.lock"
if ! flock -n 9; then
  echo "[ERROR] Another parallel collector is already using $OUTPUT" >&2
  exit 1
fi
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LAUNCH_LOG="${PROJECT_DIR}/logs/collect_heatmap_parallel_${NUM_NEW_CLIPS}c_${NUM_WORKERS}w_${RUN_STAMP}.log"
exec > >(tee -a "$LAUNCH_LOG") 2>&1

MAX_EXISTING_ID="$({
  find "$OUTPUT" -mindepth 3 -maxdepth 3 -type f -name meta.json -print0 2>/dev/null \
    | xargs -0 -r -n1 dirname \
    | sed -n 's#^.*/clip_\([0-9][0-9]*\)$#\1#p'
} | sort -n | tail -n1)"
MAX_EXISTING_ID="${MAX_EXISTING_ID:-0}"
START_ID="${HEATMAP_START_CLIP_ID:-$((10#$MAX_EXISTING_ID + 1))}"

echo "============================================================"
echo "Parallel Heatmap Patrol Collection"
echo "============================================================"
echo "Output:          $OUTPUT"
echo "New clips:       $NUM_NEW_CLIPS"
echo "Workers:         $NUM_WORKERS"
echo "First clip id:   $START_ID"
echo "Waypoints:       $NUM_WAYPOINTS"
echo "Base display:    :$BASE_DISPLAY"
echo "Launcher log:    $LAUNCH_LOG"
echo "============================================================"

for ((worker = 0; worker < NUM_WORKERS; worker++)); do
  display=$((BASE_DISPLAY + worker))
  if ! DISPLAY="localhost:${display}.0" timeout 10 xdpyinfo >/dev/null 2>&1; then
    echo "[ERROR] X display localhost:${display}.0 is unavailable" >&2
    exit 1
  fi
done

BASE_COUNT=$((NUM_NEW_CLIPS / NUM_WORKERS))
REMAINDER=$((NUM_NEW_CLIPS % NUM_WORKERS))
OFFSET=0
PIDS=()

run_worker() {
  local worker="$1"
  local count="$2"
  local worker_start="$3"
  local display="$4"
  local worker_seed="$5"
  local stats_file="$6"
  local max_retries="${HEATMAP_WORKER_RETRIES:-2}"
  local attempt=0

  while true; do
    if env \
      VLNCE_DISPLAY="localhost:${display}.0" \
      HEATMAP_WORKER_ID="$worker" \
      HEATMAP_EPISODE_SHARD_ID="$worker" \
      HEATMAP_EPISODE_SHARD_COUNT="$NUM_WORKERS" \
      HEATMAP_EPISODE_OFFSET="${HEATMAP_EPISODE_OFFSET:-0}" \
      HEATMAP_SCENE_BLOCK_SIZE="${HEATMAP_SCENE_BLOCK_SIZE:-8}" \
      HEATMAP_CLIP_ID_START="$worker_start" \
      HEATMAP_STATS_FILE="$stats_file" \
      HEATMAP_SEED="$worker_seed" \
      HEATMAP_RENDER_THREADS="${HEATMAP_RENDER_THREADS:-4}" \
      HEATMAP_MIN_KEYFRAME_TRANSLATION="${HEATMAP_MIN_KEYFRAME_TRANSLATION:-0.20}" \
      HEATMAP_IO_WORKERS="${HEATMAP_IO_WORKERS:-4}" \
      HEATMAP_MAX_PENDING_IO="${HEATMAP_MAX_PENDING_IO:-64}" \
      bash "${PROJECT_DIR}/run_collect_heatmap.sh" \
        "$OUTPUT" "$count" "$NUM_WAYPOINTS" "$STORAGE_FORMAT" 0; then
      return 0
    fi

    if ((attempt >= max_retries)); then
      echo "[FAILED] worker=$worker exhausted $max_retries retries" >&2
      return 1
    fi
    attempt=$((attempt + 1))
    echo "[RETRY] worker=$worker attempt=$attempt/$max_retries; preserving valid clips and retrying its id range" >&2
  done
}

terminate_children() {
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap terminate_children INT TERM

for ((worker = 0; worker < NUM_WORKERS; worker++)); do
  count="$BASE_COUNT"
  if ((worker < REMAINDER)); then
    count=$((count + 1))
  fi
  worker_start=$((START_ID + OFFSET))
  display=$((BASE_DISPLAY + worker))
  worker_seed=$((${HEATMAP_SEED_BASE:-42} + worker))
  stats_file="collection_stats_worker_${worker}_${RUN_STAMP}.json"

  echo "[LAUNCH] worker=$worker clips=$count ids=${worker_start}..$((worker_start + count - 1)) display=:$display seed=$worker_seed"
  run_worker \
    "$worker" "$count" "$worker_start" "$display" "$worker_seed" "$stats_file" &
  PIDS+=("$!")
  OFFSET=$((OFFSET + count))
done

STATUS=0
for index in "${!PIDS[@]}"; do
  if wait "${PIDS[$index]}"; then
    echo "[DONE] worker=$index pid=${PIDS[$index]}"
  else
    echo "[FAILED] worker=$index pid=${PIDS[$index]}" >&2
    STATUS=1
  fi
done
if [[ "$STATUS" -ne 0 ]]; then
  echo "[ERROR] At least one collection worker failed; audit not started." >&2
  exit "$STATUS"
fi

source /opt/conda/etc/profile.d/conda.sh
conda activate "${ALLOWED_ROOT}/envs/vlnce"
AUDIT_ARGS=("$OUTPUT")
if [[ -n "${HEATMAP_AUDIT_MAX_CLIPS:-}" ]]; then
  AUDIT_ARGS+=(--max-clips "$HEATMAP_AUDIT_MAX_CLIPS")
fi
python "${PROJECT_DIR}/scripts/audit_heatmap_collection.py" "${AUDIT_ARGS[@]}"

echo "============================================================"
echo "[DONE] Parallel collection and structural audit passed"
echo "Output: $OUTPUT"
echo "Log:    $LAUNCH_LOG"
echo "============================================================"
