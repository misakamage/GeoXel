#!/usr/bin/env bash
# Example:
# UAVSCENES_ROOT=/path/to/uavscene CHECKPOINT=/path/to/checkpoints.pth \
#   bash scripts/run_uavscenes_amtown.sh
# Set DRY_RUN=1 to validate paths and frame selection without running the model.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UAVSCENES_ROOT="${UAVSCENES_ROOT:?Set UAVSCENES_ROOT to the UAVScenes dataset root}"
CHECKPOINT="${CHECKPOINT:-$PROJECT_ROOT/ckpt/checkpoints.pth}"
PYTHON_BIN="${PYTHON_BIN:-python}"

ARGS=(
  --uavscenes-root "$UAVSCENES_ROOT"
  --sequence "${SEQUENCE:-interval1_AMtown01}"
  --checkpoint "$CHECKPOINT"
  --output "${OUTPUT_DIR:-$PROJECT_ROOT/outputs/uavscenes_amtown}"
  --device "${DEVICE:-cuda:0}"
  --start-frame "${START_FRAME:-200}"
  --stride "${STRIDE:-5}"
  --num-frames "${NUM_FRAMES:-200}"
  --max-frames-in-memory "${MAX_FRAMES_IN_MEMORY:-512}"
  --segment-length "${SEGMENT_LENGTH:-32}"
  --overlap "${OVERLAP:-8}"
  --anchor-stride "${ANCHOR_STRIDE:-1}"
  --ttt-steps "${TTT_STEPS:-15}"
  --ttt-lr "${TTT_LR:-1e-4}"
  --ttt-layers "${TTT_LAYERS:-17,23}"
  --dom-point-quality-min-road-frame-ratio "${DOM_POINT_QUALITY_MIN_ROAD_FRAME_RATIO:-0}"
  --seg-pgo-w-dom-xy "${SEG_PGO_W_DOM_XY:-5}"
  --seg-pgo-w-dom-z "${SEG_PGO_W_DOM_Z:-2}"
  --seg-pgo-w-point-xy "${SEG_PGO_W_POINT_XY:-0.5}"
  --seg-pgo-w-point-z "${SEG_PGO_W_POINT_Z:-0.1}"
  --pgo-w-first-frame "${PGO_W_FIRST_FRAME:-1000}"
  --pgo-w-overlap-dom-xy "${PGO_W_OVERLAP_DOM_XY:-5}"
  --pgo-w-overlap-dom-z "${PGO_W_OVERLAP_DOM_Z:-1}"
  --pgo-w-overlap-consist "${PGO_W_OVERLAP_CONSIST:-10}"
  --pgo-w-point-xy "${PGO_W_POINT_XY:-0.5}"
  --pgo-w-point-z "${PGO_W_POINT_Z:-0}"
  --pgo-w-z-dem "${PGO_W_Z_DEM:-5}"
  --pgo-w-camera-agl "${PGO_W_CAMERA_AGL:-5}"
  --pgo-w-scale "${PGO_W_SCALE:-50}"
  --pgo-max-iter "${PGO_MAX_ITER:-200}"
  --roma-variant "${ROMA_VARIANT:-full}"
  --seed "${SEED:-0}"
)

[[ -z "${DOM_TIF:-}" ]] || ARGS+=(--dom "$DOM_TIF")
[[ -z "${DEM_TIF:-}" ]] || ARGS+=(--dem "$DEM_TIF")
[[ -z "${BUILDING_MASK_TIF:-}" ]] || ARGS+=(--building-mask "$BUILDING_MASK_TIF")
[[ -z "${ROAD_MASK_TIF:-}" ]] || ARGS+=(--road-mask "$ROAD_MASK_TIF")
[[ "${DISABLE_ROMA:-0}" != "1" ]] || ARGS+=(--disable-roma)
[[ "${DOM_POINT_CAMERA_TEACHER_PGO_ENABLE:-1}" != "0" ]] || ARGS+=(--no-dom-point-camera-teacher-pgo-enable)
[[ "${DEM_Z_DIRECT_CORRECT:-1}" != "0" ]] || ARGS+=(--no-dem-z-direct-correct)
[[ "${DEM_Z_ROAD_DIRECT_CORRECT:-1}" != "0" ]] || ARGS+=(--no-dem-z-road-direct-correct)
[[ "${NO_POST_PGO:-0}" != "1" ]] || ARGS+=(--no-post-pgo)
[[ "${DRY_RUN:-0}" != "1" ]] || ARGS+=(--dry-run)

exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/run_uavscenes_amtown.py" "${ARGS[@]}"
