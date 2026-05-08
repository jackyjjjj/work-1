#!/usr/bin/env bash
set -euo pipefail

# Run all configured heatmap methods through the same work-1 pipeline:
# heatmap JSONL -> pseudo-mask -> DINOv2 mask features -> few-shot classification.
#
# This wrapper keeps going when a model is missing weights and writes a summary
# instead of stopping before the other methods are evaluated.

WORKSPACE_DIR="${WORKSPACE_DIR:-/home/jack/workspace}"
WORK1_DIR="${WORK1_DIR:-${WORKSPACE_DIR}/work-1}"
HEATMAP_ROOT="${HEATMAP_ROOT:-${WORKSPACE_DIR}/heatmap}"
CONDA_EXE="${CONDA_EXE:-/home/jack/miniconda3/bin/conda}"
RUN_ENV="${RUN_ENV:-AnomalyDINO}"
WORK1_ENV="${WORK1_ENV:-work-1}"
DEVICE="${DEVICE:-cuda:0}"
SPLIT="${SPLIT:-test}"
EPISODES="${EPISODES:-100}"
Q_QUERIES="${Q_QUERIES:-5}"
EVAL_GRID="${EVAL_GRID:-5:1,5:3,5:5}"

# Set FORCE_RERUN=0 to reuse an existing classification result instead of re-running work-1 post-processing.
FORCE_RERUN="${FORCE_RERUN:-1}"
# Keep AnomalyDINO quick by default when its raw heatmaps already exist.
ANOMALYDINO_SKIP_HEATMAP_GENERATION="${ANOMALYDINO_SKIP_HEATMAP_GENERATION:-1}"

SUMMARY_DIR="${SUMMARY_DIR:-${WORK1_DIR}/outputs/results}"
SUMMARY_MD="${SUMMARY_MD:-${SUMMARY_DIR}/three_heatmap_models_summary.md}"
SUMMARY_JSONL="${SUMMARY_JSONL:-${SUMMARY_DIR}/three_heatmap_models_summary.jsonl}"

mkdir -p "${SUMMARY_DIR}"
: > "${SUMMARY_JSONL}"

cd "${WORK1_DIR}"

write_jsonl() {
  local model="$1"
  local status="$2"
  local result_md="$3"
  local message="$4"
  python3 - "$SUMMARY_JSONL" "$model" "$status" "$result_md" "$message" <<'PY'
import json
import sys
path, model, status, result_md, message = sys.argv[1:]
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps({
        "model": model,
        "status": status,
        "result_md": result_md,
        "message": message,
    }, ensure_ascii=False) + "\n")
PY
}

has_realnet_checkpoints() {
  local missing=0
  local object_name
  for object_name in bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile transistor wood zipper; do
    local ckpt="${HEATMAP_ROOT}/RealNet/experiments/MVTec-AD/realnet_checkpoints/${object_name}/ckpt_best.pth.tar"
    [[ -f "${ckpt}" ]] || missing=$((missing + 1))
  done
  [[ "${missing}" == "0" ]]
}

has_efficientad_weights() {
  local model_size="$1"
  local missing=0
  local object_name
  for object_name in bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile transistor wood zipper; do
    local base="${HEATMAP_ROOT}/EfficientAD/output/1/trainings/mvtec_ad/${object_name}"
    [[ -f "${base}/student_final.pth" ]] || missing=$((missing + 1))
    [[ -f "${base}/autoencoder_final.pth" ]] || missing=$((missing + 1))
  done
  [[ -f "${HEATMAP_ROOT}/EfficientAD/models/teacher_${model_size}.pth" ]] || missing=$((missing + 1))
  [[ "${missing}" == "0" ]]
}

run_or_record() {
  local model="$1"
  local command="$2"
  local result_md="$3"

  if [[ "${FORCE_RERUN}" != "1" && -f "${result_md}" ]]; then
    echo "[${model}] result exists, skipping: ${result_md}"
    write_jsonl "${model}" "existing" "${result_md}" "result already exists"
    return 0
  fi

  echo "[${model}] running..."
  set +e
  bash -lc "${command}"
  local exit_code=$?
  set -e
  if [[ "${exit_code}" == "0" && -f "${result_md}" ]]; then
    write_jsonl "${model}" "ok" "${result_md}" "completed"
    return 0
  fi
  write_jsonl "${model}" "failed" "${result_md}" "command exited ${exit_code}"
  return 0
}

ANOMALYDINO_RESULT="${WORK1_DIR}/outputs/results/anomalydino_${SPLIT}_1shot_seed0_mask_prototype_grid.md"
run_or_record \
  "AnomalyDINO" \
  "PROJECT=anomalydino RUN_ENV='${RUN_ENV}' WORK1_ENV='${WORK1_ENV}' DEVICE='${DEVICE}' SPLIT='${SPLIT}' EPISODES='${EPISODES}' Q_QUERIES='${Q_QUERIES}' EVAL_GRID='${EVAL_GRID}' SKIP_HEATMAP_GENERATION='${ANOMALYDINO_SKIP_HEATMAP_GENERATION}' bash '${WORK1_DIR}/run_heatmap_to_work1.sh'" \
  "${ANOMALYDINO_RESULT}"

REALNET_RESULT="${WORK1_DIR}/outputs/results/realnet_${SPLIT}_mask_prototype_grid.md"
if has_realnet_checkpoints; then
  run_or_record \
    "RealNet" \
    "PROJECT=realnet RUN_ENV='${RUN_ENV}' WORK1_ENV='${WORK1_ENV}' DEVICE='${DEVICE}' SPLIT='${SPLIT}' EPISODES='${EPISODES}' Q_QUERIES='${Q_QUERIES}' EVAL_GRID='${EVAL_GRID}' bash '${WORK1_DIR}/run_heatmap_to_work1.sh'" \
    "${REALNET_RESULT}"
else
  echo "[RealNet] skipped: missing per-object ckpt_best.pth.tar files"
  write_jsonl "RealNet" "skipped" "${REALNET_RESULT}" "missing per-object RealNet checkpoints"
fi

EFFICIENTAD_MODEL_SIZE="${EFFICIENTAD_MODEL_SIZE:-small}"
EFFICIENTAD_RESULT="${WORK1_DIR}/outputs/results/efficientad_${EFFICIENTAD_MODEL_SIZE}_${SPLIT}_mask_prototype_grid.md"
if has_efficientad_weights "${EFFICIENTAD_MODEL_SIZE}"; then
  run_or_record \
    "EfficientAD-${EFFICIENTAD_MODEL_SIZE}" \
    "PROJECT=efficientad MODEL_SIZE='${EFFICIENTAD_MODEL_SIZE}' RUN_ENV='${RUN_ENV}' WORK1_ENV='${WORK1_ENV}' DEVICE='${DEVICE}' SPLIT='${SPLIT}' EPISODES='${EPISODES}' Q_QUERIES='${Q_QUERIES}' EVAL_GRID='${EVAL_GRID}' bash '${WORK1_DIR}/run_heatmap_to_work1.sh'" \
    "${EFFICIENTAD_RESULT}"
else
  echo "[EfficientAD-${EFFICIENTAD_MODEL_SIZE}] skipped: missing per-object student/autoencoder weights"
  write_jsonl "EfficientAD-${EFFICIENTAD_MODEL_SIZE}" "skipped" "${EFFICIENTAD_RESULT}" "missing per-object EfficientAD student_final.pth / autoencoder_final.pth weights"
fi

"${CONDA_EXE}" run -n "${WORK1_ENV}" python scripts/summarize_heatmap_model_results.py \
  --summary-jsonl "${SUMMARY_JSONL}" \
  --output-md "${SUMMARY_MD}"

echo "Summary: ${SUMMARY_MD}"
