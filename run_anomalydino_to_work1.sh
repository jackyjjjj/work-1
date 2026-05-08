#!/usr/bin/env bash
set -euo pipefail

# End-to-end bridge: AnomalyDINO heatmaps -> work-1 pseudo-bbox manifest.
# Directory layout expected by default:
#   /home/jack/workspace/data/MVTec-FS
#   /home/jack/workspace/heatmap/AnomalyDINO
#   /home/jack/workspace/work-1
WORKSPACE_DIR="${WORKSPACE_DIR:-/home/jack/workspace}"
WORK1_DIR="${WORK1_DIR:-${WORKSPACE_DIR}/work-1}"
ANOMALYDINO_DIR="${ANOMALYDINO_DIR:-${WORKSPACE_DIR}/heatmap/AnomalyDINO}"
DATA_ROOT="${DATA_ROOT:-${WORKSPACE_DIR}/data/MVTec-FS}"
MANIFEST="${MANIFEST:-${WORK1_DIR}/data/manifests/mvtec_fs.csv}"

CONDA_EXE="${CONDA_EXE:-/home/jack/miniconda3/bin/conda}"
WORK1_ENV="${WORK1_ENV:-work-1}"
ANOMALYDINO_ENV="${ANOMALYDINO_ENV:-AnomalyDINO}"
# Use an environment with numpy/cv2 for fast heatmap upsampling.
BBOX_ENV="${BBOX_ENV:-${ANOMALYDINO_ENV}}"

MODEL="${MODEL:-dinov2_vits14}"
DEVICE="${DEVICE:-cuda:0}"
IMAGE_SIZE="${IMAGE_SIZE:-518}"
SHOT="${SHOT:-1}"
SEED="${SEED:-0}"
NUM_SEEDS="${NUM_SEEDS:-1}"
HEATMAP_CLASS="${HEATMAP_CLASS:-gt}"
OBJECTS="${OBJECTS:-bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile transistor wood zipper}"

# AnomalyDINO reads CONFIG/<object>_config1/valid.csv; in the dataset paths these rows
# live under image/<object>/testing/..., which work-1 maps to split=test.
SPLIT="${SPLIT:-test}"
PERCENTILE="${PERCENTILE:-0.90}"
MIN_AREA_RATIO="${MIN_AREA_RATIO:-0.001}"
COMPONENT="${COMPONENT:-max-score}"
DUPLICATE_POLICY="${DUPLICATE_POLICY:-max}"

RAW_OUTPUT_DIR="${RAW_OUTPUT_DIR:-${WORK1_DIR}/outputs/heatmaps/anomalydino_raw}"
HEATMAP_DIR="${HEATMAP_DIR:-${WORK1_DIR}/outputs/heatmaps}"
MERGED_HEATMAP="${MERGED_HEATMAP:-${HEATMAP_DIR}/anomalydino_${SPLIT}_${SHOT}shot_seed${SEED}_merged.jsonl}"
DEDUP_HEATMAP="${DEDUP_HEATMAP:-${HEATMAP_DIR}/anomalydino_${SPLIT}_${SHOT}shot_seed${SEED}_dedup_${DUPLICATE_POLICY}.jsonl}"
PSEUDO_MANIFEST="${PSEUDO_MANIFEST:-${WORK1_DIR}/data/manifests/mvtec_fs_anomalydino_pseudo_bbox_${SPLIT}_${SHOT}shot_seed${SEED}.csv}"

mkdir -p "${RAW_OUTPUT_DIR}" "${HEATMAP_DIR}" "$(dirname "${MANIFEST}")" "$(dirname "${PSEUDO_MANIFEST}")"

if [[ ! -x "${CONDA_EXE}" ]]; then
  echo "Missing conda executable: ${CONDA_EXE}" >&2
  exit 1
fi
if [[ ! -d "${WORK1_DIR}" ]]; then
  echo "Missing work-1 dir: ${WORK1_DIR}" >&2
  exit 1
fi
if [[ ! -d "${ANOMALYDINO_DIR}" ]]; then
  echo "Missing AnomalyDINO dir: ${ANOMALYDINO_DIR}" >&2
  exit 1
fi
if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "Missing DATA_ROOT: ${DATA_ROOT}" >&2
  exit 1
fi

cd "${WORK1_DIR}"
if [[ ! -f "${MANIFEST}" ]]; then
  echo "Manifest not found; building: ${MANIFEST}"
  "${CONDA_EXE}" run -n "${WORK1_ENV}" python scripts/build_mvtec_fs_manifest.py \
    --dataset-root "${DATA_ROOT}" \
    --output "${MANIFEST}"
fi

cd "${ANOMALYDINO_DIR}"
"${CONDA_EXE}" run -n "${ANOMALYDINO_ENV}" python run_mvtecfs_anomalydino.py \
  --data_root "${DATA_ROOT}" \
  --output_dir "${RAW_OUTPUT_DIR}" \
  --objects ${OBJECTS} \
  --shots "${SHOT}" \
  --num_seeds "${NUM_SEEDS}" \
  --model_name "${MODEL}" \
  --image_size "${IMAGE_SIZE}" \
  --device "${DEVICE}" \
  --mv_method mso \
  --k_neighbors 1 \
  --heatmap_class "${HEATMAP_CLASS}"

shopt -s nullglob
heatmap_parts=("${RAW_OUTPUT_DIR}/${MODEL}"/*_"${SHOT}"shot_seed"${SEED}".jsonl)
if (( ${#heatmap_parts[@]} == 0 )); then
  echo "No AnomalyDINO heatmap parts found under ${RAW_OUTPUT_DIR}/${MODEL}" >&2
  exit 1
fi
cat "${heatmap_parts[@]}" > "${MERGED_HEATMAP}"
echo "Merged ${#heatmap_parts[@]} object heatmap files -> ${MERGED_HEATMAP}"

# Normalize duplicates before calling work-1. This also keeps the deduped JSONL reusable.
"${CONDA_EXE}" run -n "${WORK1_ENV}" python "${WORK1_DIR}/scripts/dedupe_heatmap_jsonl.py" \
  --input "${MERGED_HEATMAP}" \
  --output "${DEDUP_HEATMAP}" \
  --policy "${DUPLICATE_POLICY}" \
  --overwrite
cd "${WORK1_DIR}"
"${CONDA_EXE}" run -n "${BBOX_ENV}" python scripts/build_pseudo_bbox_manifest.py \
  --manifest "${MANIFEST}" \
  --heatmap-file "${DEDUP_HEATMAP}" \
  --output "${PSEUDO_MANIFEST}" \
  --split "${SPLIT}" \
  --percentile "${PERCENTILE}" \
  --min-area-ratio "${MIN_AREA_RATIO}" \
  --component "${COMPONENT}" \
  --upsample-heatmap-to-image \
  --missing-policy error \
  --duplicate-policy "${DUPLICATE_POLICY}" \
  --overwrite

echo "Done."
echo "Manifest: ${MANIFEST}"
echo "Raw merged heatmaps: ${MERGED_HEATMAP}"
echo "Deduped heatmaps: ${DEDUP_HEATMAP}"
echo "Pseudo-bbox manifest: ${PSEUDO_MANIFEST}"