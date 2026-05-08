#!/usr/bin/env bash
set -euo pipefail

# End-to-end bridge: heatmap project -> work-1 pseudo-mask / classification.
# Supported PROJECT values: anomalydino, realnet, efficientad, external.
# For new methods, either add a PROJECT branch below or use PROJECT=external with
# HEATMAP_INPUT_FILES / HEATMAP_INPUT_DIR pointing at compatible JSONL heatmaps.
WORKSPACE_DIR="${WORKSPACE_DIR:-/home/jack/workspace}"
WORK1_DIR="${WORK1_DIR:-${WORKSPACE_DIR}/work-1}"
HEATMAP_ROOT="${HEATMAP_ROOT:-${WORKSPACE_DIR}/heatmap}"
DATA_ROOT="${DATA_ROOT:-${WORKSPACE_DIR}/data/MVTec-FS}"
MANIFEST="${MANIFEST:-${WORK1_DIR}/data/manifests/mvtec_fs.csv}"

CONDA_EXE="${CONDA_EXE:-/home/jack/miniconda3/bin/conda}"
RUN_ENV="${RUN_ENV:-AnomalyDINO}"
WORK1_ENV="${WORK1_ENV:-work-1}"
PROJECT="${PROJECT:-anomalydino}"
DEVICE="${DEVICE:-cuda:0}"
SPLIT="${SPLIT:-test}"
FS_SPLIT="${FS_SPLIT:-valid}"
METHOD_NAME="${METHOD_NAME:-}"
PERCENTILE="${PERCENTILE:-0.90}"
MIN_AREA_RATIO="${MIN_AREA_RATIO:-0.001}"
MASK_COMPONENT="${MASK_COMPONENT:-all}"
BBOX_COMPONENT="${BBOX_COMPONENT:-max-score}"
DUPLICATE_POLICY="${DUPLICATE_POLICY:-max}"
VALIDATE_HEATMAPS="${VALIDATE_HEATMAPS:-1}"
RUN_PSEUDO_BBOX="${RUN_PSEUDO_BBOX:-1}"
RUN_PSEUDO_MASK="${RUN_PSEUDO_MASK:-1}"
RUN_FEATURES="${RUN_FEATURES:-1}"
RUN_CLASSIFICATION="${RUN_CLASSIFICATION:-1}"
OBJECTS="${OBJECTS:-bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile transistor wood zipper}"

FEATURE_REGION="${FEATURE_REGION:-mask}"
MASK_BACKGROUND="${MASK_BACKGROUND:-black}"
FEATURE_DIM="${FEATURE_DIM:-384}"
EVAL_GRID="${EVAL_GRID:-5:1,5:3,5:5}"
Q_QUERIES="${Q_QUERIES:-5}"
EPISODES="${EPISODES:-100}"
DINOV2_REPO_OR_DIR="${DINOV2_REPO_OR_DIR:-/home/jack/.cache/torch/hub/facebookresearch_dinov2_main}"

mkdir -p "${WORK1_DIR}/outputs/heatmaps" "${WORK1_DIR}/outputs/masks" "${WORK1_DIR}/outputs/features" "${WORK1_DIR}/outputs/results"

if [[ ! -x "${CONDA_EXE}" ]]; then
  echo "Missing conda executable: ${CONDA_EXE}" >&2
  exit 1
fi
if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "Missing DATA_ROOT: ${DATA_ROOT}" >&2
  exit 1
fi

cd "${WORK1_DIR}"
if [[ ! -f "${MANIFEST}" ]]; then
  "${CONDA_EXE}" run -n "${WORK1_ENV}" python scripts/build_mvtec_fs_manifest.py \
    --dataset-root "${DATA_ROOT}" \
    --output "${MANIFEST}"
fi

case "${PROJECT}" in
  anomalydino)
    SOURCE_DIR="${HEATMAP_ROOT}/AnomalyDINO"
    MODEL="${MODEL:-dinov2_vits14}"
    IMAGE_SIZE="${IMAGE_SIZE:-518}"
    SHOT="${SHOT:-1}"
    SEED="${SEED:-0}"
    NUM_SEEDS="${NUM_SEEDS:-1}"
    HEATMAP_CLASS="${HEATMAP_CLASS:-gt}"
    RAW_OUTPUT_DIR="${RAW_OUTPUT_DIR:-${WORK1_DIR}/outputs/heatmaps/anomalydino_raw}"
    HEATMAP_GLOB_DIR="${RAW_OUTPUT_DIR}/${MODEL}"
    MERGED_HEATMAP="${MERGED_HEATMAP:-${WORK1_DIR}/outputs/heatmaps/anomalydino_${SPLIT}_${SHOT}shot_seed${SEED}_merged.jsonl}"
    DEDUP_HEATMAP="${DEDUP_HEATMAP:-${WORK1_DIR}/outputs/heatmaps/anomalydino_${SPLIT}_${SHOT}shot_seed${SEED}_dedup_${DUPLICATE_POLICY}.jsonl}"
    PREFIX="${PREFIX:-anomalydino_${SPLIT}_${SHOT}shot_seed${SEED}}"

    cd "${SOURCE_DIR}"
    if [[ "${SKIP_HEATMAP_GENERATION:-0}" != "1" ]]; then
      "${CONDA_EXE}" run -n "${RUN_ENV}" python run_mvtecfs_anomalydino.py \
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
    fi

    shopt -s nullglob
    heatmap_parts=("${HEATMAP_GLOB_DIR}"/*_"${SHOT}"shot_seed"${SEED}".jsonl)
    ;;

  realnet)
    SOURCE_DIR="${HEATMAP_ROOT}/RealNet"
    LOCALIZER="${LOCALIZER:-realnet}"
    RAW_OUTPUT_DIR="${RAW_OUTPUT_DIR:-${WORK1_DIR}/outputs/heatmaps/realnet_raw}"
    HEATMAP_GLOB_DIR="${RAW_OUTPUT_DIR}/${LOCALIZER}"
    MERGED_HEATMAP="${MERGED_HEATMAP:-${WORK1_DIR}/outputs/heatmaps/realnet_${SPLIT}_merged.jsonl}"
    DEDUP_HEATMAP="${DEDUP_HEATMAP:-${WORK1_DIR}/outputs/heatmaps/realnet_${SPLIT}_dedup_${DUPLICATE_POLICY}.jsonl}"
    PREFIX="${PREFIX:-realnet_${SPLIT}}"

    cd "${SOURCE_DIR}"
    if [[ "${SKIP_HEATMAP_GENERATION:-0}" != "1" ]]; then
      "${CONDA_EXE}" run -n "${RUN_ENV}" python run_mvtecfs_realnet.py \
        --data_root "${DATA_ROOT}" \
        --output_dir "${RAW_OUTPUT_DIR}" \
        --objects ${OBJECTS} \
        --split "${FS_SPLIT}" \
        --device "${DEVICE}" \
        --missing-checkpoint-policy error \
        --localizer "${LOCALIZER}"
    fi

    shopt -s nullglob
    heatmap_parts=("${HEATMAP_GLOB_DIR}"/*_"${FS_SPLIT}".jsonl)
    ;;

  efficientad)
    SOURCE_DIR="${HEATMAP_ROOT}/EfficientAD"
    MODEL_SIZE="${MODEL_SIZE:-small}"
    LOCALIZER="${LOCALIZER:-efficientad_${MODEL_SIZE}}"
    RAW_OUTPUT_DIR="${RAW_OUTPUT_DIR:-${WORK1_DIR}/outputs/heatmaps/efficientad_raw}"
    HEATMAP_GLOB_DIR="${RAW_OUTPUT_DIR}/${LOCALIZER}"
    MERGED_HEATMAP="${MERGED_HEATMAP:-${WORK1_DIR}/outputs/heatmaps/efficientad_${MODEL_SIZE}_${SPLIT}_merged.jsonl}"
    DEDUP_HEATMAP="${DEDUP_HEATMAP:-${WORK1_DIR}/outputs/heatmaps/efficientad_${MODEL_SIZE}_${SPLIT}_dedup_${DUPLICATE_POLICY}.jsonl}"
    PREFIX="${PREFIX:-efficientad_${MODEL_SIZE}_${SPLIT}}"

    cd "${SOURCE_DIR}"
    if [[ "${SKIP_HEATMAP_GENERATION:-0}" != "1" ]]; then
      "${CONDA_EXE}" run -n "${RUN_ENV}" python run_mvtecfs_efficientad.py \
        --data_root "${DATA_ROOT}" \
        --output_dir "${RAW_OUTPUT_DIR}" \
        --objects ${OBJECTS} \
        --split "${FS_SPLIT}" \
        --normalization-split train \
        --model-size "${MODEL_SIZE}" \
        --device "${DEVICE}" \
        --missing-model-policy error \
        --localizer "${LOCALIZER}"
    fi

    shopt -s nullglob
    heatmap_parts=("${HEATMAP_GLOB_DIR}"/*_"${FS_SPLIT}".jsonl)
    ;;

  external)
    # Generic method hook. The external method only needs to provide one or more
    # JSONL files following the work-1 heatmap contract:
    # image_path, heatmap, image_width, image_height.
    METHOD_NAME="${METHOD_NAME:-external}"
    PREFIX="${PREFIX:-${METHOD_NAME}_${SPLIT}}"
    MERGED_HEATMAP="${MERGED_HEATMAP:-${WORK1_DIR}/outputs/heatmaps/${PREFIX}_merged.jsonl}"
    DEDUP_HEATMAP="${DEDUP_HEATMAP:-${WORK1_DIR}/outputs/heatmaps/${PREFIX}_dedup_${DUPLICATE_POLICY}.jsonl}"
    shopt -s nullglob
    heatmap_parts=()
    if [[ -n "${HEATMAP_INPUT_FILES:-}" ]]; then
      # shellcheck disable=SC2206
      heatmap_parts=(${HEATMAP_INPUT_FILES})
    elif [[ -n "${HEATMAP_INPUT_DIR:-}" ]]; then
      HEATMAP_INPUT_GLOB="${HEATMAP_INPUT_GLOB:-*.jsonl}"
      heatmap_parts=("${HEATMAP_INPUT_DIR}"/${HEATMAP_INPUT_GLOB})
    elif [[ -n "${HEATMAP_FILE:-}" ]]; then
      heatmap_parts=("${HEATMAP_FILE}")
    else
      echo "PROJECT=external requires HEATMAP_FILE, HEATMAP_INPUT_FILES, or HEATMAP_INPUT_DIR" >&2
      exit 1
    fi
    ;;

  *)
    echo "Unsupported PROJECT=${PROJECT}; expected anomalydino, realnet, efficientad, or external" >&2
    exit 1
    ;;
esac

METHOD_NAME="${METHOD_NAME:-${PROJECT}}"

if (( ${#heatmap_parts[@]} == 0 )); then
  echo "No heatmap parts found under ${HEATMAP_GLOB_DIR}" >&2
  exit 1
fi
cat "${heatmap_parts[@]}" > "${MERGED_HEATMAP}"
echo "Merged ${#heatmap_parts[@]} heatmap files -> ${MERGED_HEATMAP}"

if [[ "${VALIDATE_HEATMAPS}" == "1" ]]; then
  "${CONDA_EXE}" run -n "${RUN_ENV}" python "${WORK1_DIR}/scripts/validate_heatmap_jsonl.py" \
    --heatmap-file "${MERGED_HEATMAP}" \
    --manifest "${MANIFEST}" \
    --split "${SPLIT}" \
    --coverage-policy error \
    --duplicate-policy "${DUPLICATE_POLICY}" \
    --require-image-size
fi

"${CONDA_EXE}" run -n "${RUN_ENV}" python "${WORK1_DIR}/scripts/dedupe_heatmap_jsonl.py" \
  --input "${MERGED_HEATMAP}" \
  --output "${DEDUP_HEATMAP}" \
  --policy "${DUPLICATE_POLICY}" \
  --overwrite

if [[ "${VALIDATE_HEATMAPS}" == "1" ]]; then
  "${CONDA_EXE}" run -n "${RUN_ENV}" python "${WORK1_DIR}/scripts/validate_heatmap_jsonl.py" \
    --heatmap-file "${DEDUP_HEATMAP}" \
    --manifest "${MANIFEST}" \
    --split "${SPLIT}" \
    --coverage-policy error \
    --duplicate-policy error \
    --require-image-size
fi

cd "${WORK1_DIR}"
PSEUDO_BBOX_MANIFEST="${PSEUDO_BBOX_MANIFEST:-${WORK1_DIR}/data/manifests/mvtec_fs_${PREFIX}_pseudo_bbox.csv}"
PSEUDO_MASK_MANIFEST="${PSEUDO_MASK_MANIFEST:-${WORK1_DIR}/data/manifests/mvtec_fs_${PREFIX}_pseudo_mask.csv}"
MASK_DIR="${MASK_DIR:-${WORK1_DIR}/outputs/masks/${PREFIX}}"
FEATURE_FILE="${FEATURE_FILE:-${WORK1_DIR}/outputs/features/dinov2_mask_${PREFIX}/mvtec_fs_${SPLIT}.jsonl}"
RESULT_JSON="${RESULT_JSON:-${WORK1_DIR}/outputs/results/${PREFIX}_mask_prototype_grid.json}"
RESULT_MD="${RESULT_MD:-${WORK1_DIR}/outputs/results/${PREFIX}_mask_prototype_grid.md}"

if [[ "${RUN_PSEUDO_BBOX}" == "1" ]]; then
  "${CONDA_EXE}" run -n "${RUN_ENV}" python scripts/build_pseudo_bbox_manifest.py \
    --manifest "${MANIFEST}" \
    --heatmap-file "${DEDUP_HEATMAP}" \
    --output "${PSEUDO_BBOX_MANIFEST}" \
    --split "${SPLIT}" \
    --percentile "${PERCENTILE}" \
    --min-area-ratio "${MIN_AREA_RATIO}" \
    --component "${BBOX_COMPONENT}" \
    --upsample-heatmap-to-image \
    --missing-policy error \
    --duplicate-policy "${DUPLICATE_POLICY}" \
    --overwrite
fi

if [[ "${RUN_PSEUDO_MASK}" == "1" ]]; then
  "${CONDA_EXE}" run -n "${RUN_ENV}" python scripts/build_pseudo_mask_manifest.py \
    --manifest "${MANIFEST}" \
    --heatmap-file "${DEDUP_HEATMAP}" \
    --mask-dir "${MASK_DIR}" \
    --output "${PSEUDO_MASK_MANIFEST}" \
    --split "${SPLIT}" \
    --percentile "${PERCENTILE}" \
    --min-area-ratio "${MIN_AREA_RATIO}" \
    --component "${MASK_COMPONENT}" \
    --upsample-heatmap-to-image \
    --missing-policy error \
    --overwrite
fi

if [[ "${RUN_FEATURES}" == "1" ]]; then
  if [[ "${RUN_PSEUDO_MASK}" != "1" && ! -f "${PSEUDO_MASK_MANIFEST}" ]]; then
    echo "RUN_FEATURES=1 requires an existing PSEUDO_MASK_MANIFEST when RUN_PSEUDO_MASK=0: ${PSEUDO_MASK_MANIFEST}" >&2
    exit 1
  fi
  "${CONDA_EXE}" run -n "${RUN_ENV}" python scripts/extract_dinov2_features.py \
    --manifest "${PSEUDO_MASK_MANIFEST}" \
    --image-root "${DATA_ROOT}" \
    --split "${SPLIT}" \
    --region "${FEATURE_REGION}" \
    --mask-background "${MASK_BACKGROUND}" \
    --repo-or-dir "${DINOV2_REPO_OR_DIR}" \
    --source local \
    --output "${FEATURE_FILE}" \
    --device "${DEVICE}" \
    --batch-size 16 \
    --num-workers 2 \
    --overwrite
fi

if [[ "${RUN_CLASSIFICATION}" == "1" ]]; then
  if [[ ! -f "${FEATURE_FILE}" ]]; then
    echo "RUN_CLASSIFICATION=1 requires FEATURE_FILE: ${FEATURE_FILE}" >&2
    exit 1
  fi
  "${CONDA_EXE}" run -n "${WORK1_ENV}" python scripts/run_fewshot_grid.py \
    --manifest "${PSEUDO_MASK_MANIFEST}" \
    --split "${SPLIT}" \
    --grid "${EVAL_GRID}" \
    --q-queries "${Q_QUERIES}" \
    --episodes "${EPISODES}" \
    --feature-source cached \
    --feature-file "${FEATURE_FILE}" \
    --feature-dim "${FEATURE_DIM}" \
    --output-json "${RESULT_JSON}" \
    --output-md "${RESULT_MD}"
fi

echo "Done."
echo "Deduped heatmaps: ${DEDUP_HEATMAP}"
echo "Pseudo-bbox manifest: ${PSEUDO_BBOX_MANIFEST}"
echo "Pseudo-mask manifest: ${PSEUDO_MASK_MANIFEST}"
echo "Mask features: ${FEATURE_FILE}"
echo "Classification result: ${RESULT_MD}"
