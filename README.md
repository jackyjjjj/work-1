# work-1

Research and code workspace for localization-guided few-shot industrial defect classification.

Current research direction:

> Pseudo-Mask Guided Region-Context Prototype Learning for Few-Shot Industrial Defect Classification

The initial research plan is documented in `docs/research_plan.md`.
The engineering plan is documented in `docs/engineering_plan.md`.

## Route

This project uses MVREC as a strong baseline and task reference, while the main Auto-Mask method is implemented independently.

```text
baselines/MVREC/        official MVREC reproduction area, kept separate
src/lg_fdc/data/        manifests and N-way K-shot episode sampling
src/lg_fdc/localization/ PatchCore / AnomalyDINO / EfficientAD heatmap interfaces
src/lg_fdc/masks/       pseudo-mask generation
src/lg_fdc/features/    DINOv2 / Alpha-CLIP / cached feature extractors
src/lg_fdc/models/      prototype and later adapter/fusion models
src/lg_fdc/evaluation/  metrics and reports
configs/                experiment configs
scripts/                runnable experiment entry points
```

## Environment

Use the existing WSL conda environment named `work-1`:

```bash
/home/jack/miniconda3/bin/conda run -n work-1 python -V
```

For real experiments, install the research dependencies in that environment when needed:

```bash
/home/jack/miniconda3/bin/conda run -n work-1 pip install -e ".[research,dev]"
```


## Local Dataset Setup

Assume MVTec-FS has been downloaded outside Git, for example:

```bash
/home/jack/datasets/MVTec-FS
```

If the dataset directory contains `image.tar.001` to `image.tar.012`, extract the images first:

```bash
cd /home/jack/datasets/MVTec-FS
cat image.tar.* | tar -xvf -
```

Build the project manifest:

```bash
cd /home/jack/workspace/work-1
mkdir -p data/manifests
/home/jack/miniconda3/bin/conda run -n work-1 python scripts/build_mvtec_fs_manifest.py \
  --dataset-root /home/jack/datasets/MVTec-FS \
  --output data/manifests/mvtec_fs.csv
```

If the generated CSV only contains `com_sample.jpg` and `data_details.png`, the archive has not been extracted or `--dataset-root` points to the wrong folder.

Then run a quick hash-feature baseline to verify the manifest and episode sampler:

```bash
/home/jack/miniconda3/bin/conda run -n work-1 python scripts/run_prototype_baseline.py \
  --manifest data/manifests/mvtec_fs.csv \
  --split train \
  --n-way 5 \
  --k-shot 1 \
  --q-queries 5 \
  --episodes 20 \
  --feature-source hash \
  --feature-dim 64
```

Hash features are only a pipeline check. Real experiments should use cached DINOv2 or Alpha-CLIP features.
## Quick Checks

Dependency-light smoke test:

```bash
/home/jack/miniconda3/bin/conda run -n work-1 python scripts/smoke_test.py
```

Prototype baseline on the example manifest:

```bash
/home/jack/miniconda3/bin/conda run -n work-1 python scripts/run_prototype_baseline.py \
  --manifest data/example_manifest.csv \
  --n-way 3 \
  --k-shot 2 \
  --q-queries 2 \
  --episodes 5 \
  --feature-source metadata \
  --feature-dim 3
```

## Planned Stages

1. Reproduce baseline few-shot defect classification experiments on MVTec-FS.
2. Build DINOv2 / Alpha-CLIP prototype baselines.
3. Generate pseudo defect masks with anomaly localization models such as PatchCore or AnomalyDINO.
4. Implement region-context prototype learning.
5. Compare against MVREC-like and anomaly-detection-plus-classifier baselines.
## DINOv2 Whole-Image Baseline

After the manifest is correct, extract real DINOv2 visual features:

```bash
mkdir -p outputs/features/dinov2
python scripts/extract_dinov2_features.py \
  --manifest data/manifests/mvtec_fs.csv \
  --image-root /home/jack/datasets/MVTec-FS \
  --split train \
  --output outputs/features/dinov2/mvtec_fs_train.jsonl \
  --model dinov2_vits14 \
  --batch-size 16 \
  --device auto \
  --overwrite
```

Then run the prototype baseline with cached DINOv2 features:

```bash
python scripts/run_prototype_baseline.py \
  --manifest data/manifests/mvtec_fs.csv \
  --split train \
  --n-way 5 \
  --k-shot 1 \
  --q-queries 5 \
  --episodes 200 \
  --feature-source cached \
  --feature-file outputs/features/dinov2/mvtec_fs_train.jsonl \
  --feature-dim 384
```

For `dinov2_vits14`, the feature dimension is usually `384`. If you switch to `dinov2_vitb14`, use `768`.