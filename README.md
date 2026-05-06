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
## Few-Shot Grid Runner

After extracting cached DINOv2 features, run the standard grid and save both JSON and Markdown results:

```bash
python scripts/run_fewshot_grid.py \
  --manifest data/manifests/mvtec_fs.csv \
  --split train \
  --grid 5:1,5:3,5:5,10:1,10:5 \
  --q-queries 5 \
  --episodes 200 \
  --feature-source cached \
  --feature-file outputs/features/dinov2/mvtec_fs_train.jsonl \
  --feature-dim 384 \
  --output-json outputs/results/dinov2_prototype_grid.json \
  --output-md outputs/results/dinov2_prototype_grid.md
```

The Markdown file can be copied directly into experiment notes or a paper draft.
## DINOv2 BBox/ROI Baseline

After the whole-image baseline, extract DINOv2 features from bbox crops inferred from LabelMe annotations:

```bash
mkdir -p outputs/features/dinov2_bbox
python scripts/extract_dinov2_features.py \
  --manifest data/manifests/mvtec_fs.csv \
  --image-root /home/jack/datasets/MVTec-FS \
  --split train \
  --output outputs/features/dinov2_bbox/mvtec_fs_train.jsonl \
  --model dinov2_vits14 \
  --region bbox \
  --bbox-padding 0.15 \
  --min-crop-size 32 \
  --batch-size 16 \
  --device auto \
  --overwrite
```

Then run the same few-shot grid:

```bash
python scripts/run_fewshot_grid.py \
  --manifest data/manifests/mvtec_fs.csv \
  --split train \
  --grid 5:1,5:3,5:5,10:1,10:5 \
  --q-queries 5 \
  --episodes 200 \
  --feature-source cached \
  --feature-file outputs/features/dinov2_bbox/mvtec_fs_train.jsonl \
  --feature-dim 384 \
  --output-json outputs/results/dinov2_bbox_prototype_grid.json \
  --output-md outputs/results/dinov2_bbox_prototype_grid.md
```

This is the second baseline table: DINOv2 bbox/ROI prototype. Compare it against the whole-image table to check whether localization helps.
## DINOv2 Region-Global Fusion Baseline

After whole-image and bbox feature caches are ready, fuse them into a region-global representation.

Concat fusion doubles the feature dimension from 384 to 768:

```bash
mkdir -p outputs/features/dinov2_fusion_concat
python scripts/fuse_feature_cache.py \
  --whole-file outputs/features/dinov2/mvtec_fs_train.jsonl \
  --region-file outputs/features/dinov2_bbox/mvtec_fs_train.jsonl \
  --method concat \
  --output outputs/features/dinov2_fusion_concat/mvtec_fs_train.jsonl \
  --overwrite
```

Run the grid with `--feature-dim 768`:

```bash
python scripts/run_fewshot_grid.py \
  --manifest data/manifests/mvtec_fs.csv \
  --split train \
  --grid 5:1,5:3,5:5,10:1,10:5 \
  --q-queries 5 \
  --episodes 200 \
  --feature-source cached \
  --feature-file outputs/features/dinov2_fusion_concat/mvtec_fs_train.jsonl \
  --feature-dim 768 \
  --output-json outputs/results/dinov2_fusion_concat_grid.json \
  --output-md outputs/results/dinov2_fusion_concat_grid.md
```

Weighted-sum fusion keeps the feature dimension at 384:

```bash
mkdir -p outputs/features/dinov2_fusion_alpha05
python scripts/fuse_feature_cache.py \
  --whole-file outputs/features/dinov2/mvtec_fs_train.jsonl \
  --region-file outputs/features/dinov2_bbox/mvtec_fs_train.jsonl \
  --method weighted-sum \
  --alpha 0.5 \
  --normalize-input \
  --normalize-output \
  --output outputs/features/dinov2_fusion_alpha05/mvtec_fs_train.jsonl \
  --overwrite
```

Use fusion results to test whether context-rich whole-image features and defect-focused ROI features complement each other.

## Pseudo-BBox From Anomaly Heatmaps

The next stage replaces GT LabelMe bbox with pseudo bbox generated from anomaly heatmaps. The name `example_heatmaps.jsonl` was only a placeholder: first generate a real heatmap JSONL file.

A dependency-light starter localizer is provided in `scripts/extract_dinov2_patch_heatmaps.py`. It writes JSONL rows with `image_path`, original image size, and a 2D `heatmap` array.

```bash
mkdir -p outputs/heatmaps
python scripts/extract_dinov2_patch_heatmaps.py \
  --manifest data/manifests/mvtec_fs.csv \
  --image-root /home/jack/datasets/MVTec-FS \
  --split train \
  --output outputs/heatmaps/dinov2_patch_contrast_train.jsonl \
  --mode patch-contrast \
  --model dinov2_vits14 \
  --batch-size 8 \
  --device auto \
  --overwrite
```

`--mode patch-contrast` does not require normal/good images and can be used immediately on the current MVTec-FS defect-only manifest. If you later rebuild a manifest that includes normal/good images, you can try the stronger memory-based mode:

```bash
python scripts/extract_dinov2_patch_heatmaps.py \
  --manifest data/manifests/mvtec_fs.csv \
  --image-root /home/jack/datasets/MVTec-FS \
  --split train \
  --output outputs/heatmaps/dinov2_nearest_memory_train.jsonl \
  --mode nearest-memory \
  --memory-split train \
  --memory-labels good,normal,ok \
  --max-memory-images 200 \
  --max-memory-patches 20000 \
  --model dinov2_vits14 \
  --batch-size 8 \
  --device auto \
  --overwrite
```

Then build a pseudo-bbox manifest from the generated heatmaps:

```bash
python scripts/build_pseudo_bbox_manifest.py \
  --manifest data/manifests/mvtec_fs.csv \
  --heatmap-file outputs/heatmaps/dinov2_patch_contrast_train.jsonl \
  --output data/manifests/mvtec_fs_pseudo_bbox_train.csv \
  --split train \
  --percentile 0.90 \
  --min-area-ratio 0.001 \
  --component max-score \
  --missing-policy error \
  --overwrite
```

Default `--missing-policy error` stops if any selected manifest row has no heatmap. Keep `--split` aligned with the heatmap extractor: if heatmaps were generated with `--split train`, build the pseudo-bbox manifest with `--split train` too. Use `--missing-policy clear` only for debugging incomplete heatmap caches.

Then reuse the existing bbox DINOv2 extractor on the pseudo-bbox manifest:

```bash
mkdir -p outputs/features/dinov2_pseudo_bbox
python scripts/extract_dinov2_features.py \
  --manifest data/manifests/mvtec_fs_pseudo_bbox_train.csv \
  --image-root /home/jack/datasets/MVTec-FS \
  --split train \
  --output outputs/features/dinov2_pseudo_bbox/mvtec_fs_train.jsonl \
  --model dinov2_vits14 \
  --region bbox \
  --bbox-padding 0.15 \
  --min-crop-size 32 \
  --batch-size 16 \
  --device auto \
  --overwrite
```

Finally run the same grid with `data/manifests/mvtec_fs_pseudo_bbox_train.csv` and `outputs/features/dinov2_pseudo_bbox/mvtec_fs_train.jsonl`.

Diagnose pseudo-bbox quality against the GT LabelMe bbox:

```bash
mkdir -p outputs/diagnostics
python scripts/evaluate_pseudo_bbox_iou.py \
  --gt-manifest data/manifests/mvtec_fs.csv \
  --pseudo-manifest data/manifests/mvtec_fs_pseudo_bbox_train.csv \
  --split train \
  --output-json outputs/diagnostics/pseudo_bbox_iou_train.json \
  --output-md outputs/diagnostics/pseudo_bbox_iou_train.md \
  --output-csv outputs/diagnostics/pseudo_bbox_iou_train.csv
```


Sweep pseudo-bbox extraction parameters before rerunning expensive ROI features:

```bash
mkdir -p outputs/diagnostics outputs/manifests/pseudo_bbox_sweep
python scripts/sweep_pseudo_bbox_iou.py \
  --gt-manifest data/manifests/mvtec_fs.csv \
  --heatmap-file outputs/heatmaps/dinov2_patch_contrast_train.jsonl \
  --split train \
  --percentiles 0.85,0.90,0.95 \
  --min-area-ratios 0.0005,0.001,0.005 \
  --components largest,max-score \
  --output-json outputs/diagnostics/pseudo_bbox_iou_sweep_train.json \
  --output-md outputs/diagnostics/pseudo_bbox_iou_sweep_train.md \
  --output-csv outputs/diagnostics/pseudo_bbox_iou_sweep_train.csv \
  --write-manifests-dir outputs/manifests/pseudo_bbox_sweep
```

The sweep ranks settings by mean IoU, then Recall@IoU 0.50/0.25. Use the best generated pseudo manifest only as a staging artifact; copy or rebuild the chosen setting into `data/manifests/mvtec_fs_pseudo_bbox_train.csv` before extracting DINOv2 ROI features.

To test whether low-resolution patch heatmaps benefit from bilinear upsampling before thresholding, add `--upsample-heatmap-to-image`:

```bash
mkdir -p outputs/diagnostics outputs/manifests/pseudo_bbox_sweep_upsampled
python scripts/sweep_pseudo_bbox_iou.py \
  --gt-manifest data/manifests/mvtec_fs.csv \
  --heatmap-file outputs/heatmaps/dinov2_patch_contrast_train.jsonl \
  --split train \
  --percentiles 0.85,0.90,0.95 \
  --min-area-ratios 0.0005,0.001,0.005 \
  --components largest,max-score \
  --upsample-heatmap-to-image \
  --output-json outputs/diagnostics/pseudo_bbox_iou_sweep_train_upsampled.json \
  --output-md outputs/diagnostics/pseudo_bbox_iou_sweep_train_upsampled.md \
  --output-csv outputs/diagnostics/pseudo_bbox_iou_sweep_train_upsampled.csv \
  --write-manifests-dir outputs/manifests/pseudo_bbox_sweep_upsampled
```

`--upsample-heatmap-to-image` keeps the JSONL heatmap cache unchanged, but resizes each heatmap to `image_width x image_height` with bilinear interpolation before percentile thresholding and connected-component selection. The default remains native-grid thresholding.

If pseudo-bbox ROI is weak, test whether whole-image context recovers performance by fusing whole-image and pseudo-bbox features:

```bash
mkdir -p outputs/features/dinov2_pseudo_fusion_concat
python scripts/fuse_feature_cache.py \
  --whole-file outputs/features/dinov2/mvtec_fs_train.jsonl \
  --region-file outputs/features/dinov2_pseudo_bbox/mvtec_fs_train.jsonl \
  --method concat \
  --output outputs/features/dinov2_pseudo_fusion_concat/mvtec_fs_train.jsonl \
  --overwrite

python scripts/run_fewshot_grid.py \
  --manifest data/manifests/mvtec_fs_pseudo_bbox_train.csv \
  --split train \
  --grid 5:1,5:3,5:5,10:1,10:5 \
  --q-queries 5 \
  --episodes 200 \
  --feature-source cached \
  --feature-file outputs/features/dinov2_pseudo_fusion_concat/mvtec_fs_train.jsonl \
  --feature-dim 768 \
  --output-json outputs/results/dinov2_pseudo_fusion_concat_grid.json \
  --output-md outputs/results/dinov2_pseudo_fusion_concat_grid.md
```

## Region-Context Prototype

Concat fusion combines whole-image and region features before prototype scoring. A more explicit region-context baseline combines the whole-image and pseudo-region prototype scores at prediction time:

```bash
python scripts/run_region_context_grid.py \
  --manifest data/manifests/mvtec_fs_pseudo_bbox_train.csv \
  --split train \
  --grid 5:1,5:3,5:5,10:1,10:5 \
  --q-queries 5 \
  --episodes 200 \
  --whole-feature-file outputs/features/dinov2/mvtec_fs_train.jsonl \
  --region-feature-file outputs/features/dinov2_pseudo_bbox/mvtec_fs_train.jsonl \
  --whole-feature-dim 384 \
  --region-feature-dim 384 \
  --whole-weights 0.25,0.50,0.75 \
  --output-json outputs/results/dinov2_pseudo_region_context_grid.json \
  --output-md outputs/results/dinov2_pseudo_region_context_grid.md
```

`--whole-weights` controls the score-level whole-image contribution; the region score weight is `1 - whole_weight`.


For 10-way diagnostics, compare region-context and pseudo concat fusion on the same sampled episodes and export per-class/confusion tables:

```bash
mkdir -p outputs/diagnostics
python scripts/analyze_region_context_confusion.py \
  --manifest data/manifests/mvtec_fs_pseudo_bbox_train.csv \
  --split train \
  --n-way 10 \
  --k-shot 5 \
  --q-queries 5 \
  --episodes 200 \
  --whole-feature-file outputs/features/dinov2/mvtec_fs_train.jsonl \
  --region-feature-file outputs/features/dinov2_pseudo_bbox/mvtec_fs_train.jsonl \
  --whole-feature-dim 384 \
  --region-feature-dim 384 \
  --whole-weight 0.75 \
  --baseline-feature-file outputs/features/dinov2_pseudo_fusion_concat/mvtec_fs_train.jsonl \
  --baseline-feature-dim 768 \
  --baseline-name pseudo_concat \
  --output-json outputs/diagnostics/region_context_10way5shot_confusion.json \
  --output-md outputs/diagnostics/region_context_10way5shot_confusion.md \
  --output-per-class-csv outputs/diagnostics/region_context_10way5shot_per_class.csv \
  --output-confusion-csv outputs/diagnostics/region_context_10way5shot_confusions.csv
```

