# Engineering Plan

This repository follows the "MVREC as baseline, Auto-Mask as independent method" route.

## Code Ownership

- `baselines/MVREC/`: official MVREC code or notes, used for reproduction only.
- `src/lg_fdc/data/`: manifests and N-way K-shot episode sampling.
- `src/lg_fdc/localization/`: PatchCore, AnomalyDINO, EfficientAD, or cached heatmaps.
- `src/lg_fdc/masks/`: pseudo-mask generation from anomaly heatmaps.
- `src/lg_fdc/features/`: DINOv2, Alpha-CLIP, CLIP, or cached feature extraction.
- `src/lg_fdc/models/`: prototype/adapters and later region-context fusion modules.
- `src/lg_fdc/evaluation/`: metrics and reporting.
- `configs/`: reproducible experiment configs.
- `scripts/`: command-line entry points and smoke tests.

## Conda Environment

Use the existing WSL conda environment named `work-1`:

```bash
/home/jack/miniconda3/bin/conda run -n work-1 python -V
```

## First Milestone

The first milestone is a reproducible prototype baseline:

```text
MVTec-FS manifest
  -> N-way K-shot episodes
  -> cached/DINOv2 features
  -> prototype classifier
  -> Accuracy, Macro-F1, Balanced Accuracy
```

## Second Milestone

Add the Auto-Mask branch:

```text
image
  -> anomaly heatmap
  -> adaptive pseudo mask
  -> region/context/global features
  -> prototype classifier
```

## Verification

Run the dependency-light smoke test before committing structural changes:

```bash
/home/jack/miniconda3/bin/conda run -n work-1 python scripts/smoke_test.py
```