# Region-Context Handoff Summary - 2026-05-05

This handoff supersedes the older `handoff.md` for the current stage. The old file is still useful historical context, but this document reflects the latest paired-sweep and confusion-analysis conclusions from this session.

## Goal

Build and evaluate an Auto-Mask MVREC / localization-guided few-shot industrial defect classification pipeline:

```text
DINOv2 whole-image prototype
-> GT bbox/ROI prototype
-> GT region-global fusion
-> automatic heatmap -> pseudo-bbox / pseudo-mask
-> pseudo region-context prototype
-> calibrated/adaptive region-context or stronger localization
```

The current research question is no longer whether region information can help under perfect localization; GT bbox/ROI and GT fusion already show that it can. The current bottleneck is whether automatic pseudo localization is good enough, or whether region evidence must be gated/calibrated before it can reliably improve over a strong whole-image DINOv2 baseline.

## Environment

- Local WSL project path: `/home/jack/workspace/work-1`
- Windows workspace path used by Codex: `E:\CodexWorkspace`
- Server path used by the user: `/home/think/mnt/jyl/MyWork/work-1`
- Python env: conda `work-1`
- Local Python command prefix:

```bash
/home/jack/miniconda3/bin/conda run -n work-1 python ...
```

- Current date/time for this handoff: `2026-05-05 22:57 +08:00`
- WSL terminal may print garbled Chinese warning text; files remain UTF-8.

## Constraints And Preferences

- Always check `git status --short --branch` before editing.
- Do not revert user changes or unrelated dirty files.
- Keep datasets, heatmaps, features, diagnostics, and other `outputs/` artifacts out of git.
- Update `docs/change_log.md` for every code or document change.
- Use lightweight self-check scripts instead of pytest.
- Prefer `rg`/`rg --files` for searching if available; in this WSL context `grep`/`sed` are often used through approved commands.
- User usually runs expensive experiments on the server and pastes Markdown results back.
- Give explicit `git add / git commit / git push` commands or commit locally when asked.

## Current Git State At Handoff Creation

Before creating this new handoff, `git status --short --branch` was clean:

```text
## main...origin/main
```

After creating this file, it should be committed together with the `docs/change_log.md` entry.

## Recently Completed In This Session

### 1. Region-Context Confusion Diagnostics

Implemented and committed earlier:

- `scripts/analyze_region_context_confusion.py`
- `scripts/check_region_context_confusion.py`

Capabilities:

- Runs region-context on sampled episodes.
- Optionally evaluates a cached-feature baseline on the same episodes, e.g. pseudo concat fusion.
- Outputs JSON, Markdown, per-class CSV, and top-confusion CSV.
- Supports per-class recall/F1 deltas and top confusion pairs.

Committed in:

```text
81c09d2 Add region context confusion analysis
```

### 2. Region-Context Confusion Results Recorded

Recorded 10-way 5-shot confusion at `whole_weight=0.75`, `region_weight=0.25`:

| Model | Accuracy | Balanced Acc | Macro-F1 | Queries |
|---|---:|---:|---:|---:|
| region_context | 0.7480 | 0.7523 | 0.7369 | 10000 |
| pseudo_concat | 0.7272 | 0.7315 | 0.7165 | 10000 |

Main gains at `0.75/0.25`:

- `fabric_border` recall delta `+0.1581`, F1 delta `+0.1102`
- `squeeze` recall delta `+0.0889`
- `squeezed_teeth` recall delta `+0.0746`
- `split_teeth` recall delta `+0.0737`
- `faulty_imprint` recall delta `+0.0645`
- `hole` recall delta `+0.0500`

Main regressions at `0.75/0.25`:

- `scratch_neck` recall delta `-0.0453`, F1 delta `-0.0250`
- `bent` recall delta `-0.0316`, but F1 improved
- `manipulated_front` recall delta `-0.0302`
- `color` recall delta `-0.0286`

Committed in:

```text
1a13154 Record region context confusion results
```

### 3. Fine Weight Sweep Recorded, Then Corrected

A first fine weight sweep over `whole_weight=0.60..0.90` suggested best weights:

- `10-way 1-shot`: `0.90/0.10`
- `10-way 5-shot`: `0.80/0.20`

Committed in:

```text
c417eb8 Record region context weight sweep results
```

However, during follow-up review we found `scripts/run_region_context_grid.py` used different episode seeds for different weights:

```python
seed=args.seed + weight_idx * 1_000_000 + setting_idx * 100_000
```

This made cross-weight comparisons unpaired. The script was fixed so weights now share the same episodes for each setting:

```python
seed=args.seed + setting_idx * 100_000
```

The check script now verifies paired seeds across two weights.

Committed in:

```text
9ef0d85 Pair region context weight sweep episodes
```

### 4. New-Best-Weight Confusion Results Recorded

User reran confusion with the provisional best weights:

#### 10-way 1-shot @ 0.90 / 0.10

| Model | Accuracy | Balanced Acc | Macro-F1 | Queries |
|---|---:|---:|---:|---:|
| region_context | 0.7342 | 0.7226 | 0.7120 | 10000 |
| pseudo_concat | 0.7027 | 0.6925 | 0.6836 | 10000 |

Delta vs pseudo concat:

- Accuracy: `+0.0315`
- Macro-F1: `+0.0284`

Largest recall gains:

- `cut_outer_insulation` `+0.1509`
- `bent_wire` `+0.1478`
- `print` `+0.1375`
- `metal_contamination` `+0.1353`
- `missing_wire` `+0.1321`
- `hole` `+0.1161`
- `missing_cable` `+0.1067`

Largest regressions:

- `scratch` `-0.0558`
- `scratch_neck` `-0.0541`
- `misplaced` `-0.0316`
- `cable_swap` `-0.0311`
- `color` `-0.0187`

#### 10-way 5-shot @ 0.80 / 0.20

| Model | Accuracy | Balanced Acc | Macro-F1 | Queries |
|---|---:|---:|---:|---:|
| region_context | 0.7506 | 0.7547 | 0.7393 | 10000 |
| pseudo_concat | 0.7272 | 0.7315 | 0.7165 | 10000 |

Delta vs pseudo concat:

- Accuracy: `+0.0234`
- Macro-F1: `+0.0228`

Largest recall gains:

- `fabric_border` `+0.1839`
- `squeeze` `+0.0926`
- `squeezed_teeth` `+0.0847`
- `split_teeth` `+0.0842`
- `faulty_imprint` `+0.0645`
- `thread_top` `+0.0536`
- `hole` `+0.0500`
- `missing_cable` `+0.0500`

Largest regressions:

- `scratch_neck` `-0.0604`
- `color` `-0.0393`
- `manipulated_front` `-0.0377`
- `bent` `-0.0316`
- `crack` `-0.0139`

Important interpretation: reducing region weight to `0.20` did not fix `scratch_neck`, `color`, or `manipulated_front`; a single global fixed weight is likely insufficient.

### 5. Paired Weight Sweep Result From User

After the seed fix, the user reran a paired sweep:

| Whole W | Region W | 10-way 1-shot Acc | 10-way 1-shot F1 | 10-way 5-shot Acc | 10-way 5-shot F1 |
|---:|---:|---:|---:|---:|---:|
| 0.60 | 0.40 | 71.31 | 68.40 | 74.17 | 71.64 |
| 0.65 | 0.35 | 71.93 | 69.13 | 74.51 | 72.00 |
| 0.70 | 0.30 | 72.25 | 69.43 | 74.77 | 72.24 |
| 0.75 | 0.25 | 72.50 | 69.74 | 75.13 | 72.60 |
| 0.80 | 0.20 | 72.83 | 70.18 | 75.43 | 72.91 |
| 0.85 | 0.15 | 73.00 | 70.35 | 75.51 | 72.96 |
| 0.90 | 0.10 | 73.42 | 70.74 | 75.94 | 73.43 |
| 0.95 | 0.05 | 73.73 | 71.07 | 76.31 | 73.80 |
| 1.00 | 0.00 | 74.07 | 71.53 | 76.23 | 73.67 |

Latest interpretation:

- In strict paired comparison, 10-way 1-shot is best at whole-only `1.00/0.00`.
- 10-way 5-shot is best at `0.95/0.05`, but only slightly above whole-only:
  - `76.31` vs `76.23` accuracy
  - `73.80` vs `73.67` Macro-F1
- Performance mostly improves as whole-image weight increases.
- Current pseudo-bbox region branch is therefore not a robust fixed-weight improvement over whole-image DINOv2.
- This paired result has not yet been written into `experiments/dinov2_baselines.md` or `docs/change_log.md`; do that next before further method changes.

## Key Experimental Results So Far

### Whole-Image DINOv2

Feature file: `outputs/features/dinov2/mvtec_fs_train.jsonl`

| Setting | Accuracy | Macro-F1 |
|---|---:|---:|
| 5-way 1-shot | 82.10 | 80.42 |
| 5-way 3-shot | 85.38 | 84.15 |
| 5-way 5-shot | 86.84 | 85.63 |
| 10-way 1-shot | 72.61 | 70.19 |
| 10-way 5-shot | 77.16 | 74.53 |

### GT BBox/ROI

Feature file: `outputs/features/dinov2_bbox/mvtec_fs_train.jsonl`

| Setting | Accuracy | Macro-F1 |
|---|---:|---:|
| 5-way 1-shot | 79.62 | 77.61 |
| 5-way 3-shot | 84.92 | 83.67 |
| 5-way 5-shot | 88.06 | 87.08 |
| 10-way 1-shot | 70.70 | 68.16 |
| 10-way 5-shot | 80.14 | 78.40 |

### GT Concat Fusion

Feature file: `outputs/features/dinov2_fusion_concat/mvtec_fs_train.jsonl`

| Setting | Accuracy | Macro-F1 |
|---|---:|---:|
| 5-way 1-shot | 83.62 | 81.94 |
| 5-way 3-shot | 87.14 | 85.97 |
| 5-way 5-shot | 89.66 | 88.66 |
| 10-way 1-shot | 74.81 | 72.58 |
| 10-way 5-shot | 81.88 | 79.95 |

### Pseudo-BBox ROI-Only

Manifest: `data/manifests/mvtec_fs_pseudo_bbox_train.csv`
Feature file: `outputs/features/dinov2_pseudo_bbox/mvtec_fs_train.jsonl`

| Setting | Accuracy | Macro-F1 |
|---|---:|---:|
| 5-way 1-shot | 63.80 | 60.07 |
| 5-way 3-shot | 69.20 | 66.39 |
| 5-way 5-shot | 67.92 | 64.57 |
| 10-way 1-shot | 49.97 | 45.44 |
| 10-way 5-shot | 51.40 | 46.22 |

### Pseudo-BBox IoU Sweep

Best pseudo-bbox extraction setting:

- Percentile: `0.90`
- Min area ratio: `0.0005`
- Component: `largest`
- Mean IoU: `0.1863`
- Median IoU: `0.0765`
- Recall@IoU 0.50: `0.1388`
- Mean pseudo/GT area ratio: `11.3234`

Interpretation: current `DINOv2 patch-contrast` localization is weak and tends to produce too-large boxes.

### Pseudo-BBox + Whole Concat Fusion

Feature file: `outputs/features/dinov2_pseudo_fusion_concat/mvtec_fs_train.jsonl`

| Setting | Accuracy | Macro-F1 |
|---|---:|---:|
| 5-way 1-shot | 80.50 | 78.52 |
| 5-way 3-shot | 83.92 | 82.64 |
| 5-way 5-shot | 85.80 | 84.54 |
| 10-way 1-shot | 68.43 | 65.52 |
| 10-way 5-shot | 73.38 | 70.52 |

Interpretation: global context recovers most ROI-only pseudo-bbox loss.

## Key Files

### Core Source

- `src/lg_fdc/data/manifest.py`
- `src/lg_fdc/data/episodes.py`
- `src/lg_fdc/models/prototype.py`
- `src/lg_fdc/features/cached.py`
- `src/lg_fdc/pipelines/prototype_baseline.py`
- `src/lg_fdc/pipelines/region_context.py`

### Scripts

- `scripts/run_fewshot_grid.py`
- `scripts/run_region_context_grid.py`
- `scripts/analyze_region_context_confusion.py`
- `scripts/extract_dinov2_features.py`
- `scripts/fuse_feature_cache.py`
- `scripts/extract_dinov2_patch_heatmaps.py`
- `scripts/build_pseudo_bbox_manifest.py`
- `scripts/evaluate_pseudo_bbox_iou.py`
- `scripts/sweep_pseudo_bbox_iou.py`

### Self-Checks

- `scripts/check_region_context_prototype.py`
- `scripts/check_region_context_confusion.py`
- `scripts/check_pseudo_bbox_iou_sweep.py`
- `scripts/check_pseudo_bbox_iou.py`
- `scripts/check_pseudo_bbox.py`
- `scripts/check_dinov2_patch_heatmaps.py`
- `scripts/check_feature_fusion.py`
- `scripts/check_bbox_crop.py`

### Documents

- `experiments/dinov2_baselines.md`
- `docs/change_log.md`
- `handoff.md` (older historical handoff)
- `handoff_region_context_2026-05-05.md` (this file)
- `README.md`

## Key Commands

### Check Status

```bash
git status --short --branch
```

### Validate Region-Context Code

```bash
/home/jack/miniconda3/bin/conda run -n work-1 python scripts/check_region_context_prototype.py
/home/jack/miniconda3/bin/conda run -n work-1 python -m py_compile scripts/run_region_context_grid.py scripts/check_region_context_prototype.py
```

### Validate Confusion Tool

```bash
/home/jack/miniconda3/bin/conda run -n work-1 python scripts/check_region_context_confusion.py
/home/jack/miniconda3/bin/conda run -n work-1 python -m py_compile scripts/analyze_region_context_confusion.py scripts/check_region_context_confusion.py
```

### Paired 10-way Weight Sweep

Use this command after pulling commit `9ef0d85` or later:

```bash
mkdir -p outputs/diagnostics
python scripts/run_region_context_grid.py \
  --manifest data/manifests/mvtec_fs_pseudo_bbox_train.csv \
  --split train \
  --grid 10:1,10:5 \
  --q-queries 5 \
  --episodes 200 \
  --whole-feature-file outputs/features/dinov2/mvtec_fs_train.jsonl \
  --region-feature-file outputs/features/dinov2_pseudo_bbox/mvtec_fs_train.jsonl \
  --whole-feature-dim 384 \
  --region-feature-dim 384 \
  --whole-weights 0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00 \
  --output-json outputs/diagnostics/region_context_weight_sweep_10way_paired.json \
  --output-md outputs/diagnostics/region_context_weight_sweep_10way_paired.md
```

### Confusion At Whole-Heavy Weights

```bash
mkdir -p outputs/diagnostics
python scripts/analyze_region_context_confusion.py \
  --manifest data/manifests/mvtec_fs_pseudo_bbox_train.csv \
  --split train \
  --n-way 10 \
  --k-shot 1 \
  --q-queries 5 \
  --episodes 200 \
  --whole-feature-file outputs/features/dinov2/mvtec_fs_train.jsonl \
  --region-feature-file outputs/features/dinov2_pseudo_bbox/mvtec_fs_train.jsonl \
  --whole-feature-dim 384 \
  --region-feature-dim 384 \
  --whole-weight 0.90 \
  --baseline-feature-file outputs/features/dinov2_pseudo_fusion_concat/mvtec_fs_train.jsonl \
  --baseline-feature-dim 768 \
  --baseline-name pseudo_concat \
  --output-json outputs/diagnostics/region_context_10way1shot_w0p90_confusion.json \
  --output-md outputs/diagnostics/region_context_10way1shot_w0p90_confusion.md \
  --output-per-class-csv outputs/diagnostics/region_context_10way1shot_w0p90_per_class.csv \
  --output-confusion-csv outputs/diagnostics/region_context_10way1shot_w0p90_confusions.csv
```

```bash
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
  --whole-weight 0.80 \
  --baseline-feature-file outputs/features/dinov2_pseudo_fusion_concat/mvtec_fs_train.jsonl \
  --baseline-feature-dim 768 \
  --baseline-name pseudo_concat \
  --output-json outputs/diagnostics/region_context_10way5shot_w0p80_confusion.json \
  --output-md outputs/diagnostics/region_context_10way5shot_w0p80_confusion.md \
  --output-per-class-csv outputs/diagnostics/region_context_10way5shot_w0p80_per_class.csv \
  --output-confusion-csv outputs/diagnostics/region_context_10way5shot_w0p80_confusions.csv
```

## Not Yet Done

1. Record the latest paired sweep result in `experiments/dinov2_baselines.md` and `docs/change_log.md`.
2. Decide how to update Section 7.7 after paired sweep:
   - old unpaired fine sweep should be marked historical/screening only;
   - new paired sweep shows whole-only is best for 10-way 1-shot and `0.95/0.05` is only marginally best for 10-way 5-shot.
3. Implement score normalization or confidence-based adaptive weighting only after recording the paired sweep result.
4. Consider stronger localization methods:
   - `nearest-memory` heatmaps if normal/good images can be included;
   - PatchCore / AnomalyDINO heatmaps;
   - pseudo-mask rather than pseudo-bbox.
5. Consider per-class or per-episode localization/confusion analysis:
   - correlate class-wise IoU with region-context gains/regressions;
   - inspect `scratch_neck`, `color`, `manipulated_front`, `scratch`, `fabric_border`, tooth defects.

## Recommended Immediate Next Steps

1. Record the latest paired sweep result from the user.
2. Update the project conclusion:
   - current pseudo region signal is too noisy for global fixed-weight fusion;
   - whole-image DINOv2 remains strongest under paired 10-way fixed-weight comparison;
   - a tiny region weight may help 10-way 5-shot, but the gain is marginal.
3. Commit the new handoff and paired sweep documentation.
4. Then choose one of two technical directions:
   - improve localization quality, or
   - implement adaptive region gating / score normalization.

Recommended next implementation target if continuing classifier-side work:

```text
score normalization + confidence-adaptive region weighting
```

Candidate design:

- Compute whole and region prototype scores separately.
- Normalize each branch scores per query, e.g. z-score or min-max across episode labels.
- Estimate branch confidence by top-1 minus top-2 margin.
- Let region weight shrink when region confidence is low or when whole-region predictions disagree strongly.
- Compare against fixed whole-only, fixed `0.95/0.05`, fixed `0.90/0.10`, pseudo concat, and whole baseline.

## Bottom-Line Summary

The project has a complete evaluation and diagnostic pipeline. GT region fusion validates the research motivation, but the current automatic `DINOv2 patch-contrast` pseudo-bbox localizer is weak. Pseudo ROI-only fails, pseudo concat recovers much of the loss through whole-image context, and score-level region-context helps some localized classes. After fixing paired weight comparison, however, the strict 10-way sweep shows that current pseudo region evidence is not robust as a fixed-weight branch: whole-only is best for 10-way 1-shot, and 10-way 5-shot gains from a `0.05` region weight are tiny. The next meaningful progress should come from stronger localization or confidence-adaptive region-context rather than further manual fixed-weight tuning.
