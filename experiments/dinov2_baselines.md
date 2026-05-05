# DINOv2 Prototype Baselines on MVTec-FS

## 1. Purpose

This note summarizes the first set of MVTec-FS few-shot defect classification baselines.
The goal is to establish a clear progression before moving to pseudo-mask / Auto-Mask methods:

```text
whole-image feature
  -> bbox/ROI feature
  -> region-global fusion feature
  -> future pseudo-mask region-context feature
```

The current experiments use DINOv2 whole-image and bbox/ROI features with a cosine prototype classifier.

## 2. Experimental Setup

- Dataset manifest: `data/manifests/mvtec_fs.csv`
- Split: `train`
- Backbone: `dinov2_vits14`
- Feature dimension: `384` for whole-image, bbox/ROI, and weighted-sum fusion
- Feature dimension: `768` for concat fusion
- Classifier: cosine prototype classifier
- Episodes per setting: `200`
- Query images per class: `5`
- Reported metrics: mean +/- standard deviation over episodes

Few-shot settings:

```text
5-way 1-shot
5-way 3-shot
5-way 5-shot
10-way 1-shot
10-way 5-shot
```

## 3. Accuracy Summary

| Setting | Whole | BBox/ROI | Pseudo-BBox ROI | Pseudo-BBox Fusion | Best Region-Context | GT Concat Fusion | Alpha 0.25 | Alpha 0.5 | Alpha 0.75 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5-way 1-shot | 82.10 | 79.62 | 63.80 | 80.50 | 80.32 | **83.62** | 82.48 | 83.08 | 82.48 |
| 5-way 3-shot | 85.38 | 84.92 | 69.20 | 83.92 | 86.30 | 87.14 | 86.50 | **87.22** | 86.34 |
| 5-way 5-shot | 86.84 | 88.06 | 67.92 | 85.80 | 85.72 | **89.66** | 89.42 | 89.46 | 87.90 |
| 10-way 1-shot | 72.61 | 70.70 | 49.97 | 68.43 | 70.68 | 74.81 | 73.92 | **75.12** | 74.25 |
| 10-way 5-shot | 77.16 | 80.14 | 51.40 | 73.38 | 75.45 | **81.88** | 81.53 | 81.60 | 79.32 |

## 4. Macro-F1 Summary

| Setting | Whole | BBox/ROI | Pseudo-BBox ROI | Pseudo-BBox Fusion | Best Region-Context | GT Concat Fusion | Alpha 0.25 | Alpha 0.5 | Alpha 0.75 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5-way 1-shot | 80.42 | 77.61 | 60.07 | 78.52 | 78.31 | **81.94** | 80.60 | 81.33 | 80.75 |
| 5-way 3-shot | 84.15 | 83.67 | 66.39 | 82.64 | 85.12 | **85.97** | 85.31 | **85.97** | 85.18 |
| 5-way 5-shot | 85.63 | 87.08 | 64.57 | 84.54 | 84.40 | **88.66** | 88.46 | 88.43 | 86.75 |
| 10-way 1-shot | 70.19 | 68.16 | 45.44 | 65.52 | 67.90 | 72.58 | 71.51 | **72.80** | 71.96 |
| 10-way 5-shot | 74.53 | 78.40 | 46.22 | 70.52 | 72.83 | **79.95** | 79.78 | 79.51 | 76.82 |

## 5. Detailed Tables

### 5.1 DINOv2 Whole-Image Prototype

Feature file: `outputs/features/dinov2/mvtec_fs_train.jsonl`

| Setting | Episodes | Accuracy | Balanced Acc | Macro-F1 |
|---|---:|---:|---:|---:|
| 5-way 1-shot | 200 | 82.10 +/- 13.37 | 82.10 +/- 13.37 | 80.42 +/- 14.64 |
| 5-way 3-shot | 200 | 85.38 +/- 11.52 | 85.38 +/- 11.52 | 84.15 +/- 12.59 |
| 5-way 5-shot | 200 | 86.84 +/- 11.38 | 86.84 +/- 11.38 | 85.63 +/- 12.68 |
| 10-way 1-shot | 200 | 72.61 +/- 9.98 | 72.61 +/- 9.98 | 70.19 +/- 10.81 |
| 10-way 5-shot | 200 | 77.16 +/- 8.67 | 77.16 +/- 8.67 | 74.53 +/- 9.72 |

### 5.2 DINOv2 BBox/ROI Prototype

Feature file: `outputs/features/dinov2_bbox/mvtec_fs_train.jsonl`

| Setting | Episodes | Accuracy | Balanced Acc | Macro-F1 |
|---|---:|---:|---:|---:|
| 5-way 1-shot | 200 | 79.62 +/- 13.32 | 79.62 +/- 13.32 | 77.61 +/- 14.71 |
| 5-way 3-shot | 200 | 84.92 +/- 10.87 | 84.92 +/- 10.87 | 83.67 +/- 11.86 |
| 5-way 5-shot | 200 | 88.06 +/- 9.74 | 88.06 +/- 9.74 | 87.08 +/- 10.75 |
| 10-way 1-shot | 200 | 70.70 +/- 9.39 | 70.70 +/- 9.39 | 68.16 +/- 10.16 |
| 10-way 5-shot | 200 | 80.14 +/- 7.13 | 80.14 +/- 7.13 | 78.40 +/- 7.83 |

### 5.3 DINOv2 Concat Fusion Prototype

Feature file: `outputs/features/dinov2_fusion_concat/mvtec_fs_train.jsonl`

| Setting | Episodes | Accuracy | Balanced Acc | Macro-F1 |
|---|---:|---:|---:|---:|
| 5-way 1-shot | 200 | 83.62 +/- 12.19 | 83.62 +/- 12.19 | 81.94 +/- 13.66 |
| 5-way 3-shot | 200 | 87.14 +/- 10.10 | 87.14 +/- 10.10 | 85.97 +/- 11.17 |
| 5-way 5-shot | 200 | 89.66 +/- 9.60 | 89.66 +/- 9.60 | 88.66 +/- 10.89 |
| 10-way 1-shot | 200 | 74.81 +/- 9.24 | 74.81 +/- 9.24 | 72.58 +/- 10.10 |
| 10-way 5-shot | 200 | 81.88 +/- 7.38 | 81.88 +/- 7.38 | 79.95 +/- 8.28 |

### 5.4 DINOv2 Weighted-Sum Fusion, Alpha 0.25

Feature file: `outputs/features/dinov2_fusion_alpha025/mvtec_fs_train.jsonl`

| Setting | Episodes | Accuracy | Balanced Acc | Macro-F1 |
|---|---:|---:|---:|---:|
| 5-way 1-shot | 200 | 82.48 +/- 12.62 | 82.48 +/- 12.62 | 80.60 +/- 14.23 |
| 5-way 3-shot | 200 | 86.50 +/- 10.34 | 86.50 +/- 10.34 | 85.31 +/- 11.43 |
| 5-way 5-shot | 200 | 89.42 +/- 9.49 | 89.42 +/- 9.49 | 88.46 +/- 10.59 |
| 10-way 1-shot | 200 | 73.92 +/- 9.18 | 73.92 +/- 9.18 | 71.51 +/- 10.05 |
| 10-way 5-shot | 200 | 81.53 +/- 7.35 | 81.53 +/- 7.35 | 79.78 +/- 8.18 |

### 5.5 DINOv2 Weighted-Sum Fusion, Alpha 0.5

Feature file: `outputs/features/dinov2_fusion_alpha05/mvtec_fs_train.jsonl`

| Setting | Episodes | Accuracy | Balanced Acc | Macro-F1 |
|---|---:|---:|---:|---:|
| 5-way 1-shot | 200 | 83.08 +/- 12.38 | 83.08 +/- 12.38 | 81.33 +/- 13.89 |
| 5-way 3-shot | 200 | 87.22 +/- 10.36 | 87.22 +/- 10.36 | 85.97 +/- 11.57 |
| 5-way 5-shot | 200 | 89.46 +/- 9.89 | 89.46 +/- 9.89 | 88.43 +/- 11.20 |
| 10-way 1-shot | 200 | 75.12 +/- 9.14 | 75.12 +/- 9.14 | 72.80 +/- 10.04 |
| 10-way 5-shot | 200 | 81.60 +/- 7.82 | 81.60 +/- 7.82 | 79.51 +/- 8.85 |

### 5.6 DINOv2 Weighted-Sum Fusion, Alpha 0.75

Feature file: `outputs/features/dinov2_fusion_alpha075/mvtec_fs_train.jsonl`

| Setting | Episodes | Accuracy | Balanced Acc | Macro-F1 |
|---|---:|---:|---:|---:|
| 5-way 1-shot | 200 | 82.48 +/- 12.92 | 82.48 +/- 12.92 | 80.75 +/- 14.38 |
| 5-way 3-shot | 200 | 86.34 +/- 10.93 | 86.34 +/- 10.93 | 85.18 +/- 11.99 |
| 5-way 5-shot | 200 | 87.90 +/- 10.96 | 87.90 +/- 10.96 | 86.75 +/- 12.33 |
| 10-way 1-shot | 200 | 74.25 +/- 9.52 | 74.25 +/- 9.52 | 71.96 +/- 10.42 |
| 10-way 5-shot | 200 | 79.32 +/- 8.59 | 79.32 +/- 8.59 | 76.82 +/- 9.74 |

### 5.7 DINOv2 Pseudo-BBox ROI Prototype

Feature file: `outputs/features/dinov2_pseudo_bbox/mvtec_fs_train.jsonl`
Manifest: `data/manifests/mvtec_fs_pseudo_bbox_train.csv`
Localizer: `dinov2_patch_contrast` heatmap -> pseudo bbox

| Setting | Episodes | Accuracy | Balanced Acc | Macro-F1 |
|---|---:|---:|---:|---:|
| 5-way 1-shot | 200 | 63.80 +/- 14.29 | 63.80 +/- 14.29 | 60.07 +/- 15.94 |
| 5-way 3-shot | 200 | 69.20 +/- 12.79 | 69.20 +/- 12.79 | 66.39 +/- 14.21 |
| 5-way 5-shot | 200 | 67.92 +/- 13.78 | 67.92 +/- 13.78 | 64.57 +/- 15.56 |
| 10-way 1-shot | 200 | 49.97 +/- 9.48 | 49.97 +/- 9.48 | 45.44 +/- 10.03 |
| 10-way 5-shot | 200 | 51.40 +/- 8.14 | 51.40 +/- 8.14 | 46.22 +/- 8.99 |

Observation: this starter pseudo-bbox ROI baseline is much lower than whole-image, GT bbox/ROI, and fusion. The result suggests that the first `patch-contrast` localizer is not accurate enough for ROI-only classification, and that pseudo-localization quality should be diagnosed before treating pseudo-bbox ROI as a final method.


### 5.8 DINOv2 Pseudo-BBox + Whole Concat Fusion Prototype

Feature file: `outputs/features/dinov2_pseudo_fusion_concat/mvtec_fs_train.jsonl`
Manifest: `data/manifests/mvtec_fs_pseudo_bbox_train.csv`
Localizer: `dinov2_patch_contrast` heatmap -> pseudo bbox
Fusion: whole-image DINOv2 + pseudo-bbox ROI DINOv2 concat

| Setting | Episodes | Accuracy | Balanced Acc | Macro-F1 |
|---|---:|---:|---:|---:|
| 5-way 1-shot | 200 | 80.50 +/- 13.39 | 80.50 +/- 13.39 | 78.52 +/- 15.03 |
| 5-way 3-shot | 200 | 83.92 +/- 11.72 | 83.92 +/- 11.72 | 82.64 +/- 12.85 |
| 5-way 5-shot | 200 | 85.80 +/- 12.19 | 85.80 +/- 12.19 | 84.54 +/- 13.44 |
| 10-way 1-shot | 200 | 68.43 +/- 9.05 | 68.43 +/- 9.05 | 65.52 +/- 9.80 |
| 10-way 5-shot | 200 | 73.38 +/- 8.94 | 73.38 +/- 8.94 | 70.52 +/- 9.90 |

Observation: pseudo-bbox + whole-image concat recovers most of the ROI-only performance loss despite weak pseudo localization. Accuracy improves by `+14.72` to `+21.98` points over pseudo-bbox ROI, and Macro-F1 improves by `+16.25` to `+24.30` points. The result approaches whole-image performance in 5-way settings, but remains below GT bbox + whole fusion, especially in 10-way 5-shot. This supports the region-context direction while showing that localization quality still limits the automatic pipeline.


### 5.9 DINOv2 Pseudo-BBox Region-Context Prototype

Whole feature file: `outputs/features/dinov2/mvtec_fs_train.jsonl`
Region feature file: `outputs/features/dinov2_pseudo_bbox/mvtec_fs_train.jsonl`
Manifest: `data/manifests/mvtec_fs_pseudo_bbox_train.csv`
Fusion: score-level whole-image prototype + pseudo-bbox ROI prototype

| Whole W | Region W | Setting | Episodes | Accuracy | Balanced Acc | Macro-F1 |
|---:|---:|---|---:|---:|---:|---:|
| 0.25 | 0.75 | 5-way 1-shot | 200 | 76.00 +/- 13.65 | 76.00 +/- 13.65 | 73.75 +/- 14.99 |
| 0.25 | 0.75 | 5-way 3-shot | 200 | 79.76 +/- 12.56 | 79.76 +/- 12.56 | 78.16 +/- 13.80 |
| 0.25 | 0.75 | 5-way 5-shot | 200 | 82.28 +/- 12.68 | 82.28 +/- 12.68 | 80.93 +/- 13.76 |
| 0.25 | 0.75 | 10-way 1-shot | 200 | 62.39 +/- 9.07 | 62.39 +/- 9.07 | 59.28 +/- 9.70 |
| 0.25 | 0.75 | 10-way 5-shot | 200 | 69.01 +/- 9.53 | 69.01 +/- 9.53 | 65.79 +/- 10.45 |
| 0.50 | 0.50 | 5-way 1-shot | 200 | 79.90 +/- 12.92 | 79.90 +/- 12.92 | 77.82 +/- 14.28 |
| 0.50 | 0.50 | 5-way 3-shot | 200 | 83.58 +/- 12.39 | 83.58 +/- 12.39 | 82.11 +/- 13.82 |
| 0.50 | 0.50 | 5-way 5-shot | 200 | 85.58 +/- 11.89 | 85.58 +/- 11.89 | 84.40 +/- 13.02 |
| 0.50 | 0.50 | 10-way 1-shot | 200 | 68.89 +/- 10.05 | 68.89 +/- 10.05 | 65.80 +/- 10.88 |
| 0.50 | 0.50 | 10-way 5-shot | 200 | 73.50 +/- 9.70 | 73.50 +/- 9.70 | 70.76 +/- 10.89 |
| 0.75 | 0.25 | 5-way 1-shot | 200 | 80.32 +/- 12.87 | 80.32 +/- 12.87 | 78.31 +/- 14.30 |
| 0.75 | 0.25 | 5-way 3-shot | 200 | 86.30 +/- 11.25 | 86.30 +/- 11.25 | 85.12 +/- 12.36 |
| 0.75 | 0.25 | 5-way 5-shot | 200 | 85.72 +/- 10.85 | 85.72 +/- 10.85 | 84.29 +/- 12.21 |
| 0.75 | 0.25 | 10-way 1-shot | 200 | 70.68 +/- 9.88 | 70.68 +/- 9.88 | 67.90 +/- 10.72 |
| 0.75 | 0.25 | 10-way 5-shot | 200 | 75.45 +/- 9.13 | 75.45 +/- 9.13 | 72.83 +/- 10.17 |

Observation: the best accuracy is consistently obtained with whole/region score weights `0.75/0.25`, showing that the pseudo region score is useful mainly as an auxiliary cue while localization is noisy. Compared with pseudo concat fusion, score-level region-context improves 5-way 3-shot and both 10-way settings, but is slightly lower in 5-way 1-shot and 5-way 5-shot. The method is therefore a promising explicit region-context baseline, but it still needs localization improvement or score calibration to reliably beat concat fusion.


## 6. Pseudo-BBox IoU Sweep

This sweep evaluates the first `dinov2_patch_contrast` heatmap localizer before rerunning expensive ROI feature extraction.

- GT manifest: `data/manifests/mvtec_fs.csv`
- Heatmap file: `outputs/heatmaps/dinov2_patch_contrast_train.jsonl`
- Split: `train`
- Images: `886`
- Ranking: `mean_iou_then_recall`

| Rank | Percentile | Min Area Ratio | Component | Mean IoU | Median IoU | R@0.25 | R@0.50 | Mean Area Ratio |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 0.90 | 0.0005 | largest | 0.1863 | 0.0765 | 0.2788 | 0.1388 | 11.3234 |
| 2 | 0.90 | 0.0010 | largest | 0.1863 | 0.0765 | 0.2788 | 0.1388 | 11.3234 |
| 3 | 0.90 | 0.0050 | largest | 0.1862 | 0.0765 | 0.2788 | 0.1388 | 11.3232 |
| 4 | 0.90 | 0.0050 | max-score | 0.1861 | 0.0613 | 0.2856 | 0.1467 | 10.1355 |
| 5 | 0.85 | 0.0005 | largest | 0.1822 | 0.0681 | 0.2709 | 0.1309 | 16.7627 |
| 9 | 0.95 | 0.0050 | max-score | 0.1673 | 0.0352 | 0.2585 | 0.1366 | 2.0552 |
| 13 | 0.95 | 0.0005 | max-score | 0.0369 | 0.0000 | 0.0609 | 0.0305 | 0.8270 |
| 15 | 0.90 | 0.0005 | max-score | 0.0216 | 0.0000 | 0.0339 | 0.0158 | 4.8636 |
| 17 | 0.85 | 0.0005 | max-score | 0.0160 | 0.0000 | 0.0181 | 0.0113 | 7.9257 |

Best setting:

- Percentile: `0.90`
- Min area ratio: `0.0005`
- Component: `largest`
- Mean IoU: `0.1863`
- Median IoU: `0.0765`
- Recall@IoU 0.50: `0.1388`
- Pseudo manifest: `outputs/manifests/pseudo_bbox_sweep/pseudo_bbox_p0p9_area0p0005_largest.csv`

Interpretation:

1. Mean IoU around `0.186` and median IoU around `0.077` confirm that the first `patch-contrast` pseudo-bbox localizer is weak.
2. Recall@IoU 0.50 is only `13.88%`, so ROI-only pseudo-bbox classification is expected to underperform.
3. Mean pseudo/GT area ratio is very large (`11.32`) for the best setting, indicating that the largest connected component often produces boxes much larger than GT.
4. `largest` is much more stable than `max-score` for percentile `0.85` and `0.90`; `max-score` often collapses to off-target high-score islands with near-zero median IoU.
5. Pseudo-bbox + whole-image fusion is necessary for the current localizer; the remaining gap to GT fusion should be addressed by stronger localization and more explicit region-context modeling.

## 7. Delta Analysis

### 7.1 BBox/ROI vs Whole-Image

| Setting | Delta Accuracy | Delta Macro-F1 | Observation |
|---|---:|---:|---|
| 5-way 1-shot | -2.48 | -2.81 | ROI loses useful context in the most data-scarce setting. |
| 5-way 3-shot | -0.46 | -0.48 | ROI is close to whole-image. |
| 5-way 5-shot | +1.22 | +1.45 | ROI becomes better when more support examples stabilize the defect prototype. |
| 10-way 1-shot | -1.91 | -2.03 | ROI still suffers in 1-shot. |
| 10-way 5-shot | +2.98 | +3.87 | ROI helps more when class number is larger and support is sufficient. |

### 7.2 Best Fusion vs Whole-Image

| Setting | Best Fusion | Delta Accuracy | Delta Macro-F1 |
|---|---:|---:|---:|
| 5-way 1-shot | Concat | +1.52 | +1.52 |
| 5-way 3-shot | Alpha 0.5 / Concat | +1.84 | +1.82 |
| 5-way 5-shot | Concat | +2.82 | +3.03 |
| 10-way 1-shot | Alpha 0.5 | +2.51 | +2.61 |
| 10-way 5-shot | Concat | +4.72 | +5.42 |

### 7.3 Best Fusion vs BBox/ROI

| Setting | Best Fusion | Delta Accuracy | Delta Macro-F1 |
|---|---:|---:|---:|
| 5-way 1-shot | Concat | +4.00 | +4.33 |
| 5-way 3-shot | Alpha 0.5 / Concat | +2.30 | +2.30 |
| 5-way 5-shot | Concat | +1.60 | +1.58 |
| 10-way 1-shot | Alpha 0.5 | +4.42 | +4.64 |
| 10-way 5-shot | Concat | +1.74 | +1.55 |


### 7.4 Pseudo Fusion Recovery

| Setting | Fusion vs Pseudo ROI Acc | Fusion vs Pseudo ROI F1 | Fusion vs Whole Acc | Fusion vs Whole F1 | Fusion vs GT Fusion Acc | Fusion vs GT Fusion F1 |
|---|---:|---:|---:|---:|---:|---:|
| 5-way 1-shot | +16.70 | +18.45 | -1.60 | -1.90 | -3.12 | -3.42 |
| 5-way 3-shot | +14.72 | +16.25 | -1.46 | -1.51 | -3.22 | -3.33 |
| 5-way 5-shot | +17.88 | +19.97 | -1.04 | -1.09 | -3.86 | -4.12 |
| 10-way 1-shot | +18.46 | +20.08 | -4.18 | -4.67 | -6.38 | -7.06 |
| 10-way 5-shot | +21.98 | +24.30 | -3.78 | -4.01 | -8.50 | -9.43 |

Pseudo-bbox fusion recovers most ROI-only loss and nearly matches whole-image performance in 5-way settings. The remaining gap to GT fusion grows in 10-way settings, which suggests that better localization is still important when class confusion is harder.


### 7.5 Region-Context vs Pseudo Concat

This table compares the best-accuracy region-context setting, which is `whole_weight=0.75` and `region_weight=0.25` for every few-shot setting.

| Setting | Region-Context Acc | Region-Context F1 | Delta Acc vs Pseudo Fusion | Delta F1 vs Pseudo Fusion | Delta Acc vs Whole | Delta Acc vs GT Fusion |
|---|---:|---:|---:|---:|---:|---:|
| 5-way 1-shot | 80.32 | 78.31 | -0.18 | -0.21 | -1.78 | -3.30 |
| 5-way 3-shot | 86.30 | 85.12 | +2.38 | +2.48 | +0.92 | -0.84 |
| 5-way 5-shot | 85.72 | 84.29 | -0.08 | -0.25 | -1.12 | -3.94 |
| 10-way 1-shot | 70.68 | 67.90 | +2.25 | +2.38 | -1.93 | -4.13 |
| 10-way 5-shot | 75.45 | 72.83 | +2.07 | +2.31 | -1.71 | -6.43 |

Score-level region-context improves over pseudo concat fusion in 3 of 5 settings, including both 10-way settings, and the best weight always favors whole-image context. This confirms that pseudo ROI scores are useful, but should be down-weighted until the localizer becomes more accurate.

### 7.6 Region-Context 10-way 5-shot Confusion Analysis

This diagnostic compares the best fixed-weight score fusion (`whole_weight=0.75`, `region_weight=0.25`) against pseudo-bbox concat fusion on the same 200 sampled 10-way 5-shot episodes. The metrics below are query-pooled over 10,000 query predictions, so they can differ slightly from the episode-mean grid in Section 5.9.

| Model | Accuracy | Balanced Acc | Macro-F1 | Queries |
|---|---:|---:|---:|---:|
| region_context | 0.7480 | 0.7523 | 0.7369 | 10000 |
| pseudo_concat | 0.7272 | 0.7315 | 0.7165 | 10000 |
| Delta | +0.0208 | +0.0208 | +0.0204 | - |

Top recall gains from score-level region-context:

| Class | Queries | Recall Delta | F1 Delta | Observation |
|---|---:|---:|---:|---|
| fabric_border | 310 | +0.1581 | +0.1102 | Largest gain; region score helps separate border defects from interior/broken-teeth distractors. |
| squeeze | 270 | +0.0889 | +0.0476 | Strong recall gain with already high absolute recall. |
| squeezed_teeth | 295 | +0.0746 | +0.0493 | Reduces but does not remove confusion with `split_teeth`. |
| split_teeth | 285 | +0.0737 | +0.0544 | Region signal helps fine tooth-structure classes. |
| faulty_imprint | 310 | +0.0645 | +0.0323 | Improves one of the visually ambiguous surface-mark classes. |
| hole | 240 | +0.0500 | +0.0443 | Region evidence is useful when the defect shape is localized. |
| scratch_head | 260 | +0.0462 | +0.0274 | Improves recall, but still confused with scratch-neck/manipulated-front labels. |
| missing_cable | 220 | +0.0455 | +0.0247 | High absolute recall; region score gives a small but consistent boost. |
| thread | 315 | +0.0381 | +0.0053 | Recall gain is partly offset by precision/confusion pressure. |
| thread_top | 280 | +0.0357 | +0.0215 | Still has symmetric confusion with `thread_side`. |

Classes with recall regression or mixed precision/recall trade-offs:

| Class | Queries | Recall Delta | F1 Delta | Observation |
|---|---:|---:|---:|---|
| scratch_neck | 265 | -0.0453 | -0.0250 | Largest degradation; often competes with scratch-head/thread-side/manipulated-front. |
| bent | 285 | -0.0316 | +0.0178 | Recall drops but F1 rises, indicating fewer false positives or better precision. |
| manipulated_front | 265 | -0.0302 | -0.0079 | Degrades slightly; region cue likely overfits to local scratch-like evidence. |
| color | 280 | -0.0286 | -0.0133 | Region score is not helpful for global/appearance color defects. |
| crack | 360 | -0.0111 | +0.0078 | Recall drops slightly but F1 rises. |
| fabric_interior | 295 | -0.0102 | +0.0305 | Recall drops, but precision/F1 improve versus pseudo concat. |
| gray_stroke | 250 | +0.0000 | -0.0050 | Recall unchanged; small F1 loss suggests score calibration noise. |

Top region-context confusion pairs:

| True | Pred | Count | Fraction Of True |
|---|---|---:|---:|
| rough | broken_teeth | 48 | 0.1477 |
| squeezed_teeth | split_teeth | 46 | 0.1559 |
| rough | fabric_interior | 46 | 0.1415 |
| scratch_head | scratch_neck | 43 | 0.1654 |
| thread_side | scratch_neck | 42 | 0.1273 |
| manipulated_front | scratch_neck | 41 | 0.1547 |
| thread_top | thread_side | 41 | 0.1464 |
| thread_side | thread_top | 40 | 0.1212 |
| poke | glue | 37 | 0.1423 |
| crack | faulty_imprint | 37 | 0.1028 |

Interpretation:

1. The query-pooled result confirms the grid-level finding: explicit score-level region-context beats pseudo concat in 10-way 5-shot by about `+2.08` accuracy points and `+2.04` Macro-F1 points.
2. The largest gains come from localized structural labels such as `fabric_border`, tooth defects, `hole`, and `missing_cable`, which supports keeping pseudo region evidence as an auxiliary signal.
3. Regressions are concentrated in appearance/global or very similar fine-grained labels (`color`, `manipulated_front`, `scratch_neck`), suggesting that one fixed weight is not ideal for every class or episode.
4. Persistent confusion pairs are mostly within product/defect families: tooth labels, thread-side/top labels, scratch-head/neck/manipulated-front labels, and rough/fabric/broken-teeth labels.
5. The next useful ablation is to repeat this diagnostic for 10-way 1-shot, then test narrower fixed weights (`0.60/0.40` through `0.90/0.10`) or confidence-based adaptive weighting.

## 8. Conclusions

1. DINOv2 whole-image prototype is already a strong baseline, reaching `82.10%` accuracy in 5-way 1-shot.
2. BBox/ROI features are weaker in 1-shot settings but become stronger in 5-shot settings, especially in 10-way 5-shot.
3. This suggests that ROI features focus on defect evidence but may lose important product/context cues when support samples are scarce.
4. Region-global fusion consistently improves over both whole-image and BBox/ROI baselines.
5. Concat fusion is the most stable simple fusion strategy; weighted-sum is more sensitive to the global-region balance.
6. Alpha 0.25 is generally more stable than alpha 0.75, suggesting that defect-region features are important but need global context as a complement.
7. The first automatic `patch-contrast` pseudo-bbox ROI result is much weaker than all DINOv2 baselines, so the current bottleneck is localization quality rather than the prototype classifier itself.
8. Pseudo-bbox + whole-image fusion recovers most ROI-only loss and approaches whole-image performance in 5-way settings, confirming that global context is crucial when automatic localization is noisy.
9. Score-level region-context with `whole_weight=0.75` improves over pseudo concat fusion in 5-way 3-shot and both 10-way settings, but should keep region scores auxiliary until localization quality improves.
10. Per-class confusion analysis shows that region-context mainly improves localized structural classes, while appearance/global classes and scratch-like fine-grained classes need score calibration or adaptive weights.

## 9. Paper-Writing Takeaway

A concise paper motivation can be written as:

> Whole-image DINOv2 prototypes preserve global product and context cues but are affected by background noise. BBox-region prototypes focus on localized defect evidence but can lose useful context in low-shot settings. A simple region-global fusion consistently improves few-shot defect classification, validating the need to jointly model defect regions and product-level context.

Chinese version:

> 整图 DINOv2 原型保留了产品与上下文信息，但容易受到背景干扰；BBox 区域原型聚焦局部缺陷证据，但在低样本设置下容易丢失有用上下文。简单的区域-全局融合在多个 few-shot 设置下稳定提升性能，说明小样本工业缺陷分类需要同时建模缺陷区域与产品级上下文。

## 10. Next Step

The next stage should move from ground-truth LabelMe bbox to automatic localization:

```text
current: LabelMe bbox -> ROI feature
next: anomaly heatmap -> pseudo bbox / pseudo mask -> region-context feature
```

Recommended next implementation target:

```text
stronger anomaly heatmap localizer
  -> better pseudo bbox / pseudo mask
  -> calibrated or adaptive region-context prototype
```
