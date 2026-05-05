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

| Setting | Whole | BBox/ROI | Pseudo-BBox ROI | Pseudo-BBox Fusion | GT Concat Fusion | Alpha 0.25 | Alpha 0.5 | Alpha 0.75 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5-way 1-shot | 82.10 | 79.62 | 63.80 | 80.50 | **83.62** | 82.48 | 83.08 | 82.48 |
| 5-way 3-shot | 85.38 | 84.92 | 69.20 | 83.92 | 87.14 | 86.50 | **87.22** | 86.34 |
| 5-way 5-shot | 86.84 | 88.06 | 67.92 | 85.80 | **89.66** | 89.42 | 89.46 | 87.90 |
| 10-way 1-shot | 72.61 | 70.70 | 49.97 | 68.43 | 74.81 | 73.92 | **75.12** | 74.25 |
| 10-way 5-shot | 77.16 | 80.14 | 51.40 | 73.38 | **81.88** | 81.53 | 81.60 | 79.32 |

## 4. Macro-F1 Summary

| Setting | Whole | BBox/ROI | Pseudo-BBox ROI | Pseudo-BBox Fusion | GT Concat Fusion | Alpha 0.25 | Alpha 0.5 | Alpha 0.75 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5-way 1-shot | 80.42 | 77.61 | 60.07 | 78.52 | **81.94** | 80.60 | 81.33 | 80.75 |
| 5-way 3-shot | 84.15 | 83.67 | 66.39 | 82.64 | **85.97** | 85.31 | **85.97** | 85.18 |
| 5-way 5-shot | 85.63 | 87.08 | 64.57 | 84.54 | **88.66** | 88.46 | 88.43 | 86.75 |
| 10-way 1-shot | 70.19 | 68.16 | 45.44 | 65.52 | 72.58 | 71.51 | **72.80** | 71.96 |
| 10-way 5-shot | 74.53 | 78.40 | 46.22 | 70.52 | **79.95** | 79.78 | 79.51 | 76.82 |

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

## 8. Conclusions

1. DINOv2 whole-image prototype is already a strong baseline, reaching `82.10%` accuracy in 5-way 1-shot.
2. BBox/ROI features are weaker in 1-shot settings but become stronger in 5-shot settings, especially in 10-way 5-shot.
3. This suggests that ROI features focus on defect evidence but may lose important product/context cues when support samples are scarce.
4. Region-global fusion consistently improves over both whole-image and BBox/ROI baselines.
5. Concat fusion is the most stable simple fusion strategy; weighted-sum is more sensitive to the global-region balance.
6. Alpha 0.25 is generally more stable than alpha 0.75, suggesting that defect-region features are important but need global context as a complement.
7. The first automatic `patch-contrast` pseudo-bbox ROI result is much weaker than all DINOv2 baselines, so the current bottleneck is localization quality rather than the prototype classifier itself.
8. Pseudo-bbox + whole-image fusion recovers most ROI-only loss and approaches whole-image performance in 5-way settings, confirming that global context is crucial when automatic localization is noisy.

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
  -> explicit region-context prototype beyond simple concat
```
