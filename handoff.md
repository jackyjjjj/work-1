# Handoff Summary


## Latest Update - 2026-05-05 22:29 +08:00

Use this section as the current state; older sections below are historical context.

- New-best-weight confusion was run: 10-way 1-shot at `0.90/0.10`, 10-way 5-shot at `0.80/0.20`.
- Query-pooled results: 10-way 1-shot region-context `0.7342` Acc / `0.7120` Macro-F1 vs pseudo concat `0.7027` / `0.6836`; 10-way 5-shot region-context `0.7506` / `0.7393` vs pseudo concat `0.7272` / `0.7165`.
- Main 10-way 5-shot regressions remain `scratch_neck`, `color`, and `manipulated_front`, even after reducing region weight to `0.20`.
- Important code fix: `scripts/run_region_context_grid.py` now keeps episode seeds paired across weights by removing `weight_idx` from the seed offset. Earlier fine sweep is useful as screening, but should be rerun with the patched script for strict fixed-weight selection.
- Immediate next step: rerun paired fine sweep, then implement score normalization / confidence-based adaptive weighting if the fixed-weight result still leaves scratch/color/manipulated-front regressions.


## Latest Update - 2026-05-05 22:17 +08:00

Use this section as the current state; older sections below are historical context.

- Fine 10-way region-context weight sweep has been run for `whole_weight=0.60,0.65,0.70,0.75,0.80,0.85,0.90`.
- Best 10-way 1-shot fixed weight: `whole_weight=0.90`, `region_weight=0.10`, Accuracy `73.23 +/- 10.10`, Macro-F1 `70.74 +/- 10.96`.
- Best 10-way 5-shot fixed weight: `whole_weight=0.80`, `region_weight=0.20`, Accuracy `76.57 +/- 9.28`, Macro-F1 `74.15 +/- 10.56`.
- Compared with pseudo concat fusion, these best weights improve Accuracy by `+4.80` points for 10-way 1-shot and `+3.19` points for 10-way 5-shot.
- Interpretation: current pseudo-bbox region evidence is useful as a small correction, but should be whole-heavy because localization remains noisy.
- Immediate next step: rerun per-class confusion at `0.90/0.10` for 10-way 1-shot and `0.80/0.20` for 10-way 5-shot, then decide whether to implement adaptive weighting.


## Latest Update - 2026-05-05 15:40 +08:00

Use this section as the current state; older sections below are historical context.

- `scripts/analyze_region_context_confusion.py` has been implemented and committed in `81c09d2 Add region context confusion analysis`.
- The 10-way 5-shot confusion diagnostic has been run on 200 episodes / 10,000 query predictions.
- Query-pooled result: region-context `0.7480` Accuracy / `0.7369` Macro-F1 vs pseudo concat `0.7272` Accuracy / `0.7165` Macro-F1.
- Biggest recall gains: `fabric_border` (+0.1581), `squeeze` (+0.0889), `squeezed_teeth` (+0.0746), `split_teeth` (+0.0737), `faulty_imprint` (+0.0645), `hole` (+0.0500).
- Main recall regressions: `scratch_neck` (-0.0453), `bent` (-0.0316), `manipulated_front` (-0.0302), `color` (-0.0286), `crack` (-0.0111), `fabric_interior` (-0.0102).
- Interpretation: fixed score-level region-context helps localized structural defects, but global/appearance and scratch-like fine-grained labels need calibrated or adaptive weighting.
- Immediate next step: run the same confusion diagnostic for 10-way 1-shot, then try narrower fixed weights or confidence-based adaptive weighting.

## 目标

本项目目标是实现并验证 **Auto-Mask MVREC / localization-guided few-shot industrial defect classification** 研究路线：以 MVREC 作为 baseline/reference，在主工程中独立实现一个面向小样本工业缺陷分类的定位引导方法。

当前阶段的核心问题是：

1. 已有 DINOv2 whole-image、GT bbox/ROI、region-global fusion baseline。
2. 正在从 GT LabelMe bbox 过渡到自动 heatmap 生成的 pseudo-bbox / pseudo-mask。
3. 最近一次实验显示第一版 `DINOv2 patch-contrast -> pseudo-bbox ROI` 明显低于 whole-image 和 fusion，因此下一步要诊断 pseudo-bbox 定位质量，并尝试 pseudo-bbox + whole-image fusion。

## 项目路径与环境

- 本地 WSL 项目路径：`/home/jack/workspace/work-1`
- Windows UNC 路径：`\\wsl.localhost\Ubuntu\home\jack\workspace\work-1`
- 用户服务器路径示例：`/home/think/mnt/jyl/MyWork/work-1`
- Python 环境：conda 环境 `work-1`
- 本地常用命令前缀：

```bash
/home/jack/miniconda3/bin/conda run -n work-1 python ...
```

- 服务器上用户一般已经激活环境：

```bash
(work-1) think@think-System-Product-Name:~/mnt/jyl/MyWork/work-1$
```

## 约束与偏好

- 所有后续代码/文档修改都要追加记录到 `docs/change_log.md`。
- 用户希望用 GitHub 管理代码，本地改完后给出 `git add / commit / push` 命令。
- 不要把数据集、输出特征、heatmap、结果文件提交进 Git。
- 尽量使用轻量自检脚本，不依赖 `pytest`，因为之前 `work-1` 环境里没有 pytest。
- 代码注释尽量用中文，解释关键逻辑。
- 当前项目主要在 WSL 路径下工作；访问/修改 WSL 项目通常需要用户逐步批准。
- 不要回滚用户已有修改；修改前先看 `git status`。

## 已完成的工程模块

### 数据与 episode

- `src/lg_fdc/data/manifest.py`
  - 统一读取 CSV/JSONL manifest。
- `src/lg_fdc/data/episodes.py`
  - N-way K-shot episode sampler。
- `src/lg_fdc/models/prototype.py`
  - cosine prototype classifier。
- `src/lg_fdc/features/cached.py`
  - 读取 cached feature JSONL/CSV。
- `scripts/run_prototype_baseline.py`
  - 单个 few-shot 设置运行入口。
- `scripts/run_fewshot_grid.py`
  - 标准 grid：`5:1,5:3,5:5,10:1,10:5`。

### MVTec-FS manifest

- `scripts/build_mvtec_fs_manifest.py`
- `src/lg_fdc/data/mvtec_fs.py`
- `scripts/check_mvtec_fs_builder.py`
- `scripts/check_mvtec_fs_unextracted.py`

能力：

- 递归扫描 MVTec-FS 图像。
- 读取同名 LabelMe JSON，提取 defect label、bbox、polygon 数量。
- 默认跳过未标注说明图，避免 `com_sample.jpg`、`data_details.png` 被误认为样本。
- 检测未解压 `image.tar.*` 时给出提示。

### DINOv2 特征提取

- `scripts/extract_dinov2_features.py`
  - 支持 `--region whole` 和 `--region bbox`。
  - bbox 模式使用 manifest 中 `bbox` 字段裁剪。
  - 支持 `--bbox-padding`、`--min-crop-size`。
- `scripts/check_bbox_crop.py`
  - 轻量验证 bbox parsing / crop box。

### 特征融合

- `scripts/fuse_feature_cache.py`
  - 支持 `concat` 和 `weighted-sum`。
  - 支持 `--normalize-input`、`--normalize-output`。
- `scripts/check_feature_fusion.py`

### pseudo-bbox 生成

- `scripts/extract_dinov2_patch_heatmaps.py`
  - 生成真实 heatmap JSONL。
  - `--mode patch-contrast`：不需要 normal/good 图像，直接用单图 patch 与平均 patch 的差异生成启动版 heatmap。
  - `--mode nearest-memory`：如果后续 manifest 包含 good/normal 图像，可用 normal 图像 DINOv2 patch token 建 memory bank。
- `scripts/check_dinov2_patch_heatmaps.py`

- `scripts/build_pseudo_bbox_manifest.py`
  - 输入原始 manifest 和 heatmap JSONL。
  - 输出替换 `bbox` 后的 pseudo-bbox manifest。
  - 支持 `--percentile`、`--min-area-ratio`、`--component largest|max-score`。
  - 支持 `--missing-policy error|clear|keep`。
  - 已新增 `--split`，解决 heatmap 只覆盖 train 时全量 manifest 报错的问题。
- `scripts/check_pseudo_bbox.py`

### pseudo-bbox IoU 诊断

最近新增：

- `scripts/evaluate_pseudo_bbox_iou.py`
  - 对齐 GT manifest 与 pseudo manifest。
  - 计算 per-image IoU、Pseudo/GT 面积比、Recall@IoU、最差样本。
  - 支持输出 JSON / Markdown / CSV。
- `scripts/check_pseudo_bbox_iou.py`
  - 轻量验证 bbox parsing、IoU、输出文件。

## 已记录的关键实验结果

正式记录文件：`experiments/dinov2_baselines.md`

### Whole-image DINOv2 prototype

Feature file: `outputs/features/dinov2/mvtec_fs_train.jsonl`

| Setting | Accuracy | Macro-F1 |
|---|---:|---:|
| 5-way 1-shot | 82.10 +/- 13.37 | 80.42 +/- 14.64 |
| 5-way 3-shot | 85.38 +/- 11.52 | 84.15 +/- 12.59 |
| 5-way 5-shot | 86.84 +/- 11.38 | 85.63 +/- 12.68 |
| 10-way 1-shot | 72.61 +/- 9.98 | 70.19 +/- 10.81 |
| 10-way 5-shot | 77.16 +/- 8.67 | 74.53 +/- 9.72 |

### GT bbox/ROI DINOv2 prototype

Feature file: `outputs/features/dinov2_bbox/mvtec_fs_train.jsonl`

| Setting | Accuracy | Macro-F1 |
|---|---:|---:|
| 5-way 1-shot | 79.62 +/- 13.32 | 77.61 +/- 14.71 |
| 5-way 3-shot | 84.92 +/- 10.87 | 83.67 +/- 11.86 |
| 5-way 5-shot | 88.06 +/- 9.74 | 87.08 +/- 10.75 |
| 10-way 1-shot | 70.70 +/- 9.39 | 68.16 +/- 10.16 |
| 10-way 5-shot | 80.14 +/- 7.13 | 78.40 +/- 7.83 |

### GT bbox + whole concat fusion

Feature file: `outputs/features/dinov2_fusion_concat/mvtec_fs_train.jsonl`

| Setting | Accuracy | Macro-F1 |
|---|---:|---:|
| 5-way 1-shot | 83.62 +/- 12.19 | 81.94 +/- 13.66 |
| 5-way 3-shot | 87.14 +/- 10.10 | 85.97 +/- 11.17 |
| 5-way 5-shot | 89.66 +/- 9.60 | 88.66 +/- 10.89 |
| 10-way 1-shot | 74.81 +/- 9.24 | 72.58 +/- 10.10 |
| 10-way 5-shot | 81.88 +/- 7.38 | 79.95 +/- 8.28 |

### Weighted fusion

- `alpha=0.25`、`alpha=0.5`、`alpha=0.75` 已记录在 `experiments/dinov2_baselines.md`。
- 主要结论：concat 最稳定，weighted-sum 对 alpha 敏感。

### DINOv2 patch-contrast pseudo-bbox ROI

Manifest: `data/manifests/mvtec_fs_pseudo_bbox_train.csv`
Feature file: `outputs/features/dinov2_pseudo_bbox/mvtec_fs_train.jsonl`
Localizer: `dinov2_patch_contrast`

| Setting | Accuracy | Macro-F1 |
|---|---:|---:|
| 5-way 1-shot | 63.80 +/- 14.29 | 60.07 +/- 15.94 |
| 5-way 3-shot | 69.20 +/- 12.79 | 66.39 +/- 14.21 |
| 5-way 5-shot | 67.92 +/- 13.78 | 64.57 +/- 15.56 |
| 10-way 1-shot | 49.97 +/- 9.48 | 45.44 +/- 10.03 |
| 10-way 5-shot | 51.40 +/- 8.14 | 46.22 +/- 8.99 |

关键判断：

- pseudo-bbox ROI 明显低于 whole-image、GT bbox/ROI、GT fusion。
- 当前瓶颈主要是 pseudo localization quality，不是 prototype classifier。
- 需要先跑 IoU 诊断，再跑 pseudo-bbox + whole-image fusion 看上下文能否恢复性能。

## 关键命令

### 1. 构建 MVTec-FS manifest

```bash
python scripts/build_mvtec_fs_manifest.py \
  --dataset-root /path/to/MVTec-FS \
  --output data/manifests/mvtec_fs.csv
```

如果数据集目录里有 `image.tar.001` 到 `image.tar.012`，先解压：

```bash
cd /path/to/MVTec-FS
cat image.tar.* | tar -xvf -
```

### 2. whole-image DINOv2 特征

```bash
mkdir -p outputs/features/dinov2
python scripts/extract_dinov2_features.py \
  --manifest data/manifests/mvtec_fs.csv \
  --image-root /path/to/MVTec-FS \
  --split train \
  --output outputs/features/dinov2/mvtec_fs_train.jsonl \
  --model dinov2_vits14 \
  --batch-size 16 \
  --device auto \
  --overwrite
```

### 3. GT bbox/ROI DINOv2 特征

```bash
mkdir -p outputs/features/dinov2_bbox
python scripts/extract_dinov2_features.py \
  --manifest data/manifests/mvtec_fs.csv \
  --image-root /path/to/MVTec-FS \
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

### 4. 标准 few-shot grid

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

### 5. GT bbox + whole concat fusion

```bash
mkdir -p outputs/features/dinov2_fusion_concat
python scripts/fuse_feature_cache.py \
  --whole-file outputs/features/dinov2/mvtec_fs_train.jsonl \
  --region-file outputs/features/dinov2_bbox/mvtec_fs_train.jsonl \
  --method concat \
  --output outputs/features/dinov2_fusion_concat/mvtec_fs_train.jsonl \
  --overwrite

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

### 6. 生成 DINOv2 patch-contrast heatmap

`example_heatmaps.jsonl` 只是旧文档中的占位符，不会自动存在。当前真实 heatmap 生成命令如下：

```bash
mkdir -p outputs/heatmaps
python scripts/extract_dinov2_patch_heatmaps.py \
  --manifest data/manifests/mvtec_fs.csv \
  --image-root /path/to/MVTec-FS \
  --split train \
  --output outputs/heatmaps/dinov2_patch_contrast_train.jsonl \
  --mode patch-contrast \
  --model dinov2_vits14 \
  --batch-size 8 \
  --device auto \
  --overwrite
```

### 7. heatmap -> pseudo-bbox manifest

注意：如果 heatmap 用 `--split train` 生成，pseudo-bbox builder 也必须用 `--split train`，否则会检查全量 manifest 并报 missing heatmap。

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

### 8. pseudo-bbox DINOv2 ROI 特征

```bash
mkdir -p outputs/features/dinov2_pseudo_bbox
python scripts/extract_dinov2_features.py \
  --manifest data/manifests/mvtec_fs_pseudo_bbox_train.csv \
  --image-root /path/to/MVTec-FS \
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

### 9. pseudo-bbox ROI grid

```bash
python scripts/run_fewshot_grid.py \
  --manifest data/manifests/mvtec_fs_pseudo_bbox_train.csv \
  --split train \
  --grid 5:1,5:3,5:5,10:1,10:5 \
  --q-queries 5 \
  --episodes 200 \
  --feature-source cached \
  --feature-file outputs/features/dinov2_pseudo_bbox/mvtec_fs_train.jsonl \
  --feature-dim 384 \
  --output-json outputs/results/dinov2_pseudo_bbox_grid.json \
  --output-md outputs/results/dinov2_pseudo_bbox_grid.md
```

### 10. pseudo-bbox IoU 诊断

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

### 11. pseudo-bbox + whole-image concat fusion

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

## 关键文件

### 文档

- `README.md`
  - 主要运行命令。
- `docs/change_log.md`
  - 每次修改日志，必须继续追加。
- `experiments/dinov2_baselines.md`
  - 当前 DINOv2 baseline 与 pseudo-bbox ROI 实验记录。
- `handoff.md`
  - 本交接文档。

### 脚本

- `scripts/build_mvtec_fs_manifest.py`
- `scripts/extract_dinov2_features.py`
- `scripts/run_prototype_baseline.py`
- `scripts/run_fewshot_grid.py`
- `scripts/fuse_feature_cache.py`
- `scripts/extract_dinov2_patch_heatmaps.py`
- `scripts/build_pseudo_bbox_manifest.py`
- `scripts/evaluate_pseudo_bbox_iou.py`

### 自检脚本

- `scripts/smoke_test.py`
- `scripts/check_mvtec_fs_builder.py`
- `scripts/check_mvtec_fs_unextracted.py`
- `scripts/check_cached_feature_baseline.py`
- `scripts/check_bbox_crop.py`
- `scripts/check_feature_fusion.py`
- `scripts/check_dinov2_patch_heatmaps.py`
- `scripts/check_pseudo_bbox.py`
- `scripts/check_pseudo_bbox_iou.py`

## 当前 Git 状态

生成本文件前，仓库状态为：

```text
## main...origin/main
 M README.md
 M docs/change_log.md
 M experiments/dinov2_baselines.md
?? scripts/check_pseudo_bbox_iou.py
?? scripts/evaluate_pseudo_bbox_iou.py
```

本次新增 `handoff.md` 后，也需要提交。

建议提交命令：

```bash
git add README.md docs/change_log.md experiments/dinov2_baselines.md scripts/evaluate_pseudo_bbox_iou.py scripts/check_pseudo_bbox_iou.py handoff.md
git commit -m "Add pseudo bbox diagnostics and handoff summary"
git push origin main
```

## 已验证命令

最近已验证：

```bash
/home/jack/miniconda3/bin/conda run -n work-1 python scripts/check_pseudo_bbox_iou.py
```

输出：`pseudo-bbox-iou-check-ok`

```bash
/home/jack/miniconda3/bin/conda run -n work-1 python -m py_compile scripts/evaluate_pseudo_bbox_iou.py scripts/check_pseudo_bbox_iou.py
```

结果：通过，无语法错误。

之前也验证过：

```bash
/home/jack/miniconda3/bin/conda run -n work-1 python scripts/check_pseudo_bbox.py
/home/jack/miniconda3/bin/conda run -n work-1 python scripts/check_dinov2_patch_heatmaps.py
/home/jack/miniconda3/bin/conda run -n work-1 python scripts/check_feature_fusion.py
/home/jack/miniconda3/bin/conda run -n work-1 python scripts/check_bbox_crop.py
```

## 未完成 / 下一步

### 最高优先级

1. 在服务器上运行 pseudo-bbox IoU 诊断：
   - 生成 `outputs/diagnostics/pseudo_bbox_iou_train.md`
   - 查看 mean IoU、median IoU、Recall@IoU 0.25/0.50
   - 查看 worst examples，判断是框太大、太小、偏移还是热力图背景响应错误。

2. 跑 pseudo-bbox + whole-image concat fusion：
   - 输出 `outputs/features/dinov2_pseudo_fusion_concat/mvtec_fs_train.jsonl`
   - 跑 grid，和 pseudo-bbox ROI、whole-image、GT fusion 对比。

3. 根据 IoU 诊断调 pseudo-bbox 参数：
   - `--percentile 0.85 / 0.90 / 0.95`
   - `--min-area-ratio 0.0005 / 0.001 / 0.005`
   - `--component largest` vs `--component max-score`

### 中期方向

4. 如果当前 `patch-contrast` localizer 的 IoU 很低，考虑：
   - 让 manifest 纳入 normal/good 图像后跑 `--mode nearest-memory`。
   - 或接入 PatchCore / AnomalyDINO 生成 heatmap。

5. 如果 pseudo-bbox fusion 明显高于 pseudo-bbox ROI：
   - 说明上下文补偿有效。
   - 下一步实现更正式的 region-context prototype，而不是简单 concat。

6. 如果 pseudo-bbox fusion 仍然很差：
   - 优先改 localization，不急着改 classifier。

## 重要注意事项

- 运行 `build_pseudo_bbox_manifest.py` 时，如果 heatmap 是 `--split train` 生成的，必须加：

```bash
--split train
```

否则会出现：

```text
ValueError: Missing heatmaps for ... manifest rows
```

- `data/manifests/mvtec_fs_pseudo_bbox_train.csv` 只包含 train split；后续 `extract_dinov2_features.py` 和 `run_fewshot_grid.py` 都应配套使用 `--split train`。
- pseudo-bbox ROI 当前结果差，不代表路线失败；它更像是 localization ablation，证明自动定位质量是关键瓶颈。
- GT bbox + whole fusion 已经证明“区域 + 上下文”有效，因此下一阶段要把自动 localization 做准，或用 fusion 抵消 pseudo ROI 的上下文损失。
