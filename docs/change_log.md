# Change Log

这个文件记录本项目后续每次由 Codex 进行的代码、配置、文档修改。

## 记录规则

每次修改后追加一个条目，包含：

- 日期时间；
- 修改目的；
- 涉及文件；
- 主要改动；
- 验证命令和结果；
- 后续待办。

## 2026-05-02 21:00 +08:00

### 修改目的

初始化 Auto-Mask MVREC / localization-guided few-shot defect classification 项目工程骨架，为后续 MVTec-FS、DINOv2 baseline、pseudo mask 和 region-context prototype 方法实现做准备。

### 涉及文件

- `README.md`
- `.gitignore`
- `pyproject.toml`
- `requirements.txt`
- `docs/research_plan.md`
- `docs/engineering_plan.md`
- `docs/change_log.md`
- `baselines/MVREC/README.md`
- `configs/baseline/dinov2_prototype.yaml`
- `configs/method/auto_mask_mvrec.yaml`
- `data/README.md`
- `data/example_manifest.csv`
- `experiments/README.md`
- `scripts/smoke_test.py`
- `scripts/run_prototype_baseline.py`
- `src/lg_fdc/data/manifest.py`
- `src/lg_fdc/data/episodes.py`
- `src/lg_fdc/features/base.py`
- `src/lg_fdc/features/simple.py`
- `src/lg_fdc/localization/base.py`
- `src/lg_fdc/masks/pseudo_mask.py`
- `src/lg_fdc/models/prototype.py`
- `src/lg_fdc/evaluation/metrics.py`
- `src/lg_fdc/pipelines/prototype_baseline.py`
- `tests/conftest.py`
- `tests/test_prototype_smoke.py`

### 主要改动

- 建立 `src/lg_fdc/` 主代码包。
- 实现 manifest 读取、N-way K-shot episode 采样、prototype classifier、基础评估指标。
- 添加异常热力图 `Localizer` 接口和初始 pseudo mask 生成接口。
- 添加 dependency-light smoke test。
- 添加从 manifest 跑 prototype baseline 的命令行入口。
- 添加 DINOv2 prototype baseline 和 Auto-Mask MVREC 方法配置草稿。
- 明确 MVREC 官方代码放在 `baselines/MVREC/`，作为 baseline 复现区，不和主方法代码混在一起。
- 为主要 Python 代码补充中文注释。
- 明确后续使用 WSL conda 环境 `work-1` 运行代码。

### 验证命令和结果

```bash
/home/jack/miniconda3/bin/conda run -n work-1 python scripts/smoke_test.py
```

结果：通过，`accuracy=1.000`、`balanced_accuracy=1.000`、`macro_f1=1.000`。

```bash
/home/jack/miniconda3/bin/conda run -n work-1 python scripts/run_prototype_baseline.py --manifest data/example_manifest.csv --n-way 3 --k-shot 2 --q-queries 2 --episodes 5 --feature-source metadata --feature-dim 3
```

结果：通过，5 个 episode 的 `accuracy`、`balanced_accuracy`、`macro_f1` 均值均为 `1.0`。

### 后续待办

- 实现 MVTec-FS manifest builder。
- 接入真实 DINOv2 或缓存 DINOv2 特征。
- 实现 whole-image prototype baseline。
- 实现 ROI / pseudo-mask region prototype baseline。
- 后续每次修改继续追加日志到本文件。
## 2026-05-02 21:35 +08:00

### 修改目的

在假定 MVTec-FS 数据集已经下载到本地的前提下，增加数据集 manifest 构建能力。目标是在本地或服务器上把 MVTec-FS 目录整理成项目统一的 CSV manifest，供后续 few-shot episode 采样、DINOv2 baseline 和 Auto-Mask MVREC 方法使用。

### 涉及文件

- `README.md`
- `data/README.md`
- `scripts/build_mvtec_fs_manifest.py`
- `scripts/check_mvtec_fs_builder.py`
- `src/lg_fdc/data/mvtec_fs.py`
- `tests/test_mvtec_fs_manifest.py`
- `docs/change_log.md`

### 主要改动

- 新增 `MVTecFSBuildConfig` 和 `build_mvtec_fs_manifest()`。
- 支持递归扫描本地 MVTec-FS 图片文件。
- 支持读取同名 LabelMe JSON 标注，提取缺陷类别、polygon 数量和整体 bbox。
- 生成统一 CSV schema：`image_path,label,split,mask_path,object_name,defect_name,annotation_path,bbox,polygon_count`。
- 在没有官方 split 文件的情况下，按缺陷类别做确定性 train/val/test 划分。
- 新增 `scripts/build_mvtec_fs_manifest.py` 作为命令行入口。
- 新增 `scripts/check_mvtec_fs_builder.py` 作为不依赖 pytest 的自检脚本，方便服务器环境直接验证。
- 更新 README 和 data 文档，写明假定数据集在 `/home/jack/datasets/MVTec-FS` 时如何构建 manifest。

### 验证命令和结果

```bash
/home/jack/miniconda3/bin/conda run -n work-1 python scripts/check_mvtec_fs_builder.py
```

结果：通过，输出 `mvtec-fs-builder-ok`。

```bash
/home/jack/miniconda3/bin/conda run -n work-1 python scripts/smoke_test.py
```

结果：通过，`accuracy=1.000`、`balanced_accuracy=1.000`、`macro_f1=1.000`。

```bash
/home/jack/miniconda3/bin/conda run -n work-1 python scripts/run_prototype_baseline.py --manifest data/example_manifest.csv --n-way 3 --k-shot 2 --q-queries 2 --episodes 5 --feature-source metadata --feature-dim 3
```

结果：通过，5 个 episode 的 `accuracy`、`balanced_accuracy`、`macro_f1` 均值均为 `1.0`。

```bash
/home/jack/miniconda3/bin/conda run -n work-1 python -m pytest tests/test_mvtec_fs_manifest.py tests/test_prototype_smoke.py
```

结果：未运行，当前 conda 环境 `work-1` 没有安装 `pytest`，报错 `No module named pytest`。这不影响脚本级验证。

### 后续待办

- 用户下载或同步 MVTec-FS 后，运行 `scripts/build_mvtec_fs_manifest.py` 生成真实 manifest。
- 用真实 manifest 先跑 hash-feature pipeline check。
- 接入真实 DINOv2 特征缓存，替换 hash feature。
- 基于 bbox/annotation_path 实现 ROI prototype baseline。

## 2026-05-03 00:20 +08:00

### 修改目的

修复 MVTec-FS manifest builder 在数据集未解压或 `--dataset-root` 指向 GitHub 仓库外层时，把 `com_sample.jpg`、`data_details.png` 等说明图片误识别为训练样本的问题。

### 背景问题

用户生成的 `mvtec_fs.csv` 只有两行：

```csv
com_sample.jpg,MVTec-FS,train,,unknown_object,MVTec-FS,,,0
data_details.png,MVTec-FS,train,,unknown_object,MVTec-FS,,,0
```

这说明实际 MVTec-FS 图像还没有从 `image.tar.001` 到 `image.tar.012` 中解压，或者 `--dataset-root` 指向了未解压的仓库外层。

### 涉及文件

- `README.md`
- `data/README.md`
- `scripts/build_mvtec_fs_manifest.py`
- `scripts/check_mvtec_fs_unextracted.py`
- `src/lg_fdc/data/mvtec_fs.py`
- `docs/change_log.md`

### 主要改动

- `build_mvtec_fs_manifest()` 默认只保留带 LabelMe JSON 标注的图片，跳过未标注说明图片。
- 新增 `--include-unannotated` 参数，仅在用户明确需要时才纳入无 JSON 图片。
- 检测到 `image.tar.*` 分卷但没有可用标注图片时，直接报清晰错误，提示先执行 `cat image.tar.* | tar -xvf -`。
- 支持从路径中保留官方 `train` / `val` / `test` split；没有 split 信息时才自动划分。
- 修正 LabelMe bbox 点类型判断，兼容 Python 3.10+。
- 新增 `scripts/check_mvtec_fs_unextracted.py`，验证未解压数据集会给出明确错误提示。
- 更新 README 和 `data/README.md`，补充 MVTec-FS 分卷解压说明和错误 manifest 排查说明。

### 验证命令和结果

```bash
/home/jack/miniconda3/bin/conda run -n work-1 python scripts/check_mvtec_fs_builder.py
```

结果：通过，输出 `mvtec-fs-builder-ok`。

```bash
/home/jack/miniconda3/bin/conda run -n work-1 python scripts/check_mvtec_fs_unextracted.py
```

结果：通过，输出 `mvtec-fs-unextracted-check-ok`。

```bash
/home/jack/miniconda3/bin/conda run -n work-1 python scripts/smoke_test.py
```

结果：通过，`accuracy=1.000`、`balanced_accuracy=1.000`、`macro_f1=1.000`。

```bash
/home/jack/miniconda3/bin/conda run -n work-1 python scripts/run_prototype_baseline.py --manifest data/example_manifest.csv --n-way 3 --k-shot 2 --q-queries 2 --episodes 5 --feature-source metadata --feature-dim 3
```

结果：通过，示例 manifest baseline 正常输出 JSON 指标。

### 后续待办

- 在真实数据所在机器上先进入 MVTec-FS 目录执行 `cat image.tar.* | tar -xvf -`。
- 重新运行 `scripts/build_mvtec_fs_manifest.py` 生成 manifest。
- 检查 manifest 中是否有多个缺陷 label，而不是只有 `MVTec-FS`。
- 如果真实 MVTec-FS 解压后的目录结构和当前假设不同，再根据实际 `find` 输出调整 parser。

## 2026-05-03 01:00 +08:00

### 修改目的

接入第一条有意义的真实视觉 baseline：DINOv2 whole-image feature + prototype classifier。之前 `--feature-source hash` 只用于数据流检查，准确率接近 5-way 随机水平是正常现象。

### 涉及文件

- `src/lg_fdc/features/cached.py`
- `src/lg_fdc/features/__init__.py`
- `scripts/extract_dinov2_features.py`
- `scripts/run_prototype_baseline.py`
- `scripts/check_cached_feature_baseline.py`
- `configs/baseline/dinov2_prototype.yaml`
- `README.md`
- `docs/change_log.md`

### 主要改动

- 新增 `CachedFeatureExtractor`，支持从 JSONL/CSV 读取缓存特征。
- `run_prototype_baseline.py` 新增 `--feature-source cached` 和 `--feature-file` 参数。
- 新增 `scripts/extract_dinov2_features.py`，从 manifest 读取图像并用 Torch Hub DINOv2 提取 whole-image 特征。
- 新增 `scripts/check_cached_feature_baseline.py`，不依赖 torch/PIL，用合成缓存特征验证 cached-feature pipeline。
- 更新 DINOv2 baseline 配置，加入 train/test feature cache 路径。
- README 增加 DINOv2 特征提取和 cached prototype baseline 运行命令。

### 验证命令和结果

轻量级自检命令：

```bash
python scripts/check_cached_feature_baseline.py
```

DINOv2 特征提取需要在服务器或安装了 `torch`、`torchvision`、`Pillow` 的环境中运行：

```bash
python scripts/extract_dinov2_features.py --manifest data/manifests/mvtec_fs.csv --image-root /path/to/MVTec-FS --split train --output outputs/features/dinov2/mvtec_fs_train.jsonl --model dinov2_vits14 --overwrite
```

### 后续待办

- 在服务器上安装或确认 `torch`、`torchvision`、`Pillow` 可用。
- 提取 MVTec-FS train split 的 DINOv2 特征。
- 用 `--feature-source cached` 跑 5-way 1-shot/5-shot baseline。
- 如果效果正常，再做 ROI/pseudo-mask region feature baseline。

## 2026-05-03 01:35 +08:00

### 修改目的

新增 few-shot grid runner，自动运行多组 N-way/K-shot prototype baseline，并输出 JSON 与 Markdown 实验表，方便整理 DINOv2 whole-image baseline 的第一张实验表。

### 涉及文件

- `scripts/run_fewshot_grid.py`
- `README.md`
- `docs/change_log.md`

### 主要改动

- 新增 `scripts/run_fewshot_grid.py`。
- 支持 `--grid 5:1,5:3,5:5,10:1,10:5` 形式一次性配置多组实验。
- 支持 `metadata`、`hash`、`cached` 三种特征来源。
- 对每组实验输出 Accuracy、Balanced Accuracy、Macro-F1 的均值和标准差。
- 自动保存 JSON 结果和 Markdown 表格。
- README 增加服务器运行命令。

### 验证命令和结果

轻量级自检命令：

```bash
/home/jack/miniconda3/bin/conda run -n work-1 python scripts/run_fewshot_grid.py --manifest data/example_manifest.csv --split train --grid 3:1,3:2 --q-queries 2 --episodes 3 --feature-source metadata --feature-dim 3 --output-json outputs/tmp/grid_check.json --output-md outputs/tmp/grid_check.md
```

预期：生成 `outputs/tmp/grid_check.json` 和 `outputs/tmp/grid_check.md`。

### 后续待办

- 在服务器上用 DINOv2 cached features 跑标准 grid：`5:1,5:3,5:5,10:1,10:5`。
- 将输出的 Markdown 表格作为 DINOv2 whole-image prototype baseline 的第一版实验表。
- 下一步实现 ROI / pseudo-mask region feature baseline，和 whole-image baseline 对比。

## 2026-05-04 00:20 +08:00

### 修改目的

新增 DINOv2 bbox/ROI feature extraction，用于构建第二条关键 baseline：`DINOv2 bbox/ROI prototype`。该 baseline 用 LabelMe 标注推导出的 bbox 裁剪缺陷区域，再提取 DINOv2 特征，检验 localization-guided 思路是否优于 whole-image prototype。

### 涉及文件

- `scripts/extract_dinov2_features.py`
- `scripts/check_bbox_crop.py`
- `README.md`
- `docs/change_log.md`

### 主要改动

- `extract_dinov2_features.py` 新增 `--region whole/bbox` 参数。
- 新增 `--bbox-padding`，控制 bbox crop 周围上下文扩展比例。
- 新增 `--min-crop-size`，避免极小缺陷裁剪区域过小。
- bbox 模式下从 manifest 的 `bbox` 字段读取 `x1,y1,x2,y2`，裁剪后再做 DINOv2 预处理和特征提取。
- 输出 JSONL 中增加 `region` 和 `crop_box` 字段，方便后续追踪 whole/bbox 特征来源。
- 新增 `scripts/check_bbox_crop.py`，不依赖 torch/PIL，验证 bbox 解析、padding 和边界裁剪逻辑。
- README 增加 DINOv2 bbox/ROI baseline 的服务器运行命令。

### 验证命令和结果

轻量级自检命令：

```bash
/home/jack/miniconda3/bin/conda run -n work-1 python scripts/check_bbox_crop.py
```

预期输出：`bbox-crop-check-ok`。

语法检查：

```bash
/home/jack/miniconda3/bin/conda run -n work-1 python -m py_compile scripts/extract_dinov2_features.py scripts/check_bbox_crop.py
```

### 后续待办

- 在服务器上提取 `outputs/features/dinov2_bbox/mvtec_fs_train.jsonl`。
- 跑 `dinov2_bbox_prototype_grid`，和 `dinov2_prototype_grid` 对比。
- 如果 bbox ROI 高于 whole image，继续做 pseudo-mask/region-context；如果低于 whole image，优先分析 context 丢失问题。

## 2026-05-04 00:50 +08:00

### 修改目的

新增 region-global feature fusion baseline，将 DINOv2 whole-image feature 与 DINOv2 bbox/ROI feature 融合，用于验证“整图上下文 + 局部缺陷区域”是否优于单独使用 whole image 或 ROI。

### 涉及文件

- `scripts/fuse_feature_cache.py`
- `scripts/check_feature_fusion.py`
- `README.md`
- `docs/change_log.md`

### 主要改动

- 新增 `scripts/fuse_feature_cache.py`。
- 支持 `concat` 融合：`[whole_feature, region_feature]`，维度从 384 变成 768。
- 支持 `weighted-sum` 融合：`alpha * whole + (1-alpha) * region`，维度保持 384。
- 支持 `--normalize-input` 和 `--normalize-output`，方便做归一化加权融合。
- 输出 JSONL 中保留 `fusion_method`、`fusion_alpha`、`region_crop_box` 等追踪字段。
- 新增 `scripts/check_feature_fusion.py`，不依赖重模型，验证 concat/weighted-sum 与 JSONL 输出逻辑。
- README 增加 fusion concat 和 weighted-sum 的服务器运行命令。

### 验证命令和结果

轻量级自检命令：

```bash
/home/jack/miniconda3/bin/conda run -n work-1 python scripts/check_feature_fusion.py
```

预期输出：`feature-fusion-check-ok`。

语法检查：

```bash
/home/jack/miniconda3/bin/conda run -n work-1 python -m py_compile scripts/fuse_feature_cache.py scripts/check_feature_fusion.py
```

### 后续待办

- 在服务器上生成 `dinov2_fusion_concat` 特征并跑 grid。
- 可选扫描 weighted-sum 的 `alpha=0.25,0.5,0.75`。
- 对比 whole-image、bbox/ROI、fusion 三张表，判断 region-context 融合是否成立。

## 2026-05-04 01:20 +08:00

### 修改目的

将 DINOv2 whole-image、bbox/ROI、region-global fusion 相关 baseline 实验结果固化为正式实验记录，便于后续论文写作、开题汇报和方法设计对比。

### 涉及文件

- `experiments/dinov2_baselines.md`
- `docs/change_log.md`

### 主要改动

- 新增 `experiments/dinov2_baselines.md`。
- 汇总 DINOv2 whole-image prototype、bbox/ROI prototype、concat fusion、weighted-sum alpha 0.25/0.5/0.75 的实验结果。
- 增加 Accuracy 和 Macro-F1 总表。
- 增加 BBox/ROI vs Whole、Best Fusion vs Whole、Best Fusion vs BBox/ROI 的 delta 分析。
- 写出当前实验结论和可用于论文 motivation 的中英文表述。
- 明确下一阶段应从 LabelMe bbox 过渡到 anomaly heatmap 生成的 pseudo bbox / pseudo mask。

### 当前关键结论

- Whole-image DINOv2 prototype 是强 baseline，但不是最优。
- BBox/ROI 在 1-shot 下会丢失上下文，在 5-shot 下更有优势。
- Region-global fusion 在所有设置下稳定超过 whole-image 和 bbox/ROI。
- Concat fusion 是当前最稳定的简单融合策略。

### 后续待办

- 开始实现 anomaly heatmap -> pseudo bbox / pseudo mask。
- 先做 pseudo bbox ROI prototype，再和 GT bbox ROI、fusion baseline 对比。
- 进一步分析 per-class F1 和 confusion matrix，找出最受益/最困难的缺陷类别。

## 2026-05-04 01:55 +08:00

### 修改目的

开始 pseudo-mask 阶段的第一步：实现通用的 anomaly heatmap -> pseudo bbox manifest 工具。这样后续 PatchCore、AnomalyDINO 或其它定位器只要输出 heatmap JSONL，就能转换成 pseudo bbox，并复用现有 DINOv2 bbox 特征提取与 few-shot grid runner。

### 涉及文件

- `scripts/build_pseudo_bbox_manifest.py`
- `scripts/check_pseudo_bbox.py`
- `README.md`
- `docs/change_log.md`

### 主要改动

- 新增 `scripts/build_pseudo_bbox_manifest.py`。
- 输入原始 manifest 和 heatmap JSONL，输出替换 bbox 字段后的 pseudo-bbox manifest。
- heatmap JSONL 要求包含 `image_path`、`heatmap`，建议包含 `image_width`、`image_height`。
- 支持 percentile 阈值、最小区域比例、连通域选择策略 `largest` / `max-score`。
- 新增 `--missing-policy`，默认缺失 heatmap 直接报错，避免 pseudo-bbox 实验中混入原始 GT bbox。
- 输出新增字段：`bbox_source`、`pseudo_bbox_score`、`pseudo_bbox_area`。
- 新增 `scripts/check_pseudo_bbox.py`，不依赖重模型，验证 heatmap -> bbox 和 CSV 输出逻辑。
- README 增加 pseudo-bbox manifest 生成和复用 DINOv2 bbox extractor 的命令。

### 验证命令和结果

轻量级自检命令：

```bash
/home/jack/miniconda3/bin/conda run -n work-1 python scripts/check_pseudo_bbox.py
```

结果：通过，输出 `pseudo-bbox-check-ok`。

语法检查：

```bash
/home/jack/miniconda3/bin/conda run -n work-1 python -m py_compile scripts/build_pseudo_bbox_manifest.py scripts/check_pseudo_bbox.py
```

结果：通过，无语法错误。

### 后续待办

- 实现或接入第一个 heatmap 生成器，例如 PatchCore 或 DINOv2 nearest-normal memory。
- 生成 `outputs/heatmaps/*.jsonl`。
- 构建 `data/manifests/mvtec_fs_pseudo_bbox.csv`。
- 提取 DINOv2 pseudo-bbox 特征并跑 grid，与 GT bbox/ROI 对比。

## 2026-05-04 02:30 +08:00

### 修改目的

补齐 pseudo-bbox 流程中缺失的 heatmap JSONL 生成步骤。之前 README 中的 `example_heatmaps.jsonl` 只是占位示例，用户本地不会自动存在；本次新增一个可直接运行的 DINOv2 patch heatmap 生成脚本，让后续 `build_pseudo_bbox_manifest.py` 有真实输入。

### 涉及文件

- `scripts/extract_dinov2_patch_heatmaps.py`
- `scripts/check_dinov2_patch_heatmaps.py`
- `README.md`
- `docs/change_log.md`

### 主要改动

- 新增 `scripts/extract_dinov2_patch_heatmaps.py`，输出 `outputs/heatmaps/*.jsonl`。
- 支持 `patch-contrast` 模式：不依赖 normal/good 图像，直接用单张图内部 patch 与平均 patch 的差异生成启动版 anomaly heatmap。
- 支持 `nearest-memory` 模式：当 manifest 中包含 good/normal 图像时，用 normal 图像 DINOv2 patch token 建 memory bank，再计算每个 patch 到 memory 的最近余弦距离。
- 输出 JSONL 包含 `image_path`、`image_width`、`image_height`、`heatmap_width`、`heatmap_height`、`heatmap`、`localizer` 等字段，可直接输入 `build_pseudo_bbox_manifest.py`。
- 新增 `scripts/check_dinov2_patch_heatmaps.py`，轻量验证 heatmap 归一化、patch 数量到二维网格转换、memory 样本筛选逻辑。
- 更新 README，把 `example_heatmaps.jsonl` 改为真实生成命令 `outputs/heatmaps/dinov2_patch_contrast_train.jsonl`。

### 验证命令和结果

```bash
/home/jack/miniconda3/bin/conda run -n work-1 python scripts/check_dinov2_patch_heatmaps.py
```

结果：通过，输出 `dinov2-patch-heatmap-check-ok`。

```bash
/home/jack/miniconda3/bin/conda run -n work-1 python -m py_compile scripts/extract_dinov2_patch_heatmaps.py scripts/check_dinov2_patch_heatmaps.py scripts/build_pseudo_bbox_manifest.py
```

结果：通过，无语法错误。

### 后续待办

- 在服务器上先运行 `patch-contrast` 模式生成 heatmap JSONL。
- 用生成的 heatmap JSONL 构建 `data/manifests/mvtec_fs_pseudo_bbox.csv`。
- 提取 `dinov2_pseudo_bbox` 特征并运行 few-shot grid。
- 如果后续能纳入 normal/good 图像，再尝试 `nearest-memory` 模式并和 `patch-contrast` 对比。

## 2026-05-05 00:11 +08:00

### 修改目的

修复服务器运行 `build_pseudo_bbox_manifest.py` 时出现的 missing heatmap 报错。根因是 heatmap 文件用 `--split train` 生成，只覆盖 train 行；而 pseudo-bbox builder 之前默认检查整个 manifest，包含 testing/validation 行，导致未生成 heatmap 的行被当成错误。

### 涉及文件

- `scripts/build_pseudo_bbox_manifest.py`
- `scripts/check_pseudo_bbox.py`
- `README.md`
- `docs/change_log.md`

### 主要改动

- `build_pseudo_bbox_manifest.py` 新增 `--split` 参数，默认 `all`，可设置为 `train` 只输出 train manifest。
- 缺失 heatmap 的错误提示现在会显示当前 split，并提示如果 heatmap 用 `--split train` 生成，pseudo-bbox builder 也应使用 `--split train`。
- 输出日志从 `rows` 改为 `source_rows`、`written_rows`、`split`，方便判断过滤前后数量是否符合预期。
- `check_pseudo_bbox.py` 新增 split 过滤测试和 CLI 集成测试，验证只有 train heatmap 时可以成功构建 train pseudo-bbox manifest。
- README 中的 pseudo-bbox 命令改为输出 `data/manifests/mvtec_fs_pseudo_bbox_train.csv`，并显式加入 `--split train`。

### 验证命令和结果

```bash
/home/jack/miniconda3/bin/conda run -n work-1 python scripts/check_pseudo_bbox.py
```

结果：通过，输出 `pseudo-bbox-check-ok`。

```bash
/home/jack/miniconda3/bin/conda run -n work-1 python -m py_compile scripts/build_pseudo_bbox_manifest.py scripts/check_pseudo_bbox.py
```

结果：通过，无语法错误。

### 后续待办

- 服务器 `run.sh` 中生成 train heatmap 后，构建 pseudo-bbox manifest 时同步加上 `--split train`。
- 后续如果要生成 test/val 的 pseudo-bbox manifest，需要先对应生成 test/val heatmap JSONL，或者使用 `--split all` 且 heatmap 文件覆盖全部 manifest 行。

## 2026-05-05 00:40 +08:00

### 修改目的

记录第一版 DINOv2 patch-contrast pseudo-bbox ROI few-shot 结果，并新增 pseudo bbox 与 GT bbox 的 IoU 诊断工具。当前 pseudo-bbox ROI 结果明显低于 whole-image、GT bbox/ROI 和 region-global fusion，说明瓶颈主要在自动定位质量，而不是 prototype 分类器。

### 涉及文件

- `experiments/dinov2_baselines.md`
- `scripts/evaluate_pseudo_bbox_iou.py`
- `scripts/check_pseudo_bbox_iou.py`
- `docs/change_log.md`

### 主要改动

- 在 `experiments/dinov2_baselines.md` 中加入 pseudo-bbox ROI 结果表。
- 在 Accuracy 和 Macro-F1 总表中加入 `Pseudo-BBox ROI` 列。
- 新增 `scripts/evaluate_pseudo_bbox_iou.py`，对齐 GT manifest 与 pseudo manifest，计算 per-image IoU、Pseudo/GT 面积比、Recall@IoU 和最差样本。
- 新增 `scripts/check_pseudo_bbox_iou.py`，轻量验证 bbox 解析、IoU 计算、JSON/Markdown/CSV 输出。
- README 增加 IoU 诊断命令和 pseudo-bbox fusion 命令。
- 更新实验结论：当前 `patch-contrast` 伪框定位质量不足，下一步应先诊断 IoU 并尝试 pseudo-bbox + whole-image fusion。

### 记录的实验结果

- Manifest: `data/manifests/mvtec_fs_pseudo_bbox_train.csv`
- Feature file: `outputs/features/dinov2_pseudo_bbox/mvtec_fs_train.jsonl`
- Localizer: `dinov2_patch_contrast`

| Setting | Accuracy | Macro-F1 |
|---|---:|---:|
| 5-way 1-shot | 63.80 +/- 14.29 | 60.07 +/- 15.94 |
| 5-way 3-shot | 69.20 +/- 12.79 | 66.39 +/- 14.21 |
| 5-way 5-shot | 67.92 +/- 13.78 | 64.57 +/- 15.56 |
| 10-way 1-shot | 49.97 +/- 9.48 | 45.44 +/- 10.03 |
| 10-way 5-shot | 51.40 +/- 8.14 | 46.22 +/- 8.99 |

### 验证命令和结果

```bash
/home/jack/miniconda3/bin/conda run -n work-1 python scripts/check_pseudo_bbox_iou.py
```

结果：通过，输出 `pseudo-bbox-iou-check-ok`。

### 后续待办

- 在服务器上运行 `scripts/evaluate_pseudo_bbox_iou.py`，量化 pseudo bbox 与 GT bbox 的 IoU。
- 用 `fuse_feature_cache.py` 将 whole-image 特征和 pseudo-bbox ROI 特征做 concat fusion，检查上下文是否能恢复性能。
- 根据 IoU 结果调 `--percentile`、`--min-area-ratio`，或替换为 nearest-memory / PatchCore localizer。

## 2026-05-05 10:30 +08:00

### 修改目的

新增 pseudo-bbox 参数 sweep 工具，用同一份 heatmap JSONL 快速比较不同 `percentile`、`min-area-ratio` 和 `component` 设置下的 GT bbox IoU，避免每次调参都先跑昂贵的 DINOv2 ROI 特征提取。

### 涉及文件

- `scripts/sweep_pseudo_bbox_iou.py`
- `scripts/check_pseudo_bbox_iou_sweep.py`
- `README.md`
- `docs/change_log.md`

### 主要改动

- 新增 `scripts/sweep_pseudo_bbox_iou.py`，直接复用 `build_pseudo_bbox_manifest.py` 的 heatmap-to-bbox 逻辑和 `evaluate_pseudo_bbox_iou.py` 的 IoU 统计逻辑。
- 支持一次性 sweep 多个 `--percentiles`、`--min-area-ratios` 和 `--components` 组合。
- 输出 JSON / Markdown / CSV 排名表，排名依据为 mean IoU，其次是 Recall@IoU 0.50、Recall@IoU 0.25 和 median IoU。
- 可选 `--write-manifests-dir`，为每个参数组合写出对应 pseudo-bbox manifest，方便直接挑选最佳设置继续提特征。
- 新增 `scripts/check_pseudo_bbox_iou_sweep.py`，用合成 manifest 和 heatmap 验证 CLI 输出、排名和 manifest 写出逻辑。
- README 增加服务器端 pseudo-bbox IoU sweep 命令。

### 验证命令和结果

```bash
/home/jack/miniconda3/bin/conda run -n work-1 python scripts/check_pseudo_bbox_iou_sweep.py
```

结果：通过，输出 `pseudo-bbox-iou-sweep-check-ok`。

```bash
/home/jack/miniconda3/bin/conda run -n work-1 python -m py_compile scripts/sweep_pseudo_bbox_iou.py scripts/check_pseudo_bbox_iou_sweep.py
```

结果：通过，无语法错误。

### 后续待办

- 在服务器上用真实 `outputs/heatmaps/dinov2_patch_contrast_train.jsonl` 跑 sweep，查看最佳 mean IoU、median IoU 和 Recall@IoU 0.25/0.50。
- 根据 sweep 最佳参数重建 `data/manifests/mvtec_fs_pseudo_bbox_train.csv`，再提取 pseudo-bbox ROI 特征。
- 跑 pseudo-bbox + whole-image concat fusion，确认上下文能否弥补 pseudo ROI 的定位误差。

## 2026-05-05 11:00 +08:00

### 修改目的

记录服务器端 pseudo-bbox IoU sweep 结果，并据此确认当前 `dinov2_patch_contrast` pseudo localization 是 pseudo-bbox ROI baseline 性能下降的主要瓶颈。

### 涉及文件

- `experiments/dinov2_baselines.md`
- `docs/change_log.md`

### 主要记录

- sweep 使用 GT manifest `data/manifests/mvtec_fs.csv` 和 heatmap `outputs/heatmaps/dinov2_patch_contrast_train.jsonl`。
- 评估 split 为 `train`，有效图像数为 `886`。
- 最优参数为 `percentile=0.90`、`min_area_ratio=0.0005`、`component=largest`。
- 最优 mean IoU 为 `0.1863`，median IoU 为 `0.0765`，Recall@IoU 0.50 为 `0.1388`。
- 最优设置的 mean Pseudo/GT area ratio 为 `11.3234`，说明 pseudo box 往往明显大于 GT bbox。

### 结论

- 当前 `patch-contrast` heatmap 生成的 pseudo-bbox 定位质量较弱，足以解释 pseudo-bbox ROI few-shot 结果明显低于 whole-image、GT bbox/ROI 和 GT fusion。
- `largest` 组件策略明显比 `max-score` 更稳定；`max-score` 在 0.85/0.90 percentile 下容易选到偏离 GT 的高分小岛。
- 下一步应先跑 pseudo-bbox + whole-image concat fusion，检查全局上下文是否能补偿 pseudo ROI 定位误差；同时考虑 nearest-memory / PatchCore / AnomalyDINO 等更强 localizer。

### 后续待办

- 用最佳 sweep manifest 或等价参数重建 `data/manifests/mvtec_fs_pseudo_bbox_train.csv`。
- 提取最佳参数下的 pseudo-bbox ROI 特征，必要时复跑 pseudo-bbox ROI grid。
- 将 whole-image 特征与 pseudo-bbox ROI 特征 concat，运行标准 few-shot grid。

## 2026-05-05 11:30 +08:00

### 修改目的

记录服务器端 pseudo-bbox + whole-image concat fusion few-shot 结果，验证全局上下文是否能补偿当前 pseudo-bbox ROI 定位误差。

### 涉及文件

- `experiments/dinov2_baselines.md`
- `docs/change_log.md`

### 记录的实验结果

- Manifest: `data/manifests/mvtec_fs_pseudo_bbox_train.csv`
- Feature file: `outputs/features/dinov2_pseudo_fusion_concat/mvtec_fs_train.jsonl`
- Fusion: whole-image DINOv2 + pseudo-bbox ROI DINOv2 concat

| Setting | Accuracy | Macro-F1 |
|---|---:|---:|
| 5-way 1-shot | 80.50 +/- 13.39 | 78.52 +/- 15.03 |
| 5-way 3-shot | 83.92 +/- 11.72 | 82.64 +/- 12.85 |
| 5-way 5-shot | 85.80 +/- 12.19 | 84.54 +/- 13.44 |
| 10-way 1-shot | 68.43 +/- 9.05 | 65.52 +/- 9.80 |
| 10-way 5-shot | 73.38 +/- 8.94 | 70.52 +/- 9.90 |

### 结论

- pseudo-bbox + whole-image concat 相比 pseudo-bbox ROI 有大幅提升，Accuracy 提升 `+14.72` 到 `+21.98` 个百分点，Macro-F1 提升 `+16.25` 到 `+24.30` 个百分点。
- 5-way 设置已经接近 whole-image baseline，但 10-way 设置仍明显低于 whole-image 和 GT bbox + whole fusion。
- 这说明全局上下文能有效补偿 pseudo ROI 的定位误差，但当前 automatic localization 质量仍是和 GT fusion 之间的主要差距来源。

### 后续待办

- 继续优先改 localization：尝试 nearest-memory、PatchCore 或 AnomalyDINO heatmap。
- 在已有 concat 结果基础上实现更正式的 region-context prototype，而不是只做特征拼接。
- 如时间允许，补充 pseudo fusion 与 whole-image / GT fusion 的 delta 表，便于论文分析。

