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

## 2026-05-05 12:00 +08:00

### 修改目的

新增 score-level region-context prototype baseline，作为比简单特征 concat 更正式的区域-上下文建模起点；同时补充 pseudo fusion 相对 pseudo ROI、whole-image 和 GT fusion 的 delta 分析表。

### 涉及文件

- `src/lg_fdc/pipelines/region_context.py`
- `scripts/run_region_context_grid.py`
- `scripts/check_region_context_prototype.py`
- `README.md`
- `experiments/dinov2_baselines.md`
- `docs/change_log.md`

### 主要改动

- 新增 `RegionContextPrototypeClassifier`，分别为 whole-image 特征和 region 特征建立 prototype，并在分类时按权重融合两个 cosine score。
- 新增 `run_region_context_episode()`，复用现有 episode sampler、cached feature extractor 和 metrics。
- 新增 `scripts/run_region_context_grid.py`，支持标准 few-shot grid 和 `--whole-weights` 权重扫描；region 权重自动设为 `1 - whole_weight`。
- 新增 `scripts/check_region_context_prototype.py`，用合成 cached features 验证 classifier、episode pipeline 和 CLI grid 输出。
- README 增加 region-context prototype 服务器运行命令。
- `experiments/dinov2_baselines.md` 增加 pseudo fusion delta 表，便于分析上下文补偿效果和 GT fusion 差距。

### 验证命令和结果

```bash
/home/jack/miniconda3/bin/conda run -n work-1 python scripts/check_region_context_prototype.py
```

结果：通过，输出 `region-context-prototype-check-ok`。

```bash
/home/jack/miniconda3/bin/conda run -n work-1 python -m py_compile src/lg_fdc/pipelines/region_context.py scripts/run_region_context_grid.py scripts/check_region_context_prototype.py
```

结果：通过，无语法错误。

### 后续待办

- 在服务器上运行 `scripts/run_region_context_grid.py`，比较 score-level region-context 与 pseudo concat fusion。
- 如果 region-context 优于 concat，可进一步做 per-class confusion 和权重选择分析。
- 如果 region-context 仍低于 concat，优先继续改 localization 或引入可学习/校准过的 score fusion。

## 2026-05-05 12:30 +08:00

### 修改目的

记录服务器端 score-level region-context prototype 结果，并比较其与 pseudo-bbox concat fusion、whole-image baseline 和 GT fusion 的差异。

### 涉及文件

- `experiments/dinov2_baselines.md`
- `docs/change_log.md`

### 记录的实验结果

- Manifest: `data/manifests/mvtec_fs_pseudo_bbox_train.csv`
- Whole feature file: `outputs/features/dinov2/mvtec_fs_train.jsonl`
- Region feature file: `outputs/features/dinov2_pseudo_bbox/mvtec_fs_train.jsonl`
- 权重扫描：`whole_weight=0.25/0.50/0.75`，region 权重为 `1 - whole_weight`。

最佳 Accuracy 均出现在 `whole_weight=0.75`、`region_weight=0.25`：

| Setting | Accuracy | Macro-F1 |
|---|---:|---:|
| 5-way 1-shot | 80.32 +/- 12.87 | 78.31 +/- 14.30 |
| 5-way 3-shot | 86.30 +/- 11.25 | 85.12 +/- 12.36 |
| 5-way 5-shot | 85.72 +/- 10.85 | 84.29 +/- 12.21 |
| 10-way 1-shot | 70.68 +/- 9.88 | 67.90 +/- 10.72 |
| 10-way 5-shot | 75.45 +/- 9.13 | 72.83 +/- 10.17 |

### 结论

- score-level region-context 在 5-way 3-shot、10-way 1-shot、10-way 5-shot 上超过 pseudo concat fusion，分别提升 Accuracy `+2.38`、`+2.25`、`+2.07` 个百分点。
- 5-way 1-shot 和 5-way 5-shot 略低于 pseudo concat fusion，说明当前 score-level 融合还不是稳定支配 concat 的替代方案。
- 最佳权重始终偏向 whole-image，说明 pseudo region score 在定位噪声较大时应作为辅助信号，而不应占主导。
- 5-way 3-shot 的 region-context Accuracy `86.30` 已超过 whole-image `85.38`，接近 GT concat fusion `87.14`，是当前自动定位路线中最有说服力的设置之一。

### 后续待办

- 继续优化 localizer，降低 pseudo ROI 噪声后再复测 region-context 权重。
- 增加 score calibration 或 per-episode 自适应权重，避免固定权重在 5-way 1-shot/5-shot 上略低于 concat。
- 补充 confusion/per-class 分析，判断 region-context 在 10-way 中改善了哪些易混类别。

## 2026-05-05 13:00 +08:00

### 修改目的

新增 region-context 的 per-class / confusion 分析工具，用于诊断 10-way 设置中 score-level region-context 相比 pseudo concat fusion 改善了哪些类别、恶化了哪些类别。

### 涉及文件

- `scripts/analyze_region_context_confusion.py`
- `scripts/check_region_context_confusion.py`
- `README.md`
- `docs/change_log.md`

### 主要改动

- 新增 `scripts/analyze_region_context_confusion.py`，在同一批 sampled episodes 上统计 region-context 的整体指标、per-class precision/recall/F1 和 top confusion pairs。
- 支持可选 `--baseline-feature-file`，用同一批 episodes 同时评估一个 cached-feature prototype baseline，例如 `dinov2_pseudo_fusion_concat`，并输出 per-class delta。
- 支持输出 JSON、Markdown、per-class CSV 和 confusion CSV，便于后续论文分析和手工检查难分类别。
- 新增 `scripts/check_region_context_confusion.py`，用合成 cached features 验证 region-context、baseline 对比和文件输出逻辑。
- README 增加 10-way region-context vs pseudo concat confusion 诊断命令。

### 验证命令和结果

```bash
/home/jack/miniconda3/bin/conda run -n work-1 python scripts/check_region_context_confusion.py
```

结果：通过，输出 `region-context-confusion-check-ok`。

```bash
/home/jack/miniconda3/bin/conda run -n work-1 python -m py_compile scripts/analyze_region_context_confusion.py scripts/check_region_context_confusion.py
```

结果：通过，无语法错误。

### 后续待办

- 在服务器上运行 10-way 5-shot confusion 分析，查看 region-context 提升主要来自哪些 defect labels。
- 对 10-way 1-shot 也运行同一脚本，比较低 shot 和高 shot 的混淆模式是否一致。
- 根据 per-class delta 决定下一步是做 score calibration、adaptive weight，还是优先修复特定类别的 pseudo localization。

## 2026-05-05 15:40 +08:00

### 修改目的

记录服务器端 10-way 5-shot region-context confusion 分析结果，明确 score-level region-context 相比 pseudo concat fusion 的类别级收益、退化类别和主要混淆对。

### 涉及文件

- `experiments/dinov2_baselines.md`
- `docs/change_log.md`
- `handoff.md`

### 记录的实验结果

- Manifest: `data/manifests/mvtec_fs_pseudo_bbox_train.csv`
- Setting: `10-way 5-shot`
- Episodes: `200`
- Query predictions: `10000`
- Whole/region weights: `0.75/0.25`
- Baseline: `outputs/features/dinov2_pseudo_fusion_concat/mvtec_fs_train.jsonl`

| Model | Accuracy | Balanced Acc | Macro-F1 |
|---|---:|---:|---:|
| region_context | 0.7480 | 0.7523 | 0.7369 |
| pseudo_concat | 0.7272 | 0.7315 | 0.7165 |
| Delta | +0.0208 | +0.0208 | +0.0204 |

### 结论

- 该 query-pooled 结果验证了 grid 结果：region-context 在 10-way 5-shot 上稳定优于 pseudo concat fusion，Accuracy 和 Macro-F1 约提升 2 个百分点。
- 最大 recall gain 来自 `fabric_border`、`squeeze`、`squeezed_teeth`、`split_teeth`、`faulty_imprint`、`hole` 等局部结构/局部形状相关类别，说明 pseudo region score 虽 noisy，但仍有可用定位信息。
- 主要 recall regression 集中在 `scratch_neck`、`bent`、`manipulated_front`、`color`、`crack`、`fabric_interior` 等类别；其中部分类别 F1 仍提升，说明固定权重会改变 precision/recall trade-off。
- 主要混淆仍发生在同产品/相似缺陷族内，例如 tooth labels、thread top/side、scratch head/neck/manipulated-front、rough/fabric/broken-teeth。
- 下一步建议先跑 10-way 1-shot confusion，再实现更细的固定权重扫描或 confidence-based adaptive weighting。

### 后续待办

- 在服务器上运行 10-way 1-shot confusion 分析，比较低 shot 和高 shot 的类别收益/退化是否一致。
- 增加 `whole_weight=0.60,0.70,0.80,0.85,0.90` 等更细扫描，判断最佳固定权重是否落在 `0.75` 附近。
- 设计 score calibration / adaptive weight，降低 `scratch_neck`、`color`、`manipulated_front` 等类别的固定权重副作用。

## 2026-05-05 22:17 +08:00

### 修改目的

记录 10-way region-context 细粒度固定权重扫描结果，判断 `whole_weight=0.75` 是否仍是最佳，以及当前 pseudo region score 应该占多大权重。

### 涉及文件

- `experiments/dinov2_baselines.md`
- `docs/change_log.md`
- `handoff.md`

### 记录的实验结果

- Manifest: `data/manifests/mvtec_fs_pseudo_bbox_train.csv`
- Split: `train`
- Whole feature file: `outputs/features/dinov2/mvtec_fs_train.jsonl`
- Region feature file: `outputs/features/dinov2_pseudo_bbox/mvtec_fs_train.jsonl`
- Sweep: `whole_weight=0.60,0.65,0.70,0.75,0.80,0.85,0.90`
- Settings: `10-way 1-shot`, `10-way 5-shot`

| Setting | Best Whole/Region W | Accuracy | Macro-F1 | Delta Acc vs 0.75 in sweep | Delta Acc vs Pseudo Concat | Delta Acc vs Whole |
|---|---:|---:|---:|---:|---:|---:|
| 10-way 1-shot | 0.90 / 0.10 | 73.23 +/- 10.10 | 70.74 +/- 10.96 | +2.33 | +4.80 | +0.62 |
| 10-way 5-shot | 0.80 / 0.20 | 76.57 +/- 9.28 | 74.15 +/- 10.56 | +1.02 | +3.19 | -0.59 |

### 结论

- 细权重扫描推翻了“0.75 始终最佳”的粗粒度结论：10-way 1-shot 最佳为 `0.90/0.10`，10-way 5-shot 最佳为 `0.80/0.20`。
- 当前 pseudo-bbox region score 仍然有用，但由于定位噪声较大，更适合作为小权重 correction，而不是和 whole-image score 接近等权融合。
- 10-way 1-shot 在 `0.90/0.10` 下已经超过 whole-image baseline，说明极低 shot 时少量 region evidence 可以补充全局原型。
- 10-way 5-shot 在 `0.80/0.20` 下显著超过 pseudo concat fusion，但仍略低于 whole-image accuracy，说明还需要更好的 localization 或 adaptive calibration。

### 后续待办

- 用新最佳权重重跑 per-class confusion：10-way 1-shot 使用 `whole_weight=0.90`，10-way 5-shot 使用 `whole_weight=0.80`。
- 对比新旧 confusion，确认 `scratch_neck`、`color`、`manipulated_front` 等退化类别是否随 region 权重降低而改善。
- 如果新最佳权重仍存在类别级退化，再实现 confidence-based adaptive weighting 或 score normalization。
