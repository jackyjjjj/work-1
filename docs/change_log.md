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
