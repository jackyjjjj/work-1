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
