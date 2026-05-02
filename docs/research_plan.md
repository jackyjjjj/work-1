# Localization-Guided Few-Shot Defect Classification Research Plan

## 1. 接上个会话的结论

当前最值得推进的主线不是传统工业异常检测的 normal/abnormal 二分类，而是：

> Localization-guided few-shot defect type classification

也就是先利用异常定位结果找到缺陷区域，再在少样本条件下识别缺陷类型。这个方向比单纯 anomaly detection 更容易形成完整论文，因为它同时包含定位、分类、少样本泛化和工业落地价值。

建议题目暂定为：

> Pseudo-Mask Guided Region-Context Prototype Learning for Few-Shot Industrial Defect Classification

中文题目：

> 基于伪缺陷掩码引导的区域-上下文原型学习小样本工业缺陷分类方法

目标档次：

- 稳妥目标：SCI 二区/三区；
- 冲刺目标：CCF B 相关视觉/模式识别/智能制造会议；
- 前提：必须和 MVREC、DINOv2/Alpha-CLIP prototype、PatchCore ROI classifier 等强 baseline 拉开差距。

## 2. 为什么这个方向有空间

工业异常检测领域已经有很多强方法解决“是否异常”和“异常在哪里”，但真实质检场景还需要回答：

- 这是什么缺陷类型？
- 新缺陷类型只有 1-5 张样本时能否识别？
- 定位区域不准时，分类是否还能稳？
- 不同产品、材质、数据集之间能否泛化？

MVREC 已经把 few-shot defect multi-classification 做成 AAAI 2025 方向，并提出 MVTec-FS 数据集。这说明这个任务本身是成立的，但仍有可改进空间：MVREC 依赖 mask region input 和 multi-view context，而我们的切入点可以放在“无需人工 mask 的伪掩码引导”和“异常定位模型与少样本分类的统一”。

## 3. 推荐方法框架

最小可行版本：

```text
Input image
  ↓
Anomaly localization model 生成 anomaly heatmap
  ↓
Adaptive pseudo-mask generation
  ↓
Region / context / global feature extraction
  ↓
Multi-prototype few-shot classifier
  ↓
Defect type prediction + optional unknown rejection
```

### 3.1 Anomaly heatmap 基座

优先级建议：

1. AnomalyDINO：DINOv2 patch feature + nearest neighbor，适合 few-shot/one-shot anomaly localization，工程上相对简单。
2. PatchCore：经典强 baseline，适合先快速生成可靠 heatmap。
3. EfficientAD：速度快，适合做部署效率对比。
4. AnomalyCLIP / WinCLIP：适合作为视觉语言 baseline，不建议一开始作为主方法核心。

### 3.2 Pseudo-mask generation

不要直接用测试集真值 mask，否则论文意义会被质疑。建议从 heatmap 自动生成 pseudo mask：

- top-k percentile threshold；
- Otsu threshold；
- connected component filtering；
- dilation/erosion 平滑边界；
- multi-threshold proposal，保留多个候选区域；
- mask confidence score，用于后续特征融合加权。

可以把这一块包装成：

> Confidence-aware adaptive pseudo-mask generation

### 3.3 Region-context feature

对每张图提取三类特征：

```text
region feature: 缺陷区域 masked pooling
context feature: 缺陷周围 ring/context crop
global feature: 整图产品语义
```

特征提取器建议：

- DINOv2：强 patch-level 表征，适合纹理、局部结构和工业表面；
- Alpha-CLIP：适合 mask-guided region-aware representation；
- CLIP：适合作为文本语义融合和 baseline；
- ResNet/ViT：普通 few-shot baseline。

最终表示可以先做简单拼接：

```text
f = concat(f_region, f_context, f_global)
```

再加入可学习或非参数权重：

```text
f = w_r * f_region + w_c * f_context + w_g * f_global
```

其中权重由 pseudo-mask confidence、区域面积、heatmap peak score 决定。

### 3.4 Few-shot classifier

第一版不要做太复杂，建议从 prototype classifier 开始：

```text
p_c = mean({f_i | y_i = c})
score(q, c) = cosine(f_q, p_c)
```

然后逐步增强：

- multi-prototype：每类不只一个原型，处理同一缺陷类型的形态变化；
- normal prototype suppression：用 normal prototype 抑制背景/产品语义；
- query-adaptive refinement：根据 query 的 region/context 动态调整 support prototype；
- text-visual fusion：如果类别名有语义，例如 scratch/crack/stain，可用文本原型辅助。

## 4. 与 MVREC 的差异点

MVREC 是最重要的直接相关工作，必须认真对比。我们的差异不要写成“我也做 region-context”，而要强调：

| 维度 | MVREC | 我们的潜在改进 |
|---|---|---|
| 区域来源 | mask region input | anomaly heatmap 自动生成 pseudo mask |
| 分类模式 | region-context + AlphaCLIP + adapter | localization-guided region-context prototype |
| 少样本适配 | Few-shot Zip-Adapter | multi-prototype / query-adaptive prototype |
| 背景干扰 | context augmentation | normal prototype suppression |
| 开放缺陷 | 不是主重点 | 可加入 unknown defect rejection |
| 定位误差 | 依赖 mask 质量 | 显式做 mask confidence-aware fusion |

论文创新最好落在：

1. pseudo-mask guided FSDMC，不依赖人工缺陷 mask；
2. confidence-aware region-context representation；
3. defect-normal contrastive prototype learning；
4. query-adaptive multi-prototype classifier；
5. open-set unknown defect rejection，可作为增强实验。

## 5. 数据集选择

主数据集：

- MVTec-FS：最贴合 few-shot defect multi-classification，建议作为主实验。

定位辅助/泛化验证：

- MVTec AD：用于 anomaly heatmap 生成与定位评估；
- VisA：用于跨物体、多材质泛化；
- NEU-DET：钢材缺陷分类；
- GC10-DET：钢材表面缺陷；
- DeepPCB：PCB 缺陷分类；
- Magnetic Tile Defects / AITEX：可作为补充。

建议实验组合：

```text
主实验：MVTec-FS
泛化实验：NEU-DET + DeepPCB + GC10-DET
定位验证：MVTec AD / VisA
```

## 6. Baseline 列表

普通 few-shot classification：

- ProtoNet；
- RelationNet；
- MatchingNet；
- FEAT；
- DN4；
- Meta-Baseline。

Foundation model baseline：

- CLIP zero-shot；
- CLIP linear probe；
- Tip-Adapter；
- DINOv2 + kNN；
- DINOv2 + prototype；
- Alpha-CLIP + prototype。

工业异常检测 + 分类 baseline：

- PatchCore + ROI classifier；
- EfficientAD + ROI classifier；
- AnomalyDINO + prototype classifier；
- WinCLIP / WinCLIP+；
- AnomalyCLIP；
- MVREC。

必须做的递进对比：

```text
Whole image classification
  < ROI crop classification
  < region-context classification
  < confidence-aware pseudo-mask guided prototype
```

## 7. 实验设置

Few-shot 设置：

```text
5-way 1-shot
5-way 3-shot
5-way 5-shot
10-way 1-shot
10-way 5-shot
all-way 1-shot
all-way 5-shot
```

指标：

- Accuracy；
- Macro-F1；
- Balanced Accuracy；
- per-class F1；
- confusion matrix；
- 如果报告定位：pixel AUROC / AUPRO / IoU / Dice；
- 如果做 unknown rejection：unknown AUROC / OSCR / FPR95。

消融实验：

- without pseudo-mask；
- GT mask vs pseudo-mask；
- region only vs context only vs global only；
- no normal prototype suppression；
- single prototype vs multi-prototype；
- fixed threshold vs adaptive threshold；
- DINOv2 vs Alpha-CLIP vs CLIP；
- with / without text prototype。

## 8. 推荐推进顺序

第一阶段：复现和搭 baseline

1. 跑通 MVTec-FS；
2. 跑 MVREC 官方代码或至少复现其核心设定；
3. 跑 DINOv2 + prototype；
4. 跑 Alpha-CLIP + prototype；
5. 跑 PatchCore/AnomalyDINO heatmap + ROI classifier。

第二阶段：实现自己的最小方法

1. heatmap 自动生成 pseudo mask；
2. region/context/global feature extraction；
3. prototype classifier；
4. 先做 5-way 1-shot / 5-shot；
5. 和 DINOv2 + whole image prototype、ROI crop prototype 对比。

第三阶段：加创新模块

1. confidence-aware fusion；
2. normal prototype suppression；
3. multi-prototype；
4. query-adaptive prototype refinement；
5. unknown rejection。

第四阶段：论文包装

1. 明确任务定义：few-shot defect type classification with localization guidance；
2. 画方法总图；
3. 写 related work：industrial anomaly detection、few-shot classification、VLM/foundation model；
4. 强调和 MVREC 的差异；
5. 补充跨数据集实验与失败案例分析。

## 9. 当前最推荐的下一步

不要一开始就写复杂模型。最值得马上做的是这个最小闭环：

```text
MVTec-FS
  ↓
DINOv2 whole-image prototype baseline
  ↓
AnomalyDINO/PatchCore heatmap 生成 pseudo mask
  ↓
DINOv2 masked region-context prototype
  ↓
验证 region-context 是否显著优于 whole image / ROI crop
```

如果这个最小闭环能稳定提升 3-8 个点，并且 macro-F1 提升明显，这篇论文就有基础了。后续再加入 normal suppression 和 query-adaptive prototype，把贡献点做厚。

## 10. 参考入口

- MVREC code: https://github.com/ShuaiLYU/MVREC
- MVTec-FS dataset: https://github.com/ShuaiLYU/MVTec-FS
- MVTec AD 2: https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2
- AnomalyDINO: https://arxiv.org/abs/2405.14529
- WinCLIP: https://openaccess.thecvf.com/content/CVPR2023/html/Jeong_WinCLIP_Zero-Few-Shot_Anomaly_Classification_and_Segmentation_CVPR_2023_paper.html
- AnomalyCLIP: https://openreview.net/forum?id=buC4E91xZE
- Alpha-CLIP: https://arxiv.org/abs/2312.03818
