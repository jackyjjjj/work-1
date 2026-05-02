from __future__ import annotations

from dataclasses import dataclass

from lg_fdc.localization.base import Heatmap


@dataclass(frozen=True)
class PseudoMaskConfig:
    """伪 mask 生成配置。"""

    percentile: float = 0.9
    min_area: int = 1


@dataclass(frozen=True)
class PseudoMask:
    """从异常热力图生成的伪缺陷区域。"""

    values: tuple[tuple[bool, ...], ...]
    threshold: float
    area: int
    confidence: float


def generate_pseudo_mask(heatmap: Heatmap, config: PseudoMaskConfig) -> PseudoMask:
    """从异常热力图生成一个简单阈值伪 mask。

    这是 Auto-Mask 路线的最初版 baseline。后续真正的论文方法会把这里升级成：
    多阈值候选区域、连通域筛选、区域置信度评分、context ring 生成等。
    """

    flat = [value for row in heatmap.values for value in row]
    if not flat:
        raise ValueError("Cannot generate a pseudo mask from an empty heatmap")
    if not 0.0 < config.percentile <= 1.0:
        raise ValueError("percentile must be in (0, 1]")

    # 按分位数选阈值：例如 percentile=0.9 表示保留异常分数最高的一部分区域。
    sorted_values = sorted(flat)
    threshold_idx = min(len(sorted_values) - 1, max(0, int(config.percentile * len(sorted_values)) - 1))
    threshold = sorted_values[threshold_idx]
    values = tuple(tuple(value >= threshold for value in row) for row in heatmap.values)
    area = sum(1 for row in values for value in row if value)

    # 如果阈值后区域太小，就至少保留热力图峰值点，避免返回空 mask。
    if area < config.min_area:
        peak = max(flat)
        values = tuple(tuple(value == peak for value in row) for row in heatmap.values)
        area = sum(1 for row in values for value in row if value)
        threshold = peak

    selected_scores = [
        score
        for row_scores, row_mask in zip(heatmap.values, values, strict=True)
        for score, keep in zip(row_scores, row_mask, strict=True)
        if keep
    ]
    confidence = sum(selected_scores) / len(selected_scores) if selected_scores else 0.0
    return PseudoMask(values=values, threshold=threshold, area=area, confidence=confidence)