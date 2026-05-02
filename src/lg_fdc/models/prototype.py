from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class PrototypePrediction:
    """单个 query 样本的原型分类结果。"""

    label: str
    score: float
    scores: dict[str, float]


class PrototypeClassifier:
    """基于余弦相似度的原型分类器。

    在 few-shot 任务里，每个类别只有少量 support 样本。这里先把同类 support 特征
    求均值得到类别 prototype，然后用 query 特征和各类 prototype 的余弦相似度分类。
    """

    def __init__(self) -> None:
        self.prototypes: dict[str, list[float]] = {}

    def fit(self, features: list[list[float]], labels: list[str]) -> "PrototypeClassifier":
        """根据 support 特征和标签构建每个类别的 prototype。"""

        if len(features) != len(labels):
            raise ValueError("features and labels must have the same length")
        grouped: dict[str, list[list[float]]] = defaultdict(list)
        for feature, label in zip(features, labels, strict=True):
            grouped[label].append(feature)
        self.prototypes = {label: _normalize(_mean(vectors)) for label, vectors in grouped.items()}
        return self

    def predict_one(self, feature: list[float]) -> PrototypePrediction:
        """预测单个 query 特征的缺陷类别。"""

        if not self.prototypes:
            raise RuntimeError("PrototypeClassifier must be fitted before prediction")
        feature = _normalize(feature)
        scores = {label: _dot(feature, prototype) for label, prototype in self.prototypes.items()}
        label = max(scores, key=scores.get)
        return PrototypePrediction(label=label, score=scores[label], scores=scores)

    def predict(self, features: list[list[float]]) -> list[PrototypePrediction]:
        """批量预测 query 特征。"""

        return [self.predict_one(feature) for feature in features]


def _mean(vectors: list[list[float]]) -> list[float]:
    """计算多个向量的逐维均值。"""

    if not vectors:
        raise ValueError("Cannot average an empty vector list")
    dim = len(vectors[0])
    if any(len(vector) != dim for vector in vectors):
        raise ValueError("All vectors must have the same dimension")
    return [sum(vector[idx] for vector in vectors) / len(vectors) for idx in range(dim)]


def _normalize(vector: list[float], eps: float = 1e-12) -> list[float]:
    """L2 归一化，方便用点积表示余弦相似度。"""

    norm = math.sqrt(sum(value * value for value in vector))
    if norm < eps:
        return [0.0 for _ in vector]
    return [value / norm for value in vector]


def _dot(left: list[float], right: list[float]) -> float:
    """向量点积。"""

    if len(left) != len(right):
        raise ValueError("Vectors must have the same dimension")
    return sum(a * b for a, b in zip(left, right, strict=True))