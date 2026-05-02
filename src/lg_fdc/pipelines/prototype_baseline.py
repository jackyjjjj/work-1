from __future__ import annotations

from dataclasses import dataclass

from lg_fdc.data.episodes import Episode
from lg_fdc.evaluation.metrics import accuracy, balanced_accuracy, macro_f1
from lg_fdc.features.base import FeatureExtractor
from lg_fdc.models.prototype import PrototypeClassifier


@dataclass(frozen=True)
class EpisodeResult:
    """一次 episode 的评估结果。"""

    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    y_true: tuple[str, ...]
    y_pred: tuple[str, ...]


def run_prototype_episode(episode: Episode, extractor: FeatureExtractor) -> EpisodeResult:
    """运行一次 prototype few-shot episode。

    流程是：support 提特征 -> 建类别原型 -> query 提特征 -> 预测 -> 计算指标。
    后续 DINOv2 baseline 和 Auto-Mask 方法都可以复用这条基础评估链路。
    """

    support_features = [extractor.extract(record) for record in episode.support]
    support_labels = [record.label for record in episode.support]
    query_features = [extractor.extract(record) for record in episode.query]
    y_true = [record.label for record in episode.query]

    classifier = PrototypeClassifier().fit(support_features, support_labels)
    predictions = classifier.predict(query_features)
    y_pred = [prediction.label for prediction in predictions]

    return EpisodeResult(
        accuracy=accuracy(y_true, y_pred),
        balanced_accuracy=balanced_accuracy(y_true, y_pred),
        macro_f1=macro_f1(y_true, y_pred),
        y_true=tuple(y_true),
        y_pred=tuple(y_pred),
    )