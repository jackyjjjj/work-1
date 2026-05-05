from __future__ import annotations

from dataclasses import dataclass

from lg_fdc.data.episodes import Episode
from lg_fdc.evaluation.metrics import accuracy, balanced_accuracy, macro_f1
from lg_fdc.features.base import FeatureExtractor
from lg_fdc.models.prototype import PrototypeClassifier
from lg_fdc.pipelines.prototype_baseline import EpisodeResult


@dataclass(frozen=True)
class RegionContextPrediction:
    label: str
    score: float
    scores: dict[str, float]
    whole_scores: dict[str, float]
    region_scores: dict[str, float]


class RegionContextPrototypeClassifier:
    """Combine whole-image and region prototypes at score level."""

    def __init__(self, whole_weight: float = 0.5, region_weight: float = 0.5) -> None:
        if whole_weight < 0.0 or region_weight < 0.0:
            raise ValueError("whole_weight and region_weight must be non-negative")
        weight_sum = whole_weight + region_weight
        if weight_sum <= 0.0:
            raise ValueError("At least one score weight must be positive")
        self.whole_weight = whole_weight / weight_sum
        self.region_weight = region_weight / weight_sum
        self.whole_classifier = PrototypeClassifier()
        self.region_classifier = PrototypeClassifier()

    def fit(
        self,
        whole_features: list[list[float]],
        region_features: list[list[float]],
        labels: list[str],
    ) -> "RegionContextPrototypeClassifier":
        if len(whole_features) != len(region_features) or len(whole_features) != len(labels):
            raise ValueError("whole_features, region_features, and labels must have the same length")
        self.whole_classifier.fit(whole_features, labels)
        self.region_classifier.fit(region_features, labels)
        return self

    def predict_one(self, whole_feature: list[float], region_feature: list[float]) -> RegionContextPrediction:
        whole_prediction = self.whole_classifier.predict_one(whole_feature)
        region_prediction = self.region_classifier.predict_one(region_feature)
        labels = sorted(set(whole_prediction.scores) | set(region_prediction.scores))
        scores = {
            label: self.whole_weight * whole_prediction.scores.get(label, 0.0)
            + self.region_weight * region_prediction.scores.get(label, 0.0)
            for label in labels
        }
        label = max(scores, key=scores.get)
        return RegionContextPrediction(
            label=label,
            score=scores[label],
            scores=scores,
            whole_scores=whole_prediction.scores,
            region_scores=region_prediction.scores,
        )

    def predict(
        self,
        whole_features: list[list[float]],
        region_features: list[list[float]],
    ) -> list[RegionContextPrediction]:
        if len(whole_features) != len(region_features):
            raise ValueError("whole_features and region_features must have the same length")
        return [
            self.predict_one(whole_feature, region_feature)
            for whole_feature, region_feature in zip(whole_features, region_features, strict=True)
        ]


def run_region_context_episode(
    episode: Episode,
    whole_extractor: FeatureExtractor,
    region_extractor: FeatureExtractor,
    whole_weight: float = 0.5,
    region_weight: float = 0.5,
) -> EpisodeResult:
    whole_support_features = [whole_extractor.extract(record) for record in episode.support]
    region_support_features = [region_extractor.extract(record) for record in episode.support]
    support_labels = [record.label for record in episode.support]

    whole_query_features = [whole_extractor.extract(record) for record in episode.query]
    region_query_features = [region_extractor.extract(record) for record in episode.query]
    y_true = [record.label for record in episode.query]

    classifier = RegionContextPrototypeClassifier(
        whole_weight=whole_weight,
        region_weight=region_weight,
    ).fit(whole_support_features, region_support_features, support_labels)
    predictions = classifier.predict(whole_query_features, region_query_features)
    y_pred = [prediction.label for prediction in predictions]

    return EpisodeResult(
        accuracy=accuracy(y_true, y_pred),
        balanced_accuracy=balanced_accuracy(y_true, y_pred),
        macro_f1=macro_f1(y_true, y_pred),
        y_true=tuple(y_true),
        y_pred=tuple(y_pred),
    )
