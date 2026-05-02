from __future__ import annotations

from lg_fdc.data.episodes import EpisodeConfig, sample_episode
from lg_fdc.data.manifest import ImageRecord
from lg_fdc.features.simple import MetadataFeatureExtractor
from lg_fdc.pipelines.prototype_baseline import run_prototype_episode


def test_prototype_episode_smoke() -> None:
    """验证 prototype episode 流程可以在简单合成特征上得到满分。"""

    records = []
    for label, feature in {"a": [1, 0], "b": [0, 1], "c": [-1, 0]}.items():
        for idx in range(5):
            records.append(
                ImageRecord(
                    image_path=f"{label}_{idx}.png",
                    label=label,
                    metadata={"feature": feature},
                )
            )
    episode = sample_episode(records, EpisodeConfig(n_way=3, k_shot=2, q_queries=2, seed=1))
    result = run_prototype_episode(episode, MetadataFeatureExtractor(feature_dim=2))
    assert result.accuracy == 1.0
    assert result.macro_f1 == 1.0