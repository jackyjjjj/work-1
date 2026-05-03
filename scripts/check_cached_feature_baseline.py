from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

# 允许直接从源码树运行脚本，不要求先 pip install -e。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lg_fdc.data.episodes import EpisodeConfig, sample_episode
from lg_fdc.data.manifest import ImageRecord
from lg_fdc.features.cached import CachedFeatureExtractor
from lg_fdc.pipelines.prototype_baseline import run_prototype_episode


def main() -> None:
    """无重依赖版本的 cached-feature baseline 自检。"""

    with tempfile.TemporaryDirectory() as tmp_dir:
        feature_path = Path(tmp_dir) / "features.jsonl"
        records = _make_records()
        with feature_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        {
                            "image_path": record.image_path,
                            "label": record.label,
                            "feature": record.metadata["feature"],
                        }
                    )
                    + "\n"
                )

        episode = sample_episode(records, EpisodeConfig(n_way=3, k_shot=2, q_queries=2, seed=3))
        extractor = CachedFeatureExtractor(feature_path=feature_path, feature_dim=3)
        result = run_prototype_episode(episode, extractor)
        assert result.accuracy == 1.0
        assert result.macro_f1 == 1.0

    print("cached-feature-baseline-ok")


def _make_records() -> list[ImageRecord]:
    centers = {
        "scratch": [1.0, 0.0, 0.0],
        "crack": [0.0, 1.0, 0.0],
        "stain": [0.0, 0.0, 1.0],
    }
    records = []
    for label, feature in centers.items():
        for idx in range(5):
            records.append(
                ImageRecord(
                    image_path=f"synthetic/{label}_{idx}.png",
                    label=label,
                    metadata={"feature": feature},
                )
            )
    return records


if __name__ == "__main__":
    main()