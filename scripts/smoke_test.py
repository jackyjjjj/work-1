from __future__ import annotations

from pathlib import Path
import sys

# 让脚本可以在未安装 package 的情况下直接从源码目录导入 lg_fdc。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lg_fdc.data.episodes import EpisodeConfig, sample_episode
from lg_fdc.data.manifest import ImageRecord
from lg_fdc.features.simple import MetadataFeatureExtractor
from lg_fdc.pipelines.prototype_baseline import run_prototype_episode


def main() -> None:
    """运行一个不依赖真实数据和深度学习库的最小自检。"""

    records = _make_synthetic_records()
    episode = sample_episode(records, EpisodeConfig(n_way=3, k_shot=2, q_queries=3, seed=7))
    result = run_prototype_episode(episode, MetadataFeatureExtractor(feature_dim=3))

    print("classes:", ", ".join(episode.classes))
    print(f"accuracy={result.accuracy:.3f}")
    print(f"balanced_accuracy={result.balanced_accuracy:.3f}")
    print(f"macro_f1={result.macro_f1:.3f}")

    if result.accuracy < 0.99:
        raise SystemExit("Smoke test failed: prototype classifier did not recover synthetic labels")


def _make_synthetic_records() -> list[ImageRecord]:
    """构造一批线性可分的合成样本，用来验证数据流和分类器。"""

    centers = {
        "scratch": [1.0, 0.0, 0.0],
        "crack": [0.0, 1.0, 0.0],
        "stain": [0.0, 0.0, 1.0],
        "dent": [0.7, 0.7, 0.0],
    }
    records: list[ImageRecord] = []
    for label, center in centers.items():
        for idx in range(8):
            jitter = (idx - 3.5) * 0.005
            feature = [value + jitter for value in center]
            records.append(
                ImageRecord(
                    image_path=f"synthetic/{label}_{idx:03d}.png",
                    label=label,
                    split="train",
                    metadata={"feature": feature},
                )
            )
    return records


if __name__ == "__main__":
    main()