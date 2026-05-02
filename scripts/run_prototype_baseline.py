from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

# 让命令行脚本可以直接在源码树中运行，不必先 pip install -e。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lg_fdc.data.episodes import EpisodeConfig, sample_episode
from lg_fdc.data.manifest import load_manifest
from lg_fdc.features.simple import HashFeatureExtractor, MetadataFeatureExtractor
from lg_fdc.pipelines.prototype_baseline import run_prototype_episode


def parse_args() -> argparse.Namespace:
    """解析 prototype baseline 的命令行参数。"""

    parser = argparse.ArgumentParser(description="Run a prototype few-shot baseline from a manifest.")
    parser.add_argument("--manifest", required=True, help="CSV or JSONL manifest path.")
    parser.add_argument("--split", default="train", help="Manifest split to sample from.")
    parser.add_argument("--n-way", type=int, default=5)
    parser.add_argument("--k-shot", type=int, default=1)
    parser.add_argument("--q-queries", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--feature-source",
        choices=["metadata", "hash"],
        default="metadata",
        help="Use metadata feature vectors or deterministic hash placeholder features.",
    )
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--feature-key", default="feature")
    parser.add_argument("--output", help="Optional JSON metrics output path.")
    return parser.parse_args()


def main() -> None:
    """从 manifest 连续采样多个 episode，并汇总 prototype baseline 指标。"""

    args = parse_args()
    records = [record for record in load_manifest(args.manifest) if record.split == args.split]
    if not records:
        raise SystemExit(f"No records found for split={args.split!r} in {args.manifest}")

    extractor = _build_extractor(args)
    episode_metrics = []
    for episode_idx in range(args.episodes):
        # 每个 episode 改变 seed，保证同一次实验内采样不同任务，同时整体可复现。
        config = EpisodeConfig(
            n_way=args.n_way,
            k_shot=args.k_shot,
            q_queries=args.q_queries,
            seed=args.seed + episode_idx,
        )
        episode = sample_episode(records, config)
        result = run_prototype_episode(episode, extractor)
        episode_metrics.append(
            {
                "accuracy": result.accuracy,
                "balanced_accuracy": result.balanced_accuracy,
                "macro_f1": result.macro_f1,
            }
        )

    summary = {
        "manifest": args.manifest,
        "split": args.split,
        "n_way": args.n_way,
        "k_shot": args.k_shot,
        "q_queries": args.q_queries,
        "episodes": args.episodes,
        "feature_source": args.feature_source,
        "metrics": _summarize(episode_metrics),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_extractor(args: argparse.Namespace):
    """根据参数选择特征来源。"""

    if args.feature_source == "metadata":
        return MetadataFeatureExtractor(feature_dim=args.feature_dim, key=args.feature_key)
    return HashFeatureExtractor(feature_dim=args.feature_dim)


def _summarize(episode_metrics: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """对多个 episode 的指标求均值和标准差。"""

    names = episode_metrics[0].keys()
    summary = {}
    for name in names:
        values = [metrics[name] for metrics in episode_metrics]
        summary[name] = {
            "mean": statistics.fmean(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    return summary


if __name__ == "__main__":
    main()