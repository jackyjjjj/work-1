from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any

# 允许直接从源码树运行脚本，不要求先 pip install -e。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lg_fdc.data.episodes import EpisodeConfig, sample_episode
from lg_fdc.data.manifest import load_manifest
from lg_fdc.features.cached import CachedFeatureExtractor
from lg_fdc.features.simple import HashFeatureExtractor, MetadataFeatureExtractor
from lg_fdc.pipelines.prototype_baseline import run_prototype_episode


DEFAULT_GRID = "5:1,5:3,5:5,10:1,10:5"


def parse_args() -> argparse.Namespace:
    """解析 few-shot grid 实验参数。"""

    parser = argparse.ArgumentParser(description="Run multiple N-way K-shot prototype baselines.")
    parser.add_argument("--manifest", required=True, help="CSV or JSONL manifest path.")
    parser.add_argument("--split", default="train", help="Manifest split to sample from.")
    parser.add_argument(
        "--grid",
        default=DEFAULT_GRID,
        help="Comma-separated N:K settings, e.g. '5:1,5:5,10:1'.",
    )
    parser.add_argument("--q-queries", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--feature-source",
        choices=["metadata", "hash", "cached"],
        default="cached",
        help="Use metadata features, deterministic hash features, or cached features.",
    )
    parser.add_argument("--feature-dim", type=int, default=384)
    parser.add_argument("--feature-key", default="feature")
    parser.add_argument("--feature-file", help="JSONL/CSV feature cache used when --feature-source cached.")
    parser.add_argument("--output-json", required=True, help="Output JSON summary path.")
    parser.add_argument("--output-md", help="Optional Markdown table output path.")
    return parser.parse_args()


def main() -> None:
    """运行多组 few-shot 设置，并输出 JSON 与 Markdown 表格。"""

    args = parse_args()
    settings = _parse_grid(args.grid)
    records = [record for record in load_manifest(args.manifest) if record.split == args.split]
    if not records:
        raise SystemExit(f"No records found for split={args.split!r} in {args.manifest}")

    extractor = _build_extractor(args)
    results = []
    for setting_idx, (n_way, k_shot) in enumerate(settings):
        print(f"running {n_way}-way {k_shot}-shot ...", flush=True)
        metrics = _run_setting(
            records=records,
            extractor=extractor,
            n_way=n_way,
            k_shot=k_shot,
            q_queries=args.q_queries,
            episodes=args.episodes,
            seed=args.seed + setting_idx * 100000,
        )
        results.append(
            {
                "n_way": n_way,
                "k_shot": k_shot,
                "q_queries": args.q_queries,
                "episodes": args.episodes,
                "metrics": metrics,
            }
        )

    summary = {
        "manifest": args.manifest,
        "split": args.split,
        "feature_source": args.feature_source,
        "feature_file": args.feature_file,
        "feature_dim": args.feature_dim,
        "results": results,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote JSON: {output_json}")

    output_md = Path(args.output_md) if args.output_md else output_json.with_suffix(".md")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_format_markdown_table(summary), encoding="utf-8")
    print(f"wrote Markdown: {output_md}")


def _run_setting(
    records: list[Any],
    extractor: Any,
    n_way: int,
    k_shot: int,
    q_queries: int,
    episodes: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    episode_metrics = []
    for episode_idx in range(episodes):
        config = EpisodeConfig(
            n_way=n_way,
            k_shot=k_shot,
            q_queries=q_queries,
            seed=seed + episode_idx,
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
    return _summarize(episode_metrics)


def _build_extractor(args: argparse.Namespace):
    if args.feature_source == "metadata":
        return MetadataFeatureExtractor(feature_dim=args.feature_dim, key=args.feature_key)
    if args.feature_source == "cached":
        if not args.feature_file:
            raise SystemExit("--feature-file is required when --feature-source cached")
        return CachedFeatureExtractor(feature_path=args.feature_file, feature_dim=args.feature_dim)
    return HashFeatureExtractor(feature_dim=args.feature_dim)


def _parse_grid(grid: str) -> list[tuple[int, int]]:
    settings = []
    for item in grid.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            n_way_text, k_shot_text = item.split(":", maxsplit=1)
            n_way = int(n_way_text)
            k_shot = int(k_shot_text)
        except ValueError as exc:
            raise ValueError(f"Invalid grid item {item!r}; expected format N:K, e.g. 5:1") from exc
        if n_way <= 0 or k_shot <= 0:
            raise ValueError(f"Invalid grid item {item!r}; N and K must be positive")
        settings.append((n_way, k_shot))
    if not settings:
        raise ValueError("Grid is empty")
    return settings


def _summarize(episode_metrics: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    names = episode_metrics[0].keys()
    summary = {}
    for name in names:
        values = [metrics[name] for metrics in episode_metrics]
        summary[name] = {
            "mean": statistics.fmean(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    return summary


def _format_markdown_table(summary: dict[str, Any]) -> str:
    lines = [
        "# Few-Shot Prototype Results",
        "",
        f"- Manifest: `{summary['manifest']}`",
        f"- Split: `{summary['split']}`",
        f"- Feature source: `{summary['feature_source']}`",
    ]
    if summary.get("feature_file"):
        lines.append(f"- Feature file: `{summary['feature_file']}`")
    lines.extend(
        [
            "",
            "| Setting | Episodes | Accuracy | Balanced Acc | Macro-F1 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in summary["results"]:
        setting = f"{item['n_way']}-way {item['k_shot']}-shot"
        metrics = item["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    setting,
                    str(item["episodes"]),
                    _fmt_metric(metrics["accuracy"]),
                    _fmt_metric(metrics["balanced_accuracy"]),
                    _fmt_metric(metrics["macro_f1"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _fmt_metric(metric: dict[str, float]) -> str:
    return f"{metric['mean'] * 100:.2f} ± {metric['stdev'] * 100:.2f}"


if __name__ == "__main__":
    main()