from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any

# Allow running directly from the source tree without pip install -e.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lg_fdc.data.episodes import EpisodeConfig, sample_episode
from lg_fdc.data.manifest import load_manifest
from lg_fdc.features.cached import CachedFeatureExtractor
from lg_fdc.pipelines.region_context import run_region_context_episode


DEFAULT_GRID = "5:1,5:3,5:5,10:1,10:5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run region-context prototype grids with score-level fusion.")
    parser.add_argument("--manifest", required=True, help="CSV or JSONL manifest path.")
    parser.add_argument("--split", default="train", help="Manifest split to sample from.")
    parser.add_argument("--grid", default=DEFAULT_GRID, help="Comma-separated N:K settings, e.g. '5:1,5:5'.")
    parser.add_argument("--q-queries", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--whole-feature-file", required=True, help="Whole-image cached feature JSONL/CSV.")
    parser.add_argument("--region-feature-file", required=True, help="Region/bbox cached feature JSONL/CSV.")
    parser.add_argument("--whole-feature-dim", type=int, default=384)
    parser.add_argument("--region-feature-dim", type=int, default=384)
    parser.add_argument(
        "--whole-weights",
        default="0.5",
        help="Comma-separated whole-image score weights. Region weight is 1 - whole_weight.",
    )
    parser.add_argument("--output-json", required=True, help="Output JSON summary path.")
    parser.add_argument("--output-md", help="Optional Markdown table output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = _parse_grid(args.grid)
    whole_weights = _parse_weights(args.whole_weights)
    records = [record for record in load_manifest(args.manifest) if record.split == args.split]
    if not records:
        raise SystemExit(f"No records found for split={args.split!r} in {args.manifest}")

    whole_extractor = CachedFeatureExtractor(
        feature_path=args.whole_feature_file,
        feature_dim=args.whole_feature_dim,
    )
    region_extractor = CachedFeatureExtractor(
        feature_path=args.region_feature_file,
        feature_dim=args.region_feature_dim,
    )

    results = []
    for whole_weight in whole_weights:
        region_weight = 1.0 - whole_weight
        for setting_idx, (n_way, k_shot) in enumerate(settings):
            print(
                f"running whole_weight={whole_weight:g} region_weight={region_weight:g} "
                f"{n_way}-way {k_shot}-shot ...",
                flush=True,
            )
            # Keep episode samples paired across weights so weight sweeps are comparable.
            metrics = _run_setting(
                records=records,
                whole_extractor=whole_extractor,
                region_extractor=region_extractor,
                n_way=n_way,
                k_shot=k_shot,
                q_queries=args.q_queries,
                episodes=args.episodes,
                seed=args.seed + setting_idx * 100_000,
                whole_weight=whole_weight,
                region_weight=region_weight,
            )
            results.append(
                {
                    "whole_weight": whole_weight,
                    "region_weight": region_weight,
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
        "whole_feature_file": args.whole_feature_file,
        "region_feature_file": args.region_feature_file,
        "whole_feature_dim": args.whole_feature_dim,
        "region_feature_dim": args.region_feature_dim,
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
    whole_extractor: CachedFeatureExtractor,
    region_extractor: CachedFeatureExtractor,
    n_way: int,
    k_shot: int,
    q_queries: int,
    episodes: int,
    seed: int,
    whole_weight: float,
    region_weight: float,
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
        result = run_region_context_episode(
            episode=episode,
            whole_extractor=whole_extractor,
            region_extractor=region_extractor,
            whole_weight=whole_weight,
            region_weight=region_weight,
        )
        episode_metrics.append(
            {
                "accuracy": result.accuracy,
                "balanced_accuracy": result.balanced_accuracy,
                "macro_f1": result.macro_f1,
            }
        )
    return _summarize(episode_metrics)


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


def _parse_weights(text: str) -> list[float]:
    weights = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            weight = float(item)
        except ValueError as exc:
            raise ValueError(f"Invalid whole weight: {item!r}") from exc
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"whole weight must be in [0, 1], got {weight}")
        weights.append(weight)
    if not weights:
        raise ValueError("whole-weights must not be empty")
    return weights


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
        "# Region-Context Prototype Results",
        "",
        f"- Manifest: `{summary['manifest']}`",
        f"- Split: `{summary['split']}`",
        f"- Whole feature file: `{summary['whole_feature_file']}`",
        f"- Region feature file: `{summary['region_feature_file']}`",
        "",
        "| Whole W | Region W | Setting | Episodes | Accuracy | Balanced Acc | Macro-F1 |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    for item in summary["results"]:
        setting = f"{item['n_way']}-way {item['k_shot']}-shot"
        metrics = item["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{item['whole_weight']:.2f}",
                    f"{item['region_weight']:.2f}",
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
    return f"{metric['mean'] * 100:.2f} +/- {metric['stdev'] * 100:.2f}"


if __name__ == "__main__":
    main()
