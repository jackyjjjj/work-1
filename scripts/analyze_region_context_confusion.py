from __future__ import annotations

import argparse
from collections import Counter
import csv
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
from lg_fdc.evaluation.metrics import accuracy, balanced_accuracy, macro_f1
from lg_fdc.features.cached import CachedFeatureExtractor
from lg_fdc.pipelines.prototype_baseline import run_prototype_episode
from lg_fdc.pipelines.region_context import run_region_context_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze per-class confusion for region-context prototype episodes.")
    parser.add_argument("--manifest", required=True, help="CSV or JSONL manifest path.")
    parser.add_argument("--split", default="train", help="Manifest split to sample from.")
    parser.add_argument("--n-way", type=int, required=True)
    parser.add_argument("--k-shot", type=int, required=True)
    parser.add_argument("--q-queries", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--whole-feature-file", required=True)
    parser.add_argument("--region-feature-file", required=True)
    parser.add_argument("--whole-feature-dim", type=int, default=384)
    parser.add_argument("--region-feature-dim", type=int, default=384)
    parser.add_argument("--whole-weight", type=float, default=0.75)
    parser.add_argument(
        "--baseline-feature-file",
        help="Optional cached feature file evaluated on the same episodes, e.g. pseudo concat fusion.",
    )
    parser.add_argument("--baseline-feature-dim", type=int, default=768)
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--top-k-confusions", type=int, default=20)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", help="Optional Markdown report path.")
    parser.add_argument("--output-per-class-csv", help="Optional per-class metric CSV path.")
    parser.add_argument("--output-confusion-csv", help="Optional confusion pair CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.whole_weight <= 1.0:
        raise SystemExit("--whole-weight must be in [0, 1]")
    region_weight = 1.0 - args.whole_weight

    records = [record for record in load_manifest(args.manifest) if record.split == args.split]
    if not records:
        raise SystemExit(f"No records found for split={args.split!r} in {args.manifest}")

    whole_extractor = CachedFeatureExtractor(args.whole_feature_file, feature_dim=args.whole_feature_dim)
    region_extractor = CachedFeatureExtractor(args.region_feature_file, feature_dim=args.region_feature_dim)
    baseline_extractor = (
        CachedFeatureExtractor(args.baseline_feature_file, feature_dim=args.baseline_feature_dim)
        if args.baseline_feature_file
        else None
    )

    region_true: list[str] = []
    region_pred: list[str] = []
    baseline_true: list[str] = []
    baseline_pred: list[str] = []
    region_episode_metrics = []
    baseline_episode_metrics = []

    for episode_idx in range(args.episodes):
        episode = sample_episode(
            records,
            EpisodeConfig(
                n_way=args.n_way,
                k_shot=args.k_shot,
                q_queries=args.q_queries,
                seed=args.seed + episode_idx,
            ),
        )
        region_result = run_region_context_episode(
            episode=episode,
            whole_extractor=whole_extractor,
            region_extractor=region_extractor,
            whole_weight=args.whole_weight,
            region_weight=region_weight,
        )
        region_true.extend(region_result.y_true)
        region_pred.extend(region_result.y_pred)
        region_episode_metrics.append(
            {
                "accuracy": region_result.accuracy,
                "balanced_accuracy": region_result.balanced_accuracy,
                "macro_f1": region_result.macro_f1,
            }
        )

        if baseline_extractor is not None:
            baseline_result = run_prototype_episode(episode, baseline_extractor)
            baseline_true.extend(baseline_result.y_true)
            baseline_pred.extend(baseline_result.y_pred)
            baseline_episode_metrics.append(
                {
                    "accuracy": baseline_result.accuracy,
                    "balanced_accuracy": baseline_result.balanced_accuracy,
                    "macro_f1": baseline_result.macro_f1,
                }
            )

    region_summary = summarize_model(
        name="region_context",
        y_true=region_true,
        y_pred=region_pred,
        episode_metrics=region_episode_metrics,
        top_k_confusions=args.top_k_confusions,
    )
    baseline_summary = None
    if baseline_extractor is not None:
        baseline_summary = summarize_model(
            name=args.baseline_name,
            y_true=baseline_true,
            y_pred=baseline_pred,
            episode_metrics=baseline_episode_metrics,
            top_k_confusions=args.top_k_confusions,
        )

    summary = {
        "manifest": args.manifest,
        "split": args.split,
        "n_way": args.n_way,
        "k_shot": args.k_shot,
        "q_queries": args.q_queries,
        "episodes": args.episodes,
        "seed": args.seed,
        "whole_feature_file": args.whole_feature_file,
        "region_feature_file": args.region_feature_file,
        "whole_weight": args.whole_weight,
        "region_weight": region_weight,
        "region_context": region_summary,
        "baseline": baseline_summary,
        "per_class_comparison": compare_per_class(region_summary, baseline_summary) if baseline_summary else [],
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote JSON: {output_json}")

    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(format_markdown(summary), encoding="utf-8")
        print(f"wrote Markdown: {output_md}")

    if args.output_per_class_csv:
        output_per_class = Path(args.output_per_class_csv)
        output_per_class.parent.mkdir(parents=True, exist_ok=True)
        write_per_class_csv(output_per_class, summary)
        print(f"wrote per-class CSV: {output_per_class}")

    if args.output_confusion_csv:
        output_confusion = Path(args.output_confusion_csv)
        output_confusion.parent.mkdir(parents=True, exist_ok=True)
        write_confusion_csv(output_confusion, summary)
        print(f"wrote confusion CSV: {output_confusion}")

    overall = region_summary["overall"]
    print(
        "region_context: "
        f"accuracy={overall['accuracy']:.4f} "
        f"balanced_accuracy={overall['balanced_accuracy']:.4f} "
        f"macro_f1={overall['macro_f1']:.4f}"
    )
    if baseline_summary is not None:
        baseline_overall = baseline_summary["overall"]
        print(
            f"{args.baseline_name}: "
            f"accuracy={baseline_overall['accuracy']:.4f} "
            f"balanced_accuracy={baseline_overall['balanced_accuracy']:.4f} "
            f"macro_f1={baseline_overall['macro_f1']:.4f}"
        )


def summarize_model(
    name: str,
    y_true: list[str],
    y_pred: list[str],
    episode_metrics: list[dict[str, float]],
    top_k_confusions: int,
) -> dict[str, Any]:
    confusion = Counter(zip(y_true, y_pred, strict=True))
    per_class = per_class_metrics(confusion)
    return {
        "name": name,
        "overall": {
            "accuracy": accuracy(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy(y_true, y_pred),
            "macro_f1": macro_f1(y_true, y_pred),
            "query_count": len(y_true),
        },
        "episode_summary": summarize_episode_metrics(episode_metrics),
        "per_class": per_class,
        "top_confusions": top_confusions(confusion, per_class, top_k_confusions),
    }


def summarize_episode_metrics(episode_metrics: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not episode_metrics:
        return {}
    names = episode_metrics[0].keys()
    summary = {}
    for name in names:
        values = [metrics[name] for metrics in episode_metrics]
        summary[name] = {
            "mean": statistics.fmean(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    return summary


def per_class_metrics(confusion: Counter[tuple[str, str]]) -> dict[str, dict[str, float | int]]:
    labels = sorted({label for pair in confusion for label in pair})
    rows = {}
    for label in labels:
        tp = confusion[(label, label)]
        true_count = sum(count for (true, _), count in confusion.items() if true == label)
        pred_count = sum(count for (_, pred), count in confusion.items() if pred == label)
        precision = tp / pred_count if pred_count else 0.0
        recall = tp / true_count if true_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows[label] = {
            "true_count": true_count,
            "pred_count": pred_count,
            "correct": tp,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return rows


def top_confusions(
    confusion: Counter[tuple[str, str]],
    per_class: dict[str, dict[str, float | int]],
    top_k: int,
) -> list[dict[str, Any]]:
    mistakes = []
    for (true, pred), count in confusion.items():
        if true == pred:
            continue
        true_count = int(per_class[true]["true_count"])
        mistakes.append(
            {
                "true": true,
                "pred": pred,
                "count": count,
                "fraction_of_true": count / true_count if true_count else 0.0,
            }
        )
    return sorted(mistakes, key=lambda item: (item["count"], item["fraction_of_true"]), reverse=True)[:top_k]


def compare_per_class(
    region_summary: dict[str, Any],
    baseline_summary: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if baseline_summary is None:
        return []
    region_classes = region_summary["per_class"]
    baseline_classes = baseline_summary["per_class"]
    labels = sorted(set(region_classes) | set(baseline_classes))
    rows = []
    for label in labels:
        region = region_classes.get(label, empty_class_metrics())
        baseline = baseline_classes.get(label, empty_class_metrics())
        rows.append(
            {
                "label": label,
                "true_count": region["true_count"],
                "region_recall": region["recall"],
                "baseline_recall": baseline["recall"],
                "recall_delta": region["recall"] - baseline["recall"],
                "region_f1": region["f1"],
                "baseline_f1": baseline["f1"],
                "f1_delta": region["f1"] - baseline["f1"],
                "region_correct": region["correct"],
                "baseline_correct": baseline["correct"],
                "correct_delta": region["correct"] - baseline["correct"],
            }
        )
    return sorted(rows, key=lambda item: (item["recall_delta"], item["f1_delta"], item["true_count"]), reverse=True)


def empty_class_metrics() -> dict[str, float | int]:
    return {"true_count": 0, "pred_count": 0, "correct": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}


def format_markdown(summary: dict[str, Any]) -> str:
    region = summary["region_context"]
    baseline = summary["baseline"]
    lines = [
        "# Region-Context Confusion Analysis",
        "",
        f"- Manifest: `{summary['manifest']}`",
        f"- Split: `{summary['split']}`",
        f"- Setting: `{summary['n_way']}-way {summary['k_shot']}-shot`",
        f"- Episodes: `{summary['episodes']}`",
        f"- Query images per class: `{summary['q_queries']}`",
        f"- Whole/region weights: `{summary['whole_weight']:.2f}/{summary['region_weight']:.2f}`",
        "",
        "## Overall",
        "",
        "| Model | Accuracy | Balanced Acc | Macro-F1 | Queries |",
        "|---|---:|---:|---:|---:|",
        overall_row("region_context", region["overall"]),
    ]
    if baseline is not None:
        lines.append(overall_row(baseline["name"], baseline["overall"]))

    lines.extend(["", "## Per-Class Metrics", ""])
    if baseline is not None:
        lines.extend(
            [
                "| Class | Queries | Region Recall | Baseline Recall | Recall Delta | Region F1 | Baseline F1 | F1 Delta |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in summary["per_class_comparison"]:
            lines.append(
                f"| {row['label']} | {row['true_count']} | {row['region_recall']:.4f} | "
                f"{row['baseline_recall']:.4f} | {row['recall_delta']:+.4f} | "
                f"{row['region_f1']:.4f} | {row['baseline_f1']:.4f} | {row['f1_delta']:+.4f} |"
            )
    else:
        lines.extend(
            [
                "| Class | Queries | Precision | Recall | F1 | Correct | Predicted |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for label, row in sorted(region["per_class"].items()):
            lines.append(
                f"| {label} | {row['true_count']} | {row['precision']:.4f} | {row['recall']:.4f} | "
                f"{row['f1']:.4f} | {row['correct']} | {row['pred_count']} |"
            )

    lines.extend(["", "## Top Region-Context Confusions", "", "| True | Pred | Count | Fraction Of True |", "|---|---|---:|---:|"])
    for row in region["top_confusions"]:
        lines.append(f"| {row['true']} | {row['pred']} | {row['count']} | {row['fraction_of_true']:.4f} |")

    if baseline is not None:
        lines.extend(
            [
                "",
                f"## Top {baseline['name']} Confusions",
                "",
                "| True | Pred | Count | Fraction Of True |",
                "|---|---|---:|---:|",
            ]
        )
        for row in baseline["top_confusions"]:
            lines.append(f"| {row['true']} | {row['pred']} | {row['count']} | {row['fraction_of_true']:.4f} |")

    lines.append("")
    return "\n".join(lines)


def overall_row(name: str, overall: dict[str, float | int]) -> str:
    return (
        f"| {name} | {overall['accuracy']:.4f} | {overall['balanced_accuracy']:.4f} | "
        f"{overall['macro_f1']:.4f} | {overall['query_count']} |"
    )


def write_per_class_csv(path: Path, summary: dict[str, Any]) -> None:
    baseline = summary["baseline"]
    if baseline is not None:
        fieldnames = [
            "label",
            "true_count",
            "region_recall",
            "baseline_recall",
            "recall_delta",
            "region_f1",
            "baseline_f1",
            "f1_delta",
            "region_correct",
            "baseline_correct",
            "correct_delta",
        ]
        rows = summary["per_class_comparison"]
    else:
        fieldnames = ["label", "true_count", "pred_count", "correct", "precision", "recall", "f1"]
        rows = [
            {"label": label, **metrics}
            for label, metrics in sorted(summary["region_context"]["per_class"].items())
        ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_confusion_csv(path: Path, summary: dict[str, Any]) -> None:
    fieldnames = ["model", "true", "pred", "count", "fraction_of_true"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        write_confusion_rows(writer, "region_context", summary["region_context"])
        if summary["baseline"] is not None:
            write_confusion_rows(writer, summary["baseline"]["name"], summary["baseline"])


def write_confusion_rows(writer: csv.DictWriter, model_name: str, model_summary: dict[str, Any]) -> None:
    for row in model_summary["top_confusions"]:
        writer.writerow({"model": model_name, **row})


if __name__ == "__main__":
    main()
