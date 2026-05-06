from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

# 允许直接从源码树运行脚本，不要求先 pip install -e。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_pseudo_bbox_manifest import (  # noqa: E402
    filter_rows_by_split,
    format_bbox,
    heatmap_to_bbox,
    load_heatmap_cache,
    load_manifest_rows,
)
from evaluate_pseudo_bbox_iou import (  # noqa: E402
    bbox_area,
    bbox_iou,
    fraction_at_least,
    parse_bbox,
    summarize_values,
)


DEFAULT_PERCENTILES = "0.85,0.90,0.95"
DEFAULT_MIN_AREA_RATIOS = "0.0005,0.001,0.005"
DEFAULT_COMPONENTS = "largest,max-score"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep heatmap-to-pseudo-bbox parameters and summarize IoU against GT bbox."
    )
    parser.add_argument("--gt-manifest", required=True, help="Original manifest with GT LabelMe bbox.")
    parser.add_argument("--heatmap-file", required=True, help="Heatmap JSONL produced by extract_dinov2_patch_heatmaps.py.")
    parser.add_argument("--split", default="train", help="Manifest split to evaluate; use all for every row.")
    parser.add_argument("--percentiles", default=DEFAULT_PERCENTILES, help="Comma-separated heatmap percentiles.")
    parser.add_argument(
        "--min-area-ratios",
        default=DEFAULT_MIN_AREA_RATIOS,
        help="Comma-separated minimum selected component area ratios.",
    )
    parser.add_argument(
        "--components",
        default=DEFAULT_COMPONENTS,
        help="Comma-separated component strategies: largest,max-score.",
    )
    parser.add_argument(
        "--upsample-heatmap-to-image",
        action="store_true",
        help="Bilinearly upsample heatmaps to original image size before thresholding and connected components.",
    )
    parser.add_argument(
        "--missing-policy",
        choices=["error", "skip"],
        default="error",
        help="How to handle selected manifest rows without heatmaps.",
    )
    parser.add_argument("--output-json", required=True, help="Output JSON summary path.")
    parser.add_argument("--output-md", help="Optional Markdown ranking table path.")
    parser.add_argument("--output-csv", help="Optional CSV ranking table path.")
    parser.add_argument(
        "--write-manifests-dir",
        help="Optional directory for pseudo-bbox manifests generated for each sweep setting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    percentiles = parse_float_list(args.percentiles, "percentiles")
    min_area_ratios = parse_float_list(args.min_area_ratios, "min-area-ratios")
    components = parse_components(args.components)

    print(f"loading manifest: {args.gt_manifest}", flush=True)
    rows, fieldnames = load_manifest_rows(Path(args.gt_manifest))
    rows = filter_rows_by_split(rows, args.split)
    if not rows:
        raise SystemExit(f"No manifest rows found for split={args.split!r} in {args.gt_manifest}")
    print(f"loaded manifest rows for split={args.split!r}: {len(rows)}", flush=True)

    print(f"loading heatmaps: {args.heatmap_file}", flush=True)
    heatmaps = load_heatmap_cache(Path(args.heatmap_file))
    print(f"loaded heatmaps: {len(heatmaps)}", flush=True)
    write_manifests_dir = Path(args.write_manifests_dir) if args.write_manifests_dir else None
    if write_manifests_dir:
        write_manifests_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total_settings = len(percentiles) * len(min_area_ratios) * len(components)
    setting_idx = 0
    heatmap_processing = "bilinear_to_image" if args.upsample_heatmap_to_image else "native_grid"
    for percentile in percentiles:
        for min_area_ratio in min_area_ratios:
            for component in components:
                setting_idx += 1
                print(
                    f"[{setting_idx}/{total_settings}] "
                    f"percentile={percentile:g} "
                    f"min_area_ratio={min_area_ratio:g} "
                    f"component={component} "
                    f"processing={heatmap_processing}",
                    flush=True,
                )
                result, pseudo_rows = evaluate_setting(
                    rows=rows,
                    fieldnames=fieldnames,
                    heatmaps=heatmaps,
                    percentile=percentile,
                    min_area_ratio=min_area_ratio,
                    component=component,
                    missing_policy=args.missing_policy,
                    upsample_to_image=args.upsample_heatmap_to_image,
                )
                if write_manifests_dir:
                    manifest_path = write_manifests_dir / setting_filename(
                        percentile, min_area_ratio, component, args.upsample_heatmap_to_image
                    )
                    write_pseudo_manifest(manifest_path, fieldnames, pseudo_rows)
                    result["pseudo_manifest"] = str(manifest_path)
                results.append(result)
                print(
                    f"  done: evaluated={result['counts']['evaluated_rows']} "
                    f"mean_iou={format_number(result['iou']['mean'])} "
                    f"recall@0.50={format_number(result['recall_at_iou']['0.50'])}",
                    flush=True,
                )

    ranked = rank_results(results)
    summary = {
        "gt_manifest": args.gt_manifest,
        "heatmap_file": args.heatmap_file,
        "split": args.split,
        "percentiles": percentiles,
        "min_area_ratios": min_area_ratios,
        "components": components,
        "missing_policy": args.missing_policy,
        "heatmap_processing": "bilinear_to_image" if args.upsample_heatmap_to_image else "native_grid",
        "ranking_metric": "mean_iou_then_recall",
        "best": ranked[0] if ranked else None,
        "results": ranked,
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

    if args.output_csv:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        write_summary_csv(output_csv, ranked)
        print(f"wrote CSV: {output_csv}")

    if ranked:
        best = ranked[0]
        print(
            "best: "
            f"percentile={best['percentile']} "
            f"min_area_ratio={best['min_area_ratio']} "
            f"component={best['component']} "
            f"mean_iou={format_number(best['iou']['mean'])} "
            f"recall@0.50={format_number(best['recall_at_iou']['0.50'])}"
        )


def parse_float_list(text: str, name: str) -> list[float]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = float(part)
        except ValueError as exc:
            raise ValueError(f"Invalid {name} value: {part!r}") from exc
        values.append(value)
    if not values:
        raise ValueError(f"{name} must not be empty")
    return values


def parse_components(text: str) -> list[str]:
    components = [part.strip() for part in text.split(",") if part.strip()]
    if not components:
        raise ValueError("components must not be empty")
    allowed = {"largest", "max-score"}
    invalid = sorted(set(components) - allowed)
    if invalid:
        raise ValueError(f"Invalid components: {', '.join(invalid)}; allowed: largest,max-score")
    return components


def evaluate_setting(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    heatmaps: dict[str, dict[str, Any]],
    percentile: float,
    min_area_ratio: float,
    component: str,
    missing_policy: str,
    upsample_to_image: bool = False,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    if not 0.0 < percentile <= 1.0:
        raise ValueError("percentile values must be in (0, 1]")
    if min_area_ratio < 0.0:
        raise ValueError("min-area-ratio values must be non-negative")

    missing_paths = [row["image_path"] for row in rows if get_heatmap_payload(heatmaps, row["image_path"]) is None]
    if missing_paths and missing_policy == "error":
        examples = ", ".join(missing_paths[:5])
        raise ValueError(
            f"Missing heatmaps for {len(missing_paths)} selected rows; examples: {examples}. "
            "If heatmaps were generated with --split train, run this sweep with --split train too."
        )

    pseudo_rows: list[dict[str, str]] = []
    per_image: list[dict[str, Any]] = []
    counts = {
        "gt_rows": len(rows),
        "matched_heatmap_rows": 0,
        "evaluated_rows": 0,
        "missing_heatmap_rows": 0,
        "invalid_gt_bbox_rows": 0,
    }

    for row in rows:
        image_path = row["image_path"]
        heatmap_payload = get_heatmap_payload(heatmaps, image_path)
        if heatmap_payload is None:
            counts["missing_heatmap_rows"] += 1
            continue

        counts["matched_heatmap_rows"] += 1
        pseudo = heatmap_to_bbox(
            values=heatmap_payload["heatmap"],
            image_width=int(heatmap_payload.get("image_width") or 0),
            image_height=int(heatmap_payload.get("image_height") or 0),
            percentile=percentile,
            min_area_ratio=min_area_ratio,
            component=component,
            upsample_to_image=upsample_to_image,
        )
        pseudo_bbox = pseudo["bbox"]
        pseudo_rows.append(
            format_pseudo_row(row, fieldnames, pseudo, percentile, min_area_ratio, component, upsample_to_image)
        )

        gt_bbox = parse_bbox(row.get("bbox", ""))
        if gt_bbox is None:
            counts["invalid_gt_bbox_rows"] += 1
            continue

        counts["evaluated_rows"] += 1
        gt_area = bbox_area(gt_bbox)
        pseudo_area = bbox_area(pseudo_bbox)
        iou = bbox_iou(gt_bbox, pseudo_bbox)
        per_image.append(
            {
                "image_path": image_path,
                "label": row.get("label", ""),
                "split": row.get("split", ""),
                "gt_bbox": format_bbox(gt_bbox),
                "pseudo_bbox": format_bbox(pseudo_bbox),
                "iou": iou,
                "gt_area": gt_area,
                "pseudo_area": pseudo_area,
                "area_ratio": pseudo_area / gt_area if gt_area > 0 else math.inf,
            }
        )

    ious = [float(row["iou"]) for row in per_image]
    area_ratios = [float(row["area_ratio"]) for row in per_image if math.isfinite(float(row["area_ratio"]))]
    return (
        {
            "percentile": percentile,
            "min_area_ratio": min_area_ratio,
            "component": component,
            "heatmap_processing": "bilinear_to_image" if upsample_to_image else "native_grid",
            "counts": counts,
            "iou": summarize_values(ious),
            "area_ratio": summarize_values(area_ratios),
            "recall_at_iou": {
                "0.10": fraction_at_least(ious, 0.10),
                "0.25": fraction_at_least(ious, 0.25),
                "0.50": fraction_at_least(ious, 0.50),
                "0.75": fraction_at_least(ious, 0.75),
            },
            "worst_examples": sorted(per_image, key=lambda item: item["iou"])[:5],
        },
        pseudo_rows,
    )


def get_heatmap_payload(cache: dict[str, dict[str, Any]], image_path: str) -> dict[str, Any] | None:
    return cache.get(image_path) or cache.get(image_path.replace("\\", "/"))


def format_pseudo_row(
    row: dict[str, str],
    fieldnames: list[str],
    pseudo: dict[str, Any],
    percentile: float,
    min_area_ratio: float,
    component: str,
    upsample_to_image: bool,
) -> dict[str, str]:
    output = {field: row.get(field, "") for field in fieldnames}
    output["bbox"] = format_bbox(pseudo["bbox"])
    output["bbox_source"] = "pseudo_heatmap_sweep_upsampled" if upsample_to_image else "pseudo_heatmap_sweep"
    output["pseudo_bbox_score"] = f"{pseudo['score']:.6f}"
    output["pseudo_bbox_area"] = str(pseudo["area"])
    output["pseudo_bbox_percentile"] = f"{percentile:g}"
    output["pseudo_bbox_min_area_ratio"] = f"{min_area_ratio:g}"
    output["pseudo_bbox_component"] = component
    output["pseudo_bbox_heatmap_processing"] = pseudo["heatmap_processing"]
    return output


def write_pseudo_manifest(path: Path, source_fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    fieldnames = list(source_fieldnames)
    for field in (
        "bbox",
        "bbox_source",
        "pseudo_bbox_score",
        "pseudo_bbox_area",
        "pseudo_bbox_percentile",
        "pseudo_bbox_min_area_ratio",
        "pseudo_bbox_component",
        "pseudo_bbox_heatmap_processing",
    ):
        if field not in fieldnames:
            fieldnames.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def rank_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(results, key=ranking_key, reverse=True)
    for rank, result in enumerate(ranked, start=1):
        result["rank"] = rank
    return ranked


def ranking_key(result: dict[str, Any]) -> tuple[float, float, float, float]:
    mean_iou = result["iou"].get("mean")
    recall_50 = result["recall_at_iou"].get("0.50")
    recall_25 = result["recall_at_iou"].get("0.25")
    median_iou = result["iou"].get("median")
    return (
        -1.0 if mean_iou is None else float(mean_iou),
        -1.0 if recall_50 is None else float(recall_50),
        -1.0 if recall_25 is None else float(recall_25),
        -1.0 if median_iou is None else float(median_iou),
    )


def setting_filename(percentile: float, min_area_ratio: float, component: str, upsample_to_image: bool = False) -> str:
    processing = "upsampled" if upsample_to_image else "native"
    return (
        "pseudo_bbox_"
        f"{processing}_"
        f"p{slug_float(percentile)}_"
        f"area{slug_float(min_area_ratio)}_"
        f"{component.replace('-', '_')}.csv"
    )


def slug_float(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def format_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Pseudo-BBox IoU Sweep",
        "",
        f"- GT manifest: `{summary['gt_manifest']}`",
        f"- Heatmap file: `{summary['heatmap_file']}`",
        f"- Split: `{summary['split']}`",
        f"- Ranking: `{summary['ranking_metric']}`",
        f"- Heatmap processing: `{summary.get('heatmap_processing', 'native_grid')}`",
        "",
        "| Rank | Percentile | Min Area Ratio | Component | Images | Mean IoU | Median IoU | R@0.25 | R@0.50 | Mean Area Ratio |",
        "|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in summary["results"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(result["rank"]),
                    f"{result['percentile']:g}",
                    f"{result['min_area_ratio']:g}",
                    result["component"],
                    str(result["counts"]["evaluated_rows"]),
                    format_number(result["iou"]["mean"]),
                    format_number(result["iou"]["median"]),
                    format_number(result["recall_at_iou"]["0.25"]),
                    format_number(result["recall_at_iou"]["0.50"]),
                    format_number(result["area_ratio"]["mean"]),
                ]
            )
            + " |"
        )
    if summary.get("best"):
        best = summary["best"]
        lines.extend(
            [
                "",
                "## Best Setting",
                "",
                f"- Percentile: `{best['percentile']:g}`",
                f"- Min area ratio: `{best['min_area_ratio']:g}`",
                f"- Component: `{best['component']}`",
                f"- Mean IoU: `{format_number(best['iou']['mean'])}`",
                f"- Recall@IoU 0.50: `{format_number(best['recall_at_iou']['0.50'])}`",
            ]
        )
        if best.get("pseudo_manifest"):
            lines.append(f"- Pseudo manifest: `{best['pseudo_manifest']}`")
    lines.append("")
    return "\n".join(lines)


def format_number(value: float | int | None) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def write_summary_csv(path: Path, results: list[dict[str, Any]]) -> None:
    fieldnames = [
        "rank",
        "percentile",
        "min_area_ratio",
        "component",
        "heatmap_processing",
        "gt_rows",
        "matched_heatmap_rows",
        "evaluated_rows",
        "missing_heatmap_rows",
        "invalid_gt_bbox_rows",
        "mean_iou",
        "median_iou",
        "recall_iou_025",
        "recall_iou_050",
        "mean_area_ratio",
        "pseudo_manifest",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            counts = result["counts"]
            writer.writerow(
                {
                    "rank": result["rank"],
                    "percentile": result["percentile"],
                    "min_area_ratio": result["min_area_ratio"],
                    "component": result["component"],
                    "heatmap_processing": result.get("heatmap_processing", "native_grid"),
                    "gt_rows": counts["gt_rows"],
                    "matched_heatmap_rows": counts["matched_heatmap_rows"],
                    "evaluated_rows": counts["evaluated_rows"],
                    "missing_heatmap_rows": counts["missing_heatmap_rows"],
                    "invalid_gt_bbox_rows": counts["invalid_gt_bbox_rows"],
                    "mean_iou": result["iou"]["mean"],
                    "median_iou": result["iou"]["median"],
                    "recall_iou_025": result["recall_at_iou"]["0.25"],
                    "recall_iou_050": result["recall_at_iou"]["0.50"],
                    "mean_area_ratio": result["area_ratio"]["mean"],
                    "pseudo_manifest": result.get("pseudo_manifest", ""),
                }
            )


if __name__ == "__main__":
    main()
