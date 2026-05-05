from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """解析 GT bbox 与 pseudo bbox 的 IoU 诊断参数。"""

    parser = argparse.ArgumentParser(description="Evaluate pseudo bbox quality against GT bbox in manifests.")
    parser.add_argument("--gt-manifest", required=True, help="Original manifest with GT LabelMe bbox.")
    parser.add_argument("--pseudo-manifest", required=True, help="Pseudo-bbox manifest generated from heatmaps.")
    parser.add_argument("--split", default="train", help="GT manifest split to evaluate; use all for every row.")
    parser.add_argument("--output-json", help="Optional JSON summary output path.")
    parser.add_argument("--output-md", help="Optional Markdown summary output path.")
    parser.add_argument("--output-csv", help="Optional per-image CSV output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt_rows = filter_rows_by_split(read_csv_rows(Path(args.gt_manifest)), args.split)
    pseudo_rows = index_rows_by_image_path(read_csv_rows(Path(args.pseudo_manifest)))

    per_image = []
    counts = {
        "gt_rows": len(gt_rows),
        "matched_rows": 0,
        "missing_pseudo_rows": 0,
        "invalid_gt_bbox_rows": 0,
        "invalid_pseudo_bbox_rows": 0,
    }
    for gt_row in gt_rows:
        image_path = gt_row["image_path"]
        pseudo_row = pseudo_rows.get(image_path) or pseudo_rows.get(image_path.replace("\\", "/"))
        if pseudo_row is None:
            counts["missing_pseudo_rows"] += 1
            continue

        gt_bbox = parse_bbox(gt_row.get("bbox", ""))
        pseudo_bbox = parse_bbox(pseudo_row.get("bbox", ""))
        if gt_bbox is None:
            counts["invalid_gt_bbox_rows"] += 1
            continue
        if pseudo_bbox is None:
            counts["invalid_pseudo_bbox_rows"] += 1
            continue

        counts["matched_rows"] += 1
        gt_area = bbox_area(gt_bbox)
        pseudo_area = bbox_area(pseudo_bbox)
        iou = bbox_iou(gt_bbox, pseudo_bbox)
        per_image.append(
            {
                "image_path": image_path,
                "label": gt_row.get("label", ""),
                "split": gt_row.get("split", ""),
                "object_name": gt_row.get("object_name", ""),
                "defect_name": gt_row.get("defect_name", ""),
                "gt_bbox": format_bbox(gt_bbox),
                "pseudo_bbox": format_bbox(pseudo_bbox),
                "iou": iou,
                "gt_area": gt_area,
                "pseudo_area": pseudo_area,
                "area_ratio": pseudo_area / gt_area if gt_area > 0 else math.inf,
                "bbox_source": pseudo_row.get("bbox_source", ""),
                "pseudo_bbox_score": pseudo_row.get("pseudo_bbox_score", ""),
                "pseudo_bbox_area": pseudo_row.get("pseudo_bbox_area", ""),
            }
        )

    summary = summarize(per_image, counts, args)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))

    if args.output_json:
        write_text(Path(args.output_json), json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    if args.output_md:
        write_text(Path(args.output_md), format_markdown(summary))
    if args.output_csv:
        write_per_image_csv(Path(args.output_csv), per_image)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """读取 CSV manifest，保留原始字符串字段。"""

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def filter_rows_by_split(rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    """按 split 过滤；all 表示不过滤。"""

    if split == "all":
        return rows
    return [row for row in rows if row.get("split") == split]


def index_rows_by_image_path(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """用 image_path 建索引，方便 GT manifest 与 pseudo manifest 对齐。"""

    index: dict[str, dict[str, str]] = {}
    for row in rows:
        image_path = row.get("image_path", "")
        if image_path:
            index[image_path] = row
            index[image_path.replace("\\", "/")] = row
    return index


def parse_bbox(text: str | None) -> tuple[float, float, float, float] | None:
    """解析 x1,y1,x2,y2；空值或非法框返回 None。"""

    if text is None or not str(text).strip():
        return None
    parts = [part.strip() for part in str(text).replace(";", ",").split(",") if part.strip()]
    if len(parts) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(part) for part in parts)
    except ValueError:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def bbox_area(bbox: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter = bbox_area((inter_x1, inter_y1, inter_x2, inter_y2)) if inter_x2 > inter_x1 and inter_y2 > inter_y1 else 0.0
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def summarize(per_image: list[dict[str, Any]], counts: dict[str, int], args: argparse.Namespace) -> dict[str, Any]:
    ious = [float(row["iou"]) for row in per_image]
    area_ratios = [float(row["area_ratio"]) for row in per_image if math.isfinite(float(row["area_ratio"]))]
    return {
        "gt_manifest": args.gt_manifest,
        "pseudo_manifest": args.pseudo_manifest,
        "split": args.split,
        "counts": counts,
        "iou": summarize_values(ious),
        "area_ratio": summarize_values(area_ratios),
        "recall_at_iou": {
            "0.10": fraction_at_least(ious, 0.10),
            "0.25": fraction_at_least(ious, 0.25),
            "0.50": fraction_at_least(ious, 0.50),
            "0.75": fraction_at_least(ious, 0.75),
        },
        "worst_examples": sorted(per_image, key=lambda row: row["iou"])[:10],
    }


def summarize_values(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "stdev": None, "min": None, "p25": None, "p75": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "p25": percentile(values, 0.25),
        "p75": percentile(values, 0.75),
        "max": max(values),
    }


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def fraction_at_least(values: list[float], threshold: float) -> float | None:
    if not values:
        return None
    return sum(value >= threshold for value in values) / len(values)


def format_bbox(bbox: tuple[float, float, float, float]) -> str:
    return ",".join(f"{value:.2f}" for value in bbox)


def format_number(value: float | int | None) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def format_markdown(summary: dict[str, Any]) -> str:
    iou = summary["iou"]
    area_ratio = summary["area_ratio"]
    recall = summary["recall_at_iou"]
    counts = summary["counts"]
    lines = [
        "# Pseudo-BBox IoU Diagnosis",
        "",
        f"- GT manifest: `{summary['gt_manifest']}`",
        f"- Pseudo manifest: `{summary['pseudo_manifest']}`",
        f"- Split: `{summary['split']}`",
        "",
        "## Counts",
        "",
        "| Item | Value |",
        "|---|---:|",
    ]
    for key, value in counts.items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| Metric | Mean | Median | Stdev | Min | P25 | P75 | Max |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            f"| IoU | {format_number(iou['mean'])} | {format_number(iou['median'])} | {format_number(iou['stdev'])} | {format_number(iou['min'])} | {format_number(iou['p25'])} | {format_number(iou['p75'])} | {format_number(iou['max'])} |",
            f"| Pseudo/GT area | {format_number(area_ratio['mean'])} | {format_number(area_ratio['median'])} | {format_number(area_ratio['stdev'])} | {format_number(area_ratio['min'])} | {format_number(area_ratio['p25'])} | {format_number(area_ratio['p75'])} | {format_number(area_ratio['max'])} |",
            "",
            "## Recall At IoU",
            "",
            "| Threshold | Fraction |",
            "|---|---:|",
        ]
    )
    for threshold, value in recall.items():
        lines.append(f"| >= {threshold} | {format_number(value)} |")
    lines.extend(["", "## Worst Examples", "", "| Image | Label | IoU | GT BBox | Pseudo BBox |", "|---|---|---:|---|---|"])
    for row in summary["worst_examples"]:
        lines.append(
            f"| `{row['image_path']}` | {row['label']} | {row['iou']:.4f} | `{row['gt_bbox']}` | `{row['pseudo_bbox']}` |"
        )
    return "\n".join(lines) + "\n"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_per_image_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_path",
        "label",
        "split",
        "object_name",
        "defect_name",
        "gt_bbox",
        "pseudo_bbox",
        "iou",
        "gt_area",
        "pseudo_area",
        "area_ratio",
        "bbox_source",
        "pseudo_bbox_score",
        "pseudo_bbox_area",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
