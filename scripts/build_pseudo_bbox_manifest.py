from __future__ import annotations

import argparse
import csv
import json
import math
from collections import deque
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """解析 heatmap -> pseudo bbox manifest 参数。"""

    parser = argparse.ArgumentParser(description="Build a pseudo-bbox manifest from anomaly heatmap JSONL.")
    parser.add_argument("--manifest", required=True, help="Original CSV manifest.")
    parser.add_argument("--heatmap-file", required=True, help="JSONL heatmap cache keyed by image_path.")
    parser.add_argument("--output", required=True, help="Output CSV manifest with bbox replaced by pseudo bbox.")
    parser.add_argument("--split", default="all", help="Manifest split to write; use all for every row.")
    parser.add_argument("--percentile", type=float, default=0.9, help="Heatmap percentile threshold in (0, 1].")
    parser.add_argument("--min-area-ratio", type=float, default=0.001, help="Minimum selected component area ratio.")
    parser.add_argument("--component", choices=["largest", "max-score"], default="max-score")
    parser.add_argument(
        "--missing-policy",
        choices=["error", "clear", "keep"],
        default="error",
        help="How to handle manifest rows without heatmaps: error, clear bbox, or keep original bbox.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {output}. Use --overwrite to replace it.")

    heatmaps = load_heatmap_cache(Path(args.heatmap_file))
    rows, fieldnames = load_manifest_rows(Path(args.manifest))
    total_rows = len(rows)
    rows = filter_rows_by_split(rows, args.split)
    if not rows:
        raise ValueError(f"No manifest rows found for split={args.split!r}")
    missing_paths = [row["image_path"] for row in rows if _get_heatmap_payload(heatmaps, row["image_path"]) is None]
    if missing_paths and args.missing_policy == "error":
        examples = ", ".join(missing_paths[:5])
        raise ValueError(
            f"Missing heatmaps for {len(missing_paths)} manifest rows after split={args.split!r}; "
            f"examples: {examples}. If the heatmap file was generated with --split train, "
            "run this script with --split train too. Use --missing-policy clear only for debugging "
            "partial heatmap caches."
        )

    output.parent.mkdir(parents=True, exist_ok=True)

    replaced = 0
    with output.open("w", encoding="utf-8", newline="") as dst:
        for field in ("bbox", "bbox_source", "pseudo_bbox_score", "pseudo_bbox_area"):
            if field not in fieldnames:
                fieldnames.append(field)
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            image_path = row["image_path"]
            heatmap_payload = _get_heatmap_payload(heatmaps, image_path)
            if heatmap_payload is not None:
                pseudo = heatmap_to_bbox(
                    values=heatmap_payload["heatmap"],
                    image_width=int(heatmap_payload.get("image_width") or 0),
                    image_height=int(heatmap_payload.get("image_height") or 0),
                    percentile=args.percentile,
                    min_area_ratio=args.min_area_ratio,
                    component=args.component,
                )
                row["bbox"] = format_bbox(pseudo["bbox"])
                row["bbox_source"] = "pseudo_heatmap"
                row["pseudo_bbox_score"] = f"{pseudo['score']:.6f}"
                row["pseudo_bbox_area"] = str(pseudo["area"])
                replaced += 1
            else:
                if args.missing_policy == "clear":
                    row["bbox"] = ""
                row["bbox_source"] = f"missing_heatmap_{args.missing_policy}"
                row["pseudo_bbox_score"] = ""
                row["pseudo_bbox_area"] = ""
            writer.writerow(row)

    print(f"wrote pseudo-bbox manifest: {output}")
    print(f"source_rows: {total_rows}")
    print(f"written_rows: {len(rows)}")
    print(f"split: {args.split}")
    print(f"pseudo_bbox_rows: {replaced}")
    print(f"missing_heatmap_rows: {len(missing_paths)}")


def load_manifest_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """读取原始 CSV manifest，并保留已有列顺序。"""

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {path}")
        rows = [dict(row) for row in reader]
        fieldnames = list(reader.fieldnames)
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    if any("image_path" not in row or not row["image_path"] for row in rows):
        raise ValueError(f"Manifest rows must contain image_path: {path}")
    return rows, fieldnames



def filter_rows_by_split(rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    """按 split 过滤 manifest；默认 all 保留所有行。"""

    if split == "all":
        return list(rows)
    return [row for row in rows if row.get("split") == split]

def load_heatmap_cache(path: Path) -> dict[str, dict[str, Any]]:
    """读取 heatmap JSONL，要求至少包含 image_path 和 heatmap。"""

    cache: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            image_path = str(payload["image_path"])
            if image_path in cache:
                raise ValueError(f"Duplicate image_path in {path}:{line_no}: {image_path}")
            if "heatmap" not in payload:
                raise ValueError(f"Missing heatmap in {path}:{line_no}")
            cache[image_path] = payload
    if not cache:
        raise ValueError(f"Empty heatmap cache: {path}")
    return cache


def heatmap_to_bbox(
    values: list[list[float]],
    image_width: int,
    image_height: int,
    percentile: float,
    min_area_ratio: float,
    component: str,
) -> dict[str, Any]:
    """把二维 heatmap 转成图像坐标系下的 pseudo bbox。"""

    height = len(values)
    width = len(values[0]) if height else 0
    if height == 0 or width == 0:
        raise ValueError("Heatmap must not be empty")
    if any(len(row) != width for row in values):
        raise ValueError("Heatmap rows must have the same width")
    if not 0.0 < percentile <= 1.0:
        raise ValueError("percentile must be in (0, 1]")
    if image_width <= 0:
        image_width = width
    if image_height <= 0:
        image_height = height

    threshold = percentile_threshold(values, percentile)
    mask = [[score >= threshold for score in row] for row in values]
    if min_area_ratio < 0.0:
        raise ValueError("min_area_ratio must be non-negative")
    min_area = max(1, int(round(width * height * min_area_ratio)))
    components = connected_components(mask, values)
    components = [item for item in components if item["area"] >= min_area]
    if not components:
        components = [peak_component(values)]

    if component == "largest":
        chosen = max(components, key=lambda item: (item["area"], item["score"]))
    elif component == "max-score":
        chosen = max(components, key=lambda item: (item["score"], item["area"]))
    else:
        raise ValueError(f"Unsupported component strategy: {component}")

    x1, y1, x2, y2 = chosen["bbox"]
    scale_x = image_width / width
    scale_y = image_height / height
    # x2/y2 是 inclusive heatmap index，转换到图像坐标时扩展到下一格边界。
    image_bbox = (
        x1 * scale_x,
        y1 * scale_y,
        min(image_width, (x2 + 1) * scale_x),
        min(image_height, (y2 + 1) * scale_y),
    )
    return {"bbox": image_bbox, "score": chosen["score"], "area": chosen["area"]}


def percentile_threshold(values: list[list[float]], percentile: float) -> float:
    """按分位数从低到高选择阈值，percentile 越大，保留的高响应区域越少。"""

    flat = sorted(score for row in values for score in row)
    # 使用 len(flat)-1 作为尺度，避免 0.75 在小 heatmap 上落到过低的重复低值。
    index = min(len(flat) - 1, max(0, math.ceil(percentile * (len(flat) - 1))))
    return flat[index]


def connected_components(mask: list[list[bool]], values: list[list[float]]) -> list[dict[str, Any]]:
    height = len(mask)
    width = len(mask[0]) if height else 0
    visited = [[False for _ in range(width)] for _ in range(height)]
    components = []
    for y in range(height):
        for x in range(width):
            if not mask[y][x] or visited[y][x]:
                continue
            components.append(_flood_fill(mask, values, visited, x, y))
    return components


def _flood_fill(
    mask: list[list[bool]], values: list[list[float]], visited: list[list[bool]], start_x: int, start_y: int
) -> dict[str, Any]:
    height = len(mask)
    width = len(mask[0])
    queue: deque[tuple[int, int]] = deque([(start_x, start_y)])
    visited[start_y][start_x] = True
    xs = []
    ys = []
    scores = []
    while queue:
        x, y = queue.popleft()
        xs.append(x)
        ys.append(y)
        scores.append(values[y][x])
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < width and 0 <= ny < height and mask[ny][nx] and not visited[ny][nx]:
                visited[ny][nx] = True
                queue.append((nx, ny))
    return {
        "bbox": (min(xs), min(ys), max(xs), max(ys)),
        "area": len(xs),
        "score": sum(scores) / len(scores),
    }


def peak_component(values: list[list[float]]) -> dict[str, Any]:
    best_x = 0
    best_y = 0
    best_score = values[0][0]
    for y, row in enumerate(values):
        for x, score in enumerate(row):
            if score > best_score:
                best_x = x
                best_y = y
                best_score = score
    return {"bbox": (best_x, best_y, best_x, best_y), "area": 1, "score": best_score}


def _get_heatmap_payload(cache: dict[str, dict[str, Any]], image_path: str) -> dict[str, Any] | None:
    return cache.get(image_path) or cache.get(image_path.replace("\\", "/"))


def format_bbox(bbox: tuple[float, float, float, float]) -> str:
    return ",".join(f"{value:.2f}" for value in bbox)


if __name__ == "__main__":
    main()