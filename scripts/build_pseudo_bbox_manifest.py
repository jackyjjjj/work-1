from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


DUPLICATE_POLICIES = ("error", "first", "last", "max")


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
        "--upsample-heatmap-to-image",
        action="store_true",
        help="Bilinearly upsample heatmaps to original image size before thresholding and connected components.",
    )
    parser.add_argument(
        "--missing-policy",
        choices=["error", "clear", "keep"],
        default="error",
        help="How to handle manifest rows without heatmaps: error, clear bbox, or keep original bbox.",
    )
    parser.add_argument(
        "--duplicate-policy",
        choices=DUPLICATE_POLICIES,
        default="max",
        help=(
            "How to handle repeated image_path rows in heatmap JSONL. "
            "Use max for AnomalyDINO per-instance rows from the same image."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {output}. Use --overwrite to replace it.")

    heatmaps = load_heatmap_cache(Path(args.heatmap_file), duplicate_policy=args.duplicate_policy)
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
        for field in (
            "bbox",
            "bbox_source",
            "pseudo_bbox_score",
            "pseudo_bbox_area",
            "pseudo_bbox_heatmap_processing",
        ):
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
                    upsample_to_image=args.upsample_heatmap_to_image,
                )
                row["bbox"] = format_bbox(pseudo["bbox"])
                row["bbox_source"] = "pseudo_heatmap_upsampled" if args.upsample_heatmap_to_image else "pseudo_heatmap"
                row["pseudo_bbox_score"] = f"{pseudo['score']:.6f}"
                row["pseudo_bbox_area"] = str(pseudo["area"])
                row["pseudo_bbox_heatmap_processing"] = pseudo["heatmap_processing"]
                replaced += 1
            else:
                if args.missing_policy == "clear":
                    row["bbox"] = ""
                row["bbox_source"] = f"missing_heatmap_{args.missing_policy}"
                row["pseudo_bbox_score"] = ""
                row["pseudo_bbox_area"] = ""
                row["pseudo_bbox_heatmap_processing"] = ""
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

def load_heatmap_cache(path: Path, duplicate_policy: str = "max") -> dict[str, dict[str, Any]]:
    """Read a heatmap JSONL cache keyed by image_path."""

    if duplicate_policy not in DUPLICATE_POLICIES:
        raise ValueError(f"Unsupported duplicate_policy: {duplicate_policy}")

    cache: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            image_path = str(payload["image_path"])
            if "heatmap" not in payload:
                raise ValueError(f"Missing heatmap in {path}:{line_no}")
            if image_path in cache:
                cache[image_path] = merge_duplicate_heatmap_payload(
                    cache[image_path],
                    payload,
                    policy=duplicate_policy,
                    source=path,
                    line_no=line_no,
                    image_path=image_path,
                )
                continue
            cache[image_path] = payload
    if not cache:
        raise ValueError(f"Empty heatmap cache: {path}")
    return cache


def merge_duplicate_heatmap_payload(
    existing: dict[str, Any],
    incoming: dict[str, Any],
    policy: str,
    source: Path,
    line_no: int,
    image_path: str,
) -> dict[str, Any]:
    """Resolve duplicate heatmaps emitted for multiple instances of one image."""

    duplicate_count = int(existing.get("_duplicate_count") or 1) + 1
    if policy == "error":
        raise ValueError(f"Duplicate image_path in {source}:{line_no}: {image_path}")
    if policy == "first":
        output = dict(existing)
        output["_duplicate_count"] = duplicate_count
        return output
    if policy == "last":
        output = dict(incoming)
        output["_duplicate_count"] = duplicate_count
        return output
    if policy != "max":
        raise ValueError(f"Unsupported duplicate_policy: {policy}")

    merged = dict(existing)
    merged["heatmap"] = merge_heatmaps_max(existing["heatmap"], incoming["heatmap"], source, line_no, image_path)
    merged["_duplicate_count"] = duplicate_count
    for key in ("image_width", "image_height", "heatmap_width", "heatmap_height"):
        merged[key] = merge_duplicate_metadata(existing.get(key), incoming.get(key), key, source, line_no, image_path)
    merged["duplicate_policy"] = policy
    return merged


def merge_duplicate_metadata(
    existing: Any,
    incoming: Any,
    key: str,
    source: Path,
    line_no: int,
    image_path: str,
) -> Any:
    """Keep compatible metadata while rejecting conflicting image dimensions."""

    if existing in (None, ""):
        return incoming
    if incoming in (None, ""):
        return existing
    if str(existing) != str(incoming):
        raise ValueError(
            f"Duplicate image_path in {source}:{line_no} has conflicting {key}: "
            f"{image_path}: {existing!r} != {incoming!r}"
        )
    return existing


def merge_heatmaps_max(
    existing: Any,
    incoming: Any,
    source: Path,
    line_no: int,
    image_path: str,
) -> list[list[float]]:
    """Merge two same-sized heatmaps with pixel-wise max scores."""

    existing_height, existing_width = heatmap_shape(existing, source, line_no, image_path)
    incoming_height, incoming_width = heatmap_shape(incoming, source, line_no, image_path)
    if (existing_height, existing_width) != (incoming_height, incoming_width):
        raise ValueError(
            f"Duplicate image_path in {source}:{line_no} has incompatible heatmap sizes: "
            f"{image_path}: {existing_width}x{existing_height} != {incoming_width}x{incoming_height}"
        )
    return [
        [max(float(a), float(b)) for a, b in zip(existing_row, incoming_row, strict=True)]
        for existing_row, incoming_row in zip(existing, incoming, strict=True)
    ]


def heatmap_shape(values: Any, source: Path, line_no: int, image_path: str) -> tuple[int, int]:
    """Validate a 2D heatmap enough to safely merge duplicates."""

    if not isinstance(values, list) or not values:
        raise ValueError(f"Invalid heatmap for {image_path} in {source}:{line_no}")
    width = len(values[0]) if isinstance(values[0], list) else 0
    if width == 0:
        raise ValueError(f"Invalid heatmap for {image_path} in {source}:{line_no}")
    if any(not isinstance(row, list) or len(row) != width for row in values):
        raise ValueError(f"Heatmap rows must have the same width for {image_path} in {source}:{line_no}")
    return len(values), width

def heatmap_to_bbox(
    values: list[list[float]],
    image_width: int,
    image_height: int,
    percentile: float,
    min_area_ratio: float,
    component: str,
    upsample_to_image: bool = False,
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
    if upsample_to_image and (image_width <= 0 or image_height <= 0):
        raise ValueError("image_width and image_height are required when upsampling heatmaps to image size")
    if image_width <= 0:
        image_width = width
    if image_height <= 0:
        image_height = height

    if min_area_ratio < 0.0:
        raise ValueError("min_area_ratio must be non-negative")

    heatmap_processing = "bilinear_to_image" if upsample_to_image else "native_grid"
    if upsample_to_image and (width != image_width or height != image_height):
        fast_result = heatmap_to_bbox_upsampled_fast(
            values=values,
            image_width=image_width,
            image_height=image_height,
            percentile=percentile,
            min_area_ratio=min_area_ratio,
            component=component,
        )
        if fast_result is not None:
            return fast_result

        threshold = percentile_threshold_upsampled(values, percentile, image_width, image_height)
        components = connected_components_upsampled(
            values=values,
            threshold=threshold,
            image_width=image_width,
            image_height=image_height,
        )
        component_area_scale = image_width * image_height
        output_scale_x = 1.0
        output_scale_y = 1.0
    else:
        threshold = percentile_threshold(values, percentile)
        mask = [[score >= threshold for score in row] for row in values]
        components = connected_components(mask, values)
        component_area_scale = width * height
        output_scale_x = image_width / width
        output_scale_y = image_height / height

    min_area = max(1, int(round(component_area_scale * min_area_ratio)))
    components = [item for item in components if item["area"] >= min_area]
    if not components:
        components = [
            peak_component_upsampled(values, image_width, image_height)
            if upsample_to_image
            else peak_component(values)
        ]

    if component == "largest":
        chosen = max(components, key=lambda item: (item["area"], item["score"]))
    elif component == "max-score":
        chosen = max(components, key=lambda item: (item["score"], item["area"]))
    else:
        raise ValueError(f"Unsupported component strategy: {component}")

    x1, y1, x2, y2 = chosen["bbox"]
    # x2/y2 是 inclusive heatmap index，转换到图像坐标时扩展到下一格边界。
    image_bbox = (
        x1 * output_scale_x,
        y1 * output_scale_y,
        min(image_width, (x2 + 1) * output_scale_x),
        min(image_height, (y2 + 1) * output_scale_y),
    )
    return {"bbox": image_bbox, "score": chosen["score"], "area": chosen["area"], "heatmap_processing": heatmap_processing}


def heatmap_to_bbox_upsampled_fast(
    values: list[list[float]],
    image_width: int,
    image_height: int,
    percentile: float,
    min_area_ratio: float,
    component: str,
) -> dict[str, Any] | None:
    """Use NumPy/OpenCV for the image-sized upsample path when available."""

    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np
    except ImportError:
        return None

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.size == 0:
        raise ValueError("Heatmap must be a non-empty 2D array")
    resized = cv2.resize(array, (int(image_width), int(image_height)), interpolation=cv2.INTER_LINEAR)
    flat = resized.reshape(-1)
    threshold_idx = min(len(flat) - 1, max(0, int(math.ceil(percentile * (len(flat) - 1)))))
    threshold = float(np.partition(flat, threshold_idx)[threshold_idx])
    mask = (resized >= threshold).astype("uint8", copy=False)
    min_area = max(1, int(round(image_width * image_height * min_area_ratio)))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    if num_labels <= 1:
        return _peak_component_from_resized(resized, image_width, image_height)

    sums = np.bincount(labels.reshape(-1), weights=resized.reshape(-1), minlength=num_labels)
    components = []
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        left = int(stats[label_idx, cv2.CC_STAT_LEFT])
        top = int(stats[label_idx, cv2.CC_STAT_TOP])
        width = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        height = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        components.append(
            {
                "bbox": (left, top, left + width - 1, top + height - 1),
                "area": area,
                "score": float(sums[label_idx] / area),
            }
        )
    if not components:
        return _peak_component_from_resized(resized, image_width, image_height)

    if component == "largest":
        chosen = max(components, key=lambda item: (item["area"], item["score"]))
    elif component == "max-score":
        chosen = max(components, key=lambda item: (item["score"], item["area"]))
    else:
        raise ValueError(f"Unsupported component strategy: {component}")

    x1, y1, x2, y2 = chosen["bbox"]
    return {
        "bbox": (float(x1), float(y1), min(image_width, float(x2 + 1)), min(image_height, float(y2 + 1))),
        "score": chosen["score"],
        "area": chosen["area"],
        "heatmap_processing": "bilinear_to_image",
    }


def _peak_component_from_resized(resized: Any, image_width: int, image_height: int) -> dict[str, Any]:
    flat_idx = int(resized.argmax())
    y, x = divmod(flat_idx, image_width)
    score = float(resized[y, x])
    return {
        "bbox": (float(x), float(y), min(image_width, float(x + 1)), min(image_height, float(y + 1))),
        "score": score,
        "area": 1,
        "heatmap_processing": "bilinear_to_image",
    }


def upsampled_score_at(
    values: list[list[float]],
    target_x: int,
    target_y: int,
    target_width: int,
    target_height: int,
) -> float:
    """Return the bilinear score at one image-grid pixel without materializing the full heatmap."""

    source_height = len(values)
    source_width = len(values[0]) if source_height else 0
    if source_height == 0 or source_width == 0:
        raise ValueError("Heatmap must not be empty")
    if target_width <= 0 or target_height <= 0:
        raise ValueError("target_width and target_height must be positive")
    if source_width == target_width and source_height == target_height:
        return float(values[target_y][target_x])

    scale_x = source_width / target_width
    scale_y = source_height / target_height
    source_y = (target_y + 0.5) * scale_y - 0.5
    y0 = max(0, min(source_height - 1, math.floor(source_y)))
    y1 = max(0, min(source_height - 1, y0 + 1))
    wy = 0.0 if source_y < 0.0 else source_y - y0
    source_x = (target_x + 0.5) * scale_x - 0.5
    x0 = max(0, min(source_width - 1, math.floor(source_x)))
    x1 = max(0, min(source_width - 1, x0 + 1))
    wx = 0.0 if source_x < 0.0 else source_x - x0
    top = values[y0][x0] * (1.0 - wx) + values[y0][x1] * wx
    bottom = values[y1][x0] * (1.0 - wx) + values[y1][x1] * wx
    return float(top * (1.0 - wy) + bottom * wy)


def resize_heatmap_bilinear(values: list[list[float]], target_width: int, target_height: int) -> list[list[float]]:
    """Resize a heatmap with bilinear interpolation using pixel-center alignment."""

    source_height = len(values)
    source_width = len(values[0]) if source_height else 0
    if source_height == 0 or source_width == 0:
        raise ValueError("Heatmap must not be empty")
    if target_width <= 0 or target_height <= 0:
        raise ValueError("target_width and target_height must be positive")
    if any(len(row) != source_width for row in values):
        raise ValueError("Heatmap rows must have the same width")
    if source_width == target_width and source_height == target_height:
        return [[float(value) for value in row] for row in values]

    resized: list[list[float]] = []
    scale_x = source_width / target_width
    scale_y = source_height / target_height
    for target_y in range(target_height):
        source_y = (target_y + 0.5) * scale_y - 0.5
        y0 = max(0, min(source_height - 1, math.floor(source_y)))
        y1 = max(0, min(source_height - 1, y0 + 1))
        wy = 0.0 if source_y < 0.0 else source_y - y0
        row: list[float] = []
        for target_x in range(target_width):
            source_x = (target_x + 0.5) * scale_x - 0.5
            x0 = max(0, min(source_width - 1, math.floor(source_x)))
            x1 = max(0, min(source_width - 1, x0 + 1))
            wx = 0.0 if source_x < 0.0 else source_x - x0
            top = values[y0][x0] * (1.0 - wx) + values[y0][x1] * wx
            bottom = values[y1][x0] * (1.0 - wx) + values[y1][x1] * wx
            row.append(float(top * (1.0 - wy) + bottom * wy))
        resized.append(row)
    return resized


def percentile_threshold(values: list[list[float]], percentile: float) -> float:
    """按分位数从低到高选择阈值，percentile 越大，保留的高响应区域越少。"""

    flat = sorted(score for row in values for score in row)
    # 使用 len(flat)-1 作为尺度，避免 0.75 在小 heatmap 上落到过低的重复低值。
    index = min(len(flat) - 1, max(0, math.ceil(percentile * (len(flat) - 1))))
    return flat[index]


def percentile_threshold_upsampled(
    values: list[list[float]], percentile: float, image_width: int, image_height: int
) -> float:
    """Exact percentile threshold for the upsampled grid without storing a 2D resized heatmap."""

    flat = sorted(
        upsampled_score_at(values, x, y, image_width, image_height)
        for y in range(image_height)
        for x in range(image_width)
    )
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
            components.append(_flood_fill_mask(mask, values, visited, x, y))
    return components


def connected_components_upsampled(
    values: list[list[float]],
    threshold: float,
    image_width: int,
    image_height: int,
) -> list[dict[str, Any]]:
    """Run connected components on an upsampled grid without storing an image-sized heatmap."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width and image_height must be positive")
    visited = bytearray(image_width * image_height)
    components = []
    for y in range(image_height):
        row_offset = y * image_width
        for x in range(image_width):
            index = row_offset + x
            if visited[index]:
                continue
            score = upsampled_score_at(values, x, y, image_width, image_height)
            if score < threshold:
                visited[index] = 1
                continue
            components.append(_flood_fill_upsampled(values, visited, x, y, image_width, image_height, threshold, score))
    return components


def _flood_fill_mask(
    mask: list[list[bool]], values: list[list[float]], visited: list[list[bool]], start_x: int, start_y: int
) -> dict[str, Any]:
    height = len(mask)
    width = len(mask[0])
    stack = [(start_x, start_y)]
    visited[start_y][start_x] = True
    min_x = max_x = start_x
    min_y = max_y = start_y
    area = 0
    score_sum = 0.0
    while stack:
        x, y = stack.pop()
        area += 1
        score_sum += values[y][x]
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < width and 0 <= ny < height and mask[ny][nx] and not visited[ny][nx]:
                visited[ny][nx] = True
                stack.append((nx, ny))
    return {"bbox": (min_x, min_y, max_x, max_y), "area": area, "score": score_sum / area}


def _flood_fill_upsampled(
    values: list[list[float]],
    visited: bytearray,
    start_x: int,
    start_y: int,
    image_width: int,
    image_height: int,
    threshold: float,
    start_score: float,
) -> dict[str, Any]:
    stack = [(start_x, start_y, start_score)]
    visited[start_y * image_width + start_x] = 1
    min_x = max_x = start_x
    min_y = max_y = start_y
    area = 0
    score_sum = 0.0
    while stack:
        x, y, score = stack.pop()
        area += 1
        score_sum += score
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if not (0 <= nx < image_width and 0 <= ny < image_height):
                continue
            index = ny * image_width + nx
            if visited[index]:
                continue
            next_score = upsampled_score_at(values, nx, ny, image_width, image_height)
            if next_score < threshold:
                visited[index] = 1
                continue
            visited[index] = 1
            stack.append((nx, ny, next_score))
    return {"bbox": (min_x, min_y, max_x, max_y), "area": area, "score": score_sum / area}


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


def peak_component_upsampled(values: list[list[float]], image_width: int, image_height: int) -> dict[str, Any]:
    best_x = 0
    best_y = 0
    best_score = upsampled_score_at(values, 0, 0, image_width, image_height)
    for y in range(image_height):
        for x in range(image_width):
            score = upsampled_score_at(values, x, y, image_width, image_height)
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
