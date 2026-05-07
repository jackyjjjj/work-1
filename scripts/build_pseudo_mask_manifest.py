from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys
from typing import Any

# Allow running directly from the source tree without pip install -e.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_pseudo_bbox_manifest import (  # noqa: E402
    filter_rows_by_split,
    format_bbox,
    load_heatmap_cache,
    load_manifest_rows,
    percentile_threshold,
    resize_heatmap_bilinear,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a pseudo-mask manifest from anomaly heatmap JSONL.")
    parser.add_argument("--manifest", required=True, help="Original CSV manifest.")
    parser.add_argument("--heatmap-file", required=True, help="JSONL heatmap cache keyed by image_path.")
    parser.add_argument("--mask-dir", required=True, help="Directory used to write binary pseudo-mask PNG files.")
    parser.add_argument("--output", required=True, help="Output CSV manifest with mask_path and tight bbox updated.")
    parser.add_argument("--split", default="all", help="Manifest split to write; use all for every row.")
    parser.add_argument("--percentile", type=float, default=0.9, help="Heatmap percentile threshold in (0, 1].")
    parser.add_argument("--min-area-ratio", type=float, default=0.001, help="Minimum kept component area ratio.")
    parser.add_argument(
        "--component",
        choices=["all", "largest", "max-score"],
        default="all",
        help="Keep all surviving components, or only the largest / highest-score component.",
    )
    parser.add_argument(
        "--upsample-heatmap-to-image",
        action="store_true",
        help="Bilinearly upsample heatmaps to original image size before thresholding and connected components.",
    )
    parser.add_argument(
        "--missing-policy",
        choices=["error", "clear", "keep"],
        default="error",
        help="How to handle rows without heatmaps: error, clear mask/bbox, or keep original fields.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {output}. Use --overwrite to replace it.")

    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("This script requires Pillow. Install research dependencies first.") from exc

    heatmaps = load_heatmap_cache(Path(args.heatmap_file))
    rows, fieldnames = load_manifest_rows(Path(args.manifest))
    total_rows = len(rows)
    rows = filter_rows_by_split(rows, args.split)
    if not rows:
        raise ValueError(f"No manifest rows found for split={args.split!r}")

    missing_paths = [row["image_path"] for row in rows if get_heatmap_payload(heatmaps, row["image_path"]) is None]
    if missing_paths and args.missing_policy == "error":
        examples = ", ".join(missing_paths[:5])
        raise ValueError(
            f"Missing heatmaps for {len(missing_paths)} manifest rows after split={args.split!r}; "
            f"examples: {examples}. If the heatmap file was generated with --split train, "
            "run this script with --split train too. Use --missing-policy clear only for debugging "
            "partial heatmap caches."
        )

    mask_dir = Path(args.mask_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    written_masks = 0
    with output.open("w", encoding="utf-8", newline="") as dst:
        for field in (
            "mask_path",
            "mask_source",
            "bbox",
            "bbox_source",
            "pseudo_mask_score",
            "pseudo_mask_area",
            "pseudo_mask_component",
            "pseudo_mask_heatmap_processing",
        ):
            if field not in fieldnames:
                fieldnames.append(field)
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            image_path = row["image_path"]
            heatmap_payload = get_heatmap_payload(heatmaps, image_path)
            if heatmap_payload is not None:
                pseudo = heatmap_to_mask(
                    values=heatmap_payload["heatmap"],
                    image_width=int(heatmap_payload.get("image_width") or 0),
                    image_height=int(heatmap_payload.get("image_height") or 0),
                    percentile=args.percentile,
                    min_area_ratio=args.min_area_ratio,
                    component=args.component,
                    upsample_to_image=args.upsample_heatmap_to_image,
                )
                mask_path = build_mask_output_path(mask_dir=mask_dir, image_path=image_path)
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                save_mask_png(mask=pseudo["mask"], path=mask_path, image_module=Image)

                row["mask_path"] = str(mask_path.resolve())
                row["mask_source"] = (
                    "pseudo_heatmap_mask_upsampled" if args.upsample_heatmap_to_image else "pseudo_heatmap_mask"
                )
                row["bbox"] = format_bbox(pseudo["bbox"])
                row["bbox_source"] = (
                    "pseudo_mask_tight_box_upsampled" if args.upsample_heatmap_to_image else "pseudo_mask_tight_box"
                )
                row["pseudo_mask_score"] = f"{pseudo['score']:.6f}"
                row["pseudo_mask_area"] = str(pseudo["area"])
                row["pseudo_mask_component"] = args.component
                row["pseudo_mask_heatmap_processing"] = pseudo["heatmap_processing"]
                written_masks += 1
            else:
                if args.missing_policy == "clear":
                    row["mask_path"] = ""
                    row["bbox"] = ""
                row["mask_source"] = f"missing_heatmap_{args.missing_policy}"
                row["bbox_source"] = f"missing_heatmap_{args.missing_policy}"
                row["pseudo_mask_score"] = ""
                row["pseudo_mask_area"] = ""
                row["pseudo_mask_component"] = ""
                row["pseudo_mask_heatmap_processing"] = ""
            writer.writerow(row)

    print(f"wrote pseudo-mask manifest: {output}")
    print(f"source_rows: {total_rows}")
    print(f"written_rows: {len(rows)}")
    print(f"split: {args.split}")
    print(f"pseudo_mask_rows: {written_masks}")
    print(f"missing_heatmap_rows: {len(missing_paths)}")


def heatmap_to_mask(
    values: list[list[float]],
    image_width: int,
    image_height: int,
    percentile: float,
    min_area_ratio: float,
    component: str,
    upsample_to_image: bool = False,
) -> dict[str, Any]:
    source_height = len(values)
    source_width = len(values[0]) if source_height else 0
    if source_height == 0 or source_width == 0:
        raise ValueError("Heatmap must not be empty")
    if any(len(row) != source_width for row in values):
        raise ValueError("Heatmap rows must have the same width")
    if not 0.0 < percentile <= 1.0:
        raise ValueError("percentile must be in (0, 1]")
    if min_area_ratio < 0.0:
        raise ValueError("min_area_ratio must be non-negative")
    if component not in {"all", "largest", "max-score"}:
        raise ValueError(f"Unsupported component strategy: {component}")
    if upsample_to_image and (image_width <= 0 or image_height <= 0):
        raise ValueError("image_width and image_height are required when upsampling heatmaps to image size")
    if image_width <= 0:
        image_width = source_width
    if image_height <= 0:
        image_height = source_height

    if upsample_to_image:
        fast_result = heatmap_to_mask_upsampled_fast(
            values=values,
            image_width=image_width,
            image_height=image_height,
            percentile=percentile,
            min_area_ratio=min_area_ratio,
            component=component,
        )
        if fast_result is not None:
            return fast_result
        work_values = (
            resize_heatmap_bilinear(values, target_width=image_width, target_height=image_height)
            if source_width != image_width or source_height != image_height
            else [[float(value) for value in row] for row in values]
        )
        threshold = percentile_threshold(work_values, percentile)
        work_mask = [[score >= threshold for score in row] for row in work_values]
        min_area = max(1, int(round(image_width * image_height * min_area_ratio)))
        kept_mask, score = select_mask_components(work_mask, work_values, component=component, min_area=min_area)
        bbox = tight_bbox_from_mask(kept_mask)
        return {
            "mask": kept_mask,
            "bbox": bbox,
            "score": score,
            "area": mask_area(kept_mask),
            "heatmap_processing": "bilinear_to_image",
        }

    threshold = percentile_threshold(values, percentile)
    grid_mask = [[score >= threshold for score in row] for row in values]
    min_area = max(1, int(round(source_width * source_height * min_area_ratio)))
    kept_grid_mask, score = select_mask_components(grid_mask, values, component=component, min_area=min_area)
    image_mask = resize_binary_mask_nearest(kept_grid_mask, target_width=image_width, target_height=image_height)
    bbox = tight_bbox_from_mask(image_mask)
    return {
        "mask": image_mask,
        "bbox": bbox,
        "score": score,
        "area": mask_area(image_mask),
        "heatmap_processing": "native_grid",
    }


def heatmap_to_mask_upsampled_fast(
    values: list[list[float]],
    image_width: int,
    image_height: int,
    percentile: float,
    min_area_ratio: float,
    component: str,
) -> dict[str, Any] | None:
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
    binary = (resized >= threshold).astype("uint8", copy=False)
    min_area = max(1, int(round(image_width * image_height * min_area_ratio)))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=4)
    if num_labels <= 1:
        peak_y, peak_x = divmod(int(resized.argmax()), image_width)
        mask = [[False for _ in range(image_width)] for _ in range(image_height)]
        mask[peak_y][peak_x] = True
        return {
            "mask": mask,
            "bbox": (float(peak_x), float(peak_y), float(peak_x + 1), float(peak_y + 1)),
            "score": float(resized[peak_y, peak_x]),
            "area": 1,
            "heatmap_processing": "bilinear_to_image",
        }

    sums = np.bincount(labels.reshape(-1), weights=resized.reshape(-1), minlength=num_labels)
    candidates = []
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        candidates.append(
            {
                "label_idx": label_idx,
                "area": area,
                "score": float(sums[label_idx] / area),
            }
        )

    if not candidates:
        peak_y, peak_x = divmod(int(resized.argmax()), image_width)
        mask = [[False for _ in range(image_width)] for _ in range(image_height)]
        mask[peak_y][peak_x] = True
        return {
            "mask": mask,
            "bbox": (float(peak_x), float(peak_y), float(peak_x + 1), float(peak_y + 1)),
            "score": float(resized[peak_y, peak_x]),
            "area": 1,
            "heatmap_processing": "bilinear_to_image",
        }

    if component == "largest":
        keep_labels = [max(candidates, key=lambda item: (item["area"], item["score"]))["label_idx"]]
    elif component == "max-score":
        keep_labels = [max(candidates, key=lambda item: (item["score"], item["area"]))["label_idx"]]
    else:
        keep_labels = [item["label_idx"] for item in candidates]

    kept = np.isin(labels, keep_labels)
    kept_area = int(kept.sum())
    if kept_area <= 0:
        return None

    ys, xs = np.nonzero(kept)
    bbox = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
    score = float(resized[kept].mean())
    mask = kept.astype(bool).tolist()
    return {
        "mask": mask,
        "bbox": bbox,
        "score": score,
        "area": kept_area,
        "heatmap_processing": "bilinear_to_image",
    }


def select_mask_components(
    mask: list[list[bool]],
    values: list[list[float]],
    component: str,
    min_area: int,
) -> tuple[list[list[bool]], float]:
    components = component_masks(mask, values)
    kept = [item for item in components if item["area"] >= min_area]
    if not kept:
        return peak_mask(values)
    if component == "largest":
        selected = [max(kept, key=lambda item: (item["area"], item["score"]))]
    elif component == "max-score":
        selected = [max(kept, key=lambda item: (item["score"], item["area"]))]
    else:
        selected = kept

    height = len(mask)
    width = len(mask[0]) if height else 0
    merged = [[False for _ in range(width)] for _ in range(height)]
    score_sum = 0.0
    area_sum = 0
    for item in selected:
        for x, y in item["pixels"]:
            merged[y][x] = True
        score_sum += item["score"] * item["area"]
        area_sum += item["area"]
    return merged, score_sum / area_sum


def component_masks(mask: list[list[bool]], values: list[list[float]]) -> list[dict[str, Any]]:
    height = len(mask)
    width = len(mask[0]) if height else 0
    visited = [[False for _ in range(width)] for _ in range(height)]
    components = []
    for y in range(height):
        for x in range(width):
            if not mask[y][x] or visited[y][x]:
                continue
            components.append(flood_fill_component(mask, values, visited, start_x=x, start_y=y))
    return components


def flood_fill_component(
    mask: list[list[bool]],
    values: list[list[float]],
    visited: list[list[bool]],
    start_x: int,
    start_y: int,
) -> dict[str, Any]:
    height = len(mask)
    width = len(mask[0]) if height else 0
    stack = [(start_x, start_y)]
    visited[start_y][start_x] = True
    pixels: list[tuple[int, int]] = []
    score_sum = 0.0
    while stack:
        x, y = stack.pop()
        pixels.append((x, y))
        score_sum += values[y][x]
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < width and 0 <= ny < height and mask[ny][nx] and not visited[ny][nx]:
                visited[ny][nx] = True
                stack.append((nx, ny))

    area = len(pixels)
    return {
        "pixels": pixels,
        "area": area,
        "score": score_sum / area,
    }


def peak_mask(values: list[list[float]]) -> tuple[list[list[bool]], float]:
    height = len(values)
    width = len(values[0]) if height else 0
    best_x = 0
    best_y = 0
    best_score = values[0][0]
    for y, row in enumerate(values):
        for x, score in enumerate(row):
            if score > best_score:
                best_x = x
                best_y = y
                best_score = score
    mask = [[False for _ in range(width)] for _ in range(height)]
    mask[best_y][best_x] = True
    return mask, float(best_score)


def resize_binary_mask_nearest(
    mask: list[list[bool]],
    target_width: int,
    target_height: int,
) -> list[list[bool]]:
    source_height = len(mask)
    source_width = len(mask[0]) if source_height else 0
    if source_height == 0 or source_width == 0:
        raise ValueError("Mask must not be empty")
    if target_width <= 0 or target_height <= 0:
        raise ValueError("target_width and target_height must be positive")
    if source_width == target_width and source_height == target_height:
        return [[bool(value) for value in row] for row in mask]

    resized: list[list[bool]] = []
    for target_y in range(target_height):
        source_y = min(source_height - 1, int(target_y * source_height / target_height))
        row: list[bool] = []
        for target_x in range(target_width):
            source_x = min(source_width - 1, int(target_x * source_width / target_width))
            row.append(bool(mask[source_y][source_x]))
        resized.append(row)
    return resized


def tight_bbox_from_mask(mask: list[list[bool]]) -> tuple[float, float, float, float]:
    coords = [(x, y) for y, row in enumerate(mask) for x, value in enumerate(row) if value]
    if not coords:
        return (0.0, 0.0, 1.0, 1.0)
    xs = [x for x, _ in coords]
    ys = [y for _, y in coords]
    return (float(min(xs)), float(min(ys)), float(max(xs) + 1), float(max(ys) + 1))


def mask_area(mask: list[list[bool]]) -> int:
    return sum(1 for row in mask for value in row if value)


def save_mask_png(mask: list[list[bool]], path: Path, image_module: Any) -> None:
    pixels = [255 if value else 0 for row in mask for value in row]
    height = len(mask)
    width = len(mask[0]) if height else 0
    image = image_module.new("L", (width, height))
    image.putdata(pixels)
    image.save(path)


def build_mask_output_path(mask_dir: Path, image_path: str) -> Path:
    normalized = image_path.replace("\\", "/")
    path = Path(normalized)
    if not path.is_absolute():
        return (mask_dir / path).with_suffix(".png")
    safe_name = normalized.strip("/").replace(":", "").replace("/", "__")
    return mask_dir / f"{safe_name}.png"


def get_heatmap_payload(cache: dict[str, dict[str, Any]], image_path: str) -> dict[str, Any] | None:
    return cache.get(image_path) or cache.get(image_path.replace("\\", "/"))


if __name__ == "__main__":
    main()
