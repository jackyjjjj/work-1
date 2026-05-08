from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


DUPLICATE_POLICIES = ("error", "first", "last", "max", "allow")
COVERAGE_POLICIES = ("error", "warn", "ignore")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a method-agnostic heatmap JSONL before work-1 converts it "
            "to pseudo-bbox / pseudo-mask manifests."
        )
    )
    parser.add_argument("--heatmap-file", required=True, help="Heatmap JSONL to validate.")
    parser.add_argument("--manifest", help="Optional work-1 manifest CSV used for coverage checks.")
    parser.add_argument("--split", default="all", help="Manifest split to check; use all for every row.")
    parser.add_argument(
        "--coverage-policy",
        choices=COVERAGE_POLICIES,
        default="error",
        help="How to handle manifest rows without a matching heatmap.",
    )
    parser.add_argument(
        "--duplicate-policy",
        choices=DUPLICATE_POLICIES,
        default="allow",
        help=(
            "How to handle duplicate image_path rows. Use allow after explicit "
            "dedupe, or max/first/last/error to validate raw method output."
        ),
    )
    parser.add_argument(
        "--require-image-size",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require image_width and image_height; needed by upsample-to-image pseudo masks.",
    )
    parser.add_argument("--max-examples", type=int, default=8, help="Maximum example paths shown per issue.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    heatmap_path = Path(args.heatmap_file)
    records = validate_heatmap_file(
        heatmap_path,
        duplicate_policy=args.duplicate_policy,
        require_image_size=args.require_image_size,
        max_examples=args.max_examples,
    )

    manifest_rows = []
    missing_paths: list[str] = []
    extra_paths: list[str] = []
    if args.manifest:
        manifest_rows = load_manifest_rows(Path(args.manifest), args.split)
        manifest_paths = {normalize_path(row["image_path"]) for row in manifest_rows}
        heatmap_paths = {normalize_path(path) for path in records}
        missing_paths = sorted(manifest_paths.difference(heatmap_paths))
        extra_paths = sorted(heatmap_paths.difference(manifest_paths))
        if missing_paths and args.coverage_policy == "error":
            examples = ", ".join(missing_paths[: args.max_examples])
            raise ValueError(
                f"Heatmap coverage check failed: {len(missing_paths)} manifest rows "
                f"for split={args.split!r} are missing heatmaps. Examples: {examples}"
            )
        if missing_paths and args.coverage_policy == "warn":
            examples = ", ".join(missing_paths[: args.max_examples])
            print(f"WARNING missing_heatmaps={len(missing_paths)} examples={examples}")

    print(f"validated_heatmap_file: {heatmap_path}")
    print(f"records: {sum(item.count for item in records.values())}")
    print(f"unique_image_paths: {len(records)}")
    print(f"duplicate_image_paths: {sum(1 for item in records.values() if item.count > 1)}")
    print(f"require_image_size: {args.require_image_size}")
    if args.manifest:
        print(f"manifest: {args.manifest}")
        print(f"split: {args.split}")
        print(f"manifest_rows: {len(manifest_rows)}")
        print(f"missing_heatmap_rows: {len(missing_paths)}")
        print(f"extra_heatmap_rows: {len(extra_paths)}")
        if extra_paths:
            examples = ", ".join(extra_paths[: args.max_examples])
            print(f"extra_heatmap_examples: {examples}")


class RecordStats:
    def __init__(self, width: int, height: int, image_width: int | None, image_height: int | None) -> None:
        self.width = width
        self.height = height
        self.image_width = image_width
        self.image_height = image_height
        self.count = 1


def validate_heatmap_file(
    path: Path,
    duplicate_policy: str,
    require_image_size: bool,
    max_examples: int,
) -> dict[str, RecordStats]:
    if duplicate_policy not in DUPLICATE_POLICIES:
        raise ValueError(f"Unsupported duplicate policy: {duplicate_policy}")
    if not path.exists():
        raise FileNotFoundError(f"Heatmap JSONL not found: {path}")

    records: dict[str, RecordStats] = {}
    empty_lines = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                empty_lines += 1
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Record must be a JSON object at {path}:{line_no}")

            image_path = validate_image_path(payload, path, line_no)
            heatmap_height, heatmap_width = validate_heatmap(payload.get("heatmap"), path, line_no, image_path)
            validate_heatmap_size_metadata(payload, heatmap_width, heatmap_height, path, line_no, image_path)
            image_width = parse_positive_int(payload.get("image_width"), "image_width", path, line_no, image_path)
            image_height = parse_positive_int(payload.get("image_height"), "image_height", path, line_no, image_path)
            if require_image_size and (image_width is None or image_height is None):
                raise ValueError(
                    f"image_width/image_height are required for {image_path} at {path}:{line_no}; "
                    "they are needed when work-1 upsamples heatmaps to image size."
                )

            normalized_path = normalize_path(image_path)
            if normalized_path in records:
                records[normalized_path].count += 1
                validate_duplicate_metadata(
                    records[normalized_path],
                    heatmap_width,
                    heatmap_height,
                    image_width,
                    image_height,
                    path,
                    line_no,
                    image_path,
                )
                if duplicate_policy == "error":
                    raise ValueError(f"Duplicate image_path at {path}:{line_no}: {image_path}")
                continue
            records[normalized_path] = RecordStats(heatmap_width, heatmap_height, image_width, image_height)

    if not records:
        raise ValueError(f"Empty heatmap JSONL: {path}")
    if empty_lines:
        print(f"WARNING empty_lines_ignored={empty_lines}")

    if duplicate_policy in {"first", "last", "max"}:
        duplicates = [path for path, item in records.items() if item.count > 1]
        if duplicates:
            examples = ", ".join(duplicates[:max_examples])
            print(
                f"WARNING duplicate_image_paths={len(duplicates)} with policy={duplicate_policy}; "
                f"examples={examples}"
            )
    return records


def validate_image_path(payload: dict[str, Any], path: Path, line_no: int) -> str:
    image_path = payload.get("image_path")
    if not isinstance(image_path, str) or not image_path.strip():
        raise ValueError(f"Missing non-empty image_path at {path}:{line_no}")
    return image_path.strip()


def validate_heatmap(values: Any, path: Path, line_no: int, image_path: str) -> tuple[int, int]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"heatmap must be a non-empty 2D list for {image_path} at {path}:{line_no}")
    if not isinstance(values[0], list) or not values[0]:
        raise ValueError(f"heatmap must contain non-empty rows for {image_path} at {path}:{line_no}")
    width = len(values[0])
    for row_index, row in enumerate(values):
        if not isinstance(row, list) or len(row) != width:
            raise ValueError(f"ragged heatmap row {row_index} for {image_path} at {path}:{line_no}")
        for col_index, value in enumerate(row):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(
                    f"heatmap value must be finite numeric for {image_path} at {path}:{line_no} "
                    f"row={row_index} col={col_index}: {value!r}"
                )
    return len(values), width


def validate_heatmap_size_metadata(
    payload: dict[str, Any],
    heatmap_width: int,
    heatmap_height: int,
    path: Path,
    line_no: int,
    image_path: str,
) -> None:
    meta_width = parse_positive_int(payload.get("heatmap_width"), "heatmap_width", path, line_no, image_path)
    meta_height = parse_positive_int(payload.get("heatmap_height"), "heatmap_height", path, line_no, image_path)
    if meta_width is not None and meta_width != heatmap_width:
        raise ValueError(
            f"heatmap_width mismatch for {image_path} at {path}:{line_no}: "
            f"metadata={meta_width} actual={heatmap_width}"
        )
    if meta_height is not None and meta_height != heatmap_height:
        raise ValueError(
            f"heatmap_height mismatch for {image_path} at {path}:{line_no}: "
            f"metadata={meta_height} actual={heatmap_height}"
        )


def parse_positive_int(value: Any, field: str, path: Path, line_no: int, image_path: str) -> int | None:
    if value in (None, ""):
        return None
    try:
        parsed = int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer for {image_path} at {path}:{line_no}: {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{field} must be positive for {image_path} at {path}:{line_no}: {value!r}")
    return parsed


def validate_duplicate_metadata(
    existing: RecordStats,
    heatmap_width: int,
    heatmap_height: int,
    image_width: int | None,
    image_height: int | None,
    path: Path,
    line_no: int,
    image_path: str,
) -> None:
    if (existing.width, existing.height) != (heatmap_width, heatmap_height):
        raise ValueError(
            f"Duplicate image_path has incompatible heatmap size at {path}:{line_no}: "
            f"{image_path}: {existing.width}x{existing.height} != {heatmap_width}x{heatmap_height}"
        )
    if existing.image_width is not None and image_width is not None and existing.image_width != image_width:
        raise ValueError(
            f"Duplicate image_path has conflicting image_width at {path}:{line_no}: "
            f"{image_path}: {existing.image_width} != {image_width}"
        )
    if existing.image_height is not None and image_height is not None and existing.image_height != image_height:
        raise ValueError(
            f"Duplicate image_path has conflicting image_height at {path}:{line_no}: "
            f"{image_path}: {existing.image_height} != {image_height}"
        )


def load_manifest_rows(path: Path, split: str) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "image_path" not in reader.fieldnames:
            raise ValueError(f"Manifest must contain image_path column: {path}")
        rows = [dict(row) for row in reader]
    if split != "all":
        rows = [row for row in rows if row.get("split") == split]
    if not rows:
        raise ValueError(f"No manifest rows found for split={split!r} in {path}")
    return rows


def normalize_path(value: str) -> str:
    return value.replace("\\", "/").strip()


if __name__ == "__main__":
    main()
