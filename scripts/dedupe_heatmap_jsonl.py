from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


POLICIES = ("error", "first", "last", "max")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deduplicate heatmap JSONL by image_path.")
    parser.add_argument("--input", required=True, help="Merged heatmap JSONL.")
    parser.add_argument("--output", required=True, help="Deduplicated output JSONL.")
    parser.add_argument("--policy", choices=POLICIES, default="max")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.input)
    dst = Path(args.output)
    if dst.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {dst}. Use --overwrite to replace it.")

    records: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    total_rows = 0
    duplicate_rows = 0

    with src.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            total_rows += 1
            payload = json.loads(line)
            image_path = str(payload["image_path"])
            if "heatmap" not in payload:
                raise ValueError(f"missing heatmap at {src}:{line_no}")
            if image_path not in records:
                records[image_path] = payload
                order.append(image_path)
                continue

            duplicate_rows += 1
            records[image_path] = merge_duplicate(records[image_path], payload, args.policy, src, line_no, image_path)

    if not order:
        raise ValueError(f"empty heatmap file: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as handle:
        for image_path in order:
            handle.write(json.dumps(records[image_path], ensure_ascii=False) + "\n")

    print(f"deduped_heatmaps: {dst}")
    print(f"source_rows: {total_rows}")
    print(f"unique_image_paths: {len(order)}")
    print(f"duplicate_rows_merged: {duplicate_rows}")
    print(f"duplicate_policy: {args.policy}")


def merge_duplicate(
    existing: dict[str, Any],
    incoming: dict[str, Any],
    policy: str,
    source: Path,
    line_no: int,
    image_path: str,
) -> dict[str, Any]:
    duplicate_count = int(existing.get("_duplicate_count") or 1) + 1
    if policy == "error":
        raise ValueError(f"duplicate image_path at {source}:{line_no}: {image_path}")
    if policy == "first":
        output = dict(existing)
        output["_duplicate_count"] = duplicate_count
        return output
    if policy == "last":
        output = dict(incoming)
        output["_duplicate_count"] = duplicate_count
        return output
    if policy != "max":
        raise ValueError(f"unsupported duplicate policy: {policy}")

    output = dict(existing)
    output["heatmap"] = merge_heatmaps_max(existing["heatmap"], incoming["heatmap"], source, line_no, image_path)
    output["_duplicate_count"] = duplicate_count
    output["duplicate_policy"] = policy
    for key in ("image_width", "image_height", "heatmap_width", "heatmap_height"):
        output[key] = merge_metadata(existing.get(key), incoming.get(key), key, source, line_no, image_path)
    return output


def merge_metadata(existing: Any, incoming: Any, key: str, source: Path, line_no: int, image_path: str) -> Any:
    if existing in (None, ""):
        return incoming
    if incoming in (None, ""):
        return existing
    if str(existing) != str(incoming):
        raise ValueError(
            f"conflicting {key} for duplicate image_path at {source}:{line_no}: "
            f"{image_path}: {existing!r} != {incoming!r}"
        )
    return existing


def merge_heatmaps_max(existing: Any, incoming: Any, source: Path, line_no: int, image_path: str) -> list[list[float]]:
    existing_height, existing_width = heatmap_shape(existing, source, line_no, image_path)
    incoming_height, incoming_width = heatmap_shape(incoming, source, line_no, image_path)
    if (existing_height, existing_width) != (incoming_height, incoming_width):
        raise ValueError(
            f"incompatible duplicate heatmap shape at {source}:{line_no}: "
            f"{image_path}: {existing_width}x{existing_height} != {incoming_width}x{incoming_height}"
        )
    return [
        [max(float(a), float(b)) for a, b in zip(existing_row, incoming_row, strict=True)]
        for existing_row, incoming_row in zip(existing, incoming, strict=True)
    ]


def heatmap_shape(values: Any, source: Path, line_no: int, image_path: str) -> tuple[int, int]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"invalid heatmap for {image_path} at {source}:{line_no}")
    width = len(values[0]) if isinstance(values[0], list) else 0
    if width == 0:
        raise ValueError(f"invalid heatmap for {image_path} at {source}:{line_no}")
    if any(not isinstance(row, list) or len(row) != width for row in values):
        raise ValueError(f"ragged heatmap for {image_path} at {source}:{line_no}")
    return len(values), width


if __name__ == "__main__":
    main()