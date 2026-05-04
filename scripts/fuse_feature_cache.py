from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """解析 whole/region 特征融合参数。"""

    parser = argparse.ArgumentParser(description="Fuse two JSONL feature caches by image_path.")
    parser.add_argument("--whole-file", required=True, help="Whole-image feature JSONL cache.")
    parser.add_argument("--region-file", required=True, help="Region/bbox feature JSONL cache.")
    parser.add_argument("--output", required=True, help="Output fused JSONL feature cache.")
    parser.add_argument(
        "--method",
        default="concat",
        choices=["concat", "weighted-sum"],
        help="Fusion method. concat doubles feature dim; weighted-sum keeps original dim.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Whole-image weight for weighted-sum: alpha * whole + (1-alpha) * region.",
    )
    parser.add_argument("--normalize-input", action="store_true", help="L2-normalize each input vector before fusion.")
    parser.add_argument("--normalize-output", action="store_true", help="L2-normalize fused feature before writing.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise SystemExit("--alpha must be in [0, 1]")

    whole = _load_jsonl_cache(Path(args.whole_file))
    region = _load_jsonl_cache(Path(args.region_file))
    missing = sorted(set(whole) ^ set(region))
    if missing:
        examples = ", ".join(missing[:5])
        raise SystemExit(f"Feature caches have different image_path keys; examples: {examples}")

    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {output_path}. Use --overwrite to replace it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    feature_dim = None
    with output_path.open("w", encoding="utf-8") as handle:
        for image_path in sorted(whole):
            whole_payload = whole[image_path]
            region_payload = region[image_path]
            whole_feature = _to_float_list(whole_payload["feature"])
            region_feature = _to_float_list(region_payload["feature"])
            fused = fuse_features(
                whole_feature=whole_feature,
                region_feature=region_feature,
                method=args.method,
                alpha=args.alpha,
                normalize_input=args.normalize_input,
                normalize_output=args.normalize_output,
            )
            feature_dim = len(fused)
            payload = dict(whole_payload)
            payload["feature"] = fused
            payload["fusion_method"] = args.method
            payload["fusion_alpha"] = args.alpha if args.method == "weighted-sum" else None
            payload["whole_region"] = whole_payload.get("region", "whole")
            payload["region_region"] = region_payload.get("region", "region")
            payload["region_crop_box"] = region_payload.get("crop_box")
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1

    print(f"wrote fused features: {output_path}")
    print(f"records: {written}")
    print(f"method: {args.method}")
    print(f"feature_dim: {feature_dim if feature_dim is not None else 'unknown'}")


def fuse_features(
    whole_feature: list[float],
    region_feature: list[float],
    method: str,
    alpha: float = 0.5,
    normalize_input: bool = False,
    normalize_output: bool = False,
) -> list[float]:
    """融合 whole-image 和 region/bbox 特征。"""

    if not whole_feature or not region_feature:
        raise ValueError("Features must not be empty")
    if normalize_input:
        whole_feature = _l2_normalize(whole_feature)
        region_feature = _l2_normalize(region_feature)

    if method == "concat":
        fused = whole_feature + region_feature
    elif method == "weighted-sum":
        if len(whole_feature) != len(region_feature):
            raise ValueError("weighted-sum requires equal feature dimensions")
        fused = [alpha * a + (1.0 - alpha) * b for a, b in zip(whole_feature, region_feature, strict=True)]
    else:
        raise ValueError(f"Unsupported fusion method: {method}")

    if normalize_output:
        fused = _l2_normalize(fused)
    return fused


def _load_jsonl_cache(path: Path) -> dict[str, dict[str, Any]]:
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
            cache[image_path] = payload
    if not cache:
        raise ValueError(f"Empty feature cache: {path}")
    return cache


def _to_float_list(value: Any) -> list[float]:
    return [float(item) for item in value]


def _l2_normalize(vector: list[float], eps: float = 1e-12) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm < eps:
        return [0.0 for _ in vector]
    return [value / norm for value in vector]


if __name__ == "__main__":
    main()