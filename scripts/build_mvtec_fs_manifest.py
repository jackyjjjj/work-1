from __future__ import annotations

import argparse
from pathlib import Path
import sys

# 允许直接从源码树运行脚本，不要求先 pip install -e。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lg_fdc.data.mvtec_fs import MVTecFSBuildConfig, build_mvtec_fs_manifest


def parse_args() -> argparse.Namespace:
    """解析 MVTec-FS manifest 构建参数。"""

    parser = argparse.ArgumentParser(description="Build a CSV manifest from a local MVTec-FS dataset.")
    parser.add_argument("--dataset-root", required=True, help="Path to the downloaded MVTec-FS root.")
    parser.add_argument(
        "--output",
        default="data/manifests/mvtec_fs.csv",
        help="Output CSV manifest path inside this project.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Store absolute image/annotation paths instead of paths relative to dataset root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = build_mvtec_fs_manifest(
        MVTecFSBuildConfig(
            dataset_root=Path(args.dataset_root),
            output_csv=Path(args.output),
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            absolute_paths=args.absolute_paths,
        )
    )
    split_counts: dict[str, int] = {}
    labels = set()
    for record in records:
        split_counts[record.split] = split_counts.get(record.split, 0) + 1
        labels.add(record.label)
    print(f"wrote manifest: {args.output}")
    print(f"records: {len(records)}")
    print(f"labels: {len(labels)}")
    print("splits:", ", ".join(f"{key}={value}" for key, value in sorted(split_counts.items())))


if __name__ == "__main__":
    main()