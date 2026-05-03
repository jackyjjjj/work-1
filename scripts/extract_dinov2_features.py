from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

# 允许直接从源码树运行脚本，不要求先 pip install -e。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lg_fdc.data.manifest import ImageRecord, load_manifest


def parse_args() -> argparse.Namespace:
    """解析 DINOv2 whole-image 特征提取参数。"""

    parser = argparse.ArgumentParser(description="Extract DINOv2 whole-image features from a manifest.")
    parser.add_argument("--manifest", required=True, help="CSV/JSONL manifest produced by build_mvtec_fs_manifest.py.")
    parser.add_argument("--image-root", required=True, help="Root directory used to resolve relative image_path values.")
    parser.add_argument("--output", required=True, help="Output JSONL feature cache path.")
    parser.add_argument("--split", default="train", help="Manifest split to extract. Use 'all' for all splits.")
    parser.add_argument("--model", default="dinov2_vits14", help="Torch Hub DINOv2 model name.")
    parser.add_argument("--repo-or-dir", default="facebookresearch/dinov2", help="Torch Hub repo or local DINOv2 repo.")
    parser.add_argument("--source", default="github", choices=["github", "local"], help="Torch Hub source.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit", type=int, help="Optional limit for quick checks.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_heavy_deps()

    import torch
    from PIL import Image
    from torch.utils.data import DataLoader
    from torchvision import transforms

    records = load_manifest(args.manifest)
    if args.split != "all":
        records = [record for record in records if record.split == args.split]
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise SystemExit(f"No records found for split={args.split!r} in {args.manifest}")

    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {output_path}. Use --overwrite to replace it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(args.device)
    transform = _build_transform(args.image_size, transforms)
    dataset = ManifestImageDataset(records=records, image_root=Path(args.image_root), transform=transform, image_cls=Image)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate_batch,
    )

    model = torch.hub.load(args.repo_or_dir, args.model, source=args.source)
    model.eval().to(device)

    written = 0
    feature_dim = None
    with output_path.open("w", encoding="utf-8") as handle:
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device, non_blocking=True)
                features = model(images)
                if isinstance(features, (tuple, list)):
                    features = features[0]
                features = features.detach().float().cpu()
                feature_dim = int(features.shape[-1])
                for idx, feature in enumerate(features):
                    record = batch["record"][idx]
                    payload: dict[str, Any] = {
                        "image_path": record.image_path,
                        "label": record.label,
                        "split": record.split,
                        "object_name": record.object_name,
                        "defect_name": record.defect_name,
                        "feature": feature.tolist(),
                    }
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    written += 1
                print(f"extracted {written}/{len(dataset)}", flush=True)

    print(f"wrote features: {output_path}")
    print(f"records: {written}")
    print(f"feature_dim: {feature_dim if feature_dim is not None else 'unknown'}")


class ManifestImageDataset:
    """从 manifest 读取图片，返回 DINOv2 输入 tensor。"""

    def __init__(self, records: list[ImageRecord], image_root: Path, transform: Any, image_cls: Any) -> None:
        self.records = records
        self.image_root = image_root.expanduser().resolve()
        self.transform = transform
        self.image_cls = image_cls

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        image_path = _resolve_image_path(record.image_path, self.image_root)
        image = self.image_cls.open(image_path).convert("RGB")
        return {"image": self.transform(image), "record": record}


def _build_transform(image_size: int, transforms: Any) -> Any:
    """DINOv2 常用 ImageNet 归一化预处理。"""

    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def _collate_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    return {
        "image": torch.stack([item["image"] for item in items], dim=0),
        "record": [item["record"] for item in items],
    }


def _resolve_image_path(image_path: str, image_root: Path) -> Path:
    path = Path(image_path)
    resolved = path if path.is_absolute() else image_root / path
    if not resolved.exists():
        raise FileNotFoundError(f"Image not found: {resolved}")
    return resolved


def _resolve_device(device: str) -> str:
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _ensure_heavy_deps() -> None:
    missing = []
    for module_name in ("torch", "torchvision", "PIL"):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
    if missing:
        raise SystemExit(
            "Missing dependencies for DINOv2 extraction: "
            + ", ".join(missing)
            + ". Install project research dependencies first, for example: "
            + "pip install -e '.[research]'"
        )


if __name__ == "__main__":
    main()