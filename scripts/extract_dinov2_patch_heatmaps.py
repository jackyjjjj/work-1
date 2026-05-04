from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
import sys
from typing import Any

# 允许直接从源码树运行脚本，不要求先 pip install -e。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lg_fdc.data.manifest import ImageRecord, load_manifest


def parse_args() -> argparse.Namespace:
    """解析 DINOv2 patch heatmap 提取参数。"""

    parser = argparse.ArgumentParser(description="Extract DINOv2 patch heatmaps as JSONL.")
    parser.add_argument("--manifest", required=True, help="CSV/JSONL manifest path.")
    parser.add_argument("--image-root", required=True, help="Root directory used to resolve relative image_path values.")
    parser.add_argument("--output", required=True, help="Output heatmap JSONL path.")
    parser.add_argument("--split", default="train", help="Target split to write. Use 'all' for all splits.")
    parser.add_argument(
        "--mode",
        choices=["patch-contrast", "nearest-memory"],
        default="patch-contrast",
        help="patch-contrast needs no normal images; nearest-memory uses normal/good images as memory.",
    )
    parser.add_argument("--memory-split", default="train", help="Split used to build nearest-memory cache.")
    parser.add_argument(
        "--memory-labels",
        default="good,normal,ok",
        help="Comma-separated normal labels for nearest-memory mode, or 'all' for all labels.",
    )
    parser.add_argument("--max-memory-images", type=int, default=200, help="Maximum memory images.")
    parser.add_argument("--max-memory-patches", type=int, default=20000, help="Maximum memory patch tokens.")
    parser.add_argument("--distance-chunk-size", type=int, default=1024, help="Patch chunk size for distance search.")
    parser.add_argument("--model", default="dinov2_vits14", help="Torch Hub DINOv2 model name.")
    parser.add_argument("--repo-or-dir", default="facebookresearch/dinov2", help="Torch Hub repo or local DINOv2 repo.")
    parser.add_argument("--source", default="github", choices=["github", "local"], help="Torch Hub source.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=518, help="Square DINOv2 input size; should be divisible by patch size.")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit", type=int, help="Optional target record limit for quick checks.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """提取 patch 级 heatmap，并写成 build_pseudo_bbox_manifest.py 可读取的 JSONL。"""

    args = parse_args()
    _ensure_heavy_deps()

    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torch.utils.data import DataLoader
    from torchvision import transforms

    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {output}. Use --overwrite to replace it.")

    all_records = load_manifest(args.manifest)
    target_records = _filter_split(all_records, args.split)
    if args.limit:
        target_records = target_records[: args.limit]
    if not target_records:
        raise SystemExit(f"No target records found for split={args.split!r} in {args.manifest}")

    device = _resolve_device(args.device)
    transform = _build_transform(args.image_size, transforms)
    image_root = Path(args.image_root)

    model = torch.hub.load(args.repo_or_dir, args.model, source=args.source)
    model.eval().to(device)

    memory = None
    if args.mode == "nearest-memory":
        memory_records = select_memory_records(
            records=all_records,
            split=args.memory_split,
            label_set=parse_label_set(args.memory_labels),
            max_images=args.max_memory_images,
            seed=args.seed,
        )
        if not memory_records:
            raise SystemExit(
                "No memory records found. If your MVTec-FS manifest has no good/normal images, "
                "run with --mode patch-contrast first, or rebuild the manifest with normal images included."
            )
        memory_loader = DataLoader(
            ManifestImageDataset(memory_records, image_root, transform, Image),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=_collate_batch,
        )
        memory = build_memory_bank(
            model=model,
            loader=memory_loader,
            device=device,
            max_memory_patches=args.max_memory_patches,
            seed=args.seed,
            torch_module=torch,
            functional=F,
        )
        print(f"memory_images: {len(memory_records)}", flush=True)
        print(f"memory_patches: {int(memory.shape[0])}", flush=True)

    target_loader = DataLoader(
        ManifestImageDataset(target_records, image_root, transform, Image),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate_batch,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output.open("w", encoding="utf-8") as handle:
        with torch.no_grad():
            for batch in target_loader:
                images = batch["image"].to(device, non_blocking=True)
                tokens = extract_patch_tokens(model, images)
                tokens = F.normalize(tokens.float(), dim=-1)
                if args.mode == "nearest-memory":
                    scores = nearest_memory_scores(tokens, memory, args.distance_chunk_size)
                else:
                    scores = patch_contrast_scores(tokens, functional=F)

                for idx in range(tokens.shape[0]):
                    heatmap = scores_to_heatmap(scores[idx].detach().cpu().tolist())
                    record = batch["record"][idx]
                    image_width, image_height = batch["image_size"][idx]
                    payload = {
                        "image_path": record.image_path,
                        "label": record.label,
                        "split": record.split,
                        "object_name": record.object_name,
                        "defect_name": record.defect_name,
                        "image_width": int(image_width),
                        "image_height": int(image_height),
                        "heatmap_width": len(heatmap[0]) if heatmap else 0,
                        "heatmap_height": len(heatmap),
                        "heatmap": heatmap,
                        "localizer": f"dinov2_{args.mode}",
                        "model": args.model,
                        "image_size": args.image_size,
                        "score_normalization": "per_image_minmax",
                    }
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    written += 1
                print(f"wrote heatmaps {written}/{len(target_records)}", flush=True)

    print(f"wrote heatmap JSONL: {output}")
    print(f"records: {written}")
    print(f"mode: {args.mode}")


class ManifestImageDataset:
    """从 manifest 读取图像，并保留原始图像尺寸用于 heatmap 坐标映射。"""

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
        image_size = image.size
        return {"image": self.transform(image), "record": record, "image_size": image_size}


def parse_label_set(text: str) -> set[str] | None:
    """解析 normal/good 标签集合；返回 None 表示使用全部标签。"""

    if text.strip().lower() == "all":
        return None
    return {item.strip().lower() for item in text.split(",") if item.strip()}


def select_memory_records(
    records: list[ImageRecord], split: str, label_set: set[str] | None, max_images: int, seed: int
) -> list[ImageRecord]:
    """选择 nearest-memory 模式的记忆库图像。"""

    candidates = _filter_split(records, split)
    if label_set is not None:
        candidates = [record for record in candidates if _record_matches_label_set(record, label_set)]
    if max_images > 0 and len(candidates) > max_images:
        rng = random.Random(seed)
        candidates = list(candidates)
        rng.shuffle(candidates)
        candidates = candidates[:max_images]
    return sorted(candidates, key=lambda item: item.image_path)


def build_memory_bank(
    model: Any,
    loader: Any,
    device: str,
    max_memory_patches: int,
    seed: int,
    torch_module: Any,
    functional: Any,
) -> Any:
    """提取 memory 图像的 DINOv2 patch token，并可随机下采样控制显存。"""

    chunks = []
    with torch_module.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            tokens = extract_patch_tokens(model, images)
            tokens = functional.normalize(tokens.float(), dim=-1)
            chunks.append(tokens.reshape(-1, tokens.shape[-1]).cpu())
    if not chunks:
        raise ValueError("Cannot build an empty memory bank")
    memory = torch_module.cat(chunks, dim=0)
    if max_memory_patches > 0 and memory.shape[0] > max_memory_patches:
        generator = torch_module.Generator().manual_seed(seed)
        indices = torch_module.randperm(memory.shape[0], generator=generator)[:max_memory_patches]
        memory = memory[indices]
    return memory.to(device)


def extract_patch_tokens(model: Any, images: Any) -> Any:
    """兼容不同 DINOv2 Hub 模型接口，取出 patch token。"""

    if hasattr(model, "forward_features"):
        features = model.forward_features(images)
        if isinstance(features, dict):
            for key in ("x_norm_patchtokens", "patch_tokens", "x_patchtokens"):
                if key in features:
                    return features[key]
        if not isinstance(features, dict) and getattr(features, "ndim", 0) == 3:
            return features

    if hasattr(model, "get_intermediate_layers"):
        try:
            layers = model.get_intermediate_layers(images, n=1, reshape=False, return_class_token=False)
        except TypeError:
            layers = model.get_intermediate_layers(images, n=1)
        tokens = layers[0] if isinstance(layers, (list, tuple)) else layers
        if isinstance(tokens, (list, tuple)):
            tokens = tokens[0]
        if getattr(tokens, "ndim", 0) == 4:
            tokens = tokens.flatten(2).transpose(1, 2)
        return tokens

    raise RuntimeError("The selected model does not expose DINOv2 patch tokens")


def patch_contrast_scores(tokens: Any, functional: Any) -> Any:
    """无 normal 图像时的启动版 heatmap：离当前图像平均 patch 越远，分数越高。"""

    center = functional.normalize(tokens.mean(dim=1, keepdim=True), dim=-1)
    similarity = (tokens * center).sum(dim=-1)
    return 1.0 - similarity


def nearest_memory_scores(tokens: Any, memory: Any, chunk_size: int) -> Any:
    """计算每个 patch 到 memory bank 的最近余弦距离。"""

    import torch

    if memory is None:
        raise ValueError("nearest-memory mode requires a memory bank")
    flat = tokens.reshape(-1, tokens.shape[-1])
    distances = []
    chunk_size = max(1, int(chunk_size))
    for start in range(0, flat.shape[0], chunk_size):
        chunk = flat[start : start + chunk_size]
        max_similarity = (chunk @ memory.T).max(dim=1).values
        distances.append(1.0 - max_similarity)
    return torch.cat(distances, dim=0).reshape(tokens.shape[0], tokens.shape[1])


def scores_to_heatmap(scores: list[float]) -> list[list[float]]:
    """把一维 patch 分数还原成二维 heatmap。"""

    grid_size = square_grid_size(len(scores))
    normalized = normalize_scores(scores)
    return [normalized[row * grid_size : (row + 1) * grid_size] for row in range(grid_size)]


def normalize_scores(scores: list[float]) -> list[float]:
    """对单张图的 heatmap 分数做 min-max 归一化。"""

    if not scores:
        raise ValueError("scores must not be empty")
    low = min(scores)
    high = max(scores)
    if math.isclose(low, high):
        return [0.0 for _ in scores]
    return [(float(score) - low) / (high - low) for score in scores]


def square_grid_size(num_patches: int) -> int:
    """根据 patch token 数量推断方形 heatmap 边长。"""

    grid_size = int(math.sqrt(num_patches))
    if grid_size * grid_size != num_patches:
        raise ValueError(f"Patch token count must form a square grid, got {num_patches}")
    return grid_size


def _filter_split(records: list[ImageRecord], split: str) -> list[ImageRecord]:
    if split == "all":
        return list(records)
    return [record for record in records if record.split == split]


def _record_matches_label_set(record: ImageRecord, label_set: set[str]) -> bool:
    values = [record.label, record.defect_name, record.object_name]
    return any(str(value).lower() in label_set for value in values if value)


def _build_transform(image_size: int, transforms: Any) -> Any:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def _collate_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    return {
        "image": torch.stack([item["image"] for item in items], dim=0),
        "record": [item["record"] for item in items],
        "image_size": [item["image_size"] for item in items],
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
    try:
        import PIL  # noqa: F401
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "This script requires Pillow, torch, and torchvision. Install research dependencies first."
        ) from exc


if __name__ == "__main__":
    main()
