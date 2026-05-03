from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ANNOTATION_SUFFIX = ".json"
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
    "testing": "test",
}


@dataclass(frozen=True)
class MVTecFSBuildConfig:
    """MVTec-FS manifest 构建配置。

    dataset_root 指向已经下载并解压好的 MVTec-FS 根目录；output_csv 是输出到本项目的
    统一 manifest 路径。默认只保留带 LabelMe JSON 标注的图像，避免把 GitHub 仓库
    外层的说明图片误当成训练样本。
    """

    dataset_root: Path
    output_csv: Path
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    absolute_paths: bool = False
    include_unannotated: bool = False


@dataclass(frozen=True)
class MVTecFSRecord:
    """MVTec-FS 中的一张缺陷图像及其标注信息。"""

    image_path: str
    label: str
    split: str
    mask_path: str | None
    object_name: str
    defect_name: str
    annotation_path: str | None
    bbox: str
    polygon_count: int


def build_mvtec_fs_manifest(config: MVTecFSBuildConfig) -> list[MVTecFSRecord]:
    """扫描 MVTec-FS 数据集并生成项目统一 CSV manifest。

    MVTec-FS 官方仓库下载后需要先把 `image.tar.001` 到 `image.tar.012` 合并解压。
    如果直接把未解压的仓库目录传进来，通常只会扫到 `com_sample.jpg` 和
    `data_details.png` 两张说明图片；这里会主动识别这种情况并给出清晰提示。
    """

    dataset_root = config.dataset_root.expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"MVTec-FS root does not exist: {dataset_root}")

    image_paths = sorted(
        path for path in dataset_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        _raise_no_images(dataset_root)

    annotated_pairs: list[tuple[Path, Path | None]] = []
    skipped_unannotated: list[Path] = []
    for image_path in image_paths:
        annotation_path = _find_labelme_annotation(image_path, dataset_root)
        if annotation_path is None and not config.include_unannotated:
            skipped_unannotated.append(image_path)
            continue
        annotated_pairs.append((image_path, annotation_path))

    if not annotated_pairs:
        _raise_no_annotated_images(dataset_root, skipped_unannotated)

    raw_records = [
        _record_from_image(path, annotation_path, dataset_root, config.absolute_paths)
        for path, annotation_path in annotated_pairs
    ]
    records = _finalize_splits(raw_records, config.val_ratio, config.test_ratio, config.seed)
    _write_manifest(records, config.output_csv)
    return records


def _record_from_image(
    image_path: Path, annotation_path: Path | None, dataset_root: Path, absolute_paths: bool
) -> MVTecFSRecord:
    annotation = _load_json(annotation_path) if annotation_path else {}
    labels = _extract_shape_labels(annotation)
    explicit_split = _infer_split(image_path, dataset_root)
    object_name, defect_name = _infer_object_and_defect(image_path, dataset_root, labels)
    bbox = _extract_bbox(annotation)
    polygon_count = len(annotation.get("shapes", [])) if isinstance(annotation.get("shapes"), list) else 0

    rel_image = _format_path(image_path, dataset_root, absolute_paths)
    rel_annotation = _format_path(annotation_path, dataset_root, absolute_paths) if annotation_path else None

    return MVTecFSRecord(
        image_path=rel_image,
        label=defect_name,
        split=explicit_split or "auto",
        mask_path=None,
        object_name=object_name,
        defect_name=defect_name,
        annotation_path=rel_annotation,
        bbox=bbox,
        polygon_count=polygon_count,
    )


def _find_labelme_annotation(image_path: Path, dataset_root: Path) -> Path | None:
    """查找与图片对应的 LabelMe JSON 标注。"""

    direct = image_path.with_suffix(ANNOTATION_SUFFIX)
    if direct.exists():
        return direct

    rel = image_path.relative_to(dataset_root)
    stem_name = image_path.stem + ANNOTATION_SUFFIX

    # 常见情况：图片在 images/ 子目录，标注在 annotations/ 或 labels/ 对应位置。
    for image_dir_name, anno_dir_name in (("images", "annotations"), ("image", "annotation"), ("imgs", "labels")):
        parts = list(rel.parts)
        if image_dir_name in parts:
            idx = parts.index(image_dir_name)
            parts[idx] = anno_dir_name
            candidate = dataset_root.joinpath(*parts).with_suffix(ANNOTATION_SUFFIX)
            if candidate.exists():
                return candidate

    # 兜底：在当前样本附近温和搜索，不做全数据集搜索，避免大数据集上太慢。
    candidates = list(image_path.parent.rglob(stem_name))
    if candidates:
        return sorted(candidates)[0]
    return None


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract_shape_labels(annotation: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    shapes = annotation.get("shapes", [])
    if not isinstance(shapes, list):
        return labels
    for shape in shapes:
        if isinstance(shape, dict) and shape.get("label"):
            labels.append(str(shape["label"]))
    return labels


def _infer_split(image_path: Path, dataset_root: Path) -> str | None:
    """从路径中识别官方 train/test/val split。"""

    for part in image_path.relative_to(dataset_root).parts[:-1]:
        split = SPLIT_ALIASES.get(part.lower())
        if split:
            return split
    return None


def _infer_object_and_defect(image_path: Path, dataset_root: Path, labels: list[str]) -> tuple[str, str]:
    """优先用标注 label 推断缺陷类别，缺失时从路径兜底推断。"""

    rel_parts = image_path.relative_to(dataset_root).parts
    split_index = _find_split_index(rel_parts)

    if split_index is not None and split_index > 0:
        object_name = rel_parts[split_index - 1]
    elif len(rel_parts) >= 2:
        object_name = rel_parts[0]
    else:
        object_name = "unknown_object"

    if labels:
        defect_name = _normalize_label(labels[0])
    elif split_index is not None and split_index + 1 < len(rel_parts) - 1:
        defect_name = _normalize_label(rel_parts[split_index + 1])
    elif len(rel_parts) >= 3:
        defect_name = _normalize_label(rel_parts[-2])
    else:
        defect_name = _normalize_label(image_path.parent.name)
    return object_name, defect_name


def _find_split_index(parts: tuple[str, ...]) -> int | None:
    for idx, part in enumerate(parts[:-1]):
        if part.lower() in SPLIT_ALIASES:
            return idx
    return None


def _extract_bbox(annotation: dict[str, Any]) -> str:
    """从 LabelMe polygon/rectangle 点集中计算整体 bbox，格式为 x1,y1,x2,y2。"""

    points: list[tuple[float, float]] = []
    shapes = annotation.get("shapes", [])
    if not isinstance(shapes, list):
        return ""
    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        raw_points = shape.get("points", [])
        if not isinstance(raw_points, list):
            continue
        for point in raw_points:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                points.append((float(point[0]), float(point[1])))
    if not points:
        return ""
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return f"{min(xs):.2f},{min(ys):.2f},{max(xs):.2f},{max(ys):.2f}"


def _finalize_splits(
    records: list[MVTecFSRecord], val_ratio: float, test_ratio: float, seed: int
) -> list[MVTecFSRecord]:
    """保留官方 split；若路径没有 split 信息，再自动分层划分。"""

    explicit = [record for record in records if record.split != "auto"]
    auto = [record for record in records if record.split == "auto"]
    finalized = list(explicit)
    if auto:
        finalized.extend(_assign_splits(auto, val_ratio, test_ratio, seed))
    return sorted(finalized, key=lambda item: (item.split, item.object_name, item.defect_name, item.image_path))


def _assign_splits(
    records: list[MVTecFSRecord], val_ratio: float, test_ratio: float, seed: int
) -> list[MVTecFSRecord]:
    """按缺陷类别分层划分 train/val/test。"""

    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be non-negative and sum to less than 1")

    grouped: dict[str, list[MVTecFSRecord]] = {}
    for record in records:
        grouped.setdefault(record.label, []).append(record)

    rng = random.Random(seed)
    assigned: list[MVTecFSRecord] = []
    for label in sorted(grouped):
        items = list(grouped[label])
        rng.shuffle(items)
        total = len(items)
        n_test = _split_count(total, test_ratio)
        n_val = _split_count(total - n_test, val_ratio)
        for idx, record in enumerate(items):
            if idx < n_test:
                split = "test"
            elif idx < n_test + n_val:
                split = "val"
            else:
                split = "train"
            assigned.append(_replace_split(record, split))
    return assigned


def _split_count(total: int, ratio: float) -> int:
    if total <= 2 or ratio <= 0:
        return 0
    return max(1, int(round(total * ratio)))


def _replace_split(record: MVTecFSRecord, split: str) -> MVTecFSRecord:
    return MVTecFSRecord(
        image_path=record.image_path,
        label=record.label,
        split=split,
        mask_path=record.mask_path,
        object_name=record.object_name,
        defect_name=record.defect_name,
        annotation_path=record.annotation_path,
        bbox=record.bbox,
        polygon_count=record.polygon_count,
    )


def _write_manifest(records: list[MVTecFSRecord], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_path",
        "label",
        "split",
        "mask_path",
        "object_name",
        "defect_name",
        "annotation_path",
        "bbox",
        "polygon_count",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "image_path": record.image_path,
                    "label": record.label,
                    "split": record.split,
                    "mask_path": record.mask_path or "",
                    "object_name": record.object_name,
                    "defect_name": record.defect_name,
                    "annotation_path": record.annotation_path or "",
                    "bbox": record.bbox,
                    "polygon_count": record.polygon_count,
                }
            )


def _raise_no_images(dataset_root: Path) -> None:
    tar_parts = _find_tar_parts(dataset_root)
    if tar_parts:
        raise FileNotFoundError(_tar_parts_message(dataset_root, tar_parts))
    raise FileNotFoundError(f"No images found under MVTec-FS root: {dataset_root}")


def _raise_no_annotated_images(dataset_root: Path, skipped_unannotated: list[Path]) -> None:
    tar_parts = _find_tar_parts(dataset_root)
    if tar_parts:
        raise FileNotFoundError(_tar_parts_message(dataset_root, tar_parts))
    examples = ", ".join(path.name for path in skipped_unannotated[:5]) or "none"
    raise FileNotFoundError(
        "No annotated MVTec-FS images were found. The builder now ignores images without "
        "LabelMe JSON by default, so repository documentation images are not treated as samples. "
        f"dataset_root={dataset_root}; skipped_unannotated_examples={examples}. "
        "If you intentionally want to include unannotated images, rerun with --include-unannotated."
    )


def _find_tar_parts(dataset_root: Path) -> list[Path]:
    return sorted(dataset_root.glob("image.tar.*"))


def _tar_parts_message(dataset_root: Path, tar_parts: list[Path]) -> str:
    return (
        "MVTec-FS appears to be downloaded but not extracted. Found split archive parts such as "
        f"{tar_parts[0].name} under {dataset_root}. Run this in the dataset directory first: "
        "cat image.tar.* | tar -xvf - . Then rerun scripts/build_mvtec_fs_manifest.py with "
        "--dataset-root pointing to the extracted MVTec-FS root."
    )


def _format_path(path: Path, dataset_root: Path, absolute_paths: bool) -> str:
    if absolute_paths:
        return str(path.resolve())
    return path.relative_to(dataset_root).as_posix()


def _normalize_label(label: str) -> str:
    return label.strip().replace(" ", "_").replace("/", "_") or "unknown_defect"