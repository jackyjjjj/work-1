from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ANNOTATION_SUFFIX = ".json"


@dataclass(frozen=True)
class MVTecFSBuildConfig:
    """MVTec-FS manifest 构建配置。

    dataset_root 指向已经下载并解压好的 MVTec-FS 根目录；output_csv 是输出到本项目的
    统一 manifest 路径。val_ratio/test_ratio 用于在没有官方 split 文件时做确定性划分。
    """

    dataset_root: Path
    output_csv: Path
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    absolute_paths: bool = False


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

    目前假设 MVTec-FS 已经下载到本地。函数会尽量兼容常见目录组织方式：图片和
    LabelMe JSON 标注可以放在同一目录，也可以分布在子目录中。类别标签优先从 JSON
    的 shapes[].label 读取；如果没有 JSON，就退化为从路径推断。
    """

    dataset_root = config.dataset_root.expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"MVTec-FS root does not exist: {dataset_root}")

    image_paths = sorted(
        path for path in dataset_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found under MVTec-FS root: {dataset_root}")

    raw_records = [_record_from_image(path, dataset_root, config.absolute_paths) for path in image_paths]
    records = _assign_splits(raw_records, config.val_ratio, config.test_ratio, config.seed)
    _write_manifest(records, config.output_csv)
    return records


def _record_from_image(image_path: Path, dataset_root: Path, absolute_paths: bool) -> MVTecFSRecord:
    annotation_path = _find_labelme_annotation(image_path)
    annotation = _load_json(annotation_path) if annotation_path else {}
    labels = _extract_shape_labels(annotation)
    object_name, defect_name = _infer_object_and_defect(image_path, dataset_root, labels)
    bbox = _extract_bbox(annotation)
    polygon_count = len(annotation.get("shapes", [])) if isinstance(annotation.get("shapes"), list) else 0

    rel_image = _format_path(image_path, dataset_root, absolute_paths)
    rel_annotation = _format_path(annotation_path, dataset_root, absolute_paths) if annotation_path else None

    return MVTecFSRecord(
        image_path=rel_image,
        label=defect_name,
        split="train",
        mask_path=None,
        object_name=object_name,
        defect_name=defect_name,
        annotation_path=rel_annotation,
        bbox=bbox,
        polygon_count=polygon_count,
    )


def _find_labelme_annotation(image_path: Path) -> Path | None:
    """查找与图片同名的 LabelMe JSON 标注。"""

    direct = image_path.with_suffix(ANNOTATION_SUFFIX)
    if direct.exists():
        return direct

    # 有些数据集会把图片和标注分开放在 images/annotations 等目录，这里做一个温和搜索。
    candidates = list(image_path.parent.rglob(image_path.stem + ANNOTATION_SUFFIX))
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


def _infer_object_and_defect(image_path: Path, dataset_root: Path, labels: list[str]) -> tuple[str, str]:
    """优先用标注 label 推断缺陷类别，缺失时从路径兜底推断。"""

    rel_parts = image_path.relative_to(dataset_root).parts
    object_name = rel_parts[0] if len(rel_parts) >= 2 else "unknown_object"
    if labels:
        defect_name = _normalize_label(labels[0])
    elif len(rel_parts) >= 3:
        defect_name = _normalize_label(rel_parts[-2])
    else:
        defect_name = _normalize_label(image_path.parent.name)
    return object_name, defect_name


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
            if isinstance(point, list | tuple) and len(point) >= 2:
                points.append((float(point[0]), float(point[1])))
    if not points:
        return ""
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return f"{min(xs):.2f},{min(ys):.2f},{max(xs):.2f},{max(ys):.2f}"


def _assign_splits(
    records: list[MVTecFSRecord], val_ratio: float, test_ratio: float, seed: int
) -> list[MVTecFSRecord]:
    """按缺陷类别分层划分 train/val/test，保证少样本类别尽量都出现在训练集中。"""

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
    return sorted(assigned, key=lambda item: (item.object_name, item.defect_name, item.image_path))


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


def _format_path(path: Path, dataset_root: Path, absolute_paths: bool) -> str:
    if absolute_paths:
        return str(path.resolve())
    return path.relative_to(dataset_root).as_posix()


def _normalize_label(label: str) -> str:
    return label.strip().replace(" ", "_").replace("/", "_") or "unknown_defect"