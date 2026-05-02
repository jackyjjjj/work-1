from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ImageRecord:
    """一条图像样本记录。

    这个结构是后续所有模块之间传递样本信息的统一格式。真实数据集可能来自
    MVTec-FS、NEU-DET、DeepPCB 等，但都会先整理成 ImageRecord。
    """

    image_path: str
    label: str
    split: str = "train"
    mask_path: str | None = None
    object_name: str | None = None
    defect_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def load_manifest(path: str | Path) -> list[ImageRecord]:
    """读取 CSV 或 JSONL 格式的数据清单。

    最少需要 ``image_path`` 和 ``label`` 两列；其它列会保留到 metadata，方便后面
    继续扩展，比如缓存好的特征、异常热力图路径、产品类别等。
    """

    manifest_path = Path(path)
    suffix = manifest_path.suffix.lower()
    if suffix == ".csv":
        return _load_csv(manifest_path)
    if suffix in {".jsonl", ".ndjson"}:
        return _load_jsonl(manifest_path)
    raise ValueError(f"Unsupported manifest format: {manifest_path}")


def _load_csv(path: Path) -> list[ImageRecord]:
    """按表头读取 CSV，并逐行转为 ImageRecord。"""

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [_record_from_mapping(row, path) for row in reader]


def _load_jsonl(path: Path) -> list[ImageRecord]:
    """读取 JSONL，一行对应一个样本。"""

    records: list[ImageRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_no}") from exc
            records.append(_record_from_mapping(payload, path))
    return records


def _record_from_mapping(row: dict[str, Any], source: Path) -> ImageRecord:
    """把一行字典数据规范化为 ImageRecord。"""

    if "image_path" not in row or "label" not in row:
        raise ValueError(f"Manifest row in {source} must contain image_path and label")

    # 已知字段进入 ImageRecord 的显式属性；其它字段保留为 metadata，避免丢信息。
    known_keys = {"image_path", "label", "split", "mask_path", "object_name", "defect_name"}
    metadata = {key: value for key, value in row.items() if key not in known_keys}
    return ImageRecord(
        image_path=str(row["image_path"]),
        label=str(row["label"]),
        split=str(row.get("split") or "train"),
        mask_path=_optional_str(row.get("mask_path")),
        object_name=_optional_str(row.get("object_name")),
        defect_name=_optional_str(row.get("defect_name")),
        metadata=metadata,
    )


def _optional_str(value: Any) -> str | None:
    """把空字符串统一视为 None，避免后续路径判断出错。"""

    if value is None or value == "":
        return None
    return str(value)