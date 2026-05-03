from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lg_fdc.data.manifest import ImageRecord


@dataclass
class CachedFeatureExtractor:
    """从 JSONL/CSV 缓存文件中读取图像特征。

    DINOv2、Alpha-CLIP 等重模型特征建议先离线提取并缓存，再用这个 extractor
    跑 few-shot prototype baseline。这样实验阶段不会反复前向大模型。
    """

    feature_path: str | Path
    feature_dim: int | None = None
    key_field: str = "image_path"
    feature_field: str = "feature"
    features: dict[str, list[float]] = field(init=False)

    def __post_init__(self) -> None:
        path = Path(self.feature_path)
        self.features = load_feature_cache(path, self.key_field, self.feature_field)
        if not self.features:
            raise ValueError(f"No cached features found in {path}")
        first_dim = len(next(iter(self.features.values())))
        if self.feature_dim is None:
            self.feature_dim = first_dim
        elif self.feature_dim != first_dim:
            raise ValueError(f"feature_dim={self.feature_dim} does not match cached dim={first_dim}")

    def extract(self, record: ImageRecord) -> list[float]:
        """按 image_path 读取缓存特征。"""

        for key in _candidate_keys(record.image_path):
            feature = self.features.get(key)
            if feature is not None:
                return feature
        raise KeyError(f"No cached feature found for image_path={record.image_path!r}")


def load_feature_cache(
    path: Path, key_field: str = "image_path", feature_field: str = "feature"
) -> dict[str, list[float]]:
    """读取 JSONL 或 CSV 特征缓存。"""

    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        return _load_jsonl_features(path, key_field, feature_field)
    if suffix == ".csv":
        return _load_csv_features(path, key_field, feature_field)
    raise ValueError(f"Unsupported feature cache format: {path}")


def _load_jsonl_features(path: Path, key_field: str, feature_field: str) -> dict[str, list[float]]:
    features: dict[str, list[float]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            key = str(payload[key_field])
            features[key] = _to_float_list(payload[feature_field])
            if not features[key]:
                raise ValueError(f"Empty feature on {path}:{line_no}")
    return features


def _load_csv_features(path: Path, key_field: str, feature_field: str) -> dict[str, list[float]]:
    features: dict[str, list[float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for line_no, row in enumerate(reader, start=2):
            key = str(row[key_field])
            features[key] = _to_float_list(row[feature_field])
            if not features[key]:
                raise ValueError(f"Empty feature on {path}:{line_no}")
    return features


def _to_float_list(value: Any) -> list[float]:
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("["):
            return [float(item) for item in json.loads(value)]
        return [float(part.strip()) for part in value.replace(",", " ").split() if part.strip()]
    return [float(item) for item in value]


def _candidate_keys(image_path: str) -> list[str]:
    """给缓存读取提供几个常见路径 key 变体。"""

    path = Path(image_path)
    keys = [image_path, image_path.replace("\\", "/")]
    if path.is_absolute():
        keys.append(path.as_posix())
    return list(dict.fromkeys(keys))