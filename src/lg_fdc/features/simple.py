from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any

from lg_fdc.data.manifest import ImageRecord


@dataclass
class MetadataFeatureExtractor:
    """从 manifest 的 metadata 中读取预先计算好的特征。

    这个类适合第一阶段快速跑通 prototype baseline：先离线提取 DINOv2/Alpha-CLIP
    特征，再把特征路径或特征值写进 manifest。
    """

    feature_dim: int
    key: str = "feature"

    def extract(self, record: ImageRecord) -> list[float]:
        value = record.metadata.get(self.key)
        if value is None:
            raise KeyError(f"Record {record.image_path} does not contain metadata[{self.key!r}]")
        vector = _to_float_list(value)
        if len(vector) != self.feature_dim:
            raise ValueError(
                f"Expected feature_dim={self.feature_dim}, got {len(vector)} for {record.image_path}"
            )
        return vector


@dataclass
class HashFeatureExtractor:
    """不依赖图像库的确定性占位特征提取器。

    它不会产生有研究意义的视觉特征，只用于检查数据流、episode 采样和评估代码
    是否能跑通。真实实验要替换成 DINOv2、Alpha-CLIP 或缓存特征。
    """

    feature_dim: int = 64
    salt: str = "lg_fdc"

    def extract(self, record: ImageRecord) -> list[float]:
        # 用样本路径和标签生成固定随机种子，保证每次运行得到相同占位特征。
        key = f"{self.salt}:{record.image_path}:{record.label}".encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(key).digest()[:8], byteorder="big", signed=False)
        rng = random.Random(seed)
        return [rng.uniform(-1.0, 1.0) for _ in range(self.feature_dim)]


def _to_float_list(value: Any) -> list[float]:
    """把字符串或列表形式的特征统一转成 float list。"""

    if isinstance(value, str):
        return [float(part.strip()) for part in value.split() if part.strip()]
    return [float(item) for item in value]