from __future__ import annotations

from typing import Protocol

from lg_fdc.data.manifest import ImageRecord


class FeatureExtractor(Protocol):
    """特征提取器统一接口。

    后续 DINOv2、Alpha-CLIP、CLIP、缓存特征都会实现这个接口，分类器只关心
    ``extract(record)`` 返回的向量，不关心底层模型细节。
    """

    feature_dim: int

    def extract(self, record: ImageRecord) -> list[float]:
        """为一条样本返回一个特征向量。"""