from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from lg_fdc.data.manifest import ImageRecord


@dataclass(frozen=True)
class Heatmap:
    """轻量异常热力图容器。

    这里先用嵌套 tuple，避免第一版就强依赖 numpy/torch。后续接入 PatchCore 或
    AnomalyDINO 时，可以在实现类内部用 tensor，输出时再转成这个统一格式或扩展它。
    """

    values: tuple[tuple[float, ...], ...]

    @property
    def height(self) -> int:
        return len(self.values)

    @property
    def width(self) -> int:
        return len(self.values[0]) if self.values else 0


class Localizer(Protocol):
    """异常定位器统一接口。

    PatchCore、AnomalyDINO、EfficientAD 或缓存 heatmap 读取器都应该实现这个接口。
    """

    def predict_heatmap(self, record: ImageRecord) -> Heatmap:
        """为一张图返回异常热力图。"""