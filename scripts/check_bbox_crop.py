from __future__ import annotations

from pathlib import Path
import sys

# 允许直接从源码树运行脚本，不要求先 pip install -e。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from extract_dinov2_features import compute_padded_crop_box, parse_bbox


def main() -> None:
    """验证 bbox 解析和 padding crop 逻辑。"""

    assert parse_bbox("") is None
    assert parse_bbox("10,20,30,50") == (10.0, 20.0, 30.0, 50.0)

    # 普通 bbox：加 10% padding 后应该围绕中心扩展。
    box = compute_padded_crop_box("10,20,30,50", image_size=(100, 100), padding=0.1, min_crop_size=1)
    assert box == (8, 17, 32, 53), box

    # 靠近左上边界时，crop 应该自动平移到图像内。
    edge_box = compute_padded_crop_box("0,0,5,5", image_size=(20, 20), padding=0.2, min_crop_size=10)
    assert edge_box == (0, 0, 10, 10), edge_box

    # 空 bbox 表示没有可靠区域，退化为整图。
    full_box = compute_padded_crop_box("", image_size=(64, 48), padding=0.15, min_crop_size=32)
    assert full_box == (0, 0, 64, 48), full_box

    print("bbox-crop-check-ok")


if __name__ == "__main__":
    main()