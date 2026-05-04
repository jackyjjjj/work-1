from __future__ import annotations

from pathlib import Path
import sys

# 允许直接从源码树运行脚本，不要求先 pip install -e。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from extract_dinov2_patch_heatmaps import (
    normalize_scores,
    parse_label_set,
    scores_to_heatmap,
    select_memory_records,
    square_grid_size,
)
from lg_fdc.data.manifest import ImageRecord


def test_normalize_scores() -> None:
    assert normalize_scores([2.0, 4.0, 6.0]) == [0.0, 0.5, 1.0]
    assert normalize_scores([3.0, 3.0]) == [0.0, 0.0]


def test_scores_to_heatmap() -> None:
    assert square_grid_size(4) == 2
    assert scores_to_heatmap([0.0, 1.0, 2.0, 3.0]) == [[0.0, 1 / 3], [2 / 3, 1.0]]
    try:
        square_grid_size(3)
    except ValueError as exc:
        assert "square grid" in str(exc)
    else:
        raise AssertionError("non-square patch count should fail")


def test_select_memory_records() -> None:
    records = [
        ImageRecord(image_path="a.png", label="good", split="train", object_name="capsule", defect_name="good"),
        ImageRecord(image_path="b.png", label="scratch", split="train", object_name="capsule", defect_name="scratch"),
        ImageRecord(image_path="c.png", label="good", split="test", object_name="capsule", defect_name="good"),
    ]
    assert parse_label_set("good, normal") == {"good", "normal"}
    assert parse_label_set("all") is None
    selected = select_memory_records(records, split="train", label_set={"good"}, max_images=10, seed=42)
    assert [record.image_path for record in selected] == ["a.png"]
    all_train = select_memory_records(records, split="train", label_set=None, max_images=10, seed=42)
    assert [record.image_path for record in all_train] == ["a.png", "b.png"]


def main() -> None:
    test_normalize_scores()
    test_scores_to_heatmap()
    test_select_memory_records()
    print("dinov2-patch-heatmap-check-ok")


if __name__ == "__main__":
    main()
