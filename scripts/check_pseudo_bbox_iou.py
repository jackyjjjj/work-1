from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile

# 允许直接从源码树运行脚本，不要求先 pip install -e。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_pseudo_bbox_iou import bbox_iou, main, parse_bbox


def test_bbox_math() -> None:
    assert parse_bbox("0,0,10,10") == (0.0, 0.0, 10.0, 10.0)
    assert parse_bbox("") is None
    assert parse_bbox("0,0,0,10") is None
    assert bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert bbox_iou((0, 0, 10, 10), (10, 10, 20, 20)) == 0.0
    assert round(bbox_iou((0, 0, 10, 10), (5, 5, 15, 15)), 4) == 0.1429


def test_iou_cli_outputs() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        gt = tmp / "gt.csv"
        pseudo = tmp / "pseudo.csv"
        out_json = tmp / "iou.json"
        out_md = tmp / "iou.md"
        out_csv = tmp / "per_image.csv"
        write_rows(
            gt,
            ["image_path", "label", "split", "object_name", "defect_name", "bbox"],
            [
                {"image_path": "a.png", "label": "scratch", "split": "train", "object_name": "obj", "defect_name": "scratch", "bbox": "0,0,10,10"},
                {"image_path": "b.png", "label": "scratch", "split": "test", "object_name": "obj", "defect_name": "scratch", "bbox": "0,0,10,10"},
            ],
        )
        write_rows(
            pseudo,
            ["image_path", "label", "split", "bbox", "bbox_source", "pseudo_bbox_score", "pseudo_bbox_area"],
            [
                {"image_path": "a.png", "label": "scratch", "split": "train", "bbox": "5,5,15,15", "bbox_source": "pseudo_heatmap", "pseudo_bbox_score": "0.9", "pseudo_bbox_area": "4"},
            ],
        )

        argv = sys.argv
        sys.argv = [
            "evaluate_pseudo_bbox_iou.py",
            "--gt-manifest",
            str(gt),
            "--pseudo-manifest",
            str(pseudo),
            "--split",
            "train",
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
            "--output-csv",
            str(out_csv),
        ]
        try:
            main()
        finally:
            sys.argv = argv

        summary = json.loads(out_json.read_text(encoding="utf-8"))
        assert summary["counts"]["gt_rows"] == 1
        assert summary["counts"]["matched_rows"] == 1
        assert round(summary["iou"]["mean"], 4) == 0.1429
        assert "Pseudo-BBox IoU Diagnosis" in out_md.read_text(encoding="utf-8")
        with out_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 1
        assert rows[0]["image_path"] == "a.png"


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main_check() -> None:
    test_bbox_math()
    test_iou_cli_outputs()
    print("pseudo-bbox-iou-check-ok")


if __name__ == "__main__":
    main_check()
