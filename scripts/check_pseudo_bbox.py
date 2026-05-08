from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile

# 允许直接从源码树运行脚本，不要求先 pip install -e。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_pseudo_bbox_manifest import filter_rows_by_split, heatmap_to_bbox, load_heatmap_cache, main, resize_heatmap_bilinear


def test_heatmap_to_bbox_directly() -> None:
    heatmap = [
        [0.0, 0.1, 0.0, 0.0],
        [0.0, 0.8, 0.9, 0.0],
        [0.0, 0.7, 0.6, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    result = heatmap_to_bbox(
        values=heatmap,
        image_width=40,
        image_height=40,
        percentile=0.75,
        min_area_ratio=0.01,
        component="max-score",
    )
    assert result["bbox"] == (10.0, 10.0, 30.0, 30.0), result
    assert result["area"] == 4, result


def test_heatmap_upsample_to_image() -> None:
    heatmap = [
        [0.0, 0.0],
        [0.0, 1.0],
    ]
    resized = resize_heatmap_bilinear(heatmap, target_width=4, target_height=4)
    assert len(resized) == 4
    assert len(resized[0]) == 4
    assert resized[0][0] == 0.0
    assert resized[-1][-1] == 1.0

    native = heatmap_to_bbox(
        values=heatmap,
        image_width=4,
        image_height=4,
        percentile=1.0,
        min_area_ratio=0.0,
        component="max-score",
    )
    upsampled = heatmap_to_bbox(
        values=heatmap,
        image_width=4,
        image_height=4,
        percentile=1.0,
        min_area_ratio=0.0,
        component="max-score",
        upsample_to_image=True,
    )
    assert native["heatmap_processing"] == "native_grid"
    assert upsampled["heatmap_processing"] == "bilinear_to_image"
    assert native["bbox"] == (2.0, 2.0, 4, 4), native
    assert upsampled["bbox"] == (3.0, 3.0, 4, 4), upsampled


def test_duplicate_heatmap_policy() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        heatmaps = Path(tmp_dir) / "heatmaps.jsonl"
        heatmaps.write_text(
            json.dumps(
                {
                    "image_path": "a.png",
                    "image_width": 4,
                    "image_height": 4,
                    "heatmap": [[0.0, 0.2], [0.1, 0.0]],
                }
            )
            + "\n"
            + json.dumps(
                {
                    "image_path": "a.png",
                    "image_width": 4,
                    "image_height": 4,
                    "heatmap": [[0.3, 0.1], [0.4, 0.5]],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        cache = load_heatmap_cache(heatmaps)
        assert cache["a.png"]["heatmap"] == [[0.3, 0.2], [0.4, 0.5]]
        assert cache["a.png"]["_duplicate_count"] == 2
        try:
            load_heatmap_cache(heatmaps, duplicate_policy="error")
        except ValueError as exc:
            assert "Duplicate image_path" in str(exc)
        else:
            raise AssertionError("duplicate-policy error should fail on repeated image_path")

def test_build_pseudo_bbox_manifest_cli() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        manifest = tmp / "manifest.csv"
        heatmaps = tmp / "heatmaps.jsonl"
        output = tmp / "pseudo.csv"
        manifest.write_text(
            "image_path,label,split,mask_path,object_name,defect_name,annotation_path,bbox,polygon_count\n"
            "a.png,scratch,train,,obj,scratch,,,0\n",
            encoding="utf-8",
        )
        heatmaps.write_text(
            json.dumps(
                {
                    "image_path": "a.png",
                    "image_width": 40,
                    "image_height": 40,
                    "heatmap": [
                        [0.0, 0.1, 0.0, 0.0],
                        [0.0, 0.8, 0.9, 0.0],
                        [0.0, 0.7, 0.6, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )

        argv = sys.argv
        sys.argv = [
            "build_pseudo_bbox_manifest.py",
            "--manifest",
            str(manifest),
            "--heatmap-file",
            str(heatmaps),
            "--output",
            str(output),
            "--percentile",
            "0.75",
        ]
        try:
            main()
        finally:
            sys.argv = argv

        with output.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 1
        assert rows[0]["bbox"] == "10.00,10.00,30.00,30.00"
        assert rows[0]["bbox_source"] == "pseudo_heatmap"
        assert rows[0]["pseudo_bbox_area"] == "4"


def test_build_pseudo_bbox_manifest_upsampled_cli() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        manifest = tmp / "manifest.csv"
        heatmaps = tmp / "heatmaps.jsonl"
        output = tmp / "pseudo_upsampled.csv"
        manifest.write_text(
            "image_path,label,split,mask_path,object_name,defect_name,annotation_path,bbox,polygon_count\n"
            "a.png,scratch,train,,obj,scratch,,,0\n",
            encoding="utf-8",
        )
        heatmaps.write_text(
            json.dumps(
                {
                    "image_path": "a.png",
                    "image_width": 4,
                    "image_height": 4,
                    "heatmap": [[0.0, 0.0], [0.0, 1.0]],
                }
            )
            + "\n",
            encoding="utf-8",
        )

        argv = sys.argv
        sys.argv = [
            "build_pseudo_bbox_manifest.py",
            "--manifest",
            str(manifest),
            "--heatmap-file",
            str(heatmaps),
            "--output",
            str(output),
            "--percentile",
            "1.0",
            "--upsample-heatmap-to-image",
        ]
        try:
            main()
        finally:
            sys.argv = argv

        with output.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 1
        assert rows[0]["bbox"] == "3.00,3.00,4.00,4.00"
        assert rows[0]["bbox_source"] == "pseudo_heatmap_upsampled"
        assert rows[0]["pseudo_bbox_heatmap_processing"] == "bilinear_to_image"



def test_missing_heatmap_policy() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        manifest = tmp / "manifest.csv"
        heatmaps = tmp / "heatmaps.jsonl"
        output = tmp / "pseudo.csv"
        manifest.write_text(
            "image_path,label,split,mask_path,object_name,defect_name,annotation_path,bbox,polygon_count\n"
            "a.png,scratch,train,,obj,scratch,,1,1,2,2,0\n",
            encoding="utf-8",
        )
        heatmaps.write_text(
            json.dumps({"image_path": "b.png", "image_width": 4, "image_height": 4, "heatmap": [[0.0, 1.0]]})
            + "\n",
            encoding="utf-8",
        )

        argv = sys.argv
        sys.argv = [
            "build_pseudo_bbox_manifest.py",
            "--manifest",
            str(manifest),
            "--heatmap-file",
            str(heatmaps),
            "--output",
            str(output),
        ]
        try:
            try:
                main()
            except ValueError as exc:
                assert "Missing heatmaps" in str(exc)
            else:
                raise AssertionError("missing heatmap should fail by default")
        finally:
            sys.argv = argv


def test_split_filtering() -> None:
    rows = [
        {"image_path": "train.png", "split": "train"},
        {"image_path": "test.png", "split": "test"},
    ]
    assert [row["image_path"] for row in filter_rows_by_split(rows, "train")] == ["train.png"]
    assert [row["image_path"] for row in filter_rows_by_split(rows, "all")] == ["train.png", "test.png"]


def test_build_pseudo_bbox_manifest_split_cli() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        manifest = tmp / "manifest.csv"
        heatmaps = tmp / "heatmaps.jsonl"
        output = tmp / "pseudo_train.csv"
        manifest.write_text(
            "image_path,label,split,mask_path,object_name,defect_name,annotation_path,bbox,polygon_count\n"
            "a.png,scratch,train,,obj,scratch,,,0\n"
            "b.png,scratch,test,,obj,scratch,,,0\n",
            encoding="utf-8",
        )
        heatmaps.write_text(
            json.dumps(
                {
                    "image_path": "a.png",
                    "image_width": 40,
                    "image_height": 40,
                    "heatmap": [
                        [0.0, 0.1, 0.0, 0.0],
                        [0.0, 0.8, 0.9, 0.0],
                        [0.0, 0.7, 0.6, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )

        argv = sys.argv
        sys.argv = [
            "build_pseudo_bbox_manifest.py",
            "--manifest",
            str(manifest),
            "--heatmap-file",
            str(heatmaps),
            "--output",
            str(output),
            "--split",
            "train",
            "--percentile",
            "0.75",
        ]
        try:
            main()
        finally:
            sys.argv = argv

        with output.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 1
        assert rows[0]["image_path"] == "a.png"
        assert rows[0]["bbox_source"] == "pseudo_heatmap"


def main_check() -> None:
    test_heatmap_to_bbox_directly()
    test_heatmap_upsample_to_image()
    test_duplicate_heatmap_policy()
    test_build_pseudo_bbox_manifest_cli()
    test_build_pseudo_bbox_manifest_upsampled_cli()
    test_split_filtering()
    test_build_pseudo_bbox_manifest_split_cli()
    test_missing_heatmap_policy()
    print("pseudo-bbox-check-ok")


if __name__ == "__main__":
    main_check()