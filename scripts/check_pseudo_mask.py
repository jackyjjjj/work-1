from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile

try:
    from PIL import Image
except ImportError as exc:
    raise SystemExit("pseudo-mask checks require Pillow. Install research dependencies first.") from exc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_pseudo_bbox_manifest import filter_rows_by_split
from build_pseudo_mask_manifest import heatmap_to_mask, main
from extract_dinov2_features import apply_mask_crop


def test_heatmap_to_mask_directly() -> None:
    heatmap = [
        [0.0, 0.1, 0.0, 0.0],
        [0.0, 0.8, 0.9, 0.0],
        [0.0, 0.7, 0.6, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    result = heatmap_to_mask(
        values=heatmap,
        image_width=40,
        image_height=40,
        percentile=0.75,
        min_area_ratio=0.01,
        component="max-score",
    )
    assert result["bbox"] == (10.0, 10.0, 30.0, 30.0), result
    assert result["area"] == 400, result
    assert result["heatmap_processing"] == "native_grid"


def test_heatmap_to_mask_upsampled() -> None:
    heatmap = [
        [0.0, 0.0],
        [0.0, 1.0],
    ]
    native = heatmap_to_mask(
        values=heatmap,
        image_width=4,
        image_height=4,
        percentile=1.0,
        min_area_ratio=0.0,
        component="max-score",
    )
    upsampled = heatmap_to_mask(
        values=heatmap,
        image_width=4,
        image_height=4,
        percentile=1.0,
        min_area_ratio=0.0,
        component="max-score",
        upsample_to_image=True,
    )
    assert native["bbox"] == (2.0, 2.0, 4.0, 4.0), native
    assert native["area"] == 4, native
    assert native["heatmap_processing"] == "native_grid"
    assert upsampled["bbox"] == (3.0, 3.0, 4.0, 4.0), upsampled
    assert upsampled["area"] == 1, upsampled
    assert upsampled["heatmap_processing"] == "bilinear_to_image"


def test_build_pseudo_mask_manifest_cli() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        manifest = tmp / "manifest.csv"
        heatmaps = tmp / "heatmaps.jsonl"
        output = tmp / "pseudo_mask.csv"
        mask_dir = tmp / "masks"
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
            "build_pseudo_mask_manifest.py",
            "--manifest",
            str(manifest),
            "--heatmap-file",
            str(heatmaps),
            "--mask-dir",
            str(mask_dir),
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
        assert rows[0]["bbox_source"] == "pseudo_mask_tight_box"
        assert rows[0]["mask_source"] == "pseudo_heatmap_mask"
        assert rows[0]["pseudo_mask_area"] == "400"
        mask_path = Path(rows[0]["mask_path"])
        assert mask_path.exists(), rows[0]
        with Image.open(mask_path) as mask_image:
            assert mask_image.size == (40, 40)
            assert mask_image.getbbox() == (10, 10, 30, 30)


def test_build_pseudo_mask_manifest_upsampled_cli() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        manifest = tmp / "manifest.csv"
        heatmaps = tmp / "heatmaps.jsonl"
        output = tmp / "pseudo_mask_upsampled.csv"
        mask_dir = tmp / "masks"
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
            "build_pseudo_mask_manifest.py",
            "--manifest",
            str(manifest),
            "--heatmap-file",
            str(heatmaps),
            "--mask-dir",
            str(mask_dir),
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
        assert rows[0]["bbox_source"] == "pseudo_mask_tight_box_upsampled"
        assert rows[0]["pseudo_mask_heatmap_processing"] == "bilinear_to_image"
        mask_path = Path(rows[0]["mask_path"])
        assert mask_path.exists(), rows[0]
        with Image.open(mask_path) as mask_image:
            assert mask_image.getbbox() == (3, 3, 4, 4)


def test_missing_heatmap_policy() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        manifest = tmp / "manifest.csv"
        heatmaps = tmp / "heatmaps.jsonl"
        output = tmp / "pseudo_mask.csv"
        mask_dir = tmp / "masks"
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
            "build_pseudo_mask_manifest.py",
            "--manifest",
            str(manifest),
            "--heatmap-file",
            str(heatmaps),
            "--mask-dir",
            str(mask_dir),
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


def test_mask_crop_helpers() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        image_path = tmp / "image.png"
        mask_path = tmp / "mask.png"

        image = Image.new("RGB", (6, 6), color=(10, 20, 30))
        image.save(image_path)

        mask = Image.new("L", (6, 6), color=0)
        for y in (1, 2):
            for x in (2, 3):
                mask.putpixel((x, y), 255)
        mask.save(mask_path)

        loaded_image = Image.open(image_path).convert("RGB")
        try:
            black_crop, black_box = apply_mask_crop(
                image=loaded_image,
                mask_path=str(mask_path),
                image_root=tmp,
                image_cls=Image,
                padding=0.5,
                min_crop_size=1,
                background_mode="black",
            )
        finally:
            loaded_image.close()
        assert black_box == (1, 0, 5, 4)
        assert black_crop.size == (4, 4)
        assert black_crop.getpixel((0, 0)) == (0, 0, 0)
        assert black_crop.getpixel((1, 1)) == (10, 20, 30)

        loaded_image = Image.open(image_path).convert("RGB")
        try:
            keep_crop, keep_box = apply_mask_crop(
                image=loaded_image,
                mask_path=str(mask_path),
                image_root=tmp,
                image_cls=Image,
                padding=0.5,
                min_crop_size=1,
                background_mode="keep",
            )
        finally:
            loaded_image.close()
        assert keep_box == (1, 0, 5, 4)
        assert keep_crop.size == (4, 4)
        assert keep_crop.getpixel((0, 0)) == (10, 20, 30)


def main_check() -> None:
    test_heatmap_to_mask_directly()
    test_heatmap_to_mask_upsampled()
    test_build_pseudo_mask_manifest_cli()
    test_build_pseudo_mask_manifest_upsampled_cli()
    test_missing_heatmap_policy()
    test_split_filtering()
    test_mask_crop_helpers()
    print("pseudo-mask-check-ok")


if __name__ == "__main__":
    main_check()
