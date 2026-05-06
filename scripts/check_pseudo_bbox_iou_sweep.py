from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile

# 允许直接从源码树运行脚本，不要求先 pip install -e。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from sweep_pseudo_bbox_iou import main


def test_sweep_cli_outputs() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        manifest = tmp / "manifest.csv"
        heatmaps = tmp / "heatmaps.jsonl"
        out_json = tmp / "sweep.json"
        out_md = tmp / "sweep.md"
        out_csv = tmp / "sweep.csv"
        manifest_dir = tmp / "manifests"

        write_rows(
            manifest,
            ["image_path", "label", "split", "object_name", "defect_name", "bbox"],
            [
                {
                    "image_path": "a.png",
                    "label": "scratch",
                    "split": "train",
                    "object_name": "obj",
                    "defect_name": "scratch",
                    "bbox": "10,10,30,30",
                },
                {
                    "image_path": "b.png",
                    "label": "scratch",
                    "split": "test",
                    "object_name": "obj",
                    "defect_name": "scratch",
                    "bbox": "0,0,10,10",
                },
            ],
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
            "sweep_pseudo_bbox_iou.py",
            "--gt-manifest",
            str(manifest),
            "--heatmap-file",
            str(heatmaps),
            "--split",
            "train",
            "--percentiles",
            "0.75,0.95",
            "--min-area-ratios",
            "0.0",
            "--components",
            "max-score",
            "--upsample-heatmap-to-image",
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
            "--output-csv",
            str(out_csv),
            "--write-manifests-dir",
            str(manifest_dir),
        ]
        try:
            main()
        finally:
            sys.argv = argv

        summary = json.loads(out_json.read_text(encoding="utf-8"))
        assert len(summary["results"]) == 2
        assert summary["heatmap_processing"] == "bilinear_to_image"
        assert summary["best"]["percentile"] == 0.75
        assert 0.8 < summary["best"]["iou"]["mean"] < 0.9
        assert summary["best"]["counts"]["evaluated_rows"] == 1
        assert Path(summary["best"]["pseudo_manifest"]).exists()
        assert "upsampled" in Path(summary["best"]["pseudo_manifest"]).name
        md_text = out_md.read_text(encoding="utf-8")
        assert "Pseudo-BBox IoU Sweep" in md_text
        assert "bilinear_to_image" in md_text
        with out_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 2
        assert rows[0]["rank"] == "1"
        assert rows[0]["heatmap_processing"] == "bilinear_to_image"


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main_check() -> None:
    test_sweep_cli_outputs()
    print("pseudo-bbox-iou-sweep-check-ok")


if __name__ == "__main__":
    main_check()
