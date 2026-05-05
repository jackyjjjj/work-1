from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile

# Allow running directly from the source tree without pip install -e.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_region_context_confusion import main


def test_confusion_cli_outputs() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        manifest = tmp / "manifest.csv"
        whole_cache = tmp / "whole.jsonl"
        region_cache = tmp / "region.jsonl"
        baseline_cache = tmp / "baseline.jsonl"
        output_json = tmp / "confusion.json"
        output_md = tmp / "confusion.md"
        output_per_class = tmp / "per_class.csv"
        output_confusion = tmp / "pairs.csv"

        rows = []
        whole_rows = []
        region_rows = []
        baseline_rows = []
        feature_by_label = {"a": [1.0, 0.0, 0.0], "b": [0.0, 1.0, 0.0], "c": [0.0, 0.0, 1.0]}
        for label, feature in feature_by_label.items():
            for idx in range(4):
                image_path = f"{label}{idx}.png"
                rows.append({"image_path": image_path, "label": label, "split": "train"})
                whole_rows.append({"image_path": image_path, "feature": feature})
                region_rows.append({"image_path": image_path, "feature": feature})
                baseline_rows.append({"image_path": image_path, "feature": feature + feature})

        write_csv(manifest, ["image_path", "label", "split"], rows)
        write_jsonl(whole_cache, whole_rows)
        write_jsonl(region_cache, region_rows)
        write_jsonl(baseline_cache, baseline_rows)

        argv = sys.argv
        sys.argv = [
            "analyze_region_context_confusion.py",
            "--manifest",
            str(manifest),
            "--split",
            "train",
            "--n-way",
            "3",
            "--k-shot",
            "1",
            "--q-queries",
            "1",
            "--episodes",
            "4",
            "--whole-feature-file",
            str(whole_cache),
            "--region-feature-file",
            str(region_cache),
            "--whole-feature-dim",
            "3",
            "--region-feature-dim",
            "3",
            "--whole-weight",
            "0.75",
            "--baseline-feature-file",
            str(baseline_cache),
            "--baseline-feature-dim",
            "6",
            "--baseline-name",
            "concat",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--output-per-class-csv",
            str(output_per_class),
            "--output-confusion-csv",
            str(output_confusion),
        ]
        try:
            main()
        finally:
            sys.argv = argv

        summary = json.loads(output_json.read_text(encoding="utf-8"))
        assert summary["region_context"]["overall"]["accuracy"] == 1.0
        assert summary["baseline"]["overall"]["accuracy"] == 1.0
        assert summary["per_class_comparison"]
        assert "Region-Context Confusion Analysis" in output_md.read_text(encoding="utf-8")
        with output_per_class.open("r", encoding="utf-8", newline="") as handle:
            assert len(list(csv.DictReader(handle))) == 3
        with output_confusion.open("r", encoding="utf-8", newline="") as handle:
            assert list(csv.DictReader(handle)) == []


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main_check() -> None:
    test_confusion_cli_outputs()
    print("region-context-confusion-check-ok")


if __name__ == "__main__":
    main_check()
