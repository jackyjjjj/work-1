from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile

# Allow running directly from the source tree without pip install -e.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from lg_fdc.data.episodes import EpisodeConfig, sample_episode
from lg_fdc.data.manifest import load_manifest
from lg_fdc.features.cached import CachedFeatureExtractor
from lg_fdc.pipelines.region_context import RegionContextPrototypeClassifier, run_region_context_episode
from run_region_context_grid import main as grid_main


def test_region_context_classifier() -> None:
    classifier = RegionContextPrototypeClassifier(whole_weight=0.25, region_weight=0.75)
    classifier.fit(
        whole_features=[[1.0, 0.0], [0.0, 1.0]],
        region_features=[[1.0, 0.0], [0.0, 1.0]],
        labels=["a", "b"],
    )
    prediction = classifier.predict_one([0.9, 0.1], [1.0, 0.0])
    assert prediction.label == "a"
    assert round(classifier.whole_weight, 2) == 0.25
    assert round(classifier.region_weight, 2) == 0.75


def test_region_context_episode_and_grid() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        manifest = tmp / "manifest.csv"
        whole_cache = tmp / "whole.jsonl"
        region_cache = tmp / "region.jsonl"
        output_json = tmp / "grid.json"
        output_md = tmp / "grid.md"

        rows = []
        whole_rows = []
        region_rows = []
        for label, feature in (("a", [1.0, 0.0]), ("b", [0.0, 1.0])):
            for idx in range(3):
                image_path = f"{label}{idx}.png"
                rows.append({"image_path": image_path, "label": label, "split": "train"})
                whole_rows.append({"image_path": image_path, "feature": feature})
                region_rows.append({"image_path": image_path, "feature": feature})
        write_csv(manifest, ["image_path", "label", "split"], rows)
        write_jsonl(whole_cache, whole_rows)
        write_jsonl(region_cache, region_rows)

        records = load_manifest(manifest)
        episode = sample_episode(records, EpisodeConfig(n_way=2, k_shot=1, q_queries=1, seed=1))
        result = run_region_context_episode(
            episode=episode,
            whole_extractor=CachedFeatureExtractor(whole_cache, feature_dim=2),
            region_extractor=CachedFeatureExtractor(region_cache, feature_dim=2),
            whole_weight=0.5,
            region_weight=0.5,
        )
        assert result.accuracy == 1.0

        argv = sys.argv
        sys.argv = [
            "run_region_context_grid.py",
            "--manifest",
            str(manifest),
            "--split",
            "train",
            "--grid",
            "2:1",
            "--q-queries",
            "1",
            "--episodes",
            "3",
            "--whole-feature-file",
            str(whole_cache),
            "--region-feature-file",
            str(region_cache),
            "--whole-feature-dim",
            "2",
            "--region-feature-dim",
            "2",
            "--whole-weights",
            "0.5",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ]
        try:
            grid_main()
        finally:
            sys.argv = argv

        summary = json.loads(output_json.read_text(encoding="utf-8"))
        assert summary["results"][0]["metrics"]["accuracy"]["mean"] == 1.0
        assert "Region-Context Prototype Results" in output_md.read_text(encoding="utf-8")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main() -> None:
    test_region_context_classifier()
    test_region_context_episode_and_grid()
    print("region-context-prototype-check-ok")


if __name__ == "__main__":
    main()
