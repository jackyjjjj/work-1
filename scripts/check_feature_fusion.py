from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

# 允许直接从源码树运行脚本，不要求先 pip install -e。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from fuse_feature_cache import fuse_features, main


def test_fuse_features_directly() -> None:
    assert fuse_features([1, 2], [3, 4], method="concat") == [1, 2, 3, 4]
    assert fuse_features([1, 0], [0, 1], method="weighted-sum", alpha=0.25) == [0.25, 0.75]


def test_fuse_feature_cache_cli() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        whole = tmp / "whole.jsonl"
        region = tmp / "region.jsonl"
        out = tmp / "fused.jsonl"
        _write_cache(whole, region)

        argv = sys.argv
        sys.argv = [
            "fuse_feature_cache.py",
            "--whole-file",
            str(whole),
            "--region-file",
            str(region),
            "--output",
            str(out),
            "--method",
            "concat",
        ]
        try:
            main()
        finally:
            sys.argv = argv

        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 2
        assert rows[0]["feature"] == [1.0, 0.0, 0.0, 1.0]
        assert rows[0]["fusion_method"] == "concat"


def _write_cache(whole: Path, region: Path) -> None:
    whole_rows = [
        {"image_path": "a.png", "label": "a", "feature": [1, 0], "region": "whole"},
        {"image_path": "b.png", "label": "b", "feature": [0, 1], "region": "whole"},
    ]
    region_rows = [
        {"image_path": "a.png", "label": "a", "feature": [0, 1], "region": "bbox", "crop_box": [0, 0, 10, 10]},
        {"image_path": "b.png", "label": "b", "feature": [1, 0], "region": "bbox", "crop_box": [1, 1, 8, 8]},
    ]
    whole.write_text("\n".join(json.dumps(row) for row in whole_rows) + "\n", encoding="utf-8")
    region.write_text("\n".join(json.dumps(row) for row in region_rows) + "\n", encoding="utf-8")


def main_check() -> None:
    test_fuse_features_directly()
    test_fuse_feature_cache_cli()
    print("feature-fusion-check-ok")


if __name__ == "__main__":
    main_check()