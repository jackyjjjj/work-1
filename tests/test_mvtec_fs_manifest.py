from __future__ import annotations

import json
from pathlib import Path

from lg_fdc.data.manifest import load_manifest
from lg_fdc.data.mvtec_fs import MVTecFSBuildConfig, build_mvtec_fs_manifest


def test_build_mvtec_fs_manifest_from_labelme_json(tmp_path: Path) -> None:
    """用一个最小 MVTec-FS 风格目录验证 manifest 构建逻辑。"""

    root = tmp_path / "MVTec-FS"
    image_dir = root / "bottle" / "broken_large"
    image_dir.mkdir(parents=True)

    for idx in range(4):
        image_path = image_dir / f"sample_{idx}.png"
        image_path.write_bytes(b"fake-image")
        annotation = {
            "imagePath": image_path.name,
            "shapes": [
                {
                    "label": "broken_large",
                    "shape_type": "polygon",
                    "points": [[1, 2], [5, 2], [5, 7], [1, 7]],
                }
            ],
        }
        image_path.with_suffix(".json").write_text(json.dumps(annotation), encoding="utf-8")

    output = tmp_path / "mvtec_fs.csv"
    records = build_mvtec_fs_manifest(
        MVTecFSBuildConfig(dataset_root=root, output_csv=output, val_ratio=0.25, test_ratio=0.25)
    )

    assert output.exists()
    assert len(records) == 4
    loaded = load_manifest(output)
    assert len(loaded) == 4
    assert {record.label for record in loaded} == {"broken_large"}
    assert {record.object_name for record in loaded} == {"bottle"}
    assert all(record.metadata["annotation_path"] for record in loaded)
    assert all(record.metadata["bbox"] == "1.00,2.00,5.00,7.00" for record in loaded)
