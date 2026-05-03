from __future__ import annotations

from pathlib import Path
import sys
import tempfile

# 允许直接从源码树运行脚本，不要求先 pip install -e。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lg_fdc.data.mvtec_fs import MVTecFSBuildConfig, build_mvtec_fs_manifest


def main() -> None:
    """验证未解压 MVTec-FS 分卷时会得到明确报错。"""

    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir) / "MVTec-FS"
        root.mkdir()
        (root / "image.tar.001").write_bytes(b"")
        (root / "image.tar.002").write_bytes(b"")
        try:
            build_mvtec_fs_manifest(MVTecFSBuildConfig(dataset_root=root, output_csv=root / "out.csv"))
        except FileNotFoundError as exc:
            message = str(exc)
            assert "not extracted" in message
            assert "cat image.tar.* | tar -xvf -" in message
            print("mvtec-fs-unextracted-check-ok")
            return
    raise SystemExit("Expected FileNotFoundError for unextracted MVTec-FS archive parts")


if __name__ == "__main__":
    main()