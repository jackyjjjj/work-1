from __future__ import annotations

from pathlib import Path
import sys

# pytest 运行时把 src 加到 import path，避免测试依赖本地 editable install。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))