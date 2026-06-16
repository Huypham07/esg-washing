"""Đọc + kiểm tra config YAML (nguồn cấu hình duy nhất cho mọi bước)."""
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CONFIG_DIR = Path("configs")


def load_config(stage: str) -> dict:
    """Doc configs/<stage>.yml. Raise neu thieu file/khoa bat buoc."""
    path = CONFIG_DIR / f"{stage}.yml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)
