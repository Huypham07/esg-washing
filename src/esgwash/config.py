"""Load + validate config YAML (single source of truth cho moi stage)."""
from pathlib import Path

import yaml

CONFIG_DIR = Path("configs")


def load_config(stage: str) -> dict:
    """Doc configs/<stage>.yml. Raise neu thieu file/khoa bat buoc."""
    path = CONFIG_DIR / f"{stage}.yml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)
