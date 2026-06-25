"""Structural test for experiments/eval_gold.py — no GPU required.

Checks that BIN contains the 5 atomic flag names introduced in Task 6.
Importing eval_gold at module level must NOT trigger model loading
(load_models is called only inside main()).
"""
import sys
from pathlib import Path

# experiments/ is not a package; add it to sys.path so the import works.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import experiments.eval_gold as eg  # noqa: E402


def test_eval_gold_has_atomic_bins():
    """BIN must contain all 5 atomic flag entries."""
    names = {b[0] for b in eg.BIN}
    assert {"co_cam_ket", "co_hanh_dong_ten", "co_so_dinh_luong", "quy_ve_bank", "co_moc_tg"} <= names
