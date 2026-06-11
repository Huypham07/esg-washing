"""Xuat file cho user gan nhan: vn_eval_todo.csv (A4).

  python scripts/export_annotation.py
Can data/processed/sentences.parquet (chay stage build_corpus truoc).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from esgwash.pipeline.stages import stage_export_annotation

if __name__ == "__main__":
    stage_export_annotation()
