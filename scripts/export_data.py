"""Xuat data train-ready ra thu muc PHANG (cho Kaggle): bo cot `split`.

Tu cac bang gold (co cot split chong leak) -> 2 file moi task, KHONG con subfolder:

  data/topic_train.parquet  data/topic_test.parquet
  data/claim_train.parquet  data/claim_test.parquet

`val` gop vao `train`; code train tu cat val luc chay (topic_merge.carve_val).
Bien gioi `test` giu nguyen tu split goc — KHONG split lai ngau nhien (chong leak:
topic chia theo text_en duy nhat; claim test = split chuan ClimateBERT).

  python scripts/export_data.py
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

GOLD = Path("data/processed/gold")
OUT = Path("data")

# (file goc, cot giu lai ngoai `split`, ten tien to xuat)
TABLES = {
    "topic": ("topic_labeled.parquet",
              ["text", "text_en", "env", "soc", "gov", "sources"]),
    "claim": ("claim_table.parquet",
              ["text", "text_en", "commitment", "specificity", "source"]),
}


def export_one(name: str, src_file: str, keep: list[str]) -> dict:
    df = pd.read_parquet(GOLD / src_file)
    train = df[df["split"] != "test"][keep].reset_index(drop=True)
    test = df[df["split"] == "test"][keep].reset_index(drop=True)
    train.to_parquet(OUT / f"{name}_train.parquet", index=False)
    test.to_parquet(OUT / f"{name}_test.parquet", index=False)
    return {"train": len(train), "test": len(test),
            "files": [f"data/{name}_train.parquet", f"data/{name}_test.parquet"]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(TABLES), default=None,
                    help="chi xuat 1 task (mac dinh: tat ca)")
    args = ap.parse_args(argv)

    names = [args.only] if args.only else list(TABLES)
    for name in names:
        src_file, keep = TABLES[name]
        if not (GOLD / src_file).exists():
            print(f"[bo qua] chua co {GOLD / src_file}")
            continue
        info = export_one(name, src_file, keep)
        print(f"{name}: train={info['train']} test={info['test']} -> {info['files']}")


if __name__ == "__main__":
    main()
