"""Nhánh A — corpus sạch.

Tái dùng `is_noise_sentence` (đã siết) trên `sentences.parquet` có sẵn (đã NFC +
OCR-repair). Giữ THÔ (không tách từ — pipeline segment ở runtime).

Bước explode: nhiều khối bullet/target bị mis-split thành 1 "câu" chứa `\n`
(vd "- Tổng tài sản: tăng 7-9%\n- LNTT: tăng ≥10%..."). Đây là CAM KẾT CỤ THỂ
(tín hiệu CTI) → tách trên `\n` thành câu riêng để GIỮ, thay vì drop cả khối.
Dòng bảng (chứa `|`) sau tách vẫn bị filter loại.

Chạy:  python -m src.training.data_prep.prepare_clean_corpus
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.training.corpus.build_corpus import is_noise_sentence

IN_PATH = Path("data/corpus/sentences.parquet")
OUT_PATH = Path("data/corpus/sentences_clean.parquet")
_BULLET_PREFIX = r"^\s*[-–—•»*+▪◦·]+\s*"


def clean_corpus(in_path: Path = IN_PATH, out_path: Path = OUT_PATH) -> pd.DataFrame:
    df = pd.read_parquet(in_path)
    n0 = len(df)
    print(f"Input:                 {n0:,} sentences")

    # 0) Explode khối nhiều dòng (bullet/target list) trên '\n' -> giữ cam kết bullet
    df["sentence"] = df["sentence"].astype(str).str.split("\n")
    df = df.explode("sentence")
    df["sentence"] = df["sentence"].str.replace(_BULLET_PREFIX, "", regex=True).str.strip()
    df = df[df["sentence"].str.len() > 0].reset_index(drop=True)
    print(f"After explode '\\n':     {len(df):,} candidate sentences")

    # 1) Lọc noise (pipe nhúng, blob, digit-ratio, all-caps heading, section tài chính...)
    noise_mask = df.apply(
        lambda r: is_noise_sentence(str(r["sentence"]), section_title=str(r.get("section_title", ""))),
        axis=1,
    )
    df = df[~noise_mask].copy()
    print(f"After noise filter:     {len(df):,}")

    # 2) Khử trùng lặp chính xác trên câu (gỡ boilerplate lặp)
    df = df.drop_duplicates(subset=["sentence"]).reset_index(drop=True)
    # sent_id duy nhất — FLAT per-doc counter (KHÔNG còn mã hoá block/sent index như build()).
    df["sent_id"] = df["doc_id"].astype(str) + "_c" + df.groupby("doc_id").cumcount().astype(str)
    # ctx_prev/ctx_next stale sau explode (trỏ câu cũ, còn chứa '\n') -> bỏ.
    # Gold mới không có ctx -> classifier train sentence-only; infer cũng sentence-only cho nhất quán.
    df = df.drop(columns=["ctx_prev", "ctx_next"], errors="ignore")
    print(f"After dedup:            {len(df):,}  (giữ {100 * len(df) / n0:.1f}% so input)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved -> {out_path}")

    if {"bank", "year"}.issubset(df.columns):
        bt = df.groupby(["bank", "year"]).size()
        print(f"bank-year cells: {len(bt)} | per-cell min {bt.min()} median {int(bt.median())} max {bt.max()}")
    ln = df["sentence"].str.len()
    print(f"char length: median {int(ln.median())} | max {ln.max()}")
    return df


if __name__ == "__main__":
    clean_corpus()
