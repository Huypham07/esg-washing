"""P1 - build corpus tu raw OCR zip (spec 01 #1).

Input : data/extracted/raw_ocr_annual_report.zip (59 txt, per bank/year)
Output: blocks.parquet, sentences.parquet (schema spec 01 #1)
Tai dung logic tach block/cau tu src/pipeline/document_loader.py (code cu).
"""
import pandas as pd


def build_corpus(config: dict) -> pd.DataFrame:
    """Unzip -> parse tung txt -> blocks -> sentences -> clean -> dedup -> parquet."""
    raise NotImplementedError  # TODO(Phase A2)


def corpus_stats(sentences: pd.DataFrame) -> dict:
    """Thong ke bank x year, do dai cau, ti le loc - cho done-when A2."""
    raise NotImplementedError
