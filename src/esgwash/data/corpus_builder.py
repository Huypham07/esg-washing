"""P1 — dựng corpus từ raw OCR zip (spec 01 #1).

Input : source_data/raw_ocr_annual_report.zip (txt từ docling, theo bank/year)
Output: blocks.parquet + sentences.parquet
Schema sentences: doc_id, bank, year, section_id, block_id, sent_id, sentence,
                  ctx_prev, ctx_next, block_type, section_title
"""
from __future__ import annotations

import io
import itertools
import re
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

from esgwash.data.cleaning import (clean_raw_text, dedup_per_doc, is_noise_line,
                                   is_valid_sentence)
from esgwash.nlp.segmentation import sent_tokenize

FILE_RE = re.compile(r"raw_ocr_annual_report/([a-z_]+)/[^/]*?(\d{4})[^/]*\.txt$")
HEADING_RE = re.compile(r"^#{1,6}\s+(.*)")
LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+•▪]|\d{1,2}[.)])\s+")
TABLE_LINE_RE = re.compile(r"^\s*\|")


def _is_table_line(l: str) -> bool:
    l = l.strip()
    return l.count("|") >= 2 or l.startswith("|") or l.endswith("|")


def _block_type(lines: list[str]) -> str:
    if HEADING_RE.match(lines[0]):
        return "heading"
    if all(_is_table_line(l) for l in lines):
        return "table"
    if sum(bool(LIST_ITEM_RE.match(l)) for l in lines) >= max(1, len(lines) // 2):
        return "list"
    return "paragraph"


def _table_rows_to_texts(lines: list[str]) -> list[str]:
    texts = []
    for l in lines:
        if re.match(r"^\s*\|[\s\-|:]*$", l):
            continue
        cells = [c.strip() for c in l.strip().strip("|").split("|")]
        row = " | ".join(c for c in cells if c)
        if row:
            texts.append(row)
    return texts


def _parse_doc(text: str, bank: str, year: int, freq_lines: set[str]) -> tuple[list, list]:
    doc_id = f"{bank}_{year}"
    blocks, sentences = [], []
    section_id, section_title = 0, ""
    block_id = sent_id = 0

    raw_blocks = re.split(r"\n\s*\n", clean_raw_text(text))
    sub_blocks = []
    for rb in raw_blocks:
        lines = [l for l in rb.splitlines()
                 if not is_noise_line(l) and l.strip() not in freq_lines]
        if not lines:
            continue
        # block hon hop: tach run dong bang ra khoi dong thuong
        for is_tbl, grp in itertools.groupby(lines, key=_is_table_line):
            sub_blocks.append(list(grp))

    for lines in sub_blocks:
        btype = _block_type(lines)
        if btype == "heading":
            section_id += 1
            section_title = HEADING_RE.match(lines[0]).group(1).strip()
            continue

        block_text = "\n".join(lines)
        blocks.append({"doc_id": doc_id, "bank": bank, "year": year,
                       "section_id": section_id, "block_id": block_id,
                       "block_type": btype, "section_title": section_title,
                       "text": block_text})

        if btype == "table":
            sents = _table_rows_to_texts(lines)
        elif btype == "list":
            sents = [LIST_ITEM_RE.sub("", l).strip() for l in lines]
        else:
            sents = [s for chunk in lines for s in sent_tokenize(chunk)]
        sents = [re.sub(r"\s+", " ", s).strip() for s in sents]
        sents = [s for s in sents if is_valid_sentence(s)]

        for i, s in enumerate(sents):
            sentences.append({
                "doc_id": doc_id, "bank": bank, "year": year,
                "section_id": section_id, "block_id": block_id, "sent_id": sent_id,
                "sentence": s,
                "ctx_prev": sents[i - 1] if i > 0 else "",
                "ctx_next": sents[i + 1] if i + 1 < len(sents) else "",
                "block_type": btype, "section_title": section_title,
            })
            sent_id += 1
        block_id += 1
    return blocks, sentences


def _frequent_lines(text: str, min_count: int = 8) -> set[str]:
    """Header/footer lap lai nhieu lan trong cung doc (ten bank, 'BAO CAO THUONG NIEN'...)."""
    counts = Counter(l.strip() for l in text.splitlines() if l.strip())
    return {l for l, c in counts.items() if c >= min_count and len(l) < 80}


def build_corpus(config: dict) -> pd.DataFrame:
    zip_path = Path(config["raw_zip"])
    all_blocks, all_sents = [], []

    with zipfile.ZipFile(zip_path) as zf:
        for name in sorted(zf.namelist()):
            if "__MACOSX" in name or not name.endswith(".txt"):
                continue
            m = FILE_RE.search(name)
            if not m:
                continue
            bank, year = m.group(1), int(m.group(2))
            text = io.TextIOWrapper(zf.open(name), encoding="utf-8", errors="replace").read()
            freq = _frequent_lines(text)
            blocks, sents = _parse_doc(text, bank, year, freq)
            all_blocks.extend(blocks)
            all_sents.extend(sents)

    blocks_df = pd.DataFrame(all_blocks)
    sents_df = pd.DataFrame(all_sents)
    if config.get("dedup", {}).get("exact", True):
        near = config.get("dedup", {}).get("near_dup_minhash", True)
        sents_df = dedup_per_doc(sents_df, near_dup=near)

    out_blocks = Path(config["out_blocks"])
    out_blocks.parent.mkdir(parents=True, exist_ok=True)
    blocks_df.to_parquet(out_blocks, index=False)
    sents_df.to_parquet(config["out_sentences"], index=False)
    return sents_df


def corpus_stats(sentences: pd.DataFrame) -> dict:
    by_doc = sentences.groupby(["bank", "year"]).size()
    return {
        "n_sentences": int(len(sentences)),
        "n_docs": int(sentences["doc_id"].nunique()),
        "banks": sorted(sentences["bank"].unique().tolist()),
        "years": sorted(sentences["year"].unique().tolist()),
        "sentences_per_doc": {f"{b}_{y}": int(n) for (b, y), n in by_doc.items()},
        "block_type_share": sentences["block_type"].value_counts(normalize=True).round(3).to_dict(),
        "median_chars": float(sentences["sentence"].str.len().median()),
    }
