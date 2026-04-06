import re
import unicodedata
from pathlib import Path
import pandas as pd
from typing import Optional

try:
    from underthesea import sent_tokenize
    USE_UNDERTHESEA = True
except Exception:
    USE_UNDERTHESEA = False

INPUT_ROOT = Path("data/extracted/raw_ocr_annual_report")
OUT_BLOCKS = Path("data/corpus/blocks.parquet")
OUT_SENTS = Path("data/corpus/sentences.parquet")

MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\u00a0", " ")
    s = s.replace("…", ".")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def split_blocks(text: str) -> list[str]:
    return [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]


def clean_heading(text: str) -> str:
    t = text.strip()
    # Strip all markdown heading
    t = re.sub(r"^#{1,6}\s+", "", t, flags=re.MULTILINE)
    # Remove newlines
    t = re.sub(r"\s*\n\s*", " ", t)
    t = t.rstrip(":").strip()
    return t


def is_md_heading(block: str) -> bool:
    first_line = block.strip().split('\n')[0]
    return bool(re.match(r"^#{1,6}\s+", first_line))


def is_meta_heading(t: str) -> bool:
    return bool(re.match(r"^[A-Z]\d{2}[/\-][A-Z0-9\-]+$", t))


def is_table_like(block: str) -> bool:
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    if len(lines) < 3:
        return False
    pipe_ratio = sum(("|" in ln) for ln in lines) / len(lines)
    if pipe_ratio >= 0.5:
        return True
    colish = sum(bool(re.search(r"\S+\s{2,}\S+", ln)) for ln in lines) / len(lines)
    return colish >= 0.6


def is_bullet_like(block: str) -> bool:
    lines = [ln.lstrip() for ln in block.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False
    bullet = sum(bool(re.match(r"^(\-|\•|\♦|\+)\s+", ln)) for ln in lines)
    return bullet / len(lines) >= 0.5


def infer_block_type(block: str) -> str:
    t = block.strip()
    
    if is_md_heading(t) or (len(t) < 90 and t.isupper()):
        h = clean_heading(t)
        if is_meta_heading(h):
            return "meta_heading"
        return "heading_like"

    if is_table_like(block):
        return "table_like"

    if is_bullet_like(block):
        return "bullet_like"

    if re.search(r"%|tỷ|triệu|nghìn|tấn|kg|CO2|Scope|KPI", block, flags=re.IGNORECASE) and re.search(r"\d", block):
        return "kpi_like"

    return "paragraph"


def sent_split(block: str) -> list[str]:
    if USE_UNDERTHESEA:
        try:
            sents = [x.strip() for x in sent_tokenize(block) if x and x.strip()]
            if sents:
                return sents
        except Exception:
            pass
    parts = re.split(r"(?<=[\.\?\!])\s+", block.strip())
    return [p.strip() for p in parts if p.strip()]


def is_heading(block: str) -> bool:
    t = block.strip()
    if len(t) > 120:
        return False
    return t.isupper() or (t.endswith(":") and len(t) < 100)


NOISE_PATTERNS = [
    re.compile(r"<!\-\-\s*image\s*\-\->", re.IGNORECASE), # <!-- image -->
    re.compile(r"^\s*(trang|page)\s*\d+\s*$", re.IGNORECASE), # Page numbers
    re.compile(r"^[•\-\*\+]{1,3}\s*$"), # bullets
    re.compile(r"^[\d\.\s]+$"), # Only numbers/dots
    re.compile(r"^(\.{3,}|…+|\-{3,}|_{3,}|\*{3,})$"), # Separator lines
    re.compile(r"^\d{1,3}$"), # Page numbers
]


def is_noise_sentence(s: str) -> bool:
    stripped = s.strip()
    
    # Too short
    if len(stripped) < 15:
        return True
    
    # likely header/title
    if stripped.isupper() and len(stripped) < 80:
        return True
    
    # Noise patterns
    for pat in NOISE_PATTERNS:
        if pat.search(stripped):
            return True
    
    # Non-alphanumeric
    alpha_ratio = sum(c.isalpha() for c in stripped) / max(len(stripped), 1)
    if alpha_ratio < 0.3:
        return True
    
    return False


def _iter_txt_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".txt":
            raise ValueError(f"Input file must be .txt: {input_path}")
        return [input_path]

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    files = sorted(input_path.rglob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found under: {input_path}")
    return files


def _extract_bank_year(
    txt_path: Path,
) -> tuple[str, int]:
    bank = txt_path.parent.name
    match = re.search(r"(19|20)\d{2}", txt_path.stem)
    if match:
        return bank, int(match.group(0))

    return bank, 0


def build(
    input_path: str | Path | None = None,
    output_blocks: str | Path | None = None,
    output_sentences: str | Path | None = None
):
    input_root = Path(input_path) if input_path else INPUT_ROOT
    out_blocks = Path(output_blocks) if output_blocks else OUT_BLOCKS
    out_sents = Path(output_sentences) if output_sentences else OUT_SENTS

    block_rows = []
    sent_rows = []

    txt_files = _iter_txt_files(input_root)

    for txt_path in txt_files:
        bank_name, year_value = _extract_bank_year(txt_path)

        raw = txt_path.read_text(encoding="utf-8", errors="ignore")
        text = normalize_text(raw)
        doc_id = f"{bank_name}_{year_value}_{txt_path.stem}"

        blocks = split_blocks(text)

        current_section_title = "UNKNOWN"
        current_section_id = f"{doc_id}_sec0"
        sec_counter = 0

        for i, block in enumerate(blocks):
            btype = infer_block_type(block)

            if btype == "heading_like":
                sec_counter += 1
                current_section_title = clean_heading(block)
                current_section_id = f"{doc_id}_sec{sec_counter}"

            block_id = f"{doc_id}_b{i}"

            block_text_clean = block
            if btype in ("heading_like", "meta_heading"):
                block_text_clean = clean_heading(block)

            block_rows.append({
                "doc_id": doc_id,
                "bank": bank_name,
                "year": year_value,
                "section_id": current_section_id,
                "section_title": current_section_title,
                "block_id": block_id,
                "block_type": btype,
                "block_text": block_text_clean,
                "order_in_doc": i,
                "source_path": str(txt_path),
            })

            sents = sent_split(block_text_clean)
            sents = [s for s in sents if len(s) >= 10]
            sents = [s for s in sents if not is_noise_sentence(s)]

            for j, sent in enumerate(sents):
                prev_s = sents[j - 1] if j > 0 else ""
                next_s = sents[j + 1] if j + 1 < len(sents) else ""
                sent_rows.append({
                    "doc_id": doc_id,
                    "bank": bank_name,
                    "year": year_value,
                    "section_id": current_section_id,
                    "block_id": block_id,
                    "sent_id": f"{doc_id}_s{i}_{j}",
                    "sent_idx_in_block": j,
                    "sentence": sent,
                    "ctx_prev": prev_s,
                    "ctx_next": next_s,
                    "block_type": btype,
                    "section_title": current_section_title,
                })

    df_blocks = pd.DataFrame(block_rows)
    df_sents = pd.DataFrame(sent_rows)

    out_blocks.parent.mkdir(parents=True, exist_ok=True)
    df_blocks.to_parquet(out_blocks, index=False)
    df_sents.to_parquet(out_sents, index=False)

    print(f"Blocks: {out_blocks} ({len(df_blocks):,} rows)")
    print(f"Sentences: {out_sents} ({len(df_sents):,} rows)")
    
    print(f"\nBlock types distribution:")
    print(df_blocks["block_type"].value_counts().to_string())
    
    print(f"\nDocs processed: {df_blocks['doc_id'].nunique()}")
    print(f"Banks: {sorted(df_blocks['bank'].astype(str).unique())}")
    
    # Data quality validation (auto-run after building)
    try:
        from src.training.corpus.data_quality import run_quality_checks
        print("\n--- Running Data Quality Checks ---")
        df_checked, quality_report = run_quality_checks(
            df_sents,
            text_col="sentence",
            label_col="block_type",  # Use block_type as proxy label
            min_length=20,
            max_length=500,
        )

        cols_to_keep = [c for c in df_checked.columns if not c.startswith("_")]
        df_checked = df_checked[cols_to_keep]

        df_checked.to_parquet(out_sents, index=False)
        print(f"\nQuality summary: {quality_report['clean']}/{quality_report['total']} "
              f"sentences clean ({quality_report['clean_pct']}%)")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[Quality check skipped: {e}]")

if __name__ == "__main__":
    build()

