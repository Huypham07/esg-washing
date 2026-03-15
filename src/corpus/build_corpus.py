import re
import unicodedata
from pathlib import Path
import pandas as pd

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


def build():
    block_rows = []
    sent_rows = []

    # 10 banks
    VALID_BANKS = {
        "agribank", "bidv", "bsc", "mbbank", "shb",
        "techcombank", "ocb", "vietcombank", "viettinbank", "vpbank",
    }

    # 5 years
    VALID_YEARS = {2020, 2021, 2022, 2023, 2024}

    for bank_dir in sorted(INPUT_ROOT.glob("*")):
        if not bank_dir.is_dir():
            continue
        bank = bank_dir.name
        
        if bank not in VALID_BANKS:
            continue

        for txt_path in sorted(bank_dir.glob("bctn_*_raw.txt")):
            m = re.search(r"bctn_(\d{4})_raw\.txt", txt_path.name)
            if not m:
                continue
            year = int(m.group(1))
            
            if year not in VALID_YEARS:
                continue

            raw = txt_path.read_text(encoding="utf-8", errors="ignore")
            text = normalize_text(raw)
            doc_id = f"{bank}_{year}"

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
                
                # Clean block_text: strip markdown heading
                block_text_clean = block
                if btype in ("heading_like", "meta_heading"):
                    block_text_clean = clean_heading(block)
                
                block_rows.append({
                    "doc_id": doc_id,
                    "bank": bank,
                    "year": year,
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
                
                # Filter noise sentences
                sents = [s for s in sents if not is_noise_sentence(s)]
                
                for j, sent in enumerate(sents):
                    prev_s = sents[j - 1] if j > 0 else ""
                    next_s = sents[j + 1] if j + 1 < len(sents) else ""
                    sent_rows.append({
                        "doc_id": doc_id,
                        "bank": bank,
                        "year": year,
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

    OUT_BLOCKS.parent.mkdir(parents=True, exist_ok=True)
    df_blocks.to_parquet(OUT_BLOCKS, index=False)
    df_sents.to_parquet(OUT_SENTS, index=False)

    print(f"Blocks: {OUT_BLOCKS} ({len(df_blocks):,} rows)")
    print(f"Sentences: {OUT_SENTS} ({len(df_sents):,} rows)")
    
    print(f"\nBlock types distribution:")
    print(df_blocks["block_type"].value_counts().to_string())
    
    print(f"\nDocs processed: {df_blocks['doc_id'].nunique()}")
    print(f"Banks: {sorted(df_blocks['bank'].unique())}")


if __name__ == "__main__":
    build()
