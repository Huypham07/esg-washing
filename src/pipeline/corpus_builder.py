import pandas as pd
from pathlib import Path
from training.corpus.build_corpus import (
    normalize_text,
    split_blocks,
    infer_block_type,
    clean_heading,
    sent_split,
    is_noise_sentence
)

def build_single_report(txt_path: str | Path, bank: str, year: int) -> pd.DataFrame:
    """Build a sentence-level DataFrame from a single report text file."""
    txt_path = Path(txt_path)
    if not txt_path.exists():
        raise FileNotFoundError(f"Report not found: {txt_path}")
        
    raw = txt_path.read_text(encoding="utf-8", errors="ignore")
    text = normalize_text(raw)
    doc_id = f"{bank}_{year}"
    
    blocks = split_blocks(text)
    
    current_section_title = "UNKNOWN"
    sec_counter = 0
    sent_rows = []
    
    for i, block in enumerate(blocks):
        btype = infer_block_type(block)
        if btype == "heading_like":
            sec_counter += 1
            current_section_title = clean_heading(block)
        
        block_text_clean = block
        if btype in ("heading_like", "meta_heading"):
            block_text_clean = clean_heading(block)
            
        sents = sent_split(block_text_clean)
        sents = [s for s in sents if len(s) >= 10]
        sents = [s for s in sents if not is_noise_sentence(s)]
        
        for j, sent in enumerate(sents):
            prev_s = sents[j - 1] if j > 0 else ""
            next_s = sents[j + 1] if j + 1 < len(sents) else ""
            
            sent_rows.append({
                "doc_id": doc_id,
                "bank": bank,
                "year": year,
                "sent_id": f"{doc_id}_s{i}_{j}",
                "sentence": sent,
                "ctx_prev": prev_s,
                "ctx_next": next_s,
                "block_type": btype,
                "section_title": current_section_title,
            })
            
    df = pd.DataFrame(sent_rows)
    return df
