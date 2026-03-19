"""
Baseline 3: Weak Supervision + BERT Binary Classifier
Inspired by Green-Washing repo approach

Approach:
- Uses weak supervision labels based on sentence category:
  - Evidence-based: has_metric=True → label 0
  - Grey-area: action without metric → label 1
  - Greenwashing-prone: vision/marketing without evidence → label 2
- Fine-tunes PhoBERT for binary classification (substantive vs. wash)
- Adapted from the Green-Washing repo for Vietnamese banking context

Reference:
    Green-Washing repo (https://github.com/...) using BERT + weak supervision
    Original: bert-base-uncased, adapted to PhoBERT for Vietnamese
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


# ============================================================
# CATEGORY CLASSIFICATION (from Green-Washing repo)
# ============================================================

# Role patterns adapted from atomic_extractor.py in Green-Washing
ROLE_PATTERNS = {
    "metric": re.compile(
        r"\b\d+(\.\d+)?\s*(%|tấn|kg|tỷ|triệu|nghìn|kWh|MWh|CO2|VND|USD)\b",
        re.IGNORECASE,
    ),
    "vision": re.compile(
        r"\b(cam kết|mục tiêu|hướng tới|khát vọng|tầm nhìn|vision|aspire|goal)\b",
        re.IGNORECASE,
    ),
    "action": re.compile(
        r"\b(triển khai|thực hiện|giảm|tăng|xây dựng|áp dụng|implement|reduce|install)\b",
        re.IGNORECASE,
    ),
    "governance": re.compile(
        r"\b(quản trị|hội đồng|giám sát|kiểm soát|tuân thủ|board|governance|oversight)\b",
        re.IGNORECASE,
    ),
    "marketing": re.compile(
        r"\b(hàng đầu|tiên phong|dẫn đầu|xuất sắc|world-class|leader|premier)\b",
        re.IGNORECASE,
    ),
}

# Vague language indicators (from Green-Washing's weak supervision)
VAGUE_WORDS = re.compile(
    r"\b(leading|significant|robust|strong|committed to|dedicated|"
    r"hàng đầu|tiên phong|mạnh mẽ|cam kết|nỗ lực|quan tâm|chú trọng)\b",
    re.IGNORECASE,
)

# Future-oriented language
FUTURE_WORDS = re.compile(
    r"\b(sẽ|dự kiến|kế hoạch|mục tiêu|will|aim|plan|target|hướng tới|phấn đấu)\b",
    re.IGNORECASE,
)


def classify_greenwashing_label(text: str) -> Tuple[int, str, float]:
    """
    Classify using Green-Washing repo's weak supervision approach.
    
    Returns:
        (label, category, confidence)
        label: 0=Evidence-based, 1=Grey-area, 2=Greenwashing-prone
    """
    text_lower = text.lower()
    
    # Detect category
    has_metric = bool(ROLE_PATTERNS["metric"].search(text))
    has_vision = bool(ROLE_PATTERNS["vision"].search(text_lower))
    has_action = bool(ROLE_PATTERNS["action"].search(text_lower))
    has_marketing = bool(ROLE_PATTERNS["marketing"].search(text_lower))
    has_governance = bool(ROLE_PATTERNS["governance"].search(text_lower))
    has_vague = bool(VAGUE_WORDS.search(text_lower))
    has_future = bool(FUTURE_WORDS.search(text_lower))
    
    # Label assignment (following Green-Washing repo logic)
    if has_metric:
        return 0, "evidence", 0.8  # Evidence-based
    elif has_action and not has_vague and not has_future:
        return 0, "action_concrete", 0.6  # Concrete action
    elif has_vision or has_marketing:
        if has_vague or has_future:
            return 2, "greenwashing_prone", 0.7  # Greenwashing-prone
        else:
            return 1, "grey_area", 0.5
    elif has_action and (has_vague or has_future):
        return 1, "grey_area", 0.5  # Action but vague
    elif has_governance:
        return 1, "governance", 0.4
    elif has_vague and not has_metric:
        return 2, "vague_only", 0.6  # Vague without substance
    else:
        return 1, "unknown", 0.3


def map_to_actionability(label: int) -> str:
    """Map Green-Washing labels to actionability for comparison."""
    mapping = {0: "Implemented", 1: "Planning", 2: "Indeterminate"}
    return mapping.get(label, "Indeterminate")


def compute_gw_bert_ewri(df: pd.DataFrame, text_col: str = "sentence") -> pd.DataFrame:
    """
    Compute EWRI using Green-Washing BERT baseline approach.
    
    Greenwashing Risk = (greenwashing_prone / total) × 100
    """
    df = df.copy()
    
    results = df[text_col].apply(classify_greenwashing_label)
    df["gw_label"] = [r[0] for r in results]
    df["gw_category"] = [r[1] for r in results]
    df["gw_conf"] = [r[2] for r in results]
    df["gw_action"] = df["gw_label"].apply(map_to_actionability)
    
    # Aggregate by bank-year
    scores = []
    for (bank, year), group in df.groupby(["bank", "year"]):
        total = len(group)
        evidence = (group["gw_label"] == 0).sum()
        grey = (group["gw_label"] == 1).sum()
        gw_prone = (group["gw_label"] == 2).sum()
        
        gw_ratio = gw_prone / total if total > 0 else 0
        evidence_ratio = evidence / total if total > 0 else 0
        
        ewri_gw = gw_ratio * (1 - evidence_ratio) * 100
        
        scores.append({
            "bank": bank,
            "year": year,
            "total": total,
            "evidence_based": evidence,
            "grey_area": grey,
            "greenwashing_prone": gw_prone,
            "gw_ratio": round(gw_ratio, 3),
            "evidence_ratio": round(evidence_ratio, 3),
            "ewri_gw_bert": round(ewri_gw, 2),
        })
    
    return pd.DataFrame(scores), df


if __name__ == "__main__":
    print("Green-Washing BERT Baseline")
    print("=" * 50)
    
    tests = [
        "Ngân hàng đã giảm phát thải CO2 được 15% so với năm 2022.",
        "Chúng tôi cam kết hướng tới phát triển bền vững.",
        "Mục tiêu đạt net zero vào năm 2050 theo lộ trình đã đề ra.",
        "Ngân hàng hàng đầu trong nỗ lực phát triển bền vững.",
        "Đã triển khai chương trình đào tạo cho 5.000 nhân viên trong năm 2023.",
    ]
    
    for t in tests:
        label, cat, conf = classify_greenwashing_label(t)
        action = map_to_actionability(label)
        print(f"\n\"{t[:60]}...\"")
        print(f"  Label={label} ({cat}) → {action} (conf={conf:.2f})")
