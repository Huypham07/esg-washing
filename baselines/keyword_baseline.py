import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# ============================================================
# KEYWORD DICTIONARIES
# ============================================================

# Vague/cheap talk indicators (from Florstedt 2025, adapted for Vietnamese)
VAGUE_KEYWORDS = [
    # Hedging / commitment without action
    r"\b(cam kết|hướng tới|tăng cường|đẩy mạnh|tiếp tục)\b",
    r"\b(chú trọng|quan tâm|ưu tiên|nỗ lực)\b",
    r"\b(ngày càng|không ngừng|liên tục|dần dần)\b",
    r"\b(góp phần|đóng góp vào|hỗ trợ)\b",
    r"\b(nâng cao|cải thiện|phát huy|thúc đẩy)\b",
    # Boosting without evidence
    r"\b(hàng đầu|tiên phong|dẫn đầu|xuất sắc|vượt trội)\b",
    # Generic ESG language
    r"\b(phát triển bền vững|trách nhiệm xã hội|tăng trưởng xanh)\b",
]

# Substantive/evidence-based indicators
SUBSTANTIVE_KEYWORDS = [
    # Quantitative metrics
    r"\d+[\.,]?\d*\s*(%|phần trăm|percent)",
    r"\d+[\.,]?\d*\s*(tỷ|triệu|nghìn|ngàn)\s*(đồng|VND|USD)?",
    r"\d+[\.,]?\d*\s*(tấn|kg|ton)\s*(CO2|carbon)?",
    r"\d+[\.,]?\d*\s*(kWh|MWh|GWh|MW)",
    # Past tense action verbs
    r"\b(đã triển khai|đã thực hiện|đã hoàn thành|đã đạt được)\b",
    r"\b(đã giảm|đã tăng|đã tiết kiệm|đã cắt giảm)\b",
    r"\b(hoàn thành|ghi nhận|đạt được|thực hiện được)\b",
    # External references
    r"\b(GRI|SBTi|ISO\s*\d+|SDG|TCFD)\b",
    r"\b(kiểm toán|audited|chứng nhận|certified)\b",
    # Specific time references
    r"\b(năm|year)\s*(20\d{2})\b",
    r"\b(Q[1-4]|quý\s*[1-4])[\/\s]*(20\d{2})\b",
]


def classify_sentence_keyword(text: str) -> Tuple[str, float]:
    """
    Classify a sentence using keyword matching only.
    
    Returns:
        (label, confidence) where label ∈ {Substantive, Grey-area, Vague}
    """
    text_lower = text.lower()
    
    vague_count = sum(1 for pat in VAGUE_KEYWORDS if re.search(pat, text_lower, re.IGNORECASE))
    subst_count = sum(1 for pat in SUBSTANTIVE_KEYWORDS if re.search(pat, text_lower, re.IGNORECASE))
    
    if subst_count >= 2:
        return "Substantive", min(0.5 + 0.1 * subst_count, 1.0)
    elif subst_count >= 1 and vague_count <= 1:
        return "Substantive", 0.5
    elif vague_count >= 2 and subst_count == 0:
        return "Vague", min(0.5 + 0.1 * vague_count, 1.0)
    elif vague_count >= 1 and subst_count == 0:
        return "Grey-area", 0.5
    else:
        return "Grey-area", 0.3


def map_to_actionability(label: str) -> str:
    """Map keyword baseline labels to actionability labels for comparison."""
    mapping = {
        "Substantive": "Implemented",
        "Grey-area": "Planning",
        "Vague": "Indeterminate",
    }
    return mapping.get(label, "Indeterminate")


def compute_keyword_ewri(df: pd.DataFrame, text_col: str = "sentence") -> pd.DataFrame:
    """
    Compute keyword-baseline EWRI for each bank-year.
    
    EWRI_keyword = (num_vague / total_esg) × (1 - num_substantive / total_esg) × 100
    """
    # Classify all sentences
    results = df[text_col].apply(classify_sentence_keyword)
    df = df.copy()
    df["kw_label"] = [r[0] for r in results]
    df["kw_conf"] = [r[1] for r in results]
    df["kw_action"] = df["kw_label"].apply(map_to_actionability)
    
    # Aggregate by bank-year
    scores = []
    for (bank, year), group in df.groupby(["bank", "year"]):
        total = len(group)
        vague = (group["kw_label"] == "Vague").sum()
        substantive = (group["kw_label"] == "Substantive").sum()
        grey = (group["kw_label"] == "Grey-area").sum()
        
        vague_ratio = vague / total if total > 0 else 0
        subst_ratio = substantive / total if total > 0 else 0
        
        ewri = vague_ratio * (1 - subst_ratio) * 100
        
        scores.append({
            "bank": bank,
            "year": year,
            "total": total,
            "substantive": substantive,
            "grey_area": grey,
            "vague": vague,
            "vague_ratio": round(vague_ratio, 3),
            "subst_ratio": round(subst_ratio, 3),
            "ewri_keyword": round(ewri, 2),
        })
    
    return pd.DataFrame(scores), df


if __name__ == "__main__":
    print("Keyword Baseline (Florstedt 2025 approach)")
    print("=" * 50)
    
    tests = [
        "Ngân hàng đã giảm phát thải CO2 được 15% so với năm 2022.",
        "Chúng tôi cam kết hướng tới phát triển bền vững.",
        "Đã triển khai chương trình đào tạo cho 5.000 nhân viên trong năm 2023.",
        "Ngân hàng luôn quan tâm, chú trọng đến trách nhiệm xã hội.",
    ]
    
    for t in tests:
        label, conf = classify_sentence_keyword(t)
        action = map_to_actionability(label)
        print(f"\n\"{t[:60]}...\"")
        print(f"  Label: {label} → {action} (conf={conf:.2f})")
