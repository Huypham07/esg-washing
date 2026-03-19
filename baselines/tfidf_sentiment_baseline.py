"""
Baseline 2: TF-IDF + Sentiment Analysis for ESG-Washing Detection
Inspired by Lagasio (2024) ESG-washing Severity Index (ESGSI)

Approach:
- Uses TF-IDF features to measure content depth/specificity
- Uses sentiment analysis to measure positivity
- ESG-washing = high positivity + low content depth
- Computes ESGSI = Sentiment_Gap × (1 - Specificity)

Reference:
    Lagasio, V. (2024). ESG-washing Severity Index: A Text-Mining
    Analysis of Sustainability Reports. Sapienza University of Rome.
    Available at: ResearchGate / REPEC.
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# SENTIMENT SCORING (Rule-based Vietnamese adaptation)
# ============================================================

POSITIVE_WORDS = [
    r"\b(tăng trưởng|phát triển|cải thiện|nâng cao|đạt được)\b",
    r"\b(hoàn thành|thành công|hiệu quả|xuất sắc|tốt)\b",
    r"\b(tiên phong|dẫn đầu|hàng đầu|vượt trội)\b",
    r"\b(cam kết|đóng góp|hỗ trợ|bảo vệ|an toàn)\b",
    r"\b(bền vững|xanh|sạch|thân thiện)\b",
]

NEGATIVE_WORDS = [
    r"\b(giảm|sụt giảm|thất bại|yếu kém|thua lỗ)\b",
    r"\b(rủi ro|thách thức|khó khăn|hạn chế)\b",
    r"\b(ô nhiễm|vi phạm|xử phạt|thiệt hại)\b",
    r"\b(chưa đạt|chưa hoàn thành|không đủ)\b",
]

# Specificity indicators (things that make a statement more concrete)
SPECIFICITY_INDICATORS = [
    r"\d+[\.,]?\d*\s*(%|tỷ|triệu|nghìn|tấn|kWh|MWh)",
    r"\b(GRI|SBTi|ISO|SDG|TCFD|CDP)\b",
    r"\b(năm\s*20\d{2}|Q[1-4][\s/]*20\d{2})\b",
    r"\b(đã triển khai|đã thực hiện|đã hoàn thành)\b",
    r"\b(kiểm toán|chứng nhận|xác nhận)\b",
    r"\b(Deloitte|PwC|KPMG|EY)\b",
    r"\b(dự án|chương trình|sáng kiến)\s+[A-ZĐÀÁẢÃẠ]",
]


def compute_sentiment(text: str) -> float:
    """
    Simple rule-based sentiment score (-1 to 1).
    Positive = more ESG-positive language.
    """
    text_lower = text.lower()
    pos_count = sum(1 for pat in POSITIVE_WORDS if re.search(pat, text_lower, re.IGNORECASE))
    neg_count = sum(1 for pat in NEGATIVE_WORDS if re.search(pat, text_lower, re.IGNORECASE))
    
    total = pos_count + neg_count
    if total == 0:
        return 0.0
    
    return (pos_count - neg_count) / total


def compute_specificity(text: str) -> float:
    """
    Compute content specificity score (0 to 1).
    Higher = more specific/evidence-backed.
    """
    text_lower = text.lower()
    spec_count = sum(1 for pat in SPECIFICITY_INDICATORS if re.search(pat, text_lower, re.IGNORECASE))
    return min(spec_count / 3.0, 1.0)  # Normalize, cap at 1.0


def compute_tfidf_depth(corpus: List[str], target_idx: int = None) -> np.ndarray:
    """
    Compute TF-IDF based content depth for each sentence.
    Higher TF-IDF norm = more informative/specific content.
    """
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # L2 norm of TF-IDF vector as depth measure
    norms = np.array(tfidf_matrix.sum(axis=1)).flatten()
    # Normalize to [0, 1]
    if norms.max() > 0:
        norms = norms / norms.max()
    
    return norms


def compute_esgsi(df: pd.DataFrame, text_col: str = "sentence") -> pd.DataFrame:
    """
    Compute ESGSI (ESG-washing Severity Index) following Lagasio (2024).
    
    ESGSI = Positive_Sentiment × (1 - Specificity)
    
    High ESGSI = high positive language + low specificity = high washing risk
    """
    df = df.copy()
    
    # Sentiment scores
    df["sentiment"] = df[text_col].apply(compute_sentiment)
    
    # Specificity scores
    df["specificity"] = df[text_col].apply(compute_specificity)
    
    # TF-IDF depth (optional, for richer analysis)
    try:
        df["tfidf_depth"] = compute_tfidf_depth(df[text_col].tolist())
    except Exception:
        df["tfidf_depth"] = 0.5
    
    # Sentence-level ESGSI
    # Use only positive sentiment (negative sentiment is not washing)
    df["pos_sentiment"] = df["sentiment"].clip(lower=0)
    df["esgsi_sentence"] = df["pos_sentiment"] * (1 - df["specificity"])
    
    # Map to actionability-like labels (for comparison)
    def map_esgsi_to_action(row):
        if row["specificity"] >= 0.5:
            return "Implemented"
        elif row["pos_sentiment"] >= 0.5 and row["specificity"] < 0.3:
            return "Indeterminate"
        else:
            return "Planning"
    
    df["esgsi_action"] = df.apply(map_esgsi_to_action, axis=1)
    
    # Aggregate by bank-year
    scores = []
    for (bank, year), group in df.groupby(["bank", "year"]):
        total = len(group)
        avg_sentiment = group["sentiment"].mean()
        avg_specificity = group["specificity"].mean()
        avg_esgsi = group["esgsi_sentence"].mean()
        
        # EWRI equivalent
        ewri_esgsi = avg_esgsi * 100
        
        scores.append({
            "bank": bank,
            "year": year,
            "total": total,
            "avg_sentiment": round(avg_sentiment, 3),
            "avg_specificity": round(avg_specificity, 3),
            "avg_esgsi": round(avg_esgsi, 3),
            "ewri_esgsi": round(ewri_esgsi, 2),
        })
    
    return pd.DataFrame(scores), df


if __name__ == "__main__":
    print("TF-IDF + Sentiment Baseline (Lagasio 2024 approach)")
    print("=" * 50)
    
    tests = [
        "Ngân hàng đã giảm phát thải CO2 được 15% so với năm 2022.",
        "Chúng tôi cam kết hướng tới phát triển bền vững, tăng trưởng xanh.",
        "Đạt được chứng nhận ISO 14001, kiểm toán bởi KPMG.",
        "Ngân hàng tiên phong trong việc nâng cao nhận thức cộng đồng.",
    ]
    
    for t in tests:
        sent = compute_sentiment(t)
        spec = compute_specificity(t)
        esgsi = max(sent, 0) * (1 - spec)
        print(f"\n\"{t[:60]}...\"")
        print(f"  Sentiment={sent:.2f}, Specificity={spec:.2f}, ESGSI={esgsi:.3f}")
