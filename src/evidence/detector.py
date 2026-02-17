import re
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

ESG_SENTENCES_PATH = Path("data/corpus/esg_sentences_with_actionability.parquet")
OUTPUT_PATH = Path("data/corpus/esg_sentences_with_evidence.parquet")

# ============================================================
# EVIDENCE PATTERNS
# ============================================================

@dataclass
class EvidencePattern:
    name: str
    patterns: list[str]
    weight: float  # Default weight (can be varied for sensitivity)


EVIDENCE_TYPES = {
    "KPI": EvidencePattern(
        name="KPI",
        weight=0.35,
        patterns=[
            # Numbers with units
            r"\d+[\.,]?\d*\s*(%|phần trăm|percent)",
            r"\d+[\.,]?\d*\s*(tỷ|triệu|nghìn|ngàn)\s*(đồng|VND|USD)?",
            r"\d+[\.,]?\d*\s*(tấn|kg|ton|tonnes?)\s*(CO2|carbon)?",
            r"\d+[\.,]?\d*\s*(kWh|MWh|GWh|MW)",
            r"\d+[\.,]?\d*\s*(m2|m3|ha|hecta)",
            r"\d+[\.,]?\d*\s*(người|nhân viên|CBNV|khách hàng)",
            # Comparative metrics
            r"(giảm|tăng|đạt|tiết kiệm|cắt giảm)\s+\d+[\.,]?\d*\s*%",
            r"(so với|compared to).*\d+",
            # Scope emissions
            r"Scope\s*[123].*\d+",
        ],
    ),
    
    "Standard": EvidencePattern(
        name="Standard",
        weight=0.20,
        patterns=[
            # International standards
            r"\b(GRI\s*\d*|Global Reporting Initiative)\b",
            r"\b(SBTi|Science Based Targets?)\b",
            r"\b(ISO\s*\d+|ISO\s*14001|ISO\s*26000)\b",
            r"\b(SDG|Sustainable Development Goals?)\b",
            r"\b(Net[\-\s]?Zero|carbon neutral|trung hòa carbon)\b",
            r"\b(Paris Agreement|COP\d+)\b",
            r"\b(CDP|Carbon Disclosure Project)\b",
            r"\b(TCFD|Task Force on Climate)\b",
            r"\b(UN Global Compact|UNGC)\b",
            r"\b(ESG rating|xếp hạng ESG)\b",
            # Vietnamese banking standards
            r"\b(Thông tư\s*\d+|Nghị định\s*\d+)\b",
            r"\b(NHNN|Ngân hàng Nhà nước)\b",
        ],
    ),
    
    "Time_bound": EvidencePattern(
        name="Time_bound",
        weight=0.15,
        patterns=[
            # Specific years
            r"\b(năm|year)\s*(20\d{2})\b",
            r"\b(trong năm|in)\s*(20\d{2})\b",
            r"\b(đến|by|until)\s*(20\d{2})\b",
            r"\b(giai đoạn|period)\s*(20\d{2})\s*[-–]\s*(20\d{2})\b",
            # Quarters/months
            r"\b(Q[1-4]|quý\s*[1-4IViv]+)[\/\s]*(20\d{2})\b",
            r"\b(tháng\s*\d+|month\s*\d+)[\/\s]*(20\d{2})\b",
            # Relative time with specificity
            r"\b(trong\s+\d+\s*(năm|tháng|quý))\b",
            r"\b(mục tiêu|target)\s*(20\d{2})\b",
        ],
    ),
    
    "Third_party": EvidencePattern(
        name="Third_party",
        weight=0.25,
        patterns=[
            # Verification
            r"\b(kiểm toán|audited?|xác nhận|verified|certified)\b",
            r"\b(chứng nhận|certification|accreditation)\b",
            r"\b(đánh giá độc lập|independent assessment)\b",
            # Known auditors/verifiers
            r"\b(Deloitte|PwC|KPMG|EY|Ernst\s*&\s*Young)\b",
            r"\b(Bureau Veritas|DNV|SGS)\b",
            r"\b(FiinRatings|Vietnam Credit)\b",
            # External recognition
            r"\b(giải thưởng|award|recognition)\b",
            r"\b(top\s*\d+|ranking|xếp hạng)\b",
        ],
    ),
    
    "Initiative": EvidencePattern(
        name="Initiative",
        weight=0.05,
        patterns=[
            # Program/project names
            r"\b(dự án|project|chương trình|program)\s+[A-ZĐÀÁẢÃẠ][a-zđàáảãạ]+",
            r"\b(sáng kiến|initiative)\s+[A-ZĐÀÁẢÃẠ][a-zđàáảãạ]+",
            # Named campaigns
            r"\b(chiến dịch|campaign)\s+\"?[A-ZĐÀÁẢÃẠ][a-zđàáảãạ]+",
            # Green/ESG branded
            r"[A-Z][a-zA-Z]*\s*(Xanh|Green|ESG|Bền vững)\b",
            r"\b(Green\s*[A-Z][a-z]+|Xanh\s*[A-ZĐÀÁẢÃẠ][a-z]+)\b",
        ],
    ),
}

# Weight configurations for sensitivity analysis
WEIGHT_CONFIGS = {
    "default": {"KPI": 0.35, "Standard": 0.20, "Time_bound": 0.15, "Third_party": 0.25, "Initiative": 0.05},
    "kpi_heavy": {"KPI": 0.45, "Standard": 0.15, "Time_bound": 0.15, "Third_party": 0.20, "Initiative": 0.05},
    "uniform": {"KPI": 0.20, "Standard": 0.20, "Time_bound": 0.20, "Third_party": 0.20, "Initiative": 0.20},
    "verif_heavy": {"KPI": 0.30, "Standard": 0.15, "Time_bound": 0.10, "Third_party": 0.40, "Initiative": 0.05},
}


# ============================================================
# DETECTION FUNCTIONS
# ============================================================

def detect_evidence(text: str, context: str = "") -> dict:
    """
    Detect evidence elements in text.
    
    Returns:
        dict with:
        - has_evidence: bool
        - evidence_types: list of detected types
        - evidence_matches: dict of type -> list of matches
        - evidence_strength: float (0-1) using default weights
    """
    full_text = f"{context} {text}".lower() if context else text.lower()
    
    evidence_types = []
    evidence_matches = {}
    
    for etype, epattern in EVIDENCE_TYPES.items():
        matches = []
        for pattern in epattern.patterns:
            found = re.findall(pattern, text, re.IGNORECASE)
            if found:
                matches.extend(found if isinstance(found[0], str) else [m[0] if isinstance(m, tuple) else m for m in found])
        
        if matches:
            evidence_types.append(etype)
            evidence_matches[etype] = list(set(matches))[:5]  # Keep top 5 unique
    
# Revised weights for Neuro-Symbolic approach (User specification 2026-01-20)
# Only 4 rule types now: KPI, Standard, Time_bound, Third_party
ENHANCED_WEIGHTS = {
    "w_sim": 0.5,    # Weight for similarity component
    "w_R": 0.5,      # Weight for rule-based component (divided equally among 4 types)
}

# List of valid evidence types for the new formula
VALID_EVIDENCE_TYPES = ["KPI", "Standard", "Time_bound", "Third_party"]

def calculate_enhanced_strength(evidence_types: list, similarity_score: float = 0.0) -> float:
    """
    Calculate evidence strength using NEW formula (2026-01-20):
    
    ES = w_sim × S + (w_R / 4) × Σ(has_evidence_type_r)
    
    Where:
    - w_sim = 0.5 (similarity weight)
    - w_R = 0.5 (rule weight, divided equally among 4 types)
    - S = similarity_score ∈ [0, 1]
    - Σ(has_evidence_type_r) = count of evidence types found (max 4)
    
    Each rule type contributes: w_R / 4 = 0.5 / 4 = 0.125
    """
    w_sim = ENHANCED_WEIGHTS["w_sim"]
    w_R = ENHANCED_WEIGHTS["w_R"]
    
    # Component 1: Similarity
    sim_component = w_sim * min(max(similarity_score, 0.0), 1.0)
    
    # Component 2: Rule-based (count valid types)
    # Filter to only the 4 valid types
    valid_types_found = [t for t in evidence_types if t in VALID_EVIDENCE_TYPES]
    rule_count = len(valid_types_found)
    
    # Each type contributes w_R / 4
    rule_component = (w_R / 4.0) * rule_count
    
    # Total strength
    strength = sim_component + rule_component
    
    return min(strength, 1.0)  # Cap at 1.0


def detect_evidence(text: str, context: str = "", similarity_score: float = 0.0) -> dict:
    """
    Detect evidence elements in text.
    
    Returns:
        dict with:
        - has_evidence: bool
        - evidence_types: list of detected types
        - evidence_matches: dict of type -> list of matches
        - evidence_strength: float (0-1) using enhanced weights
        - rule_based_strength: float (0-1) using old heuristic weights
    """
    full_text = f"{context} {text}".lower() if context else text.lower()
    
    evidence_types = []
    evidence_matches = {}
    
    for etype, epattern in EVIDENCE_TYPES.items():
        matches = []
        for pattern in epattern.patterns:
            found = re.findall(pattern, text, re.IGNORECASE)
            if found:
                matches.extend(found if isinstance(found[0], str) else [m[0] if isinstance(m, tuple) else m for m in found])
        
        if matches:
            evidence_types.append(etype)
            evidence_matches[etype] = list(set(matches))[:5]  # Keep top 5 unique
    
    # Calculate strengths
    rule_strength = calculate_strength(evidence_types, "default")
    enhanced_strength = calculate_enhanced_strength(evidence_types, similarity_score)
    
    return {
        "has_evidence": len(evidence_types) > 0,
        "evidence_types": evidence_types,
        "evidence_matches": evidence_matches,
        "evidence_strength": enhanced_strength,
        "rule_based_strength": rule_strength,
    }


def calculate_strength(evidence_types: list, config: str = "default") -> float:
    """Calculate heuristic strength (Rule-based only)."""
    if not evidence_types:
        return 0.0
    
    weights = WEIGHT_CONFIGS.get(config, WEIGHT_CONFIGS["default"])
    total = sum(weights[et] for et in evidence_types if et in weights)
    max_possible = sum(weights.values())
    
    return min(total / max_possible, 1.0) if max_possible > 0 else 0.0


def extract_kpi_values(text: str) -> list[str]:
    """Extract specific KPI values from text."""
    kpi_patterns = [
        r"(\d+[\.,]?\d*\s*%)",
        r"(\d+[\.,]?\d*\s*(tỷ|triệu)\s*đồng)",
        r"(\d+[\.,]?\d*\s*(tấn|kg)\s*CO2?)",
        r"(\d+[\.,]?\d*\s*(kWh|MWh))",
    ]
    
    values = []
    for pattern in kpi_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for m in matches:
            val = m[0] if isinstance(m, tuple) else m
            values.append(val.strip())
    
    return list(set(values))[:10]


# ============================================================
# BATCH PROCESSING
# ============================================================

def process_dataframe(df: pd.DataFrame, text_col: str = "sentence") -> pd.DataFrame:
    """Process entire dataframe and add evidence columns."""
    print(f"Processing {len(df):,} sentences...")
    
    results = []
    for _, row in df.iterrows():
        text = str(row[text_col])
        ctx = f"{row.get('ctx_prev', '')} {row.get('ctx_next', '')}"
        
        result = detect_evidence(text, ctx)
        result["kpi_values"] = extract_kpi_values(text)
        results.append(result)
    
    # Add columns
    df = df.copy()
    df["has_evidence"] = [r["has_evidence"] for r in results]
    df["evidence_types"] = [r["evidence_types"] for r in results]
    df["evidence_strength"] = [r["evidence_strength"] for r in results]
    df["kpi_values"] = [r["kpi_values"] for r in results]
    
    return df


def run(input_path: Path = ESG_SENTENCES_PATH, output_path: Path = OUTPUT_PATH):
    """Run evidence detection on ESG sentences."""
    print(f"Loading data from {input_path}...")
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df):,} sentences")
    
    df = process_dataframe(df)
    
    # Summary statistics
    print("\n" + "="*60)
    print("EVIDENCE DETECTION SUMMARY")
    print("="*60)
    
    has_ev = df["has_evidence"].sum()
    print(f"Sentences with evidence: {has_ev:,} / {len(df):,} ({has_ev/len(df)*100:.1f}%)")
    
    print("\nBy Evidence Type:")
    for etype in EVIDENCE_TYPES.keys():
        count = df["evidence_types"].apply(lambda x: etype in x).sum()
        print(f"  {etype:15} {count:>6,}  ({count/len(df)*100:5.1f}%)")
    
    print("\nEvidence Strength Distribution:")
    print(f"  Mean:   {df['evidence_strength'].mean():.3f}")
    print(f"  Median: {df['evidence_strength'].median():.3f}")
    print(f"  Max:    {df['evidence_strength'].max():.3f}")
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"\nSaved: {output_path}")
    
    return df


if __name__ == "__main__":
    run()
