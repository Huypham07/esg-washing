"""
Module 5: EWRI (ESG-Washing Risk Index)
Calculate washing risk scores for each bank-year based on Actionability and Evidence.

EWRI = Symbolic_Ratio × (1 - Evidence_Ratio) × 100
- High EWRI → More Indeterminate + Less evidence → High washing risk
- Low EWRI → More Implemented + More evidence → Substantive disclosure
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

INPUT_PATH = Path("data/corpus/esg_sentences_enhanced_v2.parquet")
if not INPUT_PATH.exists():
    INPUT_PATH = Path("data/corpus/esg_sentences_with_evidence.parquet")
    print(f"Enhanced corpus not found, falling back to: {INPUT_PATH}")

OUTPUT_DIR = Path("outputs")


@dataclass
class EWRIScore:
    """EWRI score components for a bank-year"""
    bank: str
    year: int
    total_sentences: int
    
    # Actionability counts
    implemented: int
    planning: int
    indeterminate: int
    
    # Evidence counts
    with_evidence: int
    without_evidence: int
    
    # Ratios
    symbolic_ratio: float  # Indeterminate / Total
    evidence_ratio: float  # has_evidence / Total
    substantive_ratio: float  # Implemented / Total
    
    # EWRI score (Original V1)
    ewri: float
    
    
    # Risk level (Based on V2 if available, else V1)
    risk_level: str


# EWRI Calculation

def calculate_ewri(df_group: pd.DataFrame, alpha=2.0, beta=1.0, gamma=1.0) -> float:
    """
    Calculate EWRI (ESG-Washing Risk Index).
    
    Formula:
    Risk_i = α × I(Indeterminate) + β × I(Planning) + γ × (1 - ES_i)
    EWRI = [Σ(Risk_i) / (N × (α + β))] × 100
    
    Parameters:
    - α = 2.0: Penalty for Indeterminate statements
    - β = 1.0: Penalty for Planning statements
    - γ = 1.0: Penalty for evidence gap
    - Denominator (α + β + γ) normalizes by maximum possible risk
    
    Returns: EWRI score ∈ [0, 100+], higher = more washing risk
    """
    max_possible_risk = alpha + beta + gamma
    total_risk = 0.0
    
    for _, row in df_group.iterrows():
        # Actionability Penalty
        act_penalty = 0.0
        label = row.get("action_pred", "Unknown")
        if label == "Indeterminate":
            act_penalty = alpha
        elif label == "Planning":
            act_penalty = beta
            
        # Evidence Gap Penalty
        strength = row.get("evidence_strength", 1.0 if row.get("has_evidence") else 0.0)
        evidence_gap = gamma * (1.0 - strength)
        
        total_risk += (act_penalty + evidence_gap)
        
    avg_risk = total_risk / len(df_group) if len(df_group) > 0 else 0
    return (avg_risk / max_possible_risk) * 100


def get_risk_level(ewri: float) -> str:
    """Categorize EWRI into risk levels."""
    if ewri < 30:
        return "Low"
    elif ewri < 50:
        return "Medium"
    elif ewri < 70:
        return "High"
    else:
        return "Very High"


def calculate_bank_year_ewri(df: pd.DataFrame) -> list[EWRIScore]:
    """Calculate EWRI for each bank-year (V1 and V2)."""
    scores = []
    
    for (bank, year), group in df.groupby(["bank", "year"]):
        total = len(group)
        
        # Actionability counts
        implemented = (group["action_pred"] == "Implemented").sum()
        planning = (group["action_pred"] == "Planning").sum()
        indeterminate = (group["action_pred"] == "Indeterminate").sum()
        
        # Evidence counts
        with_evidence = group["has_evidence"].sum()
        without_evidence = total - with_evidence
        
        # Ratios
        symbolic_ratio = indeterminate / total if total > 0 else 0
        evidence_ratio = with_evidence / total if total > 0 else 0
        substantive_ratio = implemented / total if total > 0 else 0
        
        # Calculate EWRI
        ewri_score = calculate_ewri(group)
        
        # Determine risk level
        risk_level = get_risk_level(ewri_score)
        
        scores.append(EWRIScore(
            bank=bank,
            year=year,
            total_sentences=total,
            implemented=implemented,
            planning=planning,
            indeterminate=indeterminate,
            with_evidence=with_evidence,
            without_evidence=without_evidence,
            symbolic_ratio=symbolic_ratio,
            evidence_ratio=evidence_ratio,
            substantive_ratio=substantive_ratio,
            ewri=ewri_score,
            risk_level=risk_level,
        ))
    
    return scores


def scores_to_dataframe(scores: list[EWRIScore]) -> pd.DataFrame:
    """Convert EWRIScore list to DataFrame."""
    data = []
    for s in scores:
        data.append({
            "bank": s.bank,
            "year": s.year,
            "total_sentences": s.total_sentences,
            "implemented": s.implemented,
            "planning": s.planning,
            "indeterminate": s.indeterminate,
            "with_evidence": s.with_evidence,
            "without_evidence": s.without_evidence,
            "symbolic_ratio": round(s.symbolic_ratio, 3),
            "evidence_ratio": round(s.evidence_ratio, 3),
            "substantive_ratio": round(s.substantive_ratio, 3),
            "ewri": round(s.ewri, 2),
            "ewri": round(s.ewri, 2),
            "risk_level": s.risk_level,
        })
    return pd.DataFrame(data)


def run(input_path: Path = INPUT_PATH, output_dir: Path = OUTPUT_DIR):
    """Run EWRI calculation."""
    print(f"Loading data from {input_path}...")
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df):,} ESG sentences")
    
    # Calculate EWRI
    scores = calculate_bank_year_ewri(df)
    df_scores = scores_to_dataframe(scores)
    
    # Sort by EWRI (descending)
    df_scores = df_scores.sort_values("ewri", ascending=False)
    
    # Summary
    print("\n" + "="*60)
    print("EWRI SUMMARY (Neuro-Symbolic)")
    print("="*60)
    print(f"Total bank-years: {len(df_scores)}")
    print(f"Average EWRI: {df_scores['ewri'].mean():.2f}")
    print(f"Min EWRI: {df_scores['ewri'].min():.2f}")
    print(f"Max EWRI: {df_scores['ewri'].max():.2f}")
    
    print("\n=== TOP 10 HIGHEST EWRI (Highest Washing Risk) ===")
    print(df_scores.head(10)[["bank", "year", "ewri", "ewri", "risk_level"]].to_string(index=False))
    
    print("\n=== TOP 10 LOWEST EWRI (Most Substantive) ===")
    print(df_scores.tail(10)[["bank", "year", "ewri", "ewri", "risk_level"]].to_string(index=False))
    
    # Risk level distribution
    print("\n=== RISK LEVEL DISTRIBUTION ===")
    print(df_scores["risk_level"].value_counts())
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    df_scores.to_csv(output_dir / "ewri_scores.csv", index=False)
    print(f"\nSaved: {output_dir / 'ewri_scores.csv'}")
    
    return df_scores


if __name__ == "__main__":
    run()
