import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List

INPUT_PATH = Path("data/corpus/esg_sentences_scored_v2.parquet")
if not INPUT_PATH.exists():
    INPUT_PATH = Path("data/corpus/esg_sentences_with_evidence.parquet")
print(f"EWRI input path: {INPUT_PATH}")

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
    
    # EWRI score
    ewri: float
    
    # Risk level
    risk_level: str
    
    # Topic breakdown (NEW: per-topic EWRI)
    topic_breakdown: dict = field(default_factory=dict)
    
    # Top risk claims for traceability (NEW)
    top_risk_claims: List[dict] = field(default_factory=list)


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
    """Calculate EWRI for each bank-year with topic breakdown and traceability."""
    scores = []
    
    # Detect topic column
    topic_col = None
    for col in ["topic_label", "topic", "topic_pred"]:
        if col in df.columns:
            topic_col = col
            break
    
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
        
        # Topic breakdown (NEW)
        topic_breakdown = {}
        if topic_col is not None:
            topic_breakdown = calculate_topic_ewri(group, topic_col)
        
        # Top risk claims and full sentence traceability (NEW)
        top_risk_claims = get_top_risk_claims(group, top_n=None)
        
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
            topic_breakdown=topic_breakdown,
            top_risk_claims=top_risk_claims,
        ))
    
    return scores


def calculate_topic_ewri(
    df_group: pd.DataFrame,
    topic_col: str = "topic_label",
    alpha: float = 2.0,
    beta: float = 1.0,
    gamma: float = 1.0,
) -> dict:
    """
    Calculate EWRI per topic within a group (bank-year).
    
    Returns:
        dict: {topic: {"ewri": float, "n": int, "risk_level": str, ...}}
    """
    topic_ewri = {}
    
    esg_topics = ["E", "S_labor", "S_community", "S_product", "G"]
    
    for topic in esg_topics:
        topic_df = df_group[df_group[topic_col] == topic]
        n = len(topic_df)
        
        if n < 3:  # Need minimum sentences
            topic_ewri[topic] = {"ewri": None, "n": n, "risk_level": "N/A"}
            continue
        
        ewri = calculate_ewri(topic_df, alpha, beta, gamma)
        
        implemented = (topic_df["action_pred"] == "Implemented").sum()
        indeterminate = (topic_df["action_pred"] == "Indeterminate").sum()
        has_ev = topic_df["has_evidence"].sum() if "has_evidence" in topic_df.columns else 0
        
        topic_ewri[topic] = {
            "ewri": round(ewri, 2),
            "n": n,
            "risk_level": get_risk_level(ewri),
            "implemented_pct": round(implemented / n * 100, 1),
            "indeterminate_pct": round(indeterminate / n * 100, 1),
            "evidence_rate": round(has_ev / n * 100, 1) if n > 0 else 0,
        }
    
    return topic_ewri


def get_top_risk_claims(
    df_group: pd.DataFrame,
    top_n: Optional[int] = 10,
    alpha: float = 2.0,
    beta: float = 1.0,
    gamma: float = 1.0,
) -> List[dict]:
    """
    Get top-N highest risk claims from a group for traceability.
    
    Each claim gets a per-sentence risk score, and top-N are returned
    with their text, label, evidence, and explanation.
    """
    max_possible = alpha + beta + gamma
    
    risks = []
    for _, row in df_group.iterrows():
        # Actionability penalty
        act_penalty = 0.0
        label = row.get("action_pred", "Unknown")
        if label == "Indeterminate":
            act_penalty = alpha
        elif label == "Planning":
            act_penalty = beta
        
        # Evidence gap penalty
        strength = row.get("evidence_strength", 1.0 if row.get("has_evidence") else 0.0)
        evidence_gap = gamma * (1.0 - strength)
        
        risk = (act_penalty + evidence_gap) / max_possible * 100
        
        risks.append({
            "sent_id": row.get("sent_id", ""),
            "sentence": str(row.get("sentence", ""))[:200],
            "action_pred": label,
            "has_evidence": bool(row.get("has_evidence", False)),
            "evidence_strength": round(float(strength), 3),
            "risk_score": round(risk, 2),
            "explanation": str(row.get("action_explanation", ""))[:300],
            "topic": row.get("topic_label", row.get("topic", row.get("topic_pred", "Unknown"))),
            "section": str(row.get("section_title", "")),
        })
    
    # Sort by risk (highest first)
    risks.sort(key=lambda x: x["risk_score"], reverse=True)
    if top_n is not None:
        return risks[:top_n]
    return risks


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
            "risk_level": s.risk_level,
        })
    return pd.DataFrame(data)


def run(input_path: Path = INPUT_PATH, output_dir: Path = OUTPUT_DIR):
    """Run EWRI calculation with topic breakdown and traceability."""
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
    print(df_scores.head(10)[["bank", "year", "ewri", "risk_level"]].to_string(index=False))
    
    print("\n=== TOP 10 LOWEST EWRI (Most Substantive) ===")
    print(df_scores.tail(10)[["bank", "year", "ewri", "risk_level"]].to_string(index=False))
    
    # Risk level distribution
    print("\n=== RISK LEVEL DISTRIBUTION ===")
    print(df_scores["risk_level"].value_counts())
    
    # Topic breakdown (NEW)
    print("\n=== TOPIC BREAKDOWN ===")
    for score in scores[:5]:  # Show first 5 bank-years
        if score.topic_breakdown:
            print(f"\n{score.bank} {score.year} (Overall EWRI: {score.ewri:.1f}):")
            for topic, info in score.topic_breakdown.items():
                if info["ewri"] is not None:
                    print(f"  {topic:15s}: EWRI={info['ewri']:5.1f} ({info['risk_level']:9s}) "
                          f"N={info['n']:3d} Impl={info['implemented_pct']:5.1f}% "
                          f"Indet={info['indeterminate_pct']:5.1f}%")
    
    # Top risk claims (NEW)
    print("\n=== TOP RISK CLAIMS (example) ===")
    if scores:
        for claim in scores[0].top_risk_claims[:5]:
            print(f"  [{claim['risk_score']:.1f}] [{claim['action_pred']}] "
                  f"{claim['sentence'][:100]}...")
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    df_scores.to_csv(output_dir / "ewri_scores.csv", index=False)
    print(f"\nSaved: {output_dir / 'ewri_scores.csv'}")
    
    # Save topic breakdown as separate file
    topic_rows = []
    for s in scores:
        for topic, info in s.topic_breakdown.items():
            if info["ewri"] is not None:
                topic_rows.append({
                    "bank": s.bank, "year": s.year, "topic": topic,
                    **info,
                })
    if topic_rows:
        pd.DataFrame(topic_rows).to_csv(
            output_dir / "ewri_topic_breakdown.csv", index=False
        )
        print(f"Saved: {output_dir / 'ewri_topic_breakdown.csv'}")
    
    # Save all sentence details
    risk_rows = []
    for s in scores:
        for claim in s.top_risk_claims:
            claim["bank"] = s.bank
            claim["year"] = s.year
            risk_rows.append(claim)
    if risk_rows:
        pd.DataFrame(risk_rows).to_csv(
            output_dir / "ewri_sentence_details.csv", index=False
        )
        print(f"Saved: {output_dir / 'ewri_sentence_details.csv'}")
        
    # Save top risk claims (Top 10 per bank-year)
    top_risk_rows = []
    for s in scores:
        for claim in s.top_risk_claims[:10]:
            top_risk_rows.append(claim)
            
    if top_risk_rows:
        pd.DataFrame(top_risk_rows).to_csv(
            output_dir / "ewri_top_risk_claims.csv", index=False
        )
        print(f"Saved: {output_dir / 'ewri_top_risk_claims.csv'}")
    
    return df_scores


if __name__ == "__main__":
    run()
