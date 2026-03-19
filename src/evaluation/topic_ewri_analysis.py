"""
Topic-Level EWRI Analysis

The core justification for using 6 ESG subtopics:
shows that washing risk varies significantly ACROSS topics,
revealing which ESG dimensions banks tend to "wash" most.

Key outputs:
1. Washing rate per topic: % Indeterminate/Planning per topic
2. EWRI per topic per bank: heatmap showing where washing concentrates
3. Cross-tabulation: Topic × Actionability distribution
4. Statistical test: chi-square for independence between topic and washing

This answers: "Which ESG areas are most prone to washing?"
and justifies why a 3-label E/S/G split is insufficient.

Usage:
    python src/evaluation/topic_ewri_analysis.py \
        --data data/corpus/esg_sentences_enhanced.parquet
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def compute_topic_washing_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute washing vulnerability per ESG topic.
    
    Washing Rate = (Indeterminate + Planning) / Total
    → Higher rate = more vague/unsubstantiated claims in that topic
    """
    # Ensure columns exist
    topic_col = None
    for col in ["topic_label", "topic", "topic_pred"]:
        if col in df.columns:
            topic_col = col
            break
    
    action_col = None
    for col in ["action_label", "actionability", "action_pred"]:
        if col in df.columns:
            action_col = col
            break
    
    if topic_col is None or action_col is None:
        print(f"Required columns not found. Available: {list(df.columns)}")
        return pd.DataFrame()
    
    # Filter ESG only (exclude Non_ESG)
    esg_df = df[df[topic_col] != "Non_ESG"].copy()
    
    # Cross-tabulation
    xtab = pd.crosstab(esg_df[topic_col], esg_df[action_col], margins=True)
    
    print("\n" + "=" * 60)
    print("CROSS-TABULATION: Topic × Actionability")
    print("=" * 60)
    print(xtab)
    
    # Per-topic analysis
    topic_stats = []
    for topic in sorted(esg_df[topic_col].unique()):
        topic_df = esg_df[esg_df[topic_col] == topic]
        total = len(topic_df)
        
        if total == 0:
            continue
        
        # Count by actionability
        implemented = (topic_df[action_col] == "Implemented").sum()
        planning = (topic_df[action_col] == "Planning").sum()
        indeterminate = (topic_df[action_col] == "Indeterminate").sum()
        
        # Washing rate: (Indeterminate + Planning) / Total
        washing_rate = (indeterminate + planning) / total
        
        # Evidence rate (if available)
        has_evidence_col = "has_evidence" if "has_evidence" in df.columns else None
        evidence_rate = topic_df[has_evidence_col].mean() if has_evidence_col else np.nan
        
        # Mean evidence strength (if available)
        es_col = None
        for col in ["evidence_strength", "evidence_strength_v2"]:
            if col in df.columns:
                es_col = col
                break
        mean_es = topic_df[es_col].mean() if es_col else np.nan
        
        topic_stats.append({
            "Topic": topic,
            "Total Sentences": total,
            "Implemented": implemented,
            "Planning": planning,
            "Indeterminate": indeterminate,
            "Implemented %": round(implemented / total * 100, 1),
            "Planning %": round(planning / total * 100, 1),
            "Indeterminate %": round(indeterminate / total * 100, 1),
            "Washing Rate": round(washing_rate, 4),
            "Evidence Rate": round(evidence_rate, 4) if not np.isnan(evidence_rate) else "N/A",
            "Mean ES": round(mean_es, 4) if not np.isnan(mean_es) else "N/A",
        })
    
    stats_df = pd.DataFrame(topic_stats).sort_values("Washing Rate", ascending=False)
    
    print("\n" + "=" * 60)
    print("WASHING RATE PER TOPIC (sorted by risk)")
    print("=" * 60)
    print(stats_df.to_string(index=False))
    
    # Chi-square test: are topic and actionability independent?
    contingency = pd.crosstab(esg_df[topic_col], esg_df[action_col])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    
    print(f"\nChi-square test (Topic × Actionability independence):")
    print(f"  χ² = {chi2:.2f}, df = {dof}, p = {p_value:.6f}")
    if p_value < 0.05:
        print(f"  → SIGNIFICANT: Topic and actionability are NOT independent!")
        print(f"     This proves different ESG topics have different washing patterns.")
    else:
        print(f"  → Not significant: Topic and actionability are independent.")
    
    return stats_df


def compute_topic_ewri_by_bank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute EWRI per topic per bank-year.
    
    This is the KEY analysis that justifies fine-grained topics:
    shows that within the same bank, washing risk varies by topic.
    """
    topic_col = None
    for col in ["topic_label", "topic", "topic_pred"]:
        if col in df.columns:
            topic_col = col
            break
    
    action_col = None
    for col in ["action_label", "actionability", "action_pred"]:
        if col in df.columns:
            action_col = col
            break
    
    if topic_col is None or action_col is None:
        return pd.DataFrame()
    
    # Compute per (bank, year, topic)
    esg_df = df[df[topic_col] != "Non_ESG"].copy()
    
    # EWRI-like metric per group
    results = []
    
    for (bank, year, topic), group in esg_df.groupby(["bank", "year", topic_col]):
        total = len(group)
        if total < 3:  # Minimum sentences
            continue
        
        implemented = (group[action_col] == "Implemented").sum()
        indeterminate = (group[action_col] == "Indeterminate").sum()
        
        # Simple EWRI approximation
        washing_rate = (indeterminate) / total
        substantive_rate = implemented / total
        
        # If evidence strength available
        es_col = None
        for col in ["evidence_strength", "evidence_strength_v2"]:
            if col in group.columns:
                es_col = col
                break
        mean_es = group[es_col].mean() if es_col else 0.0
        
        ewri_approx = 1.0 - (substantive_rate * 0.5 + mean_es * 0.5)
        
        results.append({
            "Bank": bank,
            "Year": year,
            "Topic": topic,
            "N_Sentences": total,
            "Substantive Rate": round(substantive_rate, 3),
            "Washing Rate": round(washing_rate, 3),
            "Mean ES": round(mean_es, 3),
            "EWRI Approx": round(ewri_approx, 3),
        })
    
    result_df = pd.DataFrame(results)
    
    if len(result_df) == 0:
        print("No sufficient data for per-bank-topic analysis.")
        return result_df
    
    print("\n" + "=" * 60)
    print("EWRI BY BANK × TOPIC (sample)")
    print("=" * 60)
    
    # Show pivot
    pivot = result_df.pivot_table(
        values="EWRI Approx", 
        index=["Bank", "Year"], 
        columns="Topic", 
        aggfunc="mean"
    ).round(3)
    print(pivot)
    
    # Per-topic average across banks
    print(f"\nAverage EWRI by Topic (across all banks):")
    topic_avg = result_df.groupby("Topic")["EWRI Approx"].agg(["mean", "std"]).round(3)
    print(topic_avg)
    
    # Within-bank topic variance
    bank_topic_var = result_df.groupby(["Bank", "Year"])["EWRI Approx"].std().mean()
    print(f"\nAvg within-bank topic variance (σ): {bank_topic_var:.4f}")
    if bank_topic_var > 0.05:
        print("→ Significant within-bank variation across topics!")
        print("   This proves per-topic EWRI reveals patterns hidden by aggregation.")
    
    return result_df


def run(data_path: str = None):
    """Run topic-level EWRI analysis."""
    if data_path is None:
        # Try enhanced first, then fallback
        paths = [
            "data/corpus/esg_sentences_enhanced.parquet",
            "data/corpus/esg_sentences_with_evidence.parquet",
            "data/corpus/esg_sentences_with_actionability.parquet",
        ]
        for p in paths:
            if Path(p).exists():
                data_path = p
                break
    
    if data_path is None or not Path(data_path).exists():
        print("No data found. Provide path via --data")
        return
    
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} sentences from {data_path}")
    print(f"Columns: {list(df.columns)}")
    
    # Analysis 1: Washing rates per topic
    topic_stats = compute_topic_washing_rates(df)
    
    # Analysis 2: Per-bank-topic EWRI
    ewri_results = compute_topic_ewri_by_bank(df)
    
    # Save
    output_dir = Path("exports/tables")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if len(topic_stats) > 0:
        topic_stats.to_csv(output_dir / "topic_washing_rates.csv", index=False)
        print(f"\nSaved: {output_dir / 'topic_washing_rates.csv'}")
    
    if len(ewri_results) > 0:
        ewri_results.to_csv(output_dir / "topic_ewri_by_bank.csv", index=False)
        print(f"Saved: {output_dir / 'topic_ewri_by_bank.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    args = parser.parse_args()
    run(args.data)
