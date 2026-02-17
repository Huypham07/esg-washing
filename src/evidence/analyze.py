import pandas as pd
import numpy as np
from pathlib import Path
from detector import EVIDENCE_TYPES, WEIGHT_CONFIGS, calculate_strength

INPUT_PATH = Path("data/corpus/esg_sentences_with_evidence.parquet")
OUTPUT_DIR = Path("outputs/evidence_analysis")


def cross_tab_actionability_evidence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create cross-tab of Actionability × Evidence.
    This is the KEY analysis for substantive vs symbolic disclosure.
    """
    # Create cross-tab
    crosstab = pd.crosstab(
        df["action_label"],
        df["has_evidence"],
        margins=True,
        margins_name="Total"
    )
    crosstab.columns = ["No Evidence", "Has Evidence", "Total"]
    
    # Add percentages
    crosstab_pct = pd.crosstab(
        df["action_label"],
        df["has_evidence"],
        normalize="index"
    ) * 100
    crosstab_pct.columns = ["No Evidence %", "Has Evidence %"]
    
    result = crosstab.join(crosstab_pct.round(1))
    
    print("="*60)
    print("CROSS-TAB: ACTIONABILITY × EVIDENCE")
    print("="*60)
    print(result.to_string())
    print()
    
    # Key insight
    if "Implemented" in df["action_label"].values and "Indeterminate" in df["action_label"].values:
        impl_ev = df[(df["action_label"] == "Implemented") & (df["has_evidence"] == True)]
        impl_no = df[(df["action_label"] == "Implemented") & (df["has_evidence"] == False)]
        indet_ev = df[(df["action_label"] == "Indeterminate") & (df["has_evidence"] == True)]
        indet_no = df[(df["action_label"] == "Indeterminate") & (df["has_evidence"] == False)]
        
        print("KEY INSIGHTS:")
        print(f"  ✓ Implemented + Evidence (Substantive): {len(impl_ev):,}")
        print(f"  ⚠ Implemented + No Evidence (Weak claim): {len(impl_no):,}")
        print(f"  ? Indeterminate + Evidence: {len(indet_ev):,}")
        print(f"  🚨 Indeterminate + No Evidence (High risk): {len(indet_no):,}")
    
    return result


def sensitivity_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run sensitivity analysis with different weight configurations.
    Shows that bank-year ranking is stable across weight choices.
    """
    print("\n" + "="*60)
    print("SENSITIVITY ANALYSIS: EVIDENCE STRENGTH WEIGHTS")
    print("="*60)
    
    # Calculate strength for each config
    for config_name, weights in WEIGHT_CONFIGS.items():
        col_name = f"strength_{config_name}"
        df[col_name] = df["evidence_types"].apply(
            lambda x: calculate_strength(x, config_name) if isinstance(x, list) else 0.0
        )
        print(f"\n{config_name}: {weights}")
    
    # Aggregate by bank-year
    strength_cols = [f"strength_{c}" for c in WEIGHT_CONFIGS.keys()]
    agg_dict = {col: "mean" for col in strength_cols}
    agg_dict["has_evidence"] = "mean"
    
    bank_year = df.groupby(["bank", "year"]).agg(agg_dict).reset_index()
    
    # Calculate correlations between rankings
    print("\n" + "-"*40)
    print("RANKING STABILITY (Spearman correlation):")
    print("-"*40)
    
    from scipy.stats import spearmanr
    
    for i, c1 in enumerate(strength_cols):
        for c2 in strength_cols[i+1:]:
            corr, _ = spearmanr(bank_year[c1], bank_year[c2])
            print(f"  {c1} vs {c2}: ρ = {corr:.3f}")
    
    return bank_year


def evidence_by_topic_action(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze evidence distribution by ESG topic and action."""
    print("\n" + "="*60)
    print("EVIDENCE BY ESG TOPIC × ACTIONABILITY")
    print("="*60)
    
    # Group by topic and action
    analysis = df.groupby(["predicted_label", "action_label"]).agg({
        "has_evidence": ["sum", "count", "mean"],
        "evidence_strength": "mean",
    }).round(3)
    
    analysis.columns = ["with_evidence", "total", "evidence_rate", "avg_strength"]
    analysis = analysis.reset_index()
    
    print(analysis.to_string(index=False))
    return analysis


def evidence_type_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Detailed distribution of evidence types."""
    print("\n" + "="*60)
    print("EVIDENCE TYPE DISTRIBUTION")
    print("="*60)
    
    # Expand evidence types
    type_counts = {}
    for etype in EVIDENCE_TYPES.keys():
        count = df["evidence_types"].apply(lambda x: etype in x if isinstance(x, list) else False).sum()
        type_counts[etype] = count
    
    # By actionability
    print("\nBy Actionability:")
    for action in df["action_label"].unique():
        subset = df[df["action_label"] == action]
        print(f"\n  {action}:")
        for etype in EVIDENCE_TYPES.keys():
            count = subset["evidence_types"].apply(lambda x: etype in x if isinstance(x, list) else False).sum()
            pct = count / len(subset) * 100 if len(subset) > 0 else 0
            print(f"    {etype:15} {count:>5}  ({pct:5.1f}%)")
    
    return pd.DataFrame({"evidence_type": type_counts.keys(), "count": type_counts.values()})


def generate_report(df: pd.DataFrame):
    """Generate full analysis report and save outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Cross-tab (KEY for paper)
    crosstab = cross_tab_actionability_evidence(df)
    crosstab.to_csv(OUTPUT_DIR / "crosstab_action_evidence.csv")
    
    # 2. Sensitivity analysis
    bank_year = sensitivity_analysis(df)
    bank_year.to_csv(OUTPUT_DIR / "sensitivity_bank_year.csv", index=False)
    
    # 3. Topic × Action × Evidence
    topic_action = evidence_by_topic_action(df)
    topic_action.to_csv(OUTPUT_DIR / "evidence_by_topic_action.csv", index=False)
    
    # 4. Evidence type distribution
    type_dist = evidence_type_distribution(df)
    type_dist.to_csv(OUTPUT_DIR / "evidence_type_distribution.csv", index=False)
    
    print("\n" + "="*60)
    print(f"Reports saved to: {OUTPUT_DIR}")
    print("="*60)


def run():
    """Run full analysis."""
    print(f"Loading data from {INPUT_PATH}...")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df):,} sentences")
    
    generate_report(df)


if __name__ == "__main__":
    run()
