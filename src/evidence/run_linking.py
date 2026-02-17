import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import sys
sys.path.append("src")

from evidence.claim_evidence_linker import ClaimEvidenceLinker, analyze_linking_quality
from evidence.detector import detect_evidence

# Config
INPUT_PATH = Path("data/corpus/esg_sentences_with_actionability.parquet")
OUTPUT_PATH = Path("data/corpus/esg_sentences_enhanced.parquet")
OUTPUT_V2_PATH = Path("data/corpus/esg_sentences_enhanced_v2.parquet")
Stats_PATH = Path("exports/tables/linking_stats.txt")
Sample_PATH = Path("exports/tables/linking_samples.csv")


def run_v1(df):
    """Run original V1 linker (window-only, ±5)."""
    print("\n" + "=" * 50)
    print("Running V1 Claim-Evidence Linker (Window ±5)...")
    print("=" * 50)
    
    linker = ClaimEvidenceLinker(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        window_size=5,
        similarity_threshold=0.5
    )
    
    links_df = linker.link_corpus(df, text_column='sentence')
    
    # Merge linking info back
    df['best_evidence'] = links_df['best_evidence']
    df['similarity_score'] = links_df['similarity_score']
    df['linked_evidence_found'] = links_df['evidence_found']
    
    # Re-run Detector with Similarity info
    print("\nCalculating Enhanced Evidence Strength...")
    enhanced_results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Enhancing"):
        text = str(row['sentence'])
        ctx = "" 
        if 'ctx_prev' in row: ctx += str(row['ctx_prev'])
        if 'ctx_next' in row: ctx += " " + str(row['ctx_next'])
        sim_score = float(row['similarity_score'])
        res = detect_evidence(text, ctx, similarity_score=sim_score)
        enhanced_results.append(res)
    
    df['evidence_strength'] = [r['evidence_strength'] for r in enhanced_results]
    df['rule_based_strength'] = [r['rule_based_strength'] for r in enhanced_results]
    df['evidence_types'] = [r['evidence_types'] for r in enhanced_results]
    df['has_evidence'] = [r['has_evidence'] for r in enhanced_results]
    
    # Save
    print(f"\nSaving V1 enhanced corpus to {OUTPUT_PATH}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH)
    
    # Stats
    stats = analyze_linking_quality(links_df)
    print_stats(stats)
    save_stats_and_samples(stats, df)
    
    return df


def run_v2(df):
    """Run enhanced V2 linker (document-level + NLI)."""
    from evidence.enhanced_linker import EnhancedClaimEvidenceLinker
    
    print("\n" + "=" * 50)
    print("Running V2 Enhanced Linker (Document-level + NLI)...")
    print("=" * 50)
    
    linker = EnhancedClaimEvidenceLinker(
        window_size=5,
        document_level=True,
        tfidf_top_k=20,
        similarity_threshold=0.5,
        top_k_evidence=3,
        use_nli=True,
        nli_mode="rule",
        weight_config="default",
    )
    
    results_df = linker.link_corpus(df, text_column='sentence')
    
    # Merge back to original df
    df['best_evidence'] = results_df['best_evidence']
    df['similarity_score'] = results_df['similarity_score']
    df['linked_evidence_found'] = results_df['evidence_found']
    df['num_evidence_v2'] = results_df['num_evidence']
    df['nli_entailment_score'] = results_df['nli_entailment_score']
    df['nli_label'] = results_df['nli_label']
    df['evidence_strength_v2'] = results_df['evidence_strength_v2']
    df['search_method'] = results_df['search_method']
    
    # Also run rule-based detector (for consistency)
    print("\nRunning rule-based evidence detection...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Detecting"):
        text = str(row['sentence'])
        ctx = ""
        if 'ctx_prev' in row: ctx += str(row['ctx_prev'])
        if 'ctx_next' in row: ctx += " " + str(row['ctx_next'])
        res = detect_evidence(text, ctx, similarity_score=float(row['similarity_score']))
        df.at[idx, 'evidence_types'] = res['evidence_types']
        df.at[idx, 'has_evidence'] = res['has_evidence']
        df.at[idx, 'rule_based_strength'] = res['rule_based_strength']
    
    # Use V2 strength as primary strength
    df['evidence_strength'] = df['evidence_strength_v2']
    
    # Save
    print(f"\nSaving V2 enhanced corpus to {OUTPUT_V2_PATH}")
    OUTPUT_V2_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_V2_PATH)
    
    return df


def run_compare(df):
    """Compare V1 vs V2 for ablation study."""
    from evidence.enhanced_linker import compare_v1_v2
    comparison = compare_v1_v2(df, text_column='sentence')
    
    # Save comparison
    output_dir = Path("exports/tables")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(output_dir / "v1_v2_comparison.json", "w") as f:
        # Convert non-serializable items
        comp_serializable = {k: str(v) for k, v in comparison.items()}
        json.dump(comp_serializable, f, indent=2)
    
    print(f"\nSaved comparison to {output_dir / 'v1_v2_comparison.json'}")


def print_stats(stats):
    """Print linking quality stats."""
    print("\n" + "=" * 50)
    print("LINKING RESULTS SUMMARY")
    print("=" * 50)
    for k, v in stats.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for subk, subv in v.items():
                print(f"  {subk}: {subv}")
        else:
            print(f"{k}: {v}")


def save_stats_and_samples(stats, df):
    """Save stats and sample outputs."""
    Stats_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(Stats_PATH, "w") as f:
        f.write(str(stats))
    
    if 'similarity_score' in df.columns:
        cols = ['sentence', 'action_label', 'best_evidence', 'similarity_score', 'evidence_strength']
        available_cols = [c for c in cols if c in df.columns]
        samples = df[df['similarity_score'] > 0.6][available_cols].head(50)
        samples.to_csv(Sample_PATH)
        print(f"Saved samples to {Sample_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Run evidence linking pipeline")
    parser.add_argument("--enhanced", action="store_true", help="Use V2 enhanced linker")
    parser.add_argument("--compare", action="store_true", help="Compare V1 vs V2")
    args = parser.parse_args()
    
    print(f"Loading input data: {INPUT_PATH}")
    if not INPUT_PATH.exists():
        print(f"Error: {INPUT_PATH} not found")
        return
    
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df)} sentences.")
    
    if args.compare:
        run_compare(df)
    elif args.enhanced:
        run_v2(df)
    else:
        run_v1(df)


if __name__ == "__main__":
    main()

