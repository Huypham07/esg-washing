import pandas as pd
from pathlib import Path
import sys

def run_evidence_eval():
    print("Evaluating Evidence Retrieval: Keyword vs Neuro-Symbolic (V2)")
    df = pd.read_parquet("data/corpus/esg_sentences_enhanced_v2.parquet")
    
    total_esg = len(df)
    
    # Neuro-Symbolic V2 stats
    ns_linked = df[df["linked_evidence_found"] == True]
    ns_coverage = len(ns_linked) / total_esg * 100
    
    # Keyword baseline
    # For a fair comparison, a keyword approach just checks if the sentence has substantive keywords.
    # We will compute the percentage of sentences that have 'kw_label' == 'Substantive' natively.
    
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from baselines.keyword_baseline import classify_sentence_keyword
    
    # Process a sample of 10k sentences to save time
    sample_df = df.sample(n=min(10000, len(df)), random_state=42)
    kw_results = sample_df["sentence"].apply(classify_sentence_keyword)
    
    kw_coverage = sum(1 for res in kw_results if res[0] == "Substantive") / len(sample_df) * 100
    
    print(f"Total Sentences Evaluated: {total_esg} (NS) / {len(sample_df)} (Keyword)")
    print(f"Neuro-Symbolic V2 Coverage: {ns_coverage:.2f}%")
    print(f"Keyword-only Baseline Coverage: {kw_coverage:.2f}%")
    
if __name__ == "__main__":
    run_evidence_eval()
