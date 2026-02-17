
import pandas as pd
import numpy as np
from pathlib import Path

REGEX_OUTPUT = Path("data/corpus/esg_sentences_with_evidence.parquet")
NEURO_OUTPUT = Path("data/corpus/esg_sentences_enhanced.parquet")

def calculate_stats():
    # 1. Keyword-only (Regex)
    print(f"Loading Regex Output: {REGEX_OUTPUT}")
    df_kw = pd.read_parquet(REGEX_OUTPUT)
    
    total = len(df_kw)
    kw_found = len(df_kw[df_kw['has_evidence'] == True])
    kw_rate = (kw_found / total) * 100
    
    # Implemented Coverage
    if 'action_label' in df_kw.columns:
        col = 'action_label'
    else:
        col = 'actionability' # fallback
        
    kw_imp_rate = 0
    if col in df_kw.columns:
        imp_df = df_kw[df_kw[col] == 'Implemented']
        if len(imp_df) > 0:
            kw_imp_found = len(imp_df[imp_df['has_evidence'] == True])
            kw_imp_rate = (kw_imp_found / len(imp_df)) * 100

    # 2. Neuro-Symbolic
    print(f"Loading Neuro Output: {NEURO_OUTPUT}")
    if NEURO_OUTPUT.exists():
        df_ns = pd.read_parquet(NEURO_OUTPUT)
        # Note: run_linking.py updates 'has_evidence' based on linking result?
        # Let's check run_linking.py logic:
        # It calls `detect_evidence(..., similarity_score=...)` which updates evidence_strength.
        # But `has_evidence` in detect_evidence is `len(evidence_types) > 0`.
        # Wait, if detect_evidence logic relies on regex types, then has_evidence might not change 
        # unless `calculate_enhanced_strength` affects it? 
        # Actually `detect_evidence` returns `has_evidence` based on `evidence_types`.
        # AND `run_linking` sets `df['linked_evidence_found']`.
        # So we should use `linked_evidence_found` or `best_evidence` not null for Neuro stats.
        
        # In run_linking.py: df['linked_evidence_found'] = links_df['evidence_found']
        # This is the flag for "Did we find semantic evidence?"
        
        ns_found = len(df_ns[df_ns['linked_evidence_found'] == True])
        ns_rate = (ns_found / total) * 100
        
        ns_imp_rate = 0
        if col in df_ns.columns:
            imp_df_ns = df_ns[df_ns[col] == 'Implemented']
            if len(imp_df_ns) > 0:
                ns_imp_found = len(imp_df_ns[imp_df_ns['linked_evidence_found'] == True])
                ns_imp_rate = (ns_imp_found / len(imp_df_ns)) * 100
                
        # Cross-sentence linking
        # If 'best_evidence' text is different from 'sentence' text (approx), it's cross-sentence.
        # But better: check if 'best_evidence' is not null/empty and 'similarity_score' > 0.5 (threshold) 
        # AND it is NOT the sentence itself (if the linker allowed self-linking, but linker excludes self).
        # Linker code: `if 0 < distance <= self.window_size: candidates.append(idx)` -> Excludes self.
        # So ALL linked evidence is Cross-sentence or at least Cross-position.
        # Thus, Cross-sentence Links % = (Evidence Found via Linker) / (Total Evidence Found via Linker) ?
        # Or relative to Total Claims?
        # The metric "Cross-sentence Links (%)" in the chart likely means:
        # Of the evidence found, how many are cross-sentence? 
        # Since Keyword is 0 (it only looks at self), Neuro is 100% of its *linked* evidence is cross-sentence?
        # But wait, Neuro approach *combines* Regex + Linker.
        # The chart title is "Neuro-Symbolic". Does it include regex matches within the sentence?
        # If "Neuro-Symbolic" means "The final system state", it has both.
        # If `linked_evidence_found` is True, we have a link.
        # Does `has_evidence` (Regex) still count?
        # The "Evidence Found" for Neuro-Symbolic usually implies the UNION of Regex and Linker?
        # Or does the user consider "Evidence Linking" as the *incremental* step?
        # The previous chart said "Evidence Found: 86.6%". linking_stats.txt said 86.6%.
        # linking_stats.txt comes from `analyze_linking_quality(links_df)`.
        # `links_df` is purely from the Linker.
        # So the 86.6% is purely from the Linker finding semantic matches.
        # So for the Linker, ALL matches are cross-sentence (window search).
        # So Cross-sentence Links % = 100% of the *linked* evidence?
        # Or maybe "Cross-sentence Links" means "Rate of finding evidence that is cross-sentence relative to TOTAL CLAIMS"?
        # Let's assume the chart meant "Percentage of claims that have a cross-sentence link".
        # Which is exactly ns_rate (86.6%).
        # But in the chart I had "Cross-sentence Links (%)" as 73.2%.
        # Let's calculate the exact portion. 
        # Maybe I used a subset? Let's just output the raw numbers and decide.
        
        ns_cross_sentence_count = ns_found # Since linker excludes self.
        ns_cross_sentence_rate_of_claims = (ns_cross_sentence_count / total) * 100
        
    else:
        print("Neuro file not found!")
        ns_rate = 0
        ns_imp_rate = 0
        ns_cross_sentence_rate_of_claims = 0

    print("\n" + "="*40)
    print("COMPARISON STATISTICS (Calculated)")
    print("="*40)
    print(f"{'Metric':<25} | {'Keyword (Regex)':<15} | {'Neuro (Linker)':<15}")
    print("-" * 60)
    print(f"{'Evidence Found (%)':<25} | {kw_rate:<15.2f} | {ns_rate:<15.2f}")
    print(f"{'Implemented Coverage (%)':<25} | {kw_imp_rate:<15.2f} | {ns_imp_rate:<15.2f}")
    print(f"{'Cross-sentence Links (%)':<25} | {0.0:<15.1f} | {ns_cross_sentence_rate_of_claims:<15.2f}")

if __name__ == "__main__":
    calculate_stats()
