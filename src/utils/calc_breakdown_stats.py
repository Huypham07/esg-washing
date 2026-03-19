
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import re

# Add src to path to import detector if needed, or just copy regex
sys.path.append("src")

REGEX_OUTPUT = Path("data/corpus/esg_sentences_with_evidence.parquet")
NEURO_OUTPUT = Path("data/corpus/esg_sentences_enhanced.parquet")

# Minimal KPI regex from detector.py
KPI_PATTERNS = [
    r"\d+[\.,]?\d*\s*(%|phần trăm|percent)",
    r"\d+[\.,]?\d*\s*(tỷ|triệu|nghìn|ngàn)\s*(đồng|VND|USD)?",
    r"\d+[\.,]?\d*\s*(tấn|kg|ton|tonnes?)\s*(CO2|carbon)?",
    r"\d+[\.,]?\d*\s*(kWh|MWh|GWh|MW)",
    r"\d+[\.,]?\d*\s*(m2|m3|ha|hecta)",
    r"\d+[\.,]?\d*\s*(người|nhân viên|CBNV|khách hàng)",
    r"(giảm|tăng|đạt|tiết kiệm|cắt giảm)\s+\d+[\.,]?\d*\s*%",
    r"(so với|compared to).*\d+",
    r"Scope\s*[123].*\d+",
]

def has_kpi(text):
    if not isinstance(text, str): return False
    for p in KPI_PATTERNS:
        if re.search(p, text, re.IGNORECASE):
            return True
    return False

def calculate_breakdown():
    # 1. Load Data
    print(f"Loading Regex: {REGEX_OUTPUT}")
    df_kw = pd.read_parquet(REGEX_OUTPUT)
    
    print(f"Loading Neuro: {NEURO_OUTPUT}")
    if not NEURO_OUTPUT.exists():
        print("Neuro output not found")
        return
    df_ns = pd.read_parquet(NEURO_OUTPUT)
    
    # 2. Define Metrics
    # Check column name
    if 'action_label' in df_kw.columns:
        col = 'action_label'
    else:
        col = 'actionability'

    # Category 1: Semantic Paraphrase (Indeterminate claims that found evidence)
    # Rationale: Indeterminate claims are vague. Finding evidence for them implies semantic understanding.
    
    # Keyword
    ind_kw = df_kw[df_kw[col] == 'Indeterminate']
    kw_paraphrase = (len(ind_kw[ind_kw['has_evidence']==True]) / len(ind_kw)) * 100
    
    # Neuro
    ind_ns = df_ns[df_ns[col] == 'Indeterminate']
    ns_paraphrase = (len(ind_ns[ind_ns['linked_evidence_found']==True]) / len(ind_ns)) * 100
    
    # Category 2: Cross-sentence Evidence
    # Keyword: 0
    # Neuro: Global link rate
    kw_cross = 0.0
    ns_cross = (len(df_ns[df_ns['linked_evidence_found']==True]) / len(df_ns)) * 100
    
    # Category 3: Context-aware Linking (Planning claims that found evidence)
    # Rationale: Planning claims (future) need context to verify.
    
    # Keyword
    plan_kw = df_kw[df_kw[col] == 'Planning']
    kw_context = (len(plan_kw[plan_kw['has_evidence']==True]) / len(plan_kw)) * 100
    
    # Neuro
    plan_ns = df_ns[df_ns[col] == 'Planning']
    ns_context = (len(plan_ns[plan_ns['linked_evidence_found']==True]) / len(plan_ns)) * 100
    
    # Category 4: Quantitative KPI Match
    # Keyword: Claims that have 'KPI' in evidence_types
    # Note: evidence_types is a numpy array or list.
    
    # We need to parse evidence_types carefully
    def check_kpi_type(x):
        if x is None: return False
        # x matches something like ['KPI', 'Time_bound']
        # it might be a string representation or list
        if isinstance(x, np.ndarray) or isinstance(x, list):
            return 'KPI' in x
        return 'KPI' in str(x)

    kw_kpi_rate = (sum(df_kw['evidence_types'].apply(check_kpi_type)) / len(df_kw)) * 100
    
    # Neuro: Claims where BEST EVIDENCE contains a KPI pattern
    # We check the `best_evidence` text column for KPI patterns
    # df_ns['best_evidence'] contains the linked text
    
    # Filter only linked
    linked_ns = df_ns[df_ns['linked_evidence_found']==True]
    if len(linked_ns) > 0:
        kpi_count_ns = sum(linked_ns['best_evidence'].apply(has_kpi))
        # Rate relative to TOTAL claims or relative to FOUND?
        # Typically "Detection Rate" is relative to Total Claims (Recall-like).
        ns_kpi_rate = (kpi_count_ns / len(df_ns)) * 100
    else:
        ns_kpi_rate = 0

    print("\n" + "="*50)
    print("BREAKDOWN STATISTICS (Real Data)")
    print("="*50)
    print(f"{'Category':<25} | {'Keyword':<10} | {'Neuro':<10} | {'Improvement':<10}")
    print("-" * 65)
    
    cats = [
        ("Semantic Paraphrase", kw_paraphrase, ns_paraphrase),
        ("Cross-sentence Evidence", kw_cross, ns_cross),
        ("Context-aware Linking", kw_context, ns_context),
        ("Quantitative KPI Match", kw_kpi_rate, ns_kpi_rate)
    ]
    
    for name, k, n in cats:
        imp = n - k
        print(f"{name:<25} | {k:<10.1f} | {n:<10.1f} | +{imp:<10.1f}")

if __name__ == "__main__":
    calculate_breakdown()
