"""
Weight Tuning via Grid Search

Tunes the heuristic weights (w_text, w_ctx, threshold) in the grounded rules
using grid search on the validation/gold set to find optimal values.

Reports:
- Best weight configuration for each task (topic, actionability)
- Sensitivity heatmap: how much F1 changes across weight choices
- If F1 is stable → proves pipeline is robust to weight selection

Usage:
    python src/evaluation/weight_tuning.py \
        --task topic \
        --gold data/labels/topic_gold.parquet
"""

import argparse
import itertools
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.metrics import f1_score

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def tune_topic_weights(
    gold_df: pd.DataFrame,
    text_col: str = "sentence",
    label_col: str = "gold_topic",
    w_text_range: List[float] = [0.2, 0.3, 0.4, 0.5, 0.6],
    w_ctx_range: List[float] = [0.05, 0.1, 0.15, 0.2],
    threshold_range: List[float] = [0.2, 0.25, 0.3, 0.35, 0.4],
) -> pd.DataFrame:
    """
    Grid search for optimal topic classification weights.
    
    Temporarily patches the grounded_rules module with different weights,
    evaluates on gold set, and returns results.
    """
    import re
    from labeling.grounded_rules import ALL_TOPIC_RULES
    
    y_true = gold_df[label_col].tolist()
    labels = sorted(list(set(y_true)))
    
    results = []
    total = len(w_text_range) * len(w_ctx_range) * len(threshold_range)
    count = 0
    
    for w_text, w_ctx, threshold in itertools.product(w_text_range, w_ctx_range, threshold_range):
        count += 1
        if count % 20 == 0:
            print(f"  Grid search: {count}/{total}...")
        
        # Predict with these weights
        predictions = []
        for _, row in gold_df.iterrows():
            text = row[text_col].lower()
            ctx = ""
            if "ctx_prev" in row and "ctx_next" in row:
                ctx = f"{row.get('ctx_prev', '')} {row[text_col]} {row.get('ctx_next', '')}".lower()
            
            scores = {t: 0.0 for t in ALL_TOPIC_RULES}
            for topic, rules in ALL_TOPIC_RULES.items():
                for rule in rules:
                    for pattern in rule.patterns:
                        if re.search(pattern, text, re.IGNORECASE):
                            scores[topic] += w_text * rule.weight
                            break
                        elif ctx and re.search(pattern, ctx, re.IGNORECASE):
                            scores[topic] += w_ctx * rule.weight
                            break
            
            best_topic = max(scores, key=scores.get)
            best_score = scores[best_topic]
            predictions.append(best_topic if best_score >= threshold else "Non_ESG")
        
        # Evaluate
        f1_macro = f1_score(y_true, predictions, labels=labels, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true, predictions, labels=labels, average="weighted", zero_division=0)
        
        results.append({
            "w_text": w_text,
            "w_ctx": w_ctx,
            "threshold": threshold,
            "f1_macro": round(f1_macro, 4),
            "f1_weighted": round(f1_weighted, 4),
        })
    
    return pd.DataFrame(results)


def tune_action_weights(
    gold_df: pd.DataFrame,
    text_col: str = "sentence",
    label_col: str = "gold_action",
    w_text_range: List[float] = [0.2, 0.3, 0.35, 0.4, 0.5],
    w_ctx_range: List[float] = [0.05, 0.1, 0.15],
    threshold_range: List[float] = [0.2, 0.25, 0.3, 0.35],
) -> pd.DataFrame:
    """Grid search for optimal actionability weights."""
    import re
    from labeling.grounded_rules import ALL_ACTION_RULES
    
    y_true = gold_df[label_col].tolist()
    labels = ["Implemented", "Planning", "Indeterminate"]
    
    results = []
    total = len(w_text_range) * len(w_ctx_range) * len(threshold_range)
    count = 0
    
    for w_text, w_ctx, threshold in itertools.product(w_text_range, w_ctx_range, threshold_range):
        count += 1
        if count % 20 == 0:
            print(f"  Grid search: {count}/{total}...")
        
        predictions = []
        for _, row in gold_df.iterrows():
            text = row[text_col].lower()
            ctx = ""
            if "ctx_prev" in row and "ctx_next" in row:
                ctx = f"{row.get('ctx_prev', '')} {row[text_col]} {row.get('ctx_next', '')}".lower()
            
            scores = {label: 0.0 for label in ALL_ACTION_RULES}
            for label, rules in ALL_ACTION_RULES.items():
                for rule in rules:
                    for pattern in rule.patterns:
                        if re.search(pattern, text, re.IGNORECASE):
                            scores[label] += w_text * rule.weight
                            break
                        elif ctx and re.search(pattern, ctx, re.IGNORECASE):
                            scores[label] += w_ctx * rule.weight
                            break
            
            # Penalties
            has_numbers = bool(re.search(r"\d+\s*(%|tỷ|triệu|nghìn|tấn|kg|kWh|MWh)", text))
            if has_numbers:
                scores["Indeterminate"] -= 0.2
            
            best_label = max(scores, key=scores.get)
            best_score = scores[best_label]
            predictions.append(best_label if best_score >= threshold else "Indeterminate")
        
        f1_macro = f1_score(y_true, predictions, labels=labels, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true, predictions, labels=labels, average="weighted", zero_division=0)
        
        results.append({
            "w_text": w_text,
            "w_ctx": w_ctx,
            "threshold": threshold,
            "f1_macro": round(f1_macro, 4),
            "f1_weighted": round(f1_weighted, 4),
        })
    
    return pd.DataFrame(results)


def analyze_sensitivity(results_df: pd.DataFrame, task: str):
    """Analyze weight sensitivity and print report."""
    print(f"\n{'=' * 60}")
    print(f"WEIGHT TUNING RESULTS — {task.upper()}")
    print(f"{'=' * 60}")
    
    # Best configuration
    best = results_df.loc[results_df["f1_macro"].idxmax()]
    print(f"\nBest configuration:")
    print(f"  w_text={best['w_text']}, w_ctx={best['w_ctx']}, threshold={best['threshold']}")
    print(f"  F1 (macro)={best['f1_macro']:.4f}, F1 (weighted)={best['f1_weighted']:.4f}")
    
    # Sensitivity: range of F1 across all configs
    f1_range = results_df["f1_macro"].max() - results_df["f1_macro"].min()
    f1_std = results_df["f1_macro"].std()
    print(f"\nSensitivity analysis:")
    print(f"  F1 range: {results_df['f1_macro'].min():.4f} — {results_df['f1_macro'].max():.4f} (Δ={f1_range:.4f})")
    print(f"  F1 std: {f1_std:.4f}")
    
    if f1_range < 0.05:
        print(f"  → Pipeline is ROBUST to weight selection (Δ < 0.05)")
    elif f1_range < 0.10:
        print(f"  → Pipeline is MODERATELY sensitive to weights")
    else:
        print(f"  → Pipeline is SENSITIVE to weights — careful tuning needed")
    
    # Per-variable sensitivity
    for var in ["w_text", "w_ctx", "threshold"]:
        grouped = results_df.groupby(var)["f1_macro"].mean()
        print(f"\n  Mean F1 by {var}:")
        for val, f1 in grouped.items():
            print(f"    {var}={val}: F1={f1:.4f}")
    
    return best


def run(task: str = "topic", gold_path: str = None):
    """Run weight tuning."""
    if gold_path is None:
        gold_path = f"data/labels/{'topic_gold' if task == 'topic' else 'action_gold'}.parquet"
    
    path = Path(gold_path)
    if not path.exists():
        print(f"Gold set not found: {path}")
        return
    
    gold_df = pd.read_parquet(path)
    print(f"Loaded {len(gold_df)} samples from {path}")
    
    if task == "topic":
        results_df = tune_topic_weights(gold_df)
    else:
        results_df = tune_action_weights(gold_df)
    
    best = analyze_sensitivity(results_df, task)
    
    # Save
    output_dir = Path("exports/tables")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / f"weight_tuning_{task}.csv", index=False)
    print(f"\nSaved: {output_dir / f'weight_tuning_{task}.csv'}")
    
    return results_df, best


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["topic", "action"], default="topic")
    parser.add_argument("--gold", type=str, default=None)
    args = parser.parse_args()
    run(args.task, args.gold)
