"""
Label Ablation Study: 6 vs 3 vs 2 labels

Compares performance of ESG topic classification with different
label granularities to justify the 6-label taxonomy choice.

Configurations:
- 6 labels: E, S_labor, S_community, S_product, G, Non_ESG (proposed)
- 3 labels: E, S, G, Non_ESG (standard ESG split)
- 2 labels: ESG, Non_ESG (binary)

If 6 labels achieves comparable F1 to 3-label → fine-grained analysis
is possible WITHOUT sacrificing accuracy.

Usage:
    python src/evaluation/label_ablation.py --gold data/labels/topic_gold.parquet
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
from sklearn.metrics import f1_score, accuracy_score, classification_report

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# Label mapping
LABEL_MAP_6_TO_3 = {
    "E": "E",
    "S_labor": "S",
    "S_community": "S",
    "S_product": "S",
    "G": "G",
    "Non_ESG": "Non_ESG",
}

LABEL_MAP_6_TO_2 = {
    "E": "ESG",
    "S_labor": "ESG",
    "S_community": "ESG",
    "S_product": "ESG",
    "G": "ESG",
    "Non_ESG": "Non_ESG",
}


def run_ablation(
    gold_df: pd.DataFrame,
    text_col: str = "sentence",
    label_col: str = "gold_topic",
) -> pd.DataFrame:
    """Run label ablation study."""
    from labeling.grounded_rules import match_topic_grounded
    
    # Get 6-label predictions
    preds_6 = []
    for _, row in gold_df.iterrows():
        text = row[text_col]
        ctx = ""
        if "ctx_prev" in row and "ctx_next" in row:
            ctx = f"{row.get('ctx_prev', '')} {text} {row.get('ctx_next', '')}".lower()
        section = row.get("section_title", "")
        topic, conf, _ = match_topic_grounded(text, ctx, section)
        preds_6.append(topic)
    
    y_true_6 = gold_df[label_col].tolist()
    
    # Map to 3-label and 2-label
    y_true_3 = [LABEL_MAP_6_TO_3.get(l, l) for l in y_true_6]
    preds_3 = [LABEL_MAP_6_TO_3.get(p, p) for p in preds_6]
    
    y_true_2 = [LABEL_MAP_6_TO_2.get(l, l) for l in y_true_6]
    preds_2 = [LABEL_MAP_6_TO_2.get(p, p) for p in preds_6]
    
    # Evaluate each configuration
    results = []
    
    configs = [
        ("6 labels (proposed)", y_true_6, preds_6, 
         ["E", "S_labor", "S_community", "S_product", "G", "Non_ESG"]),
        ("3 labels (E/S/G)", y_true_3, preds_3, 
         ["E", "S", "G", "Non_ESG"]),
        ("2 labels (binary)", y_true_2, preds_2, 
         ["ESG", "Non_ESG"]),
    ]
    
    for name, y_true, y_pred, labels in configs:
        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        
        results.append({
            "Config": name,
            "Num Labels": len(labels),
            "Accuracy": round(acc, 4),
            "F1 (Macro)": round(f1_macro, 4),
            "F1 (Weighted)": round(f1_weighted, 4),
        })
        
        print(f"\n{'=' * 50}")
        print(f"CONFIG: {name}")
        print(f"{'=' * 50}")
        print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    
    summary_df = pd.DataFrame(results)
    
    # Print comparison
    print("\n" + "=" * 60)
    print("LABEL ABLATION SUMMARY")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    
    # Interpretation
    f1_6 = results[0]["F1 (Macro)"]
    f1_3 = results[1]["F1 (Macro)"]
    f1_2 = results[2]["F1 (Macro)"]
    
    print(f"\n6-label vs 3-label: ΔF1 = {f1_6 - f1_3:+.4f}")
    print(f"6-label vs 2-label: ΔF1 = {f1_6 - f1_2:+.4f}")
    
    if f1_6 >= f1_3 - 0.03:
        print("→ 6-label achieves comparable/better F1: fine-grained analysis justified!")
    else:
        print("→ 6-label has lower F1: consider merging some subtopics")
    
    return summary_df


def run(gold_path: str = None):
    if gold_path is None:
        gold_path = "data/labels/topic_gold.parquet"
    
    path = Path(gold_path)
    if not path.exists():
        print(f"Gold set not found: {path}")
        return
    
    gold_df = pd.read_parquet(path)
    print(f"Loaded {len(gold_df)} samples")
    
    summary = run_ablation(gold_df)
    
    output_dir = Path("exports/tables")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "label_ablation.csv", index=False)
    print(f"\nSaved: {output_dir / 'label_ablation.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=str, default=None)
    args = parser.parse_args()
    run(args.gold)
