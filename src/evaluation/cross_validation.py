"""
K-Fold Stratified Cross-Validation for ESG Classifiers

Provides rigorous evaluation of both ESG Topic Classifier and
Actionability Classifier using stratified k-fold cross-validation.

Reports mean ± std for each metric across folds.

Usage:
    python src/evaluation/cross_validation.py \
        --data data/labels/gold_set_topic.parquet \
        --task topic \
        --k 5
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    classification_report,
)


def cross_validate_classifier(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    predict_fn,
    k: int = 5,
    labels: List[str] = None,
) -> Dict:
    """
    Run stratified k-fold cross-validation.
    
    Args:
        df: DataFrame with text and labels
        text_col: Column name for text
        label_col: Column name for ground truth labels
        predict_fn: Function(texts: List[str]) -> List[str] predictions
        k: Number of folds
        labels: Label set
        
    Returns:
        Dictionary with per-fold and aggregated metrics
    """
    if labels is None:
        labels = sorted(df[label_col].unique().tolist())
    
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(df, df[label_col])):
        print(f"\n--- Fold {fold_idx + 1}/{k} ---")
        
        test_df = df.iloc[test_idx]
        y_true = test_df[label_col].tolist()
        texts = test_df[text_col].tolist()
        
        # Get predictions
        y_pred = predict_fn(texts)
        
        # Compute metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        precision = precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        
        fold_result = {
            "fold": fold_idx + 1,
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "precision": precision,
            "recall": recall,
        }
        
        # Per-class F1
        per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        for i, label in enumerate(labels):
            fold_result[f"f1_{label}"] = per_class_f1[i]
        
        fold_results.append(fold_result)
        print(f"  Accuracy: {accuracy:.4f}, F1 (macro): {f1_macro:.4f}, F1 (weighted): {f1_weighted:.4f}")
    
    # Aggregate
    fold_df = pd.DataFrame(fold_results)
    
    summary = {}
    for metric in ["accuracy", "f1_macro", "f1_weighted", "precision", "recall"]:
        values = fold_df[metric].values
        summary[metric] = {
            "mean": round(np.mean(values), 4),
            "std": round(np.std(values), 4),
            "min": round(np.min(values), 4),
            "max": round(np.max(values), 4),
        }
    
    for label in labels:
        col = f"f1_{label}"
        if col in fold_df.columns:
            values = fold_df[col].values
            summary[col] = {
                "mean": round(np.mean(values), 4),
                "std": round(np.std(values), 4),
            }
    
    return {
        "k": k,
        "num_samples": len(df),
        "labels": labels,
        "fold_results": fold_results,
        "summary": summary,
    }


def print_cv_results(results: Dict):
    """Pretty-print cross-validation results."""
    print("\n" + "=" * 60)
    print(f"{results['k']}-FOLD CROSS-VALIDATION RESULTS")
    print(f"Samples: {results['num_samples']}, Labels: {results['labels']}")
    print("=" * 60)
    
    print("\nOverall Metrics (mean ± std):")
    for metric in ["accuracy", "f1_macro", "f1_weighted", "precision", "recall"]:
        s = results["summary"][metric]
        print(f"  {metric:20s}: {s['mean']:.4f} ± {s['std']:.4f}  (range: {s['min']:.4f}-{s['max']:.4f})")
    
    print("\nPer-class F1 (mean ± std):")
    for label in results["labels"]:
        col = f"f1_{label}"
        if col in results["summary"]:
            s = results["summary"][col]
            print(f"  {label:20s}: {s['mean']:.4f} ± {s['std']:.4f}")


def run(data_path: str = None, task: str = "action", k: int = 5):
    """Run cross-validation."""
    if data_path is None:
        if task == "topic":
            data_path = "data/labels/gold_set_topic.parquet"
        else:
            data_path = "data/labels/gold_set_action.parquet"
    
    data_file = Path(data_path)
    if not data_file.exists():
        print(f"Data not found at: {data_file}")
        return
    
    df = pd.read_parquet(data_file)
    print(f"Loaded {len(df)} samples from {data_file}")
    
    if task == "topic":
        text_col = "sentence"
        label_col = "gold_topic"
        labels = ["E", "S_labor", "S_community", "S_product", "G", "Non_ESG"]
        
        # Simple predict function using weak labeler
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from labeling.grounded_rules import match_topic_grounded
        
        def predict_fn(texts):
            return [match_topic_grounded(t)[0] for t in texts]
    else:
        text_col = "sentence"
        label_col = "gold_action"
        labels = ["Implemented", "Planning", "Indeterminate"]
        
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from labeling.grounded_rules import match_actionability_grounded
        
        def predict_fn(texts):
            return [match_actionability_grounded(t)[0] for t in texts]
    
    results = cross_validate_classifier(df, text_col, label_col, predict_fn, k=k, labels=labels)
    print_cv_results(results)
    
    # Save
    output_dir = Path("exports/tables")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fold_df = pd.DataFrame(results["fold_results"])
    fold_df.to_csv(output_dir / f"cv_{task}_{k}fold.csv", index=False)
    print(f"\nSaved: {output_dir / f'cv_{task}_{k}fold.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--task", type=str, default="action", choices=["topic", "action"])
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    
    run(args.data, args.task, args.k)
