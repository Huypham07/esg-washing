import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    f1_score,
    accuracy_score,
)


# ============================================================
# EVALUATION FUNCTIONS
# ============================================================

def evaluate_method(
    y_true: List[str],
    y_pred: List[str],
    method_name: str,
    labels: List[str] = None,
) -> Dict:
    """
    Evaluate a single method against ground truth.
    
    Returns:
        Dictionary with all metrics.
    """
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))
    
    # Classification report
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    return {
        "method": method_name,
        "accuracy": round(accuracy, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "kappa": round(kappa, 4),
        "report": report,
        "confusion_matrix": cm,
        "per_class": {
            label: {
                "precision": round(report.get(label, {}).get("precision", 0), 4),
                "recall": round(report.get(label, {}).get("recall", 0), 4),
                "f1": round(report.get(label, {}).get("f1-score", 0), 4),
                "support": report.get(label, {}).get("support", 0),
            }
            for label in labels
        },
    }


def mcnemar_test(y_true: List[str], y_pred_a: List[str], y_pred_b: List[str]) -> Dict:
    """
    McNemar's test for comparing two classifiers.
    Tests if the two methods make significantly different errors.
    
    Returns:
        Dictionary with chi-square statistic and p-value.
    """
    from scipy.stats import chi2
    
    # Build contingency table
    # n_01: A correct, B wrong
    # n_10: A wrong, B correct
    n_01 = sum(1 for t, a, b in zip(y_true, y_pred_a, y_pred_b) if a == t and b != t)
    n_10 = sum(1 for t, a, b in zip(y_true, y_pred_a, y_pred_b) if a != t and b == t)
    
    # McNemar's statistic (with continuity correction)
    if n_01 + n_10 == 0:
        return {"chi2": 0, "p_value": 1.0, "significant": False}
    
    chi2_stat = (abs(n_01 - n_10) - 1) ** 2 / (n_01 + n_10)
    p_value = 1 - chi2.cdf(chi2_stat, df=1)
    
    return {
        "chi2": round(chi2_stat, 4),
        "p_value": round(p_value, 4),
        "significant": p_value < 0.05,
        "n_01_only_a_correct": n_01,
        "n_10_only_b_correct": n_10,
    }


def compare_all_methods(
    gold_df: pd.DataFrame,
    text_col: str = "sentence",
    true_label_col: str = "action_label",
) -> pd.DataFrame:
    """
    Run all baseline methods and compare against ground truth.
    
    Args:
        gold_df: DataFrame with gold-standard labels
        text_col: Column containing sentence text
        true_label_col: Column containing ground truth labels
        
    Returns:
        Summary DataFrame with all results.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    
    from keyword_baseline import classify_sentence_keyword, map_to_actionability as kw_map
    from tfidf_sentiment_baseline import compute_sentiment, compute_specificity
    from greenwashing_bert_baseline import classify_greenwashing_label, map_to_actionability as gw_map
    
    y_true = gold_df[true_label_col].tolist()
    labels = ["Implemented", "Planning", "Indeterminate"]
    
    results = []
    predictions = {}
    
    # ---- Method 1: Keyword baseline ----
    print("Running Keyword baseline...")
    kw_preds = []
    for text in gold_df[text_col]:
        label, _ = classify_sentence_keyword(text)
        kw_preds.append(kw_map(label))
    predictions["Keyword"] = kw_preds
    results.append(evaluate_method(y_true, kw_preds, "Keyword (Florstedt et al. 2025)", labels))
    
    # ---- Method 2: TF-IDF + Sentiment ----
    print("Running TF-IDF + Sentiment baseline...")
    tfidf_preds = []
    for text in gold_df[text_col]:
        sent = compute_sentiment(text)
        spec = compute_specificity(text)
        if spec >= 0.5:
            tfidf_preds.append("Implemented")
        elif max(sent, 0) >= 0.5 and spec < 0.3:
            tfidf_preds.append("Indeterminate")
        else:
            tfidf_preds.append("Planning")
    predictions["TF-IDF+Sentiment"] = tfidf_preds
    results.append(evaluate_method(y_true, tfidf_preds, "TF-IDF+Sentiment (Lagasio 2024)", labels))
    
    # ---- Method 3: Green-Washing BERT ----
    print("Running Green-Washing BERT baseline...")
    gw_preds = []
    for text in gold_df[text_col]:
        label, _, _ = classify_greenwashing_label(text)
        gw_preds.append(gw_map(label))
    predictions["GW-BERT"] = gw_preds
    results.append(evaluate_method(y_true, gw_preds, "GW-BERT (Green-Washing repo)", labels))
    
    # ---- Method 4: Proposed (if predictions available) ----
    proposed_col = "predicted_action" if "predicted_action" in gold_df.columns else None
    if proposed_col is None and "action_pred" in gold_df.columns:
        proposed_col = "action_pred"
    
    if proposed_col:
        print("Evaluating Proposed method...")
        proposed_preds = gold_df[proposed_col].tolist()
        predictions["Proposed"] = proposed_preds
        results.append(evaluate_method(y_true, proposed_preds, "Proposed (EVINCE)", labels))
    
    # ---- Summary Table ----
    summary = []
    for r in results:
        row = {
            "Method": r["method"],
            "Accuracy": r["accuracy"],
            "F1 (Macro)": r["f1_macro"],
            "F1 (Weighted)": r["f1_weighted"],
            "Cohen's κ": r["kappa"],
        }
        for label in labels:
            row[f"F1_{label}"] = r["per_class"][label]["f1"]
        summary.append(row)
    
    summary_df = pd.DataFrame(summary)
    
    # ---- Statistical Tests ----
    print("\n" + "=" * 60)
    print("STATISTICAL SIGNIFICANCE (McNemar's Test)")
    print("=" * 60)
    
    if "Proposed" in predictions:
        for method_name, method_preds in predictions.items():
            if method_name == "Proposed":
                continue
            test = mcnemar_test(y_true, predictions["Proposed"], method_preds)
            sig = "✓ Significant" if test["significant"] else "✗ Not significant"
            print(f"  Proposed vs {method_name}: χ²={test['chi2']}, p={test['p_value']} → {sig}")
    
    return summary_df, results, predictions


def print_comparison_table(summary_df: pd.DataFrame):
    """Pretty-print the comparison table."""
    print("\n" + "=" * 80)
    print("METHOD COMPARISON ON GOLD SET")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print()


def run(gold_set_path: str = None):
    """Run full evaluation."""
    if gold_set_path is None:
        gold_set_path = "data/labels/gold_set_action.parquet"
    
    gold_path = Path(gold_set_path)
    
    if not gold_path.exists():
        print(f"Gold set not found at: {gold_path}")
        print("Please provide the path to your gold-standard labeled dataset.")
        return
    
    print(f"Loading gold set from: {gold_path}")
    gold_df = pd.read_parquet(gold_path)
    print(f"Gold set size: {len(gold_df)}")
    
    summary_df, results, predictions = compare_all_methods(gold_df)
    print_comparison_table(summary_df)
    
    # Save results
    output_dir = Path("exports/tables")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_dir / "baseline_comparison.csv", index=False)
    print(f"Saved to: {output_dir / 'baseline_comparison.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ESG-Washing detection methods")
    parser.add_argument("--gold-set", type=str, default=None, help="Path to gold-standard dataset")
    args = parser.parse_args()
    
    run(args.gold_set)
