import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from baselines.keyword_baseline import classify_sentence_keyword, map_to_actionability

def evaluate_actionability():
    print("Evaluating Actionability: Keyword Baseline vs Gold Set")
    df = pd.read_parquet("data/labels/action_gold.parquet")
    
    y_true = df["gold_action"].tolist()
    y_pred_keyword = []
    
    for text in df["sentence"]:
        kw_label, _ = classify_sentence_keyword(text)
        action_pred = map_to_actionability(kw_label)
        y_pred_keyword.append(action_pred)
        
    labels = ["Implemented", "Planning", "Indeterminate"]
    
    acc = accuracy_score(y_true, y_pred_keyword)
    macro_f1 = f1_score(y_true, y_pred_keyword, labels=labels, average="macro", zero_division=0)
    
    per_class_f1 = f1_score(y_true, y_pred_keyword, labels=labels, average=None, zero_division=0)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    for lbl, score in zip(labels, per_class_f1):
        print(f"F1 {lbl}: {score:.4f}")

if __name__ == "__main__":
    evaluate_actionability()
