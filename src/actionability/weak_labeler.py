import re
import pandas as pd
from pathlib import Path

ESG_SENTENCES_PATH = Path("data/corpus/esg_sentences.parquet")
OUTPUT_PATH = Path("data/labels/action_weak.parquet")

def match_actionability(row: pd.Series) -> tuple[str, float]:
    text = row["sentence"].lower()
    ctx = f"{row['ctx_prev']} {row['sentence']} {row['ctx_next']}".lower()
    
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from labeling.grounded_rules import match_actionability_grounded
        grounded_label, grounded_conf, grounded_matched = match_actionability_grounded(
            row["sentence"], ctx
        )
    except ImportError:
        grounded_label, grounded_conf, grounded_matched = "Indeterminate", 0.0, []
    
    return grounded_label, grounded_conf


def run(min_confidence: float = 0.4):
    print("Loading ESG sentences...")
    
    if not ESG_SENTENCES_PATH.exists():
        print("Creating ESG sentences file from predictions...")
        pred_path = Path("data/corpus/sentences_with_topic.parquet")
        if not pred_path.exists():
            raise FileNotFoundError(f"Run ESG classifier first: {pred_path}")
        
        df_all = pd.read_parquet(pred_path)
        df = df[df["topic_pred"] != "Non_ESG"].copy()
        
        ESG_SENTENCES_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(ESG_SENTENCES_PATH, index=False)
        print(f"Created: {ESG_SENTENCES_PATH} ({len(df):,} ESG sentences)")
    else:
        df = pd.read_parquet(ESG_SENTENCES_PATH)
    
    print(f"Total ESG sentences: {len(df):,}")
    
    print("Applying weak labeling rules...")
    results = df.apply(match_actionability, axis=1)
    df["weak_action"] = [r[0] for r in results]
    df["weak_conf"] = [r[1] for r in results]
    
    df_labeled = df[df["weak_conf"] >= min_confidence].copy()
    
    print(f"\nWeak labels distribution (conf >= {min_confidence}):")
    print(df_labeled["weak_action"].value_counts())
    print(f"\nTotal labeled: {len(df_labeled):,} / {len(df):,} ({100*len(df_labeled)/len(df):.1f}%)")
    
    print("\nConfidence distribution:")
    print(df_labeled.groupby("weak_action")["weak_conf"].describe()[["mean", "min", "max"]])
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_labeled.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}")
    
    return df_labeled


if __name__ == "__main__":
    run()
