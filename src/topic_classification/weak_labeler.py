import re
import pandas as pd
from pathlib import Path

SENTENCES_PATH = Path("data/corpus/sentences.parquet")
OUTPUT_PATH = Path("data/labels/weak_labels.parquet")

def match_rules(row: pd.Series) -> tuple[str, float]:
    text = row["sentence"].lower()
    ctx = f"{row['ctx_prev']} {row['sentence']} {row['ctx_next']}".lower()
    section = row["section_title"].lower()
    
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from labeling.grounded_rules import match_topic_grounded
        grounded_topic, grounded_conf, grounded_matched = match_topic_grounded(
            text, ctx, section
        )
    except ImportError:
        grounded_topic, grounded_conf, grounded_matched = "Non_ESG", 0.0, []
    
    return grounded_topic, grounded_conf


def run(min_confidence: float = 0.4):
    print("Loading sentences...")
    df = pd.read_parquet(SENTENCES_PATH)
    print(f"Total sentences: {len(df):,}")
    
    print("Applying weak labeling rules...")
    results = df.apply(match_rules, axis=1)
    df["weak_topic"] = [r[0] for r in results]
    df["weak_conf"] = [r[1] for r in results]
    
    # Filter by confidence
    df_labeled = df[df["weak_conf"] >= min_confidence].copy()
    
    print(f"\nWeak labels distribution (conf >= {min_confidence}):")
    print(df_labeled["weak_topic"].value_counts())
    print(f"\nTotal labeled: {len(df_labeled):,} / {len(df):,} ({100*len(df_labeled)/len(df):.1f}%)")
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_labeled.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}")
    
    return df_labeled


if __name__ == "__main__":
    run()
