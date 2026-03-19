import pandas as pd
from pathlib import Path
import argparse

ESG_SENTENCES_PATH = Path("data/corpus/esg_sentences.parquet")
OUTPUT_CSV = Path("data/labels/action_gold_to_label.csv")
OUTPUT_PARQUET = Path("data/labels/action_gold.parquet")


def create_gold_sample(n: int = 500, seed: int = 42) -> pd.DataFrame:
    df = pd.read_parquet(ESG_SENTENCES_PATH)
    
    # Filter noise
    df = df[~df["sentence"].str.contains(r"<!-- image -->|^\d+$", regex=True, na=False)]
    df = df[df["sentence"].str.len() >= 20]
    df = df[df["sentence"].str.len() <= 500]  # Not too long
    
    print(f"Total ESG sentences after filtering: {len(df):,}")
    print(f"Topic distribution:\n{df['predicted_label'].value_counts()}")
    print(f"\nBlock type distribution:\n{df['block_type'].value_counts()}")
    
    # Stratified sampling
    samples = []
    
    # Primary stratification: ESG topic
    topics = df["predicted_label"].unique()
    n_per_topic = n // len(topics)
    
    for topic in topics:
        topic_df = df[df["predicted_label"] == topic]
        
        # Secondary stratification: block_type
        block_types = topic_df["block_type"].value_counts()
        for btype, count in block_types.items():
            btype_df = topic_df[topic_df["block_type"] == btype]
            # Proportional sampling within topic
            n_sample = max(2, int(n_per_topic * (count / len(topic_df))))
            n_sample = min(n_sample, len(btype_df))
            samples.append(btype_df.sample(n=n_sample, random_state=seed))
    
    result = pd.concat(samples).drop_duplicates(subset=["sent_id"])
    
    # If not enough, add more randomly
    if len(result) < n:
        remaining = df[~df["sent_id"].isin(result["sent_id"])]
        extra = remaining.sample(n=min(n - len(result), len(remaining)), random_state=seed)
        result = pd.concat([result, extra])
    
    result = result.head(n).reset_index(drop=True)
    
    print(f"\n=== GOLD SET STATS ===")
    print(f"Total samples: {len(result)}")
    print(f"\nBy topic:\n{result['predicted_label'].value_counts()}")
    print(f"\nBy block_type:\n{result['block_type'].value_counts()}")
    
    return result


def export_for_labeling(df: pd.DataFrame):
    export_df = df[[
        "sent_id", "sentence", "ctx_prev", "ctx_next", 
        "section_title", "block_type", "predicted_label",
        "bank", "year"
    ]].copy()
    
    # Add columns for human annotation
    export_df["gold_action"] = ""  # Implemented / Planning / Indeterminate
    export_df["gold_note"] = ""
    
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nExported for labeling: {OUTPUT_CSV}")
    print("\n=== LABELING INSTRUCTIONS ===")
    print("Fill in 'gold_action' column with one of:")
    print("  - Implemented: Đã triển khai / có kết quả / có số liệu")
    print("  - Planning: Kế hoạch / mục tiêu tương lai")
    print("  - Indeterminate: Cam kết chung chung, không rõ hành động")


def import_labeled(input_csv: Path = OUTPUT_CSV):
    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    
    # Check if manual labels exist
    valid_labels = {"Implemented", "Planning", "Indeterminate"}
    manually_labeled = df[df["gold_action"].isin(valid_labels)]
    
    if len(manually_labeled) > 0:
        print(f"Found {len(manually_labeled)} manually labeled samples")
        labeled_df = manually_labeled.copy()
    else:
        print("No manual labels found. Using LLM labels as pseudo-gold (same as topic classification)...")
        # Load LLM labels and merge
        llm_path = Path("data/labels/action_llm.parquet")
        if not llm_path.exists():
            print(f"ERROR: LLM labels not found at {llm_path}. Run llm_labeler.py first.")
            return None
        
        llm_df = pd.read_parquet(llm_path)
        df = df.merge(
            llm_df[["sent_id", "llm_action", "llm_conf"]],
            on="sent_id",
            how="left"
        )
        
        # Use LLM action as gold, filtered by confidence
        df["gold_action"] = df.apply(
            lambda r: r["llm_action"] if pd.notna(r["llm_action"]) and r["llm_conf"] >= 0.6
            else "Indeterminate",
            axis=1
        )
        labeled_df = df[df["gold_action"].isin(valid_labels)].copy()
    
    print(f"Gold set: {len(labeled_df)} samples")
    print(f"Label distribution:\n{labeled_df['gold_action'].value_counts()}")
    
    if len(labeled_df) > 0:
        labeled_df.to_parquet(OUTPUT_PARQUET, index=False)
        print(f"\nSaved: {OUTPUT_PARQUET}")
    
    return labeled_df


def create_gold_from_llm(n: int = 500, min_conf: float = 0.7, seed: int = 42):
    llm_path = Path("data/labels/action_llm.parquet")
    if not llm_path.exists():
        print(f"ERROR: Run llm_labeler.py first to generate {llm_path}")
        return None
    
    print("Loading LLM labels...")
    llm_df = pd.read_parquet(llm_path)
    
    # Filter by confidence
    high_conf = llm_df[llm_df["llm_conf"] >= min_conf]
    print(f"High confidence samples (>= {min_conf}): {len(high_conf)}")
    
    # Stratified sampling by action label
    samples = []
    for action in ["Implemented", "Planning", "Indeterminate"]:
        subset = high_conf[high_conf["llm_action"] == action]
        n_sample = min(len(subset), n // 3)
        if n_sample > 0:
            samples.append(subset.sample(n=n_sample, random_state=seed))
    
    gold_df = pd.concat(samples).reset_index(drop=True)
    gold_df["gold_action"] = gold_df["llm_action"]  # LLM as pseudo-gold
    
    print(f"\n=== PSEUDO-GOLD SET (LLM as teacher) ===")
    print(f"Total: {len(gold_df)}")
    print(f"Distribution:\n{gold_df['gold_action'].value_counts()}")
    
    gold_df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"\nSaved: {OUTPUT_PARQUET}")
    
    return gold_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500, help="Number of samples")
    parser.add_argument("--import-labeled", action="store_true", help="Import labeled CSV")
    parser.add_argument("--from-llm", action="store_true", help="Create gold from LLM (no manual labeling)")
    parser.add_argument("--min-conf", type=float, default=0.7, help="Min LLM confidence for gold")
    parser.add_argument("--input-csv", type=Path, default=OUTPUT_CSV)
    args = parser.parse_args()
    
    if args.from_llm:
        create_gold_from_llm(args.n, args.min_conf)
    elif args.import_labeled:
        import_labeled(args.input_csv)
    else:
        df = create_gold_sample(args.n)
        export_for_labeling(df)
