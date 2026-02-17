import pandas as pd
from pathlib import Path

LLM_LABELS = Path("data/labels/llm_prelabels.parquet")
WEAK_LABELS = Path("data/labels/weak_labels.parquet")
OUTPUT_PATH = Path("data/labels/hybrid_train.parquet")

TOPICS = ["E", "S_labor", "S_community", "S_product", "G", "Non_ESG"]

NON_ESG_BLOCK_TYPES = {"meta_heading"}

NON_ESG_PATTERNS = [
    r"<!-- image -->",
    r"^\d+$",
    r"^(thạc sĩ|tiến sĩ|cử nhân)",
    r"(sinh năm \d{4})",
    r"^(ông|bà)\s+[A-ZĐÀÁẢÃẠ]",
]


def create_hybrid_labels(
    llm_min_conf: float = 0.6,
    weak_non_esg_conf: float = 0.7,
) -> pd.DataFrame:
    print("Loading labels...")
    llm = pd.read_parquet(LLM_LABELS)
    weak = pd.read_parquet(WEAK_LABELS)
    
    df = llm.merge(
        weak[["sent_id", "weak_topic", "weak_conf"]],
        on="sent_id",
        how="left"
    )
    
    print(f"Total samples: {len(df)}")
    print(f"LLM labels: {len(llm)}, Weak overlap: {df['weak_topic'].notna().sum()}")
    
    # Start with LLM label
    df["final_topic"] = df["llm_topic"]
    df["label_source"] = "llm"
    
    # === HARD FILTER 1: Block type ===
    mask_block = df["block_type"].isin(NON_ESG_BLOCK_TYPES)
    df.loc[mask_block, "final_topic"] = "Non_ESG"
    df.loc[mask_block, "label_source"] = "weak_block_filter"
    print(f"Block type filter: {mask_block.sum()} → Non_ESG")
    
    # === HARD FILTER 2: Weak high-confidence Non_ESG ===
    mask_weak = (
        (df["weak_topic"] == "Non_ESG") & 
        (df["weak_conf"] >= weak_non_esg_conf) &
        (df["llm_topic"] != "Non_ESG")  # Only override if LLM disagrees
    )
    mask_weak = mask_weak & (df["llm_conf"] < 0.85)
    df.loc[mask_weak, "final_topic"] = "Non_ESG"
    df.loc[mask_weak, "label_source"] = "weak_high_conf"
    print(f"Weak high-conf filter: {mask_weak.sum()} → Non_ESG")
    
    # === AGREEMENT BOOST: Both agree on ESG topic ===
    mask_agree = (
        (df["weak_topic"] == df["llm_topic"]) & 
        (df["llm_topic"] != "Non_ESG")
    )
    df.loc[mask_agree, "label_source"] = "llm_weak_agree"
    print(f"LLM+Weak agreement (ESG): {mask_agree.sum()}")
    
    # Filter by LLM confidence
    df = df[df["llm_conf"] >= llm_min_conf].copy()
    print(f"After LLM conf filter (>={llm_min_conf}): {len(df)}")
    
    # Final distribution
    print("\n=== HYBRID LABELS DISTRIBUTION ===")
    print(df["final_topic"].value_counts())
    print("\nLabel source:")
    print(df["label_source"].value_counts())
    
    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}")
    
    return df


def prepare_train_val_split(
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split hybrid labels into train/val, stratified by topic"""
    from sklearn.model_selection import train_test_split
    
    df = pd.read_parquet(OUTPUT_PATH)
    
    train, val = train_test_split(
        df,
        test_size=val_ratio,
        stratify=df["final_topic"],
        random_state=seed,
    )
    
    train.to_parquet(OUTPUT_PATH.parent / "hybrid_train_split.parquet", index=False)
    val.to_parquet(OUTPUT_PATH.parent / "hybrid_val_split.parquet", index=False)
    
    print(f"Train: {len(train)}, Val: {len(val)}")
    print("\nTrain distribution:")
    print(train["final_topic"].value_counts())
    print("\nVal distribution:")
    print(val["final_topic"].value_counts())
    
    return train, val


if __name__ == "__main__":
    df = create_hybrid_labels()
    prepare_train_val_split()
