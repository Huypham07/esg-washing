import re
import pandas as pd
from pathlib import Path

LLM_LABELS = Path("data/labels/action_llm.parquet")
WEAK_LABELS = Path("data/labels/action_weak.parquet")
OUTPUT_PATH = Path("data/labels/action_hybrid_train.parquet")

LABELS = ["Implemented", "Planning", "Indeterminate"]

IMPLEMENTED_STRONG = [
    r"(đã|năm \d{4}).*\d+\s*(%|tỷ|triệu|tấn|kWh)",  # Past + KPI
    r"đã (triển khai|thực hiện|hoàn thành|đạt được)",
    r"(hoàn thành|ghi nhận)\s+\d+",
]

PLANNING_STRONG = [
    r"(mục tiêu|kế hoạch|dự kiến).*\b(2025|2026|2027|2028|2029|2030|2050)\b",
    r"(sẽ|dự kiến)\s+(triển khai|thực hiện|đạt)",
    r"đến năm\s+(2025|2026|2027|2028|2029|2030)",
]


def create_hybrid_labels(
    llm_min_conf: float = 0.6,
    weak_override_conf: float = 0.6,
) -> pd.DataFrame:
    print("Loading labels...")
    llm = pd.read_parquet(LLM_LABELS)
    weak = pd.read_parquet(WEAK_LABELS)
    
    df = llm.merge(
        weak[["sent_id", "weak_action", "weak_conf"]],
        on="sent_id",
        how="left"
    )
    
    print(f"Total samples: {len(df)}")
    print(f"LLM labels: {len(llm)}, Weak overlap: {df['weak_action'].notna().sum()}")
    
    # Start with LLM label
    df["final_action"] = df["llm_action"]
    df["label_source"] = "llm"
    
    # === STRONG PATTERN OVERRIDE ===
    def check_strong_patterns(text: str) -> tuple[str, bool]:
        text_lower = text.lower()
        for pat in IMPLEMENTED_STRONG:
            if re.search(pat, text_lower, re.IGNORECASE):
                return "Implemented", True
        for pat in PLANNING_STRONG:
            if re.search(pat, text_lower, re.IGNORECASE):
                return "Planning", True
        return None, False
    
    overrides = df["sentence"].apply(check_strong_patterns)
    for idx, (label, is_strong) in enumerate(overrides):
        if is_strong and df.loc[idx, "llm_conf"] < 0.85:
            df.loc[idx, "final_action"] = label
            df.loc[idx, "label_source"] = "weak_strong_pattern"
    
    strong_override = (df["label_source"] == "weak_strong_pattern").sum()
    print(f"Strong pattern override: {strong_override}")
    
    # === WEAK HIGH-CONF OVERRIDE (when LLM is uncertain) ===
    mask_weak_better = (
        (df["weak_conf"] >= weak_override_conf) &
        (df["llm_conf"] < 0.7) &
        (df["weak_action"] != df["llm_action"])
    )
    df.loc[mask_weak_better, "final_action"] = df.loc[mask_weak_better, "weak_action"]
    df.loc[mask_weak_better, "label_source"] = "weak_high_conf"
    print(f"Weak high-conf override: {mask_weak_better.sum()}")
    
    # === AGREEMENT BOOST ===
    mask_agree = (df["weak_action"] == df["llm_action"])
    df.loc[mask_agree, "label_source"] = "llm_weak_agree"
    print(f"LLM+Weak agreement: {mask_agree.sum()}")
    
    # Filter by LLM confidence
    df = df[df["llm_conf"] >= llm_min_conf].copy()
    print(f"After LLM conf filter (>={llm_min_conf}): {len(df)}")
    
    # Final distribution
    print("\n=== HYBRID LABELS DISTRIBUTION ===")
    print(df["final_action"].value_counts())
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
    """Split hybrid labels into train/val, stratified by action"""
    from sklearn.model_selection import train_test_split
    
    df = pd.read_parquet(OUTPUT_PATH)
    
    train, val = train_test_split(
        df,
        test_size=val_ratio,
        stratify=df["final_action"],
        random_state=seed,
    )
    
    train.to_parquet(OUTPUT_PATH.parent / "action_hybrid_train_split.parquet", index=False)
    val.to_parquet(OUTPUT_PATH.parent / "action_hybrid_val_split.parquet", index=False)
    
    print(f"Train: {len(train)}, Val: {len(val)}")
    print("\nTrain distribution:")
    print(train["final_action"].value_counts())
    print("\nVal distribution:")
    print(val["final_action"].value_counts())
    
    return train, val


if __name__ == "__main__":
    df = create_hybrid_labels()
    prepare_train_val_split()
