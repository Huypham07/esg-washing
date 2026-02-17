import pandas as pd
from pathlib import Path

HYBRID_LABELS = Path("data/labels/hybrid_train.parquet")
OUTPUT_CSV = Path("data/labels/topic_gold.csv")
OUTPUT_PARQUET = Path("data/labels/topic_gold.parquet")

TOPICS = ["E", "S_labor", "S_community", "S_product", "G", "Non_ESG"]

SAMPLES_PER_TOPIC = {
    "E": 80,
    "S_labor": 80,
    "S_community": 80,
    "S_product": 80,
    "G": 90,
    "Non_ESG": 90,
}


def create_gold_set(seed: int = 42) -> pd.DataFrame:
    print("Loading hybrid labels...")
    df = pd.read_parquet(HYBRID_LABELS)
    
    print(f"Total samples: {len(df)}")
    print(f"Label sources: {df['label_source'].value_counts().to_dict()}")
    
    samples = []
    
    for topic in TOPICS:
        n_target = SAMPLES_PER_TOPIC[topic]
        subset = df[df["final_topic"] == topic]
        
        agree = subset[subset["label_source"] == "llm_weak_agree"]
        llm_only = subset[subset["label_source"] == "llm"]
        
        n_agree = min(len(agree), int(n_target * 0.7))
        n_llm = min(len(llm_only), n_target - n_agree)
        
        if n_agree > 0:
            samples.append(agree.sample(n=n_agree, random_state=seed))
        if n_llm > 0:
            samples.append(llm_only.sample(n=n_llm, random_state=seed))
        
        print(f"{topic}: {n_agree + n_llm} (agree={n_agree}, llm={n_llm})")
    
    df_gold = pd.concat(samples, ignore_index=True)
    df_gold = df_gold.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    df_export = df_gold[[
        "sent_id", "doc_id", "bank", "year", "section_title",
        "sentence", "ctx_prev", "ctx_next", "block_type",
        "final_topic", "llm_conf", "label_source"
    ]].copy()
    
    df_export["gold_topic"] = ""
    df_export["annotator_note"] = ""
    
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_export.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n=== GOLD SET CREATED ===")
    print(f"Total: {len(df_export)} samples")
    print(f"CSV: {OUTPUT_CSV}")
    print(f"\nDistribution:")
    print(df_export["final_topic"].value_counts())


def load_annotated_gold() -> pd.DataFrame:
    df = pd.read_csv(OUTPUT_CSV)
    
    df["gold_topic"] = df.apply(
        lambda r: r["gold_topic"] if pd.notna(r["gold_topic"]) and r["gold_topic"] != "" 
        else r["final_topic"],
        axis=1
    )
    
    df[["sent_id", "gold_topic", "sentence", "ctx_prev", "ctx_next", 
        "section_title", "block_type"]].to_parquet(OUTPUT_PARQUET, index=False)
    
    print(f"Gold labels: {len(df)}")
    print(df["gold_topic"].value_counts())
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action="store_true")
    args = parser.parse_args()
    
    if args.load:
        load_annotated_gold()
    else:
        create_gold_set()
