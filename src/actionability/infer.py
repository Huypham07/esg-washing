import torch
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse

ESG_SENTENCES_PATH = Path("data/corpus/esg_sentences.parquet")
MODEL_PATH = Path("outputs/models/action_phobert_large")
OUTPUT_PATH = Path("data/corpus/esg_sentences_with_actionability.parquet")

LABELS = ["Implemented", "Planning", "Indeterminate"]


def load_model(model_path: Path = MODEL_PATH):
    """Load trained actionability model"""
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    print(f"Model loaded on {device}")
    return tokenizer, model, device


def predict_batch(
    texts: list[str],
    tokenizer,
    model,
    device,
    max_length: int = 256,
) -> list[dict]:
    """Predict actionability for a batch of texts"""
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        max_probs = probs.max(dim=-1).values
    
    results = []
    for pred, prob in zip(preds, max_probs):
        label = model.config.id2label[pred.item()]
        results.append({
            "action_label": label,
            "action_prob": prob.item(),
        })
    
    return results


def run_inference(batch_size: int = 64):
    """Run inference on full ESG corpus"""
    tokenizer, model, device = load_model()
    
    print(f"Loading ESG sentences from {ESG_SENTENCES_PATH}...")
    df = pd.read_parquet(ESG_SENTENCES_PATH)
    print(f"Total ESG sentences: {len(df):,}")
    
    print(f"Running inference with batch size {batch_size}...")
    sentences = df["sentence"].tolist()
    
    all_results = []
    for i in tqdm(range(0, len(sentences), batch_size)):
        batch = sentences[i:i + batch_size]
        results = predict_batch(batch, tokenizer, model, device)
        all_results.extend(results)
    
    # Add predictions to dataframe
    df["action_label"] = [r["action_label"] for r in all_results]
    df["action_prob"] = [r["action_prob"] for r in all_results]
    
    # Summary
    print("\n=== ACTIONABILITY DISTRIBUTION ===")
    print(df["action_label"].value_counts())
    print(f"\nPercentages:")
    print((df["action_label"].value_counts(normalize=True) * 100).round(1))
    
    # By bank-year
    print("\n=== BY BANK-YEAR ===")
    summary = df.groupby(["bank", "year", "action_label"]).size().unstack(fill_value=0)
    print(summary)
    
    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}")
    
    # Also save summary
    summary_path = OUTPUT_PATH.parent / "actionability_summary.csv"
    
    # Create enhanced summary with percentages
    summary_detailed = df.groupby(["bank", "year"]).apply(
        lambda x: pd.Series({
            "total_esg": len(x),
            "implemented": (x["action_label"] == "Implemented").sum(),
            "planning": (x["action_label"] == "Planning").sum(),
            "indeterminate": (x["action_label"] == "Indeterminate").sum(),
            "implemented_pct": (x["action_label"] == "Implemented").mean() * 100,
            "planning_pct": (x["action_label"] == "Planning").mean() * 100,
            "indeterminate_pct": (x["action_label"] == "Indeterminate").mean() * 100,
        })
    ).reset_index()
    
    summary_detailed.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    
    run_inference(args.batch_size)
