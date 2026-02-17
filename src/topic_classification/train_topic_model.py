import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
import argparse
import yaml

def load_config(config_path="configs/train.yml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

MODEL_NAME = "vinai/phobert-large"
OUTPUT_DIR = Path("outputs/models/topic_phobert_large")
TOPICS = ["E", "S_labor", "S_community", "S_product", "G", "Non_ESG"]
LABEL2ID = {t: i for i, t in enumerate(TOPICS)}
ID2LABEL = {i: t for t, i in LABEL2ID.items()}


class TopicDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = 256):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def prepare_text(row: pd.Series) -> str:
    """Combine sentence with context for better classification"""
    parts = []
    if row.get("ctx_prev"):
        parts.append(str(row["ctx_prev"]))
    parts.append(str(row["sentence"]))
    if row.get("ctx_next"):
        parts.append(str(row["ctx_next"]))
    return " ".join(parts)


def load_hybrid_labels(config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load hybrid train/val splits"""
    train = pd.read_parquet(config["paths"]["train_data"])
    val = pd.read_parquet(config["paths"]["val_data"])
    
    train["text"] = train.apply(prepare_text, axis=1)
    train["label"] = train["final_topic"].map(LABEL2ID)
    
    val["text"] = val.apply(prepare_text, axis=1)
    val["label"] = val["final_topic"].map(LABEL2ID)
    
    return train, val


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, preds, average="macro")
    micro_f1 = f1_score(labels, preds, average="micro")
    return {"macro_f1": macro_f1, "micro_f1": micro_f1}


def compute_class_weights(labels: list, method: str = "inverse") -> torch.Tensor:
    """Compute class weights for imbalanced data"""
    counts = Counter(labels)
    n_samples = len(labels)
    n_classes = len(TOPICS)
    
    if method == "inverse":
        weights = [n_samples / (n_classes * counts[i]) for i in range(n_classes)]
    elif method == "sqrt_inverse":
        weights = [np.sqrt(n_samples / (n_classes * counts[i])) for i in range(n_classes)]
    elif method == "effective":
        beta = 0.9999
        weights = [(1 - beta) / (1 - beta ** counts[i]) for i in range(n_classes)]
    else:
        weights = [1.0] * n_classes
    
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum() * n_classes
    return weights


class WeightedTrainer(Trainer):
    """Custom Trainer with class-weighted loss"""
    
    def __init__(self, class_weights: torch.Tensor = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss_fn = nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fn = nn.CrossEntropyLoss()
        
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def train(config_file: str):
    config = load_config(config_file)
    
    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    lr = config["training"]["learning_rate"]
    max_len = config["model"]["max_length"]
    use_weights = config["training"]["use_class_weights"]
    weight_method = config["training"]["weight_method"]
    model_name = config["model"]["name"]
    output_dir = Path(config["paths"]["output_dir"])
    
    print("Loading hybrid labels...")
    df_train, df_val = load_hybrid_labels(config)
    
    print(f"Train: {len(df_train)}, Val: {len(df_val)}")
    print("\nTrain distribution:")
    print(df_train["final_topic"].value_counts())
    
    class_weights = None
    if use_weights:
        class_weights = compute_class_weights(df_train["label"].tolist(), method=weight_method)
        print(f"\nClass weights ({weight_method}):")
        for t, w in zip(TOPICS, class_weights):
            print(f"  {t}: {w:.3f}")
    
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(TOPICS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    
    train_dataset = TopicDataset(df_train, tokenizer, max_len)
    val_dataset = TopicDataset(df_val, tokenizer, max_len)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        weight_decay=config["training"]["weight_decay"],
        warmup_ratio=config["training"]["warmup_ratio"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    
    print("\nTraining...")
    trainer.train()
    
    print("\n=== Validation Results ===")
    val_results = trainer.evaluate()
    print(val_results)
    
    # Gold test evaluation (if available)
    gold_path = Path(config["paths"]["gold_test"])
    if gold_path.exists():
        try:
            df_gold = pd.read_parquet(gold_path)
            # If CSV, we could use read_csv, but notebook uses parquet. We'll handle both.
            if gold_path.suffix == '.csv':
                 df_gold = pd.read_csv(gold_path)
                 
            # ensure gold_topic exists
            if "gold_topic" in df_gold.columns:
                df_gold = df_gold[df_gold["gold_topic"].notna() & (df_gold["gold_topic"] != "")]
                df_gold["text"] = df_gold.apply(prepare_text, axis=1)
                df_gold["label"] = df_gold["gold_topic"].map(LABEL2ID)
                
                if len(df_gold) > 0:
                    print(f"\n=== Gold Test Results ({len(df_gold)} samples) ===")
                    test_dataset = TopicDataset(df_gold, tokenizer, max_len)
                    preds = trainer.predict(test_dataset)
                    y_pred = np.argmax(preds.predictions, axis=-1)
                    y_true = df_gold["label"].values
                    
                    print(f"Macro-F1: {f1_score(y_true, y_pred, average='macro'):.4f}")
                    print("\nClassification Report:")
                    print(classification_report(y_true, y_pred, target_names=TOPICS))
        except Exception as e:
            print(f"Gold test skipped: {e}")
    
    # Save model
    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\nModel saved: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yml")
    args = parser.parse_args()
    
    train(config_file=args.config)
