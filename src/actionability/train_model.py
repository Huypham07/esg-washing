import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

TRAIN_PATH = Path("data/labels/action_hybrid_train_split.parquet")
VAL_PATH = Path("data/labels/action_hybrid_val_split.parquet")
GOLD_PATH = Path("data/labels/action_gold.parquet")
OUTPUT_DIR = Path("outputs/models/action_phobert_large")

MODEL_NAME = "vinai/phobert-large"
LABELS = ["Implemented", "Planning", "Indeterminate"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}

CONTEXT_BLOCK_TYPES = {"paragraph", "bullet_like", "kpi_like"}


class ActionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 256, use_context: bool = True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_context = use_context
        
        # Prepare texts with context for paragraph/bullet, sentence only for table
        self.texts = []
        for _, row in df.iterrows():
            if use_context and row.get("block_type", "") in CONTEXT_BLOCK_TYPES:
                # Use context: ctx_prev + sentence + ctx_next
                parts = []
                if row.get("ctx_prev"):
                    parts.append(str(row["ctx_prev"]))
                parts.append(str(row["sentence"]))
                if row.get("ctx_next"):
                    parts.append(str(row["ctx_next"]))
                text = " ".join(parts)
            else:
                text = str(row["sentence"])
            self.texts.append(text)
        
        self.labels = [LABEL2ID[label] for label in df["final_action"]]
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class WeightedTrainer(Trainer):
    """Custom Trainer with class weights for imbalanced data"""
    
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.class_weights is not None:
            weight = torch.tensor(self.class_weights, device=logits.device, dtype=logits.dtype)
            loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
        
        loss = loss_fct(logits.view(-1, len(LABELS)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    acc = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average="macro")
    per_class_f1 = f1_score(labels, predictions, average=None)
    
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "f1_implemented": per_class_f1[0],
        "f1_planning": per_class_f1[1],
        "f1_indeterminate": per_class_f1[2],
    }


def print_confusion_matrix(y_true, y_pred):
    """Print formatted confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    print("\n=== CONFUSION MATRIX ===")
    print(f"{'':>15} | {'Pred Impl':>10} | {'Pred Plan':>10} | {'Pred Indet':>10}")
    print("-" * 55)
    for i, label in enumerate(LABELS):
        row = cm[i]
        print(f"{label:>15} | {row[0]:>10} | {row[1]:>10} | {row[2]:>10}")


def train(
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 256,
    use_class_weights: bool = True,
    use_context: bool = True,
    test_run: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load data
    print("Loading training data...")
    train_df = pd.read_parquet(TRAIN_PATH)
    val_df = pd.read_parquet(VAL_PATH)
    
    if test_run:
        train_df = train_df.head(100)
        val_df = val_df.head(50)
        epochs = 1
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    print(f"Train distribution:\n{train_df['final_action'].value_counts()}")
    
    # Compute class weights for imbalanced data
    class_weights = None
    if use_class_weights:
        labels_array = train_df["final_action"].map(LABEL2ID).values
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.array([0, 1, 2]),
            y=labels_array
        )
        print(f"\nClass weights: {dict(zip(LABELS, class_weights))}")
    
    # Load tokenizer and model
    print(f"\nLoading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    
    # Create datasets
    print(f"Using context: {use_context}")
    train_dataset = ActionDataset(train_df, tokenizer, max_length, use_context)
    val_dataset = ActionDataset(val_df, tokenizer, max_length, use_context)
    
    # Training arguments
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_dir=str(OUTPUT_DIR / "logs"),
        logging_steps=50,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )
    
    # Trainer with class weights
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Evaluate on validation
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    val_results = trainer.evaluate()
    for k, v in val_results.items():
        print(f"  {k}: {v:.4f}")
    
    # Confusion matrix on validation
    val_predictions = trainer.predict(val_dataset)
    val_preds = np.argmax(val_predictions.predictions, axis=-1)
    val_labels = [LABEL2ID[l] for l in val_df["final_action"]]
    print_confusion_matrix(val_labels, val_preds)
    
    print("\nClassification Report (Validation):")
    print(classification_report(val_labels, val_preds, target_names=LABELS))
    
    # Save model
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nModel saved: {OUTPUT_DIR}")
    
    # Evaluate on gold set if available
    if GOLD_PATH.exists():
        print("\n" + "="*60)
        print("GOLD SET EVALUATION")
        print("="*60)
        gold_df = pd.read_parquet(GOLD_PATH)
        
        if "gold_action" in gold_df.columns:
            # Prepare gold dataset
            gold_df["final_action"] = gold_df["gold_action"]
            gold_dataset = ActionDataset(gold_df, tokenizer, max_length, use_context)
            
            gold_predictions = trainer.predict(gold_dataset)
            gold_preds = np.argmax(gold_predictions.predictions, axis=-1)
            gold_labels = [LABEL2ID[l] for l in gold_df["gold_action"]]
            
            gold_macro_f1 = f1_score(gold_labels, gold_preds, average="macro")
            gold_acc = accuracy_score(gold_labels, gold_preds)
            print(f"  Gold Accuracy: {gold_acc:.4f}")
            print(f"  Gold Macro-F1: {gold_macro_f1:.4f}")
            
            print_confusion_matrix(gold_labels, gold_preds)
            print("\nClassification Report (Gold):")
            print(classification_report(gold_labels, gold_preds, target_names=LABELS))
    
    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--no-class-weights", action="store_true", help="Disable class weights")
    parser.add_argument("--no-context", action="store_true", help="Disable context for paragraph/bullet")
    parser.add_argument("--test", action="store_true", help="Quick test run")
    args = parser.parse_args()
    
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_class_weights=not args.no_class_weights,
        use_context=not args.no_context,
        test_run=args.test,
    )
