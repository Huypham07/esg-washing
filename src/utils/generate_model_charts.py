
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

# Configuration
OUTPUT_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ACTION_MODEL_ID = "huypham71/esg-action"
TOPIC_MODEL_ID = "huypham71/esg-topic-classifier"

ACTION_GOLD_PATH = Path("data/labels/action_gold.parquet")
TOPIC_GOLD_PATH = Path("data/labels/topic_gold.parquet")

ACTION_LABELS = ["Implemented", "Planning", "Indeterminate"]
TOPIC_LABELS = ["E", "S_labor", "S_community", "S_product", "G", "Non_ESG"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

def plot_confusion_matrix(y_true, y_pred, labels, title, filename):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=labels, 
        yticklabels=labels,
        annot_kws={"size": 12}
    )
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()
    print(f"Saved {filename}")

def plot_distribution(df, col, title, filename, labels=None):
    plt.figure(figsize=(12, 6))
    if labels:
        # Ensure order
        order = labels
    else:
        order = df[col].value_counts().index
        
    ax = sns.countplot(data=df, x=col, order=order, palette="viridis")
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel("Label", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(rotation=45)
    
    # Add counts on top
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
        
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()
    print(f"Saved {filename}")

def predict(model, tokenizer, texts, labels, batch_size=16):
    model.eval()
    model.to(DEVICE)
    
    all_preds = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            max_length=256, 
            return_tensors="pt"
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            
        all_preds.extend([labels[p] for p in preds])
        
    return all_preds

def main():
    # 1. Actionability Model Evaluation
    if ACTION_GOLD_PATH.exists():
        print(f"\nProcessing Actionability Gold Set: {ACTION_GOLD_PATH}")
        df_action = pd.read_parquet(ACTION_GOLD_PATH)
        
        # Plot Data Distribution
        plot_distribution(
            df_action, 
            "gold_action", 
            "Actionability Gold Set Distribution", 
            "action_data_dist.png",
            labels=ACTION_LABELS
        )
        
        # Load Model & Predict
        print("Loading Actionability Model...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(ACTION_MODEL_ID)
            model = AutoModelForSequenceClassification.from_pretrained(ACTION_MODEL_ID)
            
            # Predict
            texts = df_action["text"].tolist()
            # Context handling if applicable (simple text for now as gold set format varies)
            # Assuming 'text' column is sufficient or pre-processed
            
            y_pred = predict(model, tokenizer, texts, ACTION_LABELS)
            y_true = df_action["gold_action"].tolist()
            
            # Plot Confusion Matrix
            plot_confusion_matrix(
                y_true, 
                y_pred, 
                ACTION_LABELS, 
                "Actionability Confusion Matrix (Gold Set)", 
                "action_confusion_matrix.png"
            )
        except Exception as e:
            print(f"Error evaluating Actionability Model: {e}")
    else:
        print(f"Actionability Gold Path not found: {ACTION_GOLD_PATH}")

    # 2. Topic Classifier Evaluation
    if TOPIC_GOLD_PATH.exists():
        print(f"\nProcessing Topic Gold Set: {TOPIC_GOLD_PATH}")
        df_topic = pd.read_parquet(TOPIC_GOLD_PATH)
        
        # Plot Data Distribution
        plot_distribution(
            df_topic, 
            "gold_topic", 
            "Topic Gold Set Distribution", 
            "topic_data_dist.png",
            labels=TOPIC_LABELS
        )
        
        # Load Model & Predict
        print("Loading Topic Model...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(TOPIC_MODEL_ID)
            model = AutoModelForSequenceClassification.from_pretrained(TOPIC_MODEL_ID)
            
            # Predict
            texts = df_topic["text"].tolist()
            y_pred = predict(model, tokenizer, texts, TOPIC_LABELS)
            y_true = df_topic["gold_topic"].tolist()
            
            # Plot Confusion Matrix
            plot_confusion_matrix(
                y_true, 
                y_pred, 
                TOPIC_LABELS, 
                "Topic Classification Confusion Matrix (Gold Set)", 
                "topic_confusion_matrix.png"
            )
        except Exception as e:
            print(f"Error evaluating Topic Model: {e}")
    else:
        print(f"Topic Gold Path not found: {TOPIC_GOLD_PATH}")

if __name__ == "__main__":
    main()
