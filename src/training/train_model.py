import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.neuro_symbolic import SemanticLoss, create_semantic_loss
from src.training.corpus.word_segment import word_segment_batch, unsegment

class TextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = 256,
                 word_segment: bool = False):
        texts = df['text'].tolist()
        self.texts = word_segment_batch(texts) if word_segment else texts
        self.labels = df['label'].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
        }

def prepare_text(row: pd.Series, use_context_prev: bool = True, use_context_next: bool = True) -> str:
    parts = []
    if use_context_prev and row.get('ctx_prev'):
        parts.append(str(row['ctx_prev']))
    parts.append(str(row['sentence']))
    if use_context_next and row.get('ctx_next'):
        parts.append(str(row['ctx_next']))
    return ' '.join(parts)


def load_yaml_config(config_path: Path) -> dict:
    with config_path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f'Invalid config format in {config_path}: root must be a mapping')
    return data

def resolve_runtime_config(
    raw_config: dict,
    task: str | None = None,
    output_dir: str | None = None,
) -> dict:
    defaults = raw_config.get('defaults', {})
    tasks_cfg = raw_config.get('tasks', {})

    chosen_task = task or defaults.get('task')
    if chosen_task not in tasks_cfg:
        raise ValueError(
            f"Unknown task '{chosen_task}'. Available tasks: {list(tasks_cfg.keys())}"
        )

    resolved = copy.deepcopy(tasks_cfg[chosen_task])

    if output_dir:
        resolved.setdefault('paths', {})['output_dir'] = output_dir

    resolved['runtime'] = {
        'task': chosen_task,
    }
    return resolved

def load_dataframe(data_path: Path) -> pd.DataFrame:
    suffix = data_path.suffix.lower()
    if suffix == '.parquet':
        return pd.read_parquet(data_path)
    if suffix == '.csv':
        return pd.read_csv(data_path)
    raise ValueError(f'Unsupported data format: {data_path}. Use .parquet or .csv')

def encode_labels(df: pd.DataFrame, label_column: str, label2id: dict[str, int]) -> pd.DataFrame:
    if label_column not in df.columns:
        raise ValueError(f"Column '{label_column}' not found. Available columns: {list(df.columns)}")
    result = df.copy()
    result["label"] = result[label_column].map(label2id)
    result = result.dropna(subset=["label"])
    result["label"] = result["label"].astype(int)
    return result

def build_text_column(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    use_context_prev = config.get("data", {}).get("use_context_prev", True)
    use_context_next = config.get("data", {}).get("use_context_next", True)
    result = df.copy()
    result["text"] = result.apply(
        lambda row: prepare_text(row, use_context_prev=use_context_prev, use_context_next=use_context_next),
        axis=1,
    )
    return result

def _label_col(config: dict) -> str:
    task = config.get("runtime", {}).get("task", "topic")
    return "topic_label" if task == "topic" else "action_label"

def load_train_val_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    train_df = build_text_column(load_dataframe(Path(config["paths"]["train_data"])), config)
    val_df = build_text_column(load_dataframe(Path(config["paths"]["val_data"])), config)

    labels = config["labels"]
    label2id = {name: idx for idx, name in enumerate(labels)}
    label_col = _label_col(config)

    train_df = encode_labels(train_df, label_col, label2id)
    val_df = encode_labels(val_df, label_col, label2id)
    return train_df, val_df, label2id

def load_test_data(config: dict, label2id: dict[str, int]) -> pd.DataFrame | None:
    test_path = Path(config["paths"].get("test_data", ""))
    if not test_path.exists():
        return None

    test_df = build_text_column(load_dataframe(test_path), config)
    source_col = _label_col(config)
    for col in config.get("test_label_candidates", [source_col]):
        if col in test_df.columns:
            return encode_labels(test_df, col, label2id)
    return None

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, preds, average='macro')
    micro_f1 = f1_score(labels, preds, average='micro')
    return {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
    }

def compute_class_weights(config: dict, labels: list, method: str = 'inverse') -> torch.Tensor:
    counts = Counter(labels)
    n_samples = len(labels)
    n_classes = len(config['labels'])

    def safe_count(index: int) -> int:
        return max(counts.get(index, 0), 1)

    if method == 'inverse':
        weights = [n_samples / (n_classes * safe_count(i)) for i in range(n_classes)]
    elif method == 'sqrt_inverse':
        weights = [np.sqrt(n_samples / (n_classes * safe_count(i))) for i in range(n_classes)]
    elif method == 'effective':
        beta = 0.9999
        weights = [(1 - beta) / (1 - beta**safe_count(i)) for i in range(n_classes)]
    else:
        weights = [1.0] * n_classes

    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum() * n_classes
    return weights

class NeuroSymbolicTrainer(Trainer):
    def __init__(
        self,
        class_weights: torch.Tensor = None,
        semantic_loss: SemanticLoss = None,
        tokenizer_ref=None,
        word_segment: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.semantic_loss = semantic_loss
        self.tokenizer_ref = tokenizer_ref
        self.word_segment = word_segment

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_fn = nn.CrossEntropyLoss()

        ce_loss = loss_fn(logits, labels)
        total_loss = ce_loss

        if self.semantic_loss is not None and self.tokenizer_ref is not None:
            texts = self.tokenizer_ref.batch_decode(inputs["input_ids"], skip_special_tokens=True)
            if self.word_segment:
                texts = [unsegment(t) for t in texts]
            sem_loss = self.semantic_loss(logits, texts)
            total_loss = ce_loss + sem_loss

        return (total_loss, outputs) if return_outputs else total_loss

def build_training_args(config: dict, output_dir: Path, seed: int) -> TrainingArguments:
    train_cfg = config["training"]
    eval_batch_size = train_cfg.get("eval_batch_size", train_cfg["train_batch_size"] * 2)

    fp16_cfg = train_cfg.get("fp16", "auto")
    if isinstance(fp16_cfg, str) and fp16_cfg.lower() == "auto":
        fp16_cfg = torch.cuda.is_available()

    warmup_steps = int(train_cfg.get("warmup_steps", 0))
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.0))

    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["epochs"],
        per_device_train_batch_size=train_cfg["train_batch_size"],
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay"),
        warmup_steps=warmup_steps,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=train_cfg.get("load_best_model_at_end", True),
        metric_for_best_model=train_cfg.get("metric_for_best_model", "macro_f1"),
        greater_is_better=True,
        logging_steps=50,
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        max_grad_norm=train_cfg.get("max_grad_norm"),
        label_smoothing_factor=train_cfg.get("label_smoothing_factor"),
        fp16=bool(fp16_cfg),
        report_to=train_cfg.get("report_to", "none"),
        seed=seed,
    )

def summarize_predictions(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro")),
        "report": classification_report(y_true, y_pred, target_names=labels, zero_division=0),
    }


def train_once(config: dict, save_model=True) -> dict:
    task = config["runtime"]["task"]
    labels = config["labels"]
    label_col = 'label'

    seed = int(config.get("seed", 42))
    set_seed(seed)

    print("Loading labels...")
    df_train, df_val, label2id = load_train_val_data(config)
    print(f"Train: {len(df_train)}, Val: {len(df_val)}")
    print(df_train[label_col].value_counts())

    model_name = config["model"]["name"]
    max_len = config["model"]["max_length"]
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label={i: name for i, name in enumerate(labels)},
        label2id={name: i for i, name in enumerate(labels)},
        ignore_mismatched_sizes=True,
    )

    lengths = df_train['sentence'].fillna('').astype(str).apply(lambda x: len(tokenizer.tokenize(x)))
    print(lengths.describe())

    ws_enabled = bool(config["model"].get("word_segment", False))
    train_dataset = TextDataset(df_train, tokenizer, max_len, word_segment=ws_enabled)
    val_dataset = TextDataset(df_val, tokenizer, max_len, word_segment=ws_enabled)

    class_weights = None
    if config["training"].get("use_class_weights", False):
        class_weights = compute_class_weights(
            config,
            df_train["label"].tolist(),
            method=config["training"].get("weight_method", "inverse"),
        )

    semantic_loss = None
    if "neuro_symbolic" in config:
        from sentence_transformers import SentenceTransformer
        st_model = config.get("model", {}).get("sentence_transformer")
        sem_encoder = SentenceTransformer(st_model)
        print(f"Semantic encoder: {st_model}")
        semantic_loss = create_semantic_loss(
            task=task, labels=labels, config=config, encoder=sem_encoder,
        )

    callbacks = []
    early_cfg = config.get("early_stopping", {})
    if early_cfg.get("enabled", True):
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=early_cfg.get("patience", 2),
                early_stopping_threshold=early_cfg.get("threshold", 0.0),
            )
        )

    trainer = NeuroSymbolicTrainer(
        class_weights=class_weights,
        semantic_loss=semantic_loss,
        tokenizer_ref=tokenizer,
        word_segment=ws_enabled,
        model=model,
        args=build_training_args(config, output_dir=output_dir, seed=seed),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    print('\nTraining...')
    trainer.train()
    trainer.evaluate()

    if save_model:
        final_path = output_dir / "final"
        trainer.save_model(str(final_path))
        tokenizer.save_pretrained(str(final_path))

    return {
        "trainer": trainer,
        "tokenizer": tokenizer,
        "label2id": label2id,
        "labels": labels,
    }

def evaluate_split(trainer: Trainer, dataset: TextDataset, labels: list[str]) -> dict:
    preds = trainer.predict(dataset)
    logits = preds.predictions
    y_pred = np.argmax(logits, axis=-1)
    y_true = preds.label_ids
    return summarize_predictions(y_true, y_pred, labels)

def run(config: dict) -> dict:
    result = train_once(config)
    trainer = result["trainer"]
    tokenizer = result["tokenizer"]
    labels = result["labels"]

    metrics = {"task": config["runtime"]["task"]}

    test_df = load_test_data(config, result["label2id"])
    if test_df is not None and len(test_df) > 0:
        max_len = config["model"]["max_length"]
        ws_enabled = bool(config["model"].get("word_segment", False))
        test_dataset = TextDataset(test_df, tokenizer, max_len, word_segment=ws_enabled)
        test_metrics = evaluate_split(trainer, test_dataset, labels)
        metrics["test"] = test_metrics

    output_dir = Path(config["paths"]["output_dir"])
    summary_path = output_dir / "metrics_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Saved metrics summary to: {summary_path}")

    return metrics

def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ESG classifier")
    parser.add_argument("--config", type=str, default="config/train.yml", help="Path to YAML config")
    parser.add_argument("--task", type=str, choices=["topic", "action"], default=None, help="Task to train")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    return parser.parse_args(args)

def main(args=None) -> None:
    args = parse_args(args)
    project_root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    raw_config = load_yaml_config(config_path)
    run_config = resolve_runtime_config(
        raw_config=raw_config,
        task=args.task,
        output_dir=args.output_dir,
    )
    run(run_config)

if __name__ == '__main__':
    main()