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
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.build_model import build_model
from src.training.corpus.word_segment import word_segment_batch

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
    # Gold mới dùng cột 'label' (0/1); cho override qua config['label_column'].
    return config.get("label_column", "label")

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
    return {
        'macro_f1': f1_score(labels, preds, average='macro'),
        'micro_f1': f1_score(labels, preds, average='micro'),
        # F1 lớp positive (label=1) — quan trọng khi lệch lớp; dùng chung cả bake-off.
        'f1_positive': f1_score(labels, preds, pos_label=1, average='binary', zero_division=0),
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

class WeightedTrainer(Trainer):
    """Trainer dùng class-weighted CrossEntropy (đã bỏ neuro-symbolic semantic loss)."""

    def __init__(self, class_weights: torch.Tensor = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # **kwargs hấp thụ num_items_in_batch (transformers mới) — CỐ Ý không dùng:
        # seq-classification mỗi sample 1 item -> CE trung bình theo batch đúng cho cả encoder/decoder.
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_fn = nn.CrossEntropyLoss()

        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

def build_training_args(config: dict, output_dir: Path, seed: int) -> TrainingArguments:
    train_cfg = config["training"]
    eval_batch_size = train_cfg.get("eval_batch_size", train_cfg["train_batch_size"] * 2)

    # bf16 ưu tiên trên GPU Ampere+ (ổn định loss hơn fp16); fp16 fallback GPU cũ; none trên CPU.
    precision = str(train_cfg.get("precision", train_cfg.get("fp16", "auto"))).lower()
    if precision == "bf16":
        use_bf16, use_fp16 = True, False
    elif precision in ("fp16", "true"):
        use_bf16, use_fp16 = False, True
    elif precision in ("none", "fp32", "false"):
        use_bf16, use_fp16 = False, False
    else:  # auto
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        use_fp16 = torch.cuda.is_available() and not use_bf16

    warmup_steps = int(train_cfg.get("warmup_steps", 0))
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.0))

    # QLoRA (decoder_lora) cần gradient-checkpointing + paged optimizer để vừa VRAM 16GB.
    grad_ckpt = bool(train_cfg.get("gradient_checkpointing", False))

    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["epochs"],
        per_device_train_batch_size=train_cfg["train_batch_size"],
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
        warmup_steps=warmup_steps,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy=train_cfg.get("save_strategy", "epoch"),
        save_total_limit=train_cfg.get("save_total_limit", 2),
        load_best_model_at_end=train_cfg.get("load_best_model_at_end", True),
        metric_for_best_model=train_cfg.get("metric_for_best_model", "macro_f1"),
        greater_is_better=True,
        logging_steps=50,
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        gradient_checkpointing=grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False} if grad_ckpt else None,
        optim=train_cfg.get("optim", "adamw_torch_fused"),  # = lib-default cũ -> PhoBERT KHÔNG đổi
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        label_smoothing_factor=train_cfg.get("label_smoothing_factor", 0.0),
        bf16=use_bf16,
        fp16=use_fp16,
        report_to=train_cfg.get("report_to", "none"),
        seed=seed,
    )

def summarize_predictions(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro")),
        "f1_positive": float(f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)),
        "report": classification_report(y_true, y_pred, target_names=[str(x) for x in labels], zero_division=0),
    }


def _prob_positive(logits: np.ndarray) -> np.ndarray:
    """P(class=1) qua softmax (binary). logits shape (n, 2)."""
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True))[:, 1]


# Cách C (chốt 2026-06-13): KHÔNG tự-chọn ngưỡng (tune_threshold bị bỏ). Eval/inference ở 0.5
# cố định — đã cân lệch lớp bằng class weights, 0.5 = baseline đã chạy tốt + nhất quán + dễ bảo vệ.
# Việc quét ngưỡng để dành cho sensitivity analysis (Phase 05) qua evaluate_split(threshold=...)
# + _prob_positive (vẫn giữ). Tự-chọn ngưỡng trên val nhỏ (149) overfit -> ngưỡng cực đoan 0.08.


def train_once(config: dict, save_model=True, extra_callbacks=None) -> dict:
    labels = config["labels"]
    label_col = 'label'

    seed = int(config.get("seed", 42))
    set_seed(seed)

    print("Loading labels...")
    df_train, df_val, label2id = load_train_val_data(config)
    print(f"Train: {len(df_train)}, Val: {len(df_val)}")
    print(df_train[label_col].value_counts())

    max_len = config["model"]["max_length"]
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # build_model: encoder (PhoBERT) | decoder_lora (LLM 4-bit QLoRA) — xem build_model.py
    model, tokenizer = build_model(config, labels)

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

    callbacks = list(extra_callbacks or [])
    early_cfg = config.get("early_stopping", {})
    if early_cfg.get("enabled", True):
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=early_cfg.get("patience", 2),
                early_stopping_threshold=early_cfg.get("threshold", 0.0),
            )
        )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=build_training_args(config, output_dir=output_dir, seed=seed),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    print('\nTraining...')
    trainer.train()
    # KHÔNG eval thừa ở đây: best_metric đã có trong state.log khi train; eval thừa còn tạo 1
    # "step" pruning ảo (sau khi train xong) -> có thể prune nhầm trial đã hoàn tất. (Phase 02 re-check)

    if save_model:
        final_path = output_dir / "final"
        trainer.save_model(str(final_path))
        tokenizer.save_pretrained(str(final_path))

    return {
        "trainer": trainer,
        "tokenizer": tokenizer,
        "label2id": label2id,
        "labels": labels,
        "val_dataset": val_dataset,
    }

def evaluate_split(trainer: Trainer, dataset: TextDataset, labels: list[str],
                   threshold: float | None = None) -> tuple[dict, np.ndarray, np.ndarray]:
    """threshold=None -> argmax (0.5); else y_pred = (P(class1) >= threshold)."""
    preds = trainer.predict(dataset)
    logits = preds.predictions
    y_true = preds.label_ids
    if threshold is None:
        y_pred = np.argmax(logits, axis=-1)
    else:
        y_pred = (_prob_positive(logits) >= threshold).astype(int)
    return summarize_predictions(y_true, y_pred, labels), y_true, y_pred

def _train_eval_seed(config: dict, seed: int, save_model: bool) -> dict:
    """Train 1 seed -> eval test ở NGƯỠNG 0.5 (argmax). Cách C: KHÔNG tự-chọn ngưỡng
    (đã cân lệch lớp bằng class weights; 0.5 nhất quán + là baseline đã chạy tốt)."""
    res = train_once({**config, "seed": seed}, save_model=save_model)
    trainer, tokenizer, labels = res["trainer"], res["tokenizer"], res["labels"]
    rec = {"seed": seed}

    test_df = load_test_data(config, res["label2id"])
    if test_df is not None and len(test_df) > 0:
        ws = bool(config["model"].get("word_segment", False))
        test_ds = TextDataset(test_df, tokenizer, config["model"]["max_length"], word_segment=ws)
        tm, y_true, y_pred = evaluate_split(trainer, test_ds, labels)  # threshold=None -> argmax 0.5
        rec["test"] = {k: v for k, v in tm.items() if k != "report"}
        if save_model:  # chỉ seed[0]: lưu preds + report đầy đủ
            rec["report"] = tm["report"]
            pd.DataFrame({"sentence": test_df["sentence"].tolist(),
                          "y_true": y_true, "y_pred": y_pred}).to_parquet(
                Path(config["paths"]["output_dir"]) / "test_predictions.parquet", index=False)

    del res, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rec


def _agg(per_seed: list[dict], field: str, metric: str) -> dict | None:
    vals = [p[field][metric] for p in per_seed if field in p]
    if not vals:
        return None
    return {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4)}


def run_multi_seed(config: dict, seeds: list[int]) -> dict:
    """Train nhiều seed -> mean±std (Phase 02). Lưu model seed[0] + metrics_summary.json."""
    per_seed = [_train_eval_seed(config, s, save_model=(i == 0)) for i, s in enumerate(seeds)]
    metrics = {
        "task": config["runtime"]["task"],
        "seeds": list(seeds),
        "n_seeds": len(seeds),
        "model_saved_seed": seeds[0],
        "inference_threshold": 0.5,  # cách C: ngưỡng cố định 0.5 (đã cân lệch bằng class weights)
        "per_seed": per_seed,
    }
    if all("test" in p for p in per_seed):
        metrics["test"] = {m: _agg(per_seed, "test", m)
                           for m in ("macro_f1", "micro_f1", "f1_positive")}

    out_dir = Path(config["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics_summary.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[multi-seed] {len(seeds)} seeds @ ngưỡng 0.5 | test macro_f1="
          f"{metrics.get('test', {}).get('macro_f1')} | f1_positive="
          f"{metrics.get('test', {}).get('f1_positive')}")
    print(f"Saved metrics summary to: {out_dir / 'metrics_summary.json'}")
    return metrics


def run(config: dict) -> dict:
    seeds = config["training"].get("seeds") or [int(config.get("seed", 42))]
    return run_multi_seed(config, [int(s) for s in seeds])

def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ESG classifier")
    parser.add_argument("--config", type=str, default="config/train.yml", help="Path to YAML config")
    parser.add_argument("--task", type=str, default=None, help="Task to train (validated against config tasks)")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Override seeds, vd '42,43' (mặc định đọc training.seeds)")
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
    if args.seeds:
        run_config["training"]["seeds"] = [int(s) for s in args.seeds.split(",") if s.strip()]
    run(run_config)

if __name__ == '__main__':
    main()