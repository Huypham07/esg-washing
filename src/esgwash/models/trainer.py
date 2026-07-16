"""Vong train chung cho model encoder nhieu dau sigmoid.

MultiHeadClassifier + masked BCE dung chung cho topic (env/soc/gov partial labels) va
commitment (1 dau), chi khac config heads/data. Seed control, pos_weight per head,
threshold tuning tren val, luu artifact + config.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from esgwash.nlp.segmentation import word_segment_batch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MultiHeadClassifier(nn.Module):
    def __init__(self, backbone_name: str, heads: list[str], dropout: float = 0.1):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({h: nn.Linear(hidden, 1) for h in heads})
        self.head_names = list(heads)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0])
        return torch.cat([self.heads[h](cls) for h in self.head_names], dim=1)


def masked_bce_loss(logits: torch.Tensor, targets: torch.Tensor,
                    pos_weight: torch.Tensor) -> torch.Tensor:
    """BCEWithLogits chi tinh tren o co nhan (targets NaN = bo qua)."""
    mask = ~torch.isnan(targets)
    if not mask.any():
        return logits.sum() * 0.0
    t = torch.nan_to_num(targets)
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, t, pos_weight=pos_weight, reduction="none")
    return (loss * mask).sum() / mask.sum()


class _TextDataset(Dataset):
    def __init__(self, encodings, targets):
        self.encodings = encodings
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.encodings.items()}
        item["targets"] = self.targets[i]
        return item


def _f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


class MultiHeadTrainer:
    """config keys: backbone, max_length, word_segment, heads (or labels), train{...}."""

    def __init__(self, config: dict):
        self.config = config
        self.heads = list(config.get("heads") or config["labels"])
        self.device = get_device()
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["backbone"], use_fast=not config.get("word_segment", False))
        self.model: MultiHeadClassifier | None = None
        self.thresholds = {h: 0.5 for h in self.heads}

    def _prep_texts(self, texts: list[str]) -> list[str]:
        if self.config.get("word_segment", False):
            return word_segment_batch(texts)
        return list(texts)

    def _encode(self, texts: list[str]):
        enc = self.tokenizer(self._prep_texts(texts), truncation=True, padding=True,
                             max_length=self.config.get("max_length", 256),
                             return_tensors="pt")
        return {k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")}

    def _targets(self, df: pd.DataFrame) -> torch.Tensor:
        return torch.tensor(df[self.heads].to_numpy(dtype=np.float32))

    def _pos_weight(self, df: pd.DataFrame) -> torch.Tensor:
        ws = []
        for h in self.heads:
            sub = df[h].dropna()
            pos = sub.sum()
            ws.append((len(sub) - pos) / pos if pos > 0 else 1.0)
        return torch.tensor(ws, dtype=torch.float32, device=self.device)

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
            text_col: str = "text", seed: int = 42,
            epoch_callback=None) -> dict:
        """epoch_callback(epoch, val_metrics): hook sau moi epoch — Optuna pruning
        raise TrialPruned tu day de cat som trial te."""
        from transformers import get_linear_schedule_with_warmup

        set_seed(seed)
        tc = self.config["train"]
        self.model = MultiHeadClassifier(self.config["backbone"], self.heads,
                                         dropout=self.config.get("dropout", 0.1)
                                         ).to(self.device)
        pos_weight = self._pos_weight(train_df) if self.config.get(
            "pos_weight", "auto") == "auto" else None

        ds = _TextDataset(self._encode(train_df[text_col].tolist()),
                          self._targets(train_df))
        loader = DataLoader(ds, batch_size=tc["batch_size"], shuffle=True,
                            generator=torch.Generator().manual_seed(seed))
        steps = len(loader) * tc["epochs"]
        opt = torch.optim.AdamW(self.model.parameters(), lr=float(tc["lr"]),
                                weight_decay=tc.get("weight_decay", 0.01))
        sched = get_linear_schedule_with_warmup(
            opt, int(steps * tc.get("warmup_ratio", 0.1)), steps)

        from tqdm.auto import tqdm

        best = {"macro_f1": -1.0, "state": None}
        history = []
        for epoch in range(tc["epochs"]):
            self.model.train()
            total = 0.0
            loop = tqdm(loader, desc=f"[{self.device}] epoch {epoch + 1}/{tc['epochs']}",
                        leave=False)
            for batch in loop:
                targets = batch.pop("targets").to(self.device)
                batch = {k: v.to(self.device) for k, v in batch.items()}
                logits = self.model(**batch)
                loss = masked_bce_loss(logits, targets, pos_weight)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                total += loss.item()
                loop.set_postfix(loss=f"{loss.item():.3f}")
            val_metrics = self.evaluate(val_df, text_col=text_col)
            print(f"[{self.device}] epoch {epoch + 1}/{tc['epochs']} "
                  f"loss={total / len(loader):.3f} val_macroF1={val_metrics['macro_f1']:.4f}")
            history.append({"epoch": epoch, "train_loss": total / len(loader),
                            **val_metrics})
            if val_metrics["macro_f1"] > best["macro_f1"]:
                best = {"macro_f1": val_metrics["macro_f1"],
                        "state": {k: v.detach().cpu().clone()
                                  for k, v in self.model.state_dict().items()}}
            if epoch_callback is not None:
                epoch_callback(epoch, val_metrics)
        if best["state"] is not None:
            self.model.load_state_dict(best["state"])
        self.tune_thresholds(val_df, text_col=text_col)
        return {"history": history, "best_val_macro_f1": best["macro_f1"],
                "thresholds": self.thresholds}

    @torch.no_grad()
    def predict_proba(self, texts: list[str], batch_size: int = 64) -> pd.DataFrame:
        from tqdm.auto import tqdm
        self.model.eval()
        enc = self._encode(texts)
        probs = []
        for i in tqdm(range(0, len(texts), batch_size), desc="topic", unit="batch",
                      leave=False):
            batch = {k: v[i:i + batch_size].to(self.device) for k, v in enc.items()}
            probs.append(torch.sigmoid(self.model(**batch)).cpu().numpy())
        return pd.DataFrame(np.vstack(probs), columns=self.heads)

    def tune_thresholds(self, val_df: pd.DataFrame, text_col: str = "text") -> dict:
        """Per-head maximize F1 tren val (spec 02 #1) - khong mac dinh 0.5."""
        probs = self.predict_proba(val_df[text_col].tolist())
        for h in self.heads:
            y = val_df[h].to_numpy(dtype=float)
            valid = ~np.isnan(y)
            if not valid.any():
                continue
            grid = np.arange(0.05, 0.96, 0.01)
            f1s = [_f1(y[valid], (probs[h].to_numpy()[valid] >= t).astype(int))
                   for t in grid]
            self.thresholds[h] = round(float(grid[int(np.argmax(f1s))]), 2)
        return self.thresholds

    def evaluate(self, df: pd.DataFrame, text_col: str = "text",
                 use_thresholds: bool = False) -> dict:
        probs = self.predict_proba(df[text_col].tolist())
        out, f1s = {}, []
        for h in self.heads:
            y = df[h].to_numpy(dtype=float)
            valid = ~np.isnan(y)
            if not valid.any():
                continue
            thr = self.thresholds[h] if use_thresholds else 0.5
            pred = (probs[h].to_numpy()[valid] >= thr).astype(int)
            f1 = _f1(y[valid], pred)
            out[f"f1_{h}"] = round(f1, 4)
            f1s.append(f1)
        out["macro_f1"] = round(float(np.mean(f1s)), 4) if f1s else 0.0
        return out

    def save(self, out_dir: str | Path) -> None:
        """Luu dang HF-uploadable: model.safetensors + tokenizer + config.json.

        config.json giu them `heads`/`thresholds`/`backbone` de tai lai kien truc
        multi-head tuy bien (khong phai AutoModelForSequenceClassification chuan)."""
        from safetensors.torch import save_model

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_model(self.model, str(out_dir / "model.safetensors"))
        self.tokenizer.save_pretrained(out_dir)
        (out_dir / "config.json").write_text(json.dumps(
            {**self.config, "heads": self.heads, "thresholds": self.thresholds},
            indent=2, default=str, ensure_ascii=False), encoding="utf-8")

    def load(self, out_dir: str | Path) -> "MultiHeadTrainer":
        out_dir = Path(out_dir)
        saved = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
        self.thresholds = saved.get("thresholds", self.thresholds)
        self.model = MultiHeadClassifier(self.config["backbone"], self.heads,
                                         dropout=self.config.get("dropout", 0.1))
        st = out_dir / "model.safetensors"
        if st.exists():
            from safetensors.torch import load_model
            load_model(self.model, str(st))
        else:  # tuong thich nguoc voi model.pt cu
            self.model.load_state_dict(torch.load(out_dir / "model.pt",
                                                  map_location="cpu"))
        self.model.to(self.device)
        return self


def multi_seed(config: dict, train_df: pd.DataFrame, val_df: pd.DataFrame,
               test_df: pd.DataFrame, text_col: str = "text",
               out_dir: str | Path | None = None,
               trainer_cls: type | None = None) -> dict:
    trainer_cls = trainer_cls or MultiHeadTrainer
    runs = []
    for i, seed in enumerate(config["train"]["seeds"]):
        trainer = trainer_cls(config)
        fit_info = trainer.fit(train_df, val_df, text_col=text_col, seed=seed)
        test_metrics = trainer.evaluate(test_df, text_col=text_col, use_thresholds=True)
        runs.append({"seed": seed, "test": test_metrics,
                     "thresholds": dict(trainer.thresholds),
                     "best_val_macro_f1": fit_info["best_val_macro_f1"]})
        if i == 0 and out_dir:
            trainer.save(out_dir)
    keys = runs[0]["test"].keys()
    agg = {k: {"mean": round(float(np.mean([r["test"][k] for r in runs])), 4),
               "std": round(float(np.std([r["test"][k] for r in runs])), 4)}
           for k in keys}
    return {"runs": runs, "aggregate": agg}
