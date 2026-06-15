"""Adapter cho model commitment cua cong su (thay M2-commitment cua ta tu 2026-06-14).

dqa2412/esg-washing-optimized: PhoBERT-base-v2 + AutoModelForSequenceClassification,
nhi phan (1=commitment/action, 0=khong), tokenize bang underthesea (KHAC pyvi cua ta).
Interface .predict() khop voi cho inference.classify_sentences dung (p_commitment/is_commitment).
ClaimModel cu van giu trong code (claim_model.py) nhung khong dung o luong inference nua.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from esgwash.models.trainer import get_device

DEFAULT_REPO = "dqa2412/esg-washing-optimized"


def _patched_snapshot(repo: str) -> str:
    """Tai snapshot repo + va config.json: id2label/label2id co value kieu int
    ({"0": 0}) bi huggingface_hub moi reject (yeu cau dict[*, str]). Ep ve str,
    ghi lai tai cho. Tra ve duong dan local de from_pretrained doc."""
    import json
    from pathlib import Path

    if Path(repo).exists():
        d = Path(repo)
    else:
        from huggingface_hub import snapshot_download
        d = Path(snapshot_download(repo))

    cfg_path = d / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    changed = False
    if isinstance(cfg.get("id2label"), dict):
        fixed = {str(k): str(v) for k, v in cfg["id2label"].items()}
        if fixed != cfg["id2label"]:
            cfg["id2label"] = fixed
            changed = True
    if isinstance(cfg.get("label2id"), dict):
        fixed = {str(k): int(v) for k, v in cfg["label2id"].items()}
        if fixed != cfg["label2id"]:
            cfg["label2id"] = fixed
            changed = True
    if changed:
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(d)


class CommitmentHF:
    def __init__(self, repo: str = DEFAULT_REPO, threshold: float = 0.5,
                 max_length: int = 256):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.threshold = threshold
        self.max_length = max_length
        self.device = get_device()
        local = _patched_snapshot(repo)  # va config.json id2label int -> str (validate strict)
        self.tokenizer = AutoTokenizer.from_pretrained(local)
        self.model = AutoModelForSequenceClassification.from_pretrained(local)
        self.model.to(self.device).eval()

    @staticmethod
    def _segment(texts: list[str]) -> list[str]:
        from underthesea import word_tokenize
        return [word_tokenize(str(t), format="text") for t in texts]

    @torch.no_grad()
    def predict_proba(self, sentences: list[str], batch_size: int = 64) -> np.ndarray:
        seg = self._segment(sentences)
        out = []
        for i in range(0, len(seg), batch_size):
            enc = self.tokenizer(seg[i:i + batch_size], truncation=True, padding=True,
                                 max_length=self.max_length, return_tensors="pt"
                                 ).to(self.device)
            probs = torch.softmax(self.model(**enc).logits, dim=-1)[:, 1]
            out.append(probs.cpu().numpy())
        return np.concatenate(out) if out else np.array([])

    def predict(self, sentences: list[str], batch_size: int = 64) -> pd.DataFrame:
        p = self.predict_proba(sentences, batch_size=batch_size)
        return pd.DataFrame({"p_commitment": p,
                             "is_commitment": (p >= self.threshold).astype(int)})
