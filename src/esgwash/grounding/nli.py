"""NLI claim-evidence (spec 03 #3): mDeBERTa-v3 XNLI (có tiếng Việt).

premise = evidence (+ctx nếu ngắn), hypothesis = claim.
Giữ phân phối {entail, neutral, contradict}, không argmax.
"""
from __future__ import annotations

import numpy as np
import torch

from esgwash.models.trainer import get_device


class NLIScorer:
    def __init__(self, config: dict):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        name = config.get("nli_model", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
        self.device = get_device()
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name)
        self.model.to(self.device).eval()
        # vi tri label entailment trong id2label (mDeBERTa: 0=entail,1=neutral,2=contra)
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.entail_idx = next(i for i, lab in id2label.items() if "entail" in lab)
        self.labels = [id2label[i] for i in sorted(id2label)]

    @torch.no_grad()
    def score_pairs(self, pairs: list[tuple[str, str]], batch_size: int = 32) -> np.ndarray:
        """pairs = [(premise=evidence, hypothesis=claim), ...] -> probs (n, 3)."""
        if not pairs:
            return np.empty((0, len(self.labels)))
        out = []
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i:i + batch_size]
            enc = self.tok([p for p, _ in chunk], [h for _, h in chunk],
                           truncation=True, padding=True, max_length=256,
                           return_tensors="pt").to(self.device)
            probs = torch.softmax(self.model(**enc).logits, dim=-1)
            out.append(probs.cpu().numpy())
        return np.concatenate(out)

    def entail(self, probs: np.ndarray) -> np.ndarray:
        """Cot xac suat entailment."""
        return probs[:, self.entail_idx]
