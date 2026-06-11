"""M1 - PhoBERT 3 dau sigmoid, masked BCE (spec 02 #1).

Partial labels: loss chi tinh tren cot khac NaN (xem trainer.masked_bce_loss).
Cau khong tru nao vuot nguong -> non_esg, loai khoi cac tang sau.
"""
from __future__ import annotations

import pandas as pd

from esgwash.models.trainer import MultiHeadTrainer

PILLARS = ("env", "soc", "gov")


class TopicModel(MultiHeadTrainer):
    def __init__(self, config: dict):
        super().__init__({**config, "heads": list(config.get("labels", PILLARS))})

    def predict(self, sentences: list[str]) -> pd.DataFrame:
        """-> prob 3 cot + nhan da threshold + pillar chinh (non_esg neu khong vuot)."""
        probs = self.predict_proba(sentences)
        out = probs.copy()
        for p in self.heads:
            out[f"is_{p}"] = (probs[p] >= self.thresholds[p]).astype(int)
        any_pos = out[[f"is_{p}" for p in self.heads]].any(axis=1)
        out["pillar"] = probs[list(self.heads)].idxmax(axis=1).where(any_pos, "non_esg")
        return out
