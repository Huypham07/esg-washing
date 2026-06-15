"""Phân loại trụ E/S/G: PhoBERT + 3 đầu sigmoid, masked BCE.

Nhãn thiếu (partial labels): loss chỉ tính trên cột khác NaN (trainer.masked_bce_loss).
Câu không vượt ngưỡng trụ nào -> non_esg, loại khỏi các tầng sau.
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
