"""Bộ phân loại commitment (single-head): PhoBERT + 1 đầu sigmoid.
"""
from __future__ import annotations

import pandas as pd

from esgwash.models.trainer import MultiHeadTrainer


class CommitmentModel(MultiHeadTrainer):
    def __init__(self, config: dict):
        super().__init__({**config, "heads": ["commitment"]})

    def predict(self, sentences: list[str]) -> pd.DataFrame:
        probs = self.predict_proba(sentences)["commitment"]
        return pd.DataFrame({"p_commitment": probs,
                             "is_commitment": (probs >= self.thresholds["commitment"]).astype(int)})
