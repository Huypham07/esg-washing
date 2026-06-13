"""M2 - multi-task commitment + specificity (spec 02 #2).

Hai head sigmoid tren cung PhoBERT (cung tap van ban - da xac minh trung 100%).
Fallback single-task: train 2 model 1 head rieng, chon theo val (giu ca 2 cho ablation).
Aux head env_claims da BO (quyet dinh 2026-06-12).
"""
from __future__ import annotations

import pandas as pd

from esgwash.models.trainer import MultiHeadTrainer, multi_seed

MAIN_HEADS = ("commitment", "specificity")


class ClaimModel(MultiHeadTrainer):
    def __init__(self, config: dict):
        super().__init__({**config, "heads": list(MAIN_HEADS)})

    def predict(self, sentences: list[str]) -> pd.DataFrame:
        probs = self.predict_proba(sentences)
        out = pd.DataFrame({f"p_{h}": probs[h] for h in MAIN_HEADS})
        for h in MAIN_HEADS:
            out[f"is_{h}"] = (probs[h] >= self.thresholds[h]).astype(int)
        return out


def run_single_task_fallback(config: dict, train_df, val_df, test_df) -> dict:
    """2 model 1 head rieng - so voi multi-task tren val, ghi ca 2 vao ablation."""
    results = {}
    for h in MAIN_HEADS:
        cfg = {**config, "heads": [h]}
        sub_train = train_df.dropna(subset=[h])
        results[h] = multi_seed(cfg, sub_train, val_df.dropna(subset=[h]),
                                test_df.dropna(subset=[h]))
    return results
