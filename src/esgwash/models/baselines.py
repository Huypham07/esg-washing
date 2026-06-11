"""Baselines (spec 02 #3-4): TF-IDF + LogisticRegression per head;
zero-shot XLM-R (train EN goc, infer VI) cho ma tran transfer E3/E5."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline

from esgwash.models.trainer import MultiHeadTrainer


def tfidf_lr_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame,
                      heads: list[str], text_col: str = "text", seed: int = 42) -> dict:
    out, f1s = {}, []
    for h in heads:
        tr = train_df.dropna(subset=[h])
        te = test_df.dropna(subset=[h])
        if tr.empty or te.empty:
            continue
        pipe = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), max_features=50_000, sublinear_tf=True),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))
        pipe.fit(tr[text_col], tr[h].astype(int))
        f1 = f1_score(te[h].astype(int), pipe.predict(te[text_col]))
        out[f"f1_{h}"] = round(float(f1), 4)
        f1s.append(f1)
    out["macro_f1"] = round(float(np.mean(f1s)), 4) if f1s else 0.0
    return out


def xlmr_zero_shot(config: dict, train_en: pd.DataFrame, dev_en: pd.DataFrame,
                   test_vi: pd.DataFrame, seed: int = 42) -> dict:
    """Train tren EN goc (khong dich), eval truc tiep tren VI - khong word-segment."""
    cfg = {**config, "backbone": config.get("zero_shot_backbone", "xlm-roberta-base"),
           "word_segment": False}
    trainer = MultiHeadTrainer(cfg)
    trainer.fit(train_en, dev_en, seed=seed)
    return trainer.evaluate(test_vi, use_thresholds=True)
