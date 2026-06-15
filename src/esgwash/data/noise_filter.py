"""Lọc nhiễu dịch máy (spec 01 #4): QE + Confident Learning.

QE chính: Unbabel/wmt22-cometkiwi-da (gated, cần accept license HF).
Fallback: LaBSE cosine(EN, VI). CL: cleanlab trên out-of-fold predictions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def qe_scores(texts_en: list[str], texts_vi: list[str],
              backend: str = "labse", batch_size: int = 64) -> np.ndarray:
    if backend == "cometkiwi":
        from comet import download_model, load_from_checkpoint
        model = load_from_checkpoint(download_model("Unbabel/wmt22-cometkiwi-da"))
        data = [{"src": en, "mt": vi} for en, vi in zip(texts_en, texts_vi)]
        return np.array(model.predict(data, batch_size=batch_size).scores)
    if backend == "labse":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/LaBSE")
        emb_en = model.encode(texts_en, batch_size=batch_size, normalize_embeddings=True,
                              show_progress_bar=True)
        emb_vi = model.encode(texts_vi, batch_size=batch_size, normalize_embeddings=True,
                              show_progress_bar=True)
        return (emb_en * emb_vi).sum(axis=1)
    raise ValueError(f"backend khong ho tro: {backend}")


def filter_bottom_quantile(df: pd.DataFrame, scores: np.ndarray, q: float = 0.05) -> pd.DataFrame:
    cutoff = np.quantile(scores, q)
    return df[scores > cutoff].reset_index(drop=True)


def confident_learning_flags(labels: np.ndarray, proba_oof: np.ndarray) -> np.ndarray:
    """Mask bool dong nghi sai nhan. proba_oof: (n, 2) out-of-fold; labels: 0/1."""
    from cleanlab.filter import find_label_issues
    return find_label_issues(labels=labels.astype(int), pred_probs=proba_oof,
                             return_indices_ranked_by=None)


def oof_proba_tfidf_lr(texts: list[str], labels: np.ndarray, n_folds: int = 5,
                       seed: int = 42) -> np.ndarray:
    """Out-of-fold proba re tien cho CL (khong can train transformer nhieu lan)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import make_pipeline

    pipe = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), max_features=50_000, sublinear_tf=True),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))
    return cross_val_predict(pipe, texts, labels.astype(int), cv=n_folds,
                             method="predict_proba")
