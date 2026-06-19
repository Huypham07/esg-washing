"""Interpretable lexical baseline (CPU): TF-IDF + LogisticRegression per head.

- top_tokens: which words drive each classifier (explainability, no transformer).
- cross_lingual_macro_f1: train on one language column, test on another (text=VI,
  text_en=EN) to quantify lexical cross-lingual transfer. Same TF-IDF+LR config as
  esgwash.models.baselines.tfidf_lr_baseline so numbers are comparable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline


def fit_head(train_df: pd.DataFrame, head: str, text_col: str = "text", seed: int = 42):
    tr = train_df.dropna(subset=[head, text_col])
    pipe = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), max_features=50_000, sublinear_tf=True),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))
    pipe.fit(tr[text_col].astype(str), tr[head].astype(int))
    return pipe


def top_tokens(train_df: pd.DataFrame, head: str, text_col: str = "text",
               k: int = 15, seed: int = 42):
    pipe = fit_head(train_df, head, text_col, seed)
    vec = pipe.named_steps["tfidfvectorizer"]
    clf = pipe.named_steps["logisticregression"]
    names = np.asarray(vec.get_feature_names_out())
    coef = clf.coef_[0]
    order = np.argsort(coef)[::-1][:k]
    return [(str(names[i]), float(coef[i])) for i in order]


def cross_lingual_macro_f1(train_df: pd.DataFrame, test_df: pd.DataFrame, heads,
                           train_col: str, test_col: str, seed: int = 42) -> float:
    f1s = []
    for h in heads:
        tr = train_df.dropna(subset=[h, train_col])
        te = test_df.dropna(subset=[h, test_col])
        if tr.empty or te.empty:
            continue
        pipe = fit_head(tr, h, train_col, seed)
        f1s.append(f1_score(te[h].astype(int), pipe.predict(te[test_col].astype(str))))
    return float(np.mean(f1s)) if f1s else 0.0
