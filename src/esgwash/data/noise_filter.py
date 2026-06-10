"""Loc nhieu dich may (spec 01 #4): QE + Confident Learning.

QE chinh: Unbabel/wmt22-cometkiwi-da (gated - can accept license).
Fallback: LaBSE cosine(EN, VI). CL: cleanlab tren out-of-fold predictions.
"""


def qe_scores(pairs_en_vi: list, backend: str = "cometkiwi") -> list:
    raise NotImplementedError  # TODO(Phase B4)


def filter_bottom_quantile(df, scores, q: float = 0.05):
    raise NotImplementedError


def confident_learning_flags(texts, labels, predict_proba_oof):
    """Tra ve mask dong nghi sai nhan (cleanlab.find_label_issues)."""
    raise NotImplementedError
