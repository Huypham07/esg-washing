"""Baselines (spec 02 #3-4): TF-IDF + LogisticRegression per task;
zero-shot XLM-R (train EN goc, infer VI) cho ma tran transfer E3."""


def tfidf_lr_baseline(train_df, test_df, label_col: str) -> dict:
    raise NotImplementedError  # TODO(Phase B1)


def xlmr_zero_shot(task: str, config: dict) -> dict:
    raise NotImplementedError
