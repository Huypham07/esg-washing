"""M2 - multi-task commitment + specificity (spec 02 #2).

Hai head sigmoid tren cung PhoBERT (cung tap van ban - da xac minh trung 100%).
Aux head env_claims (chi regularize). Fallback: 2 model single-task, chon theo dev.
"""


class ClaimModel:
    def __init__(self, config: dict): ...

    def fit(self, train_df, dev_df):
        raise NotImplementedError  # TODO(Phase B3)

    def predict(self, sentences: list):
        """-> DataFrame[p_commitment, p_specific, is_commitment, is_specific]."""
        raise NotImplementedError
