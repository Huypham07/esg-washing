"""M1 - PhoBERT 3 dau sigmoid, masked BCE (spec 02 #1).

Partial labels: loss chi tinh tren cot khac NaN. Threshold per-label tune tren dev.
"""
import torch


class MaskedBCELoss(torch.nn.Module):
    """BCEWithLogits co mask NaN + pos_weight per label."""

    def forward(self, logits, targets):  # targets co the chua NaN
        raise NotImplementedError  # TODO(Phase B2)


class TopicModel:
    """Wrapper train/predict; predict tra ve prob 3 cot + nhan da threshold."""

    def __init__(self, config: dict): ...

    def fit(self, train_df, dev_df):
        raise NotImplementedError

    def tune_thresholds(self, dev_df) -> dict:
        raise NotImplementedError

    def predict(self, sentences: list):
        raise NotImplementedError
