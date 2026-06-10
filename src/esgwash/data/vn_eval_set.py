"""VN human-eval set (spec 01 #5) - test set chinh cho transfer claim.

Sampling ~300 cau stratified (bank x tru du doan) -> CSV cho nguoi gan;
load lai nhan + tinh Cohen kappa.
"""


def sample_for_annotation(sentences_df, n: int = 300, seed: int = 42):
    raise NotImplementedError  # TODO(Phase A4)


def load_annotations(path: str):
    raise NotImplementedError


def inter_annotator_kappa(ann_a, ann_b) -> dict:
    raise NotImplementedError
