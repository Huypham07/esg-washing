"""Retrieval top-k per doc bang bkai vietnamese-bi-encoder (spec 03 #2).

Cosine top-k=5, san sim >= 0.5. Pool nho per-doc -> khong can FAISS.
"""


class EvidenceRetriever:
    def __init__(self, config: dict): ...

    def topk(self, claim: str, pool: list, k: int = 5):
        raise NotImplementedError  # TODO(Phase C1)
