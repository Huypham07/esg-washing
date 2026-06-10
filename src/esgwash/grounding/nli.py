"""NLI claim-evidence (spec 03 #3): mDeBERTa-v3 XNLI (co tieng Viet).

premise = evidence (+ctx neu ngan), hypothesis = claim.
GIU PHAN PHOI {entail, neutral, contradict} - khong argmax.
"""


class NLIScorer:
    def __init__(self, config: dict): ...

    def score_pairs(self, pairs: list):
        """-> array (n, 3) probs."""
        raise NotImplementedError  # TODO(Phase C2)
