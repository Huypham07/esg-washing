"""Evidence-support score (spec 03 #4).

support(claim) = max_{e in top-k} P_entail(e, claim)   (FEVER-style aggregation)
grounded@theta voi theta in {0.5, 0.7, 0.9} - sweep, khong chon cung.
"""


def support_score(entail_probs: list) -> float:
    raise NotImplementedError  # TODO(Phase C2)


def grounded_flags(support: float, thresholds: list) -> dict:
    raise NotImplementedError
