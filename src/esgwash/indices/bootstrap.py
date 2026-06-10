"""Percentile bootstrap CI (spec 04 #4): resample cau commitment trong o,
B=1000, CI 95%. Min-n: o |C| < 30 chi vao bang pooled (bank x pillar / bank x year)."""


def bootstrap_ci(values, statistic, n_resamples: int = 1000, ci: float = 0.95):
    raise NotImplementedError  # TODO(Phase D1)


def pooled_cells(claims_df, min_n: int = 30):
    raise NotImplementedError
