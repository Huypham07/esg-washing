"""Selective disclosure (spec 04 #3): share(p) = n_p / (n_E+n_S+n_G) theo (bank, year);
đo lệch so với mặt bằng ngành cùng năm. Mô tả, không gộp vào CTI.

Input `pred_long`: mỗi dòng = (câu ESG × trụ nó thuộc về), cột bank, year, pillar.
"""
from __future__ import annotations

import pandas as pd


def pillar_shares(pred_long: pd.DataFrame) -> pd.DataFrame:
    """-> bank, year, pillar, n, share, industry_share, share_dev."""
    cnt = pred_long.groupby(["bank", "year", "pillar"]).size().rename("n").reset_index()
    cnt["share"] = cnt["n"] / cnt.groupby(["bank", "year"])["n"].transform("sum")
    industry = (cnt.groupby(["year", "pillar"])["share"].mean()
                .rename("industry_share").reset_index())
    cnt = cnt.merge(industry, on=["year", "pillar"], how="left")
    cnt["share_dev"] = (cnt["share"] - cnt["industry_share"]).round(4)
    cnt["share"] = cnt["share"].round(4)
    cnt["industry_share"] = cnt["industry_share"].round(4)
    return cnt.sort_values(["bank", "year", "pillar"]).reset_index(drop=True)
