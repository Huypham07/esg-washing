"""CTI & grounded-CTI (spec 04 #1-2; Bingler et al. 2022).

CTI(b,y,p)  = |{commitment & ~specific}| / |{commitment}|
gCTI(b,y,p) = |{commitment & (~specific | (specific & support<theta))}| / |{commitment}|
Chi la ti le output classifier - khong trong so tu dat.
"""
import pandas as pd


def compute_cti(claims_df: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError  # TODO(Phase D1)


def compute_grounded_cti(claims_df: pd.DataFrame, theta: float) -> pd.DataFrame:
    raise NotImplementedError
