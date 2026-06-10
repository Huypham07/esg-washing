"""Merge 3 tap topic nhi phan -> bang masked multi-label (spec 01 #3).

Chien luoc chinh: bang [text, env, soc, gov] voi NaN = khong co nhan
(partial labels) -> masked BCE. Chien luoc phu: cross-pseudo-labeling
(confidence >= tau) dien bot NaN - ablation.
Augment optional: env_claims positives -> env=1.
"""
import pandas as pd


def build_masked_table(config: dict) -> pd.DataFrame:
    """6000 dong, moi dong co dung 1 cot nhan (tru khi augment/pseudo)."""
    raise NotImplementedError  # TODO(Phase A3)


def split_stratified(df: pd.DataFrame, seed: int = 42):
    """80/10/10 stratified TRONG tung tap nguon truoc khi gop (chong leak)."""
    raise NotImplementedError


def cross_pseudo_label(df: pd.DataFrame, models: dict, tau: float = 0.9) -> pd.DataFrame:
    """Vong 2 optional - dien NaN bang du doan confidence cao cua model trai nguon."""
    raise NotImplementedError
