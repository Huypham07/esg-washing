"""Unit test logic thuan: masked table, split khong leak, masked BCE, dedup."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from esgwash.data.cleaning import dedup_mask, is_valid_sentence
from esgwash.data.topic_merge import (PILLARS, fill_cross_labels, label_stats,
                                      split_stratified)


def make_masked(n=200, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        row = {"text": f"cau {i}", "text_en": f"sent {i}",
               "env": np.nan, "soc": np.nan, "gov": np.nan}
        p = rng.choice(PILLARS)
        row[p] = float(rng.integers(0, 2))
        rows.append(row)
    df = pd.DataFrame(rows)
    df["sources"] = df[list(PILLARS)].notna().idxmax(axis=1)
    return df


def test_split_no_leak_and_ratio():
    df = split_stratified(make_masked(500))
    assert df["text_en"].nunique() == len(df)
    sizes = df["split"].value_counts(normalize=True)
    assert abs(sizes["train"] - 0.8) < 0.05
    assert set(df["split"]) == {"train", "val", "test"}


def test_fill_cross_labels_fills_pool_keeps_test_gold():
    df = split_stratified(make_masked(100))
    probs = pd.DataFrame(0.95, index=df.index, columns=list(PILLARS))
    orig = df.copy()
    out = fill_cross_labels(df, probs, tau=0.9)
    labeled = orig[list(PILLARS)].notna()
    # nhan goc khong bi ghi de
    assert (out[list(PILLARS)].values[labeled.values]
            == orig[list(PILLARS)].values[labeled.values]).all()
    # pool train (split != test): moi o NaN confidence >= tau deu duoc dien
    pool = out["split"] != "test"
    assert out.loc[pool, list(PILLARS)].isna().sum().sum() == 0
    # test giu GOLD thuan: o non-gold van NaN (de evaluate mask -> do tren gold)
    is_test = out["split"] == "test"
    assert (out.loc[is_test, list(PILLARS)].isna().sum().sum()
            == orig.loc[is_test, list(PILLARS)].isna().sum().sum())


def test_label_stats():
    stats = label_stats(make_masked(100))
    assert stats["n_rows"] == 100
    assert all(p in stats for p in PILLARS)


def test_dedup_exact_and_near():
    texts = ["mot cau hoan toan khac biet o day",
             "mot cau hoan toan khac biet o day",
             "ngan hang cam ket giam phat thai khi nha kinh xuong muc thap nhat nam 2030",
             "ngan hang cam ket giam phat thai khi nha kinh xuong muc thap nhat nam 2031",
             "van ban thu ba khong lien quan gi den hai cau tren ca"]
    keep = dedup_mask(texts)
    assert not keep[1]
    assert keep[0] and keep[2] and keep[4]
    assert not keep[3]


def test_is_valid_sentence():
    assert not is_valid_sentence("abc")
    assert not is_valid_sentence("12 34 56 78")
    assert is_valid_sentence("Ngân hàng công bố báo cáo bền vững.")


def test_masked_bce_loss():
    torch = pytest.importorskip("torch")
    from esgwash.models.trainer import masked_bce_loss
    logits = torch.zeros(2, 3)
    targets = torch.tensor([[1.0, float("nan"), 0.0], [float("nan")] * 3])
    pw = torch.ones(3)
    loss = masked_bce_loss(logits, targets, pw)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.zeros(2), torch.tensor([1.0, 0.0]))
    assert torch.isclose(loss, expected)
    all_nan = torch.full((2, 3), float("nan"))
    assert masked_bce_loss(logits, all_nan, pw).item() == 0.0
