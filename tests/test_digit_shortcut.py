"""Test diagnostic chen-so: digit-scorer flip cao, scorer 'robust' flip 0."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from esgwash.validation.digit_shortcut import (DigitPresenceScorer, PERTURBATIONS,
                                               flip_rates)

VAGUE = ["Ngân hàng cam kết hướng tới tương lai bền vững.",
         "Chúng tôi nỗ lực vì môi trường và cộng đồng.",
         "Tiếp tục thúc đẩy phát triển xanh."]


def test_digit_scorer_flips_on_injection():
    res = flip_rates(DigitPresenceScorer(), VAGUE)
    assert res["n_base_nonspecific"] == 3       # ca 3 cau goc khong co chu so
    # moi phep chen them chu so -> digit-scorer flip het
    assert res["overall_flip_rate"] == 1.0
    assert res["per_perturbation"]["year_reference"] == 1.0


class RobustStub:
    """Luon non-specific (bo qua chu so) -> khong bao gio flip."""

    def predict(self, texts):
        return pd.DataFrame({"is_specific": [0] * len(texts)})


def test_robust_scorer_no_flip():
    res = flip_rates(RobustStub(), VAGUE)
    assert res["overall_flip_rate"] == 0.0


def test_base_filters_already_specific():
    # cau co san chu so -> digit-scorer goi la specific -> bi loai khoi base
    seeds = VAGUE + ["Giảm 30% phát thải vào 2030."]
    res = flip_rates(DigitPresenceScorer(), seeds)
    assert res["n_seed"] == 4 and res["n_base_nonspecific"] == 3


def test_perturbations_add_digits():
    for fn in PERTURBATIONS.values():
        assert any(ch.isdigit() for ch in fn("cau mo ho khong so"))
