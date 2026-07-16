"""Diagnostic chen-so (spec 05 V2, perturbation kieu CheckList): do encoder/scorer
co bat shortcut "co chu so -> specific" khong.

Y tuong: lay cau MO HO (non-specific), chen them so/ten tieu chuan VO NGHIA (khong
phai dai luong dinh luong quy ve chu the). Scorer dung phai GIU nhan non-specific.
Flip-rate cao = bat shortcut theo chu so. So sanh:
- DigitPresenceScorer: gia thuyet shortcut hien ngon (specific <=> co chu so) -> flip ~100%.
- SpecificityLLM (rubric): ky vong flip ~0 vi suy luan so co quy ve chu the khong.
- (cam duoc) encoder specificity da train khi co -> nam o giua.

Scorer protocol: .predict(list[str]) -> DataFrame co cot 'is_specific' (0/1).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# Cac phep chen: deu la so/ten tieu chuan TRANG TRI, khong phai target dinh luong cua chu the.
PERTURBATIONS = {
    "national_strategy": lambda s: s.rstrip(". ")
    + ". Hoạt động này hưởng ứng Chiến lược quốc gia giai đoạn 2021-2030, tầm nhìn đến năm 2050.",
    "standards": lambda s: s.rstrip(". ")
    + ". Việc triển khai tuân theo các tiêu chuẩn ISO 14001 và VIETGAP.",
    "decorative_count": lambda s: s.rstrip(". ")
    + ". Nội dung được thực hiện qua hơn 100 hoạt động khác nhau.",
    "year_reference": lambda s: s.rstrip(". ") + " trong năm 2023.",
}

# Vai cau curated kieu ngan hang VN (gom motif BIDV) de seed phong phu hon gold dich.
CURATED_VAGUE = [
    "Ngân hàng cam kết đồng hành cùng khách hàng hướng tới một tương lai xanh và bền vững.",
    "Chúng tôi không ngừng nỗ lực nâng cao trách nhiệm với môi trường và cộng đồng.",
    "Ngân hàng tiếp tục thúc đẩy các sáng kiến tín dụng xanh và phát triển bền vững.",
]


def predict_is_specific(scorer, texts: list[str]) -> np.ndarray:
    return scorer.predict(list(texts))["is_specific"].to_numpy().astype(int)


class DigitPresenceScorer:
    """Strawman: du doan specific <=> cau chua it nhat 1 chu so. Hien ngon hoa shortcut."""

    _re = re.compile(r"\d")

    def predict(self, texts: list[str]) -> pd.DataFrame:
        return pd.DataFrame({"is_specific": [int(bool(self._re.search(str(t)))) for t in texts]})


def flip_rates(scorer, seeds: list[str],
               perturbations: dict | None = None) -> dict:
    """Tren tap seed du doan NON-specific, do ti le bi flip sang specific sau moi phep chen.

    -> {n_base, per_perturbation: {name: flip_rate}, overall_flip_rate}.
    """
    perturbations = perturbations or PERTURBATIONS
    base_pred = predict_is_specific(scorer, seeds)
    base = [s for s, p in zip(seeds, base_pred) if p == 0]   # chi cau dang non-specific
    out = {"n_seed": len(seeds), "n_base_nonspecific": len(base), "per_perturbation": {}}
    if not base:
        out["overall_flip_rate"] = None
        return out
    all_flips = []
    for name, fn in perturbations.items():
        pred = predict_is_specific(scorer, [fn(s) for s in base])
        flips = (pred == 1)
        out["per_perturbation"][name] = round(float(flips.mean()), 4)
        all_flips.append(flips)
    out["overall_flip_rate"] = round(float(np.concatenate(all_flips).mean()), 4)
    return out
