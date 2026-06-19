"""Internal numerical axis (RQ5): reuse the specificity rubric's already-extracted
figures (no new extraction model) + per-pillar say-do gap.

say-do gap per (bank,year,pillar) = CTI_pillar - QDR_pillar (share vague minus
share quantified among that pillar's commitment chunks); >0 = says more than does.
"""
from __future__ import annotations

import json

import pandas as pd

PILLARS = ("env", "soc", "gov")
_ESG = [f"is_{p}" for p in PILLARS]

# priority-ordered keyword map (first match wins)
_CATEGORIES = [
    ("green_credit", ("tín dụng xanh", "dư nợ", "cho vay", "trái phiếu xanh", "giải ngân", "tài trợ vốn")),
    ("emissions", ("phát thải", "khí nhà kính", "carbon", "co2", "scope")),
    ("energy", ("năng lượng", "điện mặt trời", "điện gió", "tái tạo", "mw ", " mw")),
    ("trees", ("trồng", "cây xanh", "cây")),
    ("training", ("đào tạo", "tập huấn", "cán bộ", "nhân viên", "nhân sự")),
    ("social", ("ủng hộ", "từ thiện", "an sinh", "học bổng", "giáo dục", "y tế", "cộng đồng")),
]


def parse_figures(spec_rubric):
    if not spec_rubric or not isinstance(spec_rubric, str):
        return []
    try:
        obj = json.loads(spec_rubric)
    except (json.JSONDecodeError, TypeError):
        return []
    out = []
    for it in (obj.get("items") or []):
        fig = it.get("figure")
        if it.get("is_quantified") and fig:
            out.append({"action": str(it.get("action_or_event") or ""), "figure": str(fig)})
    return out


def categorize_action(action: str) -> str:
    low = str(action).lower()
    for name, kws in _CATEGORIES:
        if any(k in low for k in kws):
            return name
    return "other"


def _esg_commit(classified: pd.DataFrame) -> pd.DataFrame:
    esg = classified[_ESG].max(axis=1).astype(bool)
    return classified[(classified["is_commitment"] == 1) & esg]


def figure_table(classified: pd.DataFrame) -> pd.DataFrame:
    sub = _esg_commit(classified)
    rows = []
    for _, r in sub.iterrows():
        for f in parse_figures(r.get("spec_rubric")):
            rows.append({"bank": r["bank"], "year": r["year"],
                         "type": categorize_action(f["action"])})
    if not rows:
        return pd.DataFrame(columns=["bank", "year", "type", "n"])
    df = pd.DataFrame(rows)
    return df.groupby(["bank", "year", "type"]).size().rename("n").reset_index()


def pillar_say_do(classified: pd.DataFrame) -> pd.DataFrame:
    sub = _esg_commit(classified)
    rows = []
    for p in PILLARS:
        g = sub[sub[f"is_{p}"] == 1]
        for (bank, year), gg in g.groupby(["bank", "year"]):
            lv = gg["spec_level"].to_numpy()
            cti_p = float((lv == 0).mean())
            qdr_p = float((lv == 2).mean())
            rows.append({"bank": bank, "year": year, "pillar": p,
                         "cti_p": cti_p, "qdr_p": qdr_p,
                         "say_do": cti_p - qdr_p, "n": int(len(gg))})
    return pd.DataFrame(rows)
