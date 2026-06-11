"""Load gold EN + ban dich VI, chuan hoa cot (spec 01 #2).

- Bo cot 'Unnamed: 0'; NFC normalize text; giu text_en de QE filter/merge.
- File dich cung ten, cung thu tu dong voi file goc -> align theo index.
- KHONG load netzero_reduction (da loai khoi thiet ke, quyet dinh 2026-06-10).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from esgwash.data.cleaning import normalize_unicode

SOURCE_DIR = Path("data/source_dataset")
TRANSLATE_DIR = Path("data/translate")

TOPIC_FILES = {"env": "environmental_2k.csv", "soc": "social_2k.csv",
               "gov": "governance_2k.csv"}


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    df["text"] = df["text"].astype(str).map(normalize_unicode).str.strip()
    return df


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
    return _clean_df(df)


def _load_aligned(rel: str, subdir: str, lang: str) -> pd.DataFrame:
    """lang='vi': text tu translate/, text_en tu source; lang='en': text = text_en."""
    src = _read(SOURCE_DIR / subdir / rel)
    if lang == "en":
        src["text_en"] = src["text"]
        return src
    tr = _read(TRANSLATE_DIR / rel)
    assert len(tr) == len(src), f"{rel}: translate ({len(tr)}) != source ({len(src)})"
    tr["text_en"] = src["text"].values
    return tr


def load_topic(pillar: str, lang: str = "vi") -> pd.DataFrame:
    df = _load_aligned(TOPIC_FILES[pillar], "topic", lang)
    return df[["text", "text_en", pillar]]


def load_claim_pair(lang: str = "vi") -> pd.DataFrame:
    """Commitment + specificity chia se cung tap van ban -> merge theo text_en.

    -> DataFrame[text, text_en, commitment, specificity, split(train|test)]
    """
    parts = []
    for split in ("train", "test"):
        com = _load_aligned(f"commitments_actions.{split}.parquet", "subst", lang)
        spe = _load_aligned(f"specificity.{split}.parquet", "subst", lang)
        com = com.rename(columns={"label": "commitment"})
        spe = spe.rename(columns={"label": "specificity"})
        spe = spe.drop_duplicates("text_en")[["text_en", "specificity"]]
        df = com.merge(spe, on="text_en", how="left")
        df["split"] = split
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    return out[["text", "text_en", "commitment", "specificity", "split"]]


def load_env_claims(lang: str = "vi") -> pd.DataFrame:
    parts = []
    for split in ("train", "val", "test"):
        df = _load_aligned(f"env_claims.{split}.parquet", "subst", lang)
        df = df.rename(columns={"label": "claim"})
        df["split"] = split
        parts.append(df)
    return pd.concat(parts, ignore_index=True)[["text", "text_en", "claim", "split"]]


def load_action(lang: str = "vi") -> pd.DataFrame:
    df = _load_aligned("action_500.csv", "subst", lang)
    return df[["text", "text_en", "action"]]


TIMELINE_MAP = {
    "already": "already", "less than 2 years": "lt_2y", "within 2 years": "lt_2y",
    "within_2_years": "lt_2y", "2 to 5 years": "2_5y", "2_to_5_years": "2_5y",
    "more than 5 years": "gt_5y", "longer than 5 years": "gt_5y", "n/a": None, "nan": None,
}


def load_ml_promise(lang: str = "vi") -> pd.DataFrame:
    """Flatten csv (en/fr/ja/zh, 1346 dong) hoac ban dich VI khi co.

    lang='vi' can data/translate/ml_promise_vi.csv (user dang dich) — cung cot,
    cung thu tu dong voi source_dataset/ml_promise/ml_promise.csv.
    -> DataFrame[text, text_en?, promise, evidence, timeline, lang_src]
    """
    src = pd.read_csv(SOURCE_DIR / "ml_promise" / "ml_promise.csv")
    src = _clean_df(src).rename(columns={"lang": "lang_src"})
    if lang == "vi":
        vi_path = TRANSLATE_DIR / "ml_promise_vi.csv"
        if not vi_path.exists():
            raise FileNotFoundError(f"Chua co ban dich: {vi_path}")
        tr = _clean_df(pd.read_csv(vi_path))
        assert len(tr) == len(src), "ml_promise_vi khong align voi source"
        src["text"] = tr["text"].values
    df = src
    df["promise"] = (df["promise_status"].str.strip().str.lower() == "yes").astype(int)
    df["evidence"] = (df["evidence_status"].astype(str).str.strip().str.lower() == "yes").astype(int)
    df["timeline"] = (df["verification_timeline"].astype(str).str.strip().str.lower()
                      .map(TIMELINE_MAP))
    return df[["text", "promise", "evidence", "timeline", "lang_src"]]


def load_ml_promise_json(lang_name: str = "Japanese") -> pd.DataFrame:
    """JSON goc (utf-8-sig, co BOM) - chi JA co ESG_type de eval tach tru."""
    path = SOURCE_DIR / "ml_promise" / f"Trainset_{lang_name}.json"
    rows = json.loads(path.read_text(encoding="utf-8-sig"))
    return pd.DataFrame(rows)
