"""Nạp dữ liệu gold (EN gốc + bản dịch VI), căn theo từng dòng.

Bản dịch VI cùng số dòng và cùng thứ tự với file gốc nên ghép text_en theo index.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from esgwash.data.cleaning import normalize_unicode

SOURCE_DIR = Path("source_data/source_dataset")
TRANSLATE_DIR = Path("source_data/translate")

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
    """lang='vi' lấy text từ bản dịch, text_en từ gốc; lang='en' thì text chính là bản gốc."""
    src = _read(SOURCE_DIR / subdir / rel)
    if lang == "en":
        src["text_en"] = src["text"]
        return src
    tr = _read(TRANSLATE_DIR / rel)
    assert len(tr) == len(src), f"{rel}: bản dịch ({len(tr)}) lệch số dòng với gốc ({len(src)})"
    tr["text_en"] = src["text"].values
    return tr


def load_topic(pillar: str, lang: str = "vi") -> pd.DataFrame:
    df = _load_aligned(TOPIC_FILES[pillar], "topic", lang)
    return df[["text", "text_en", pillar]]


def load_commitment(lang: str = "vi") -> pd.DataFrame:
    """commitments_actions (ClimateBERT) -> [text, text_en, commitment, split(train|test)]."""
    parts = []
    for split in ("train", "test"):
        df = _load_aligned(f"commitments_actions.{split}.parquet", "subst", lang)
        df = df.rename(columns={"label": "commitment"})
        df["split"] = split
        parts.append(df[["text", "text_en", "commitment", "split"]])
    return pd.concat(parts, ignore_index=True)


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
    """ml_promise EN+FR+JA (1200 dòng); lang='vi' lấy text từ bản dịch căn theo dòng.

    -> [text, promise, evidence, timeline, lang_src]
    """
    src = pd.read_csv(SOURCE_DIR / "ml_promise" / "ml_promise.csv")
    src = _clean_df(src).rename(columns={"lang": "lang_src"})
    if lang == "vi":
        vi_path = TRANSLATE_DIR / "ml_promise_vi.csv"
        if not vi_path.exists():
            raise FileNotFoundError(f"Chưa có bản dịch: {vi_path}")
        tr = _clean_df(pd.read_csv(vi_path))
        assert len(tr) == len(src), "ml_promise_vi lệch số dòng với gốc"
        src["text"] = tr["text"].values
    df = src
    df["promise"] = (df["promise_status"].str.strip().str.lower() == "yes").astype(int)
    df["evidence"] = (df["evidence_status"].astype(str).str.strip().str.lower() == "yes").astype(int)
    df["timeline"] = (df["verification_timeline"].astype(str).str.strip().str.lower()
                      .map(TIMELINE_MAP))
    return df[["text", "promise", "evidence", "timeline", "lang_src"]]


def load_ml_promise_json(lang_name: str = "Japanese") -> pd.DataFrame:
    """JSON gốc (utf-8-sig, có BOM); chỉ bản JA có ESG_type để eval tách trụ."""
    path = SOURCE_DIR / "ml_promise" / f"Trainset_{lang_name}.json"
    rows = json.loads(path.read_text(encoding="utf-8-sig"))
    return pd.DataFrame(rows)
