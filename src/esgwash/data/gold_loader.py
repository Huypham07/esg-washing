"""Load gold EN + ban dich VI, chuan hoa cot (spec 01 #2).

- Bo cot 'Unnamed: 0'; NFC normalize text.
- KHONG load netzero_reduction (da loai khoi thiet ke, quyet dinh 2026-06-10).
- Tra ve dict task -> DataFrame[text, <labels...>, split].
"""
from pathlib import Path

GOLD_DIR = Path("data/en_gold")

TASKS = ("topic_env", "topic_soc", "topic_gov", "commitment_specificity",
         "env_claims", "action_500", "ml_promise")


def load_task(task: str, lang: str = "vi"):
    """lang='vi' -> translate/, lang='en' -> topic|subst/ (upper-bound E2)."""
    raise NotImplementedError  # TODO(Phase A3)


def load_ml_promise(langs: tuple = ("English", "French", "Japanese")):
    """data/external/ml_promise/Trainset_<lang>.json (spec 01 #7.1).

    - Doc bang utf-8-sig (file co BOM).
    - Chinese: chi giu dong promise_string != 'N/A' (146 positive); Korean: bo (khong co text).
    - Chuan hoa verification_timeline (gia tri khac nhau giua ngon ngu, strip space).
    - Giu cot nguon `lang` de stratify (ti le promise lech manh: EN 78% vs ZH 36% Yes).
    -> DataFrame[text, promise, evidence, evidence_quality, timeline, esg_type?, lang]
    """
    raise NotImplementedError  # TODO(Phase A3)
