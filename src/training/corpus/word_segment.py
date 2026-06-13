from __future__ import annotations

from functools import lru_cache
from typing import List

try:
    from underthesea import word_tokenize as _ut_word_tokenize
    _UT_AVAILABLE = True
except Exception:
    _ut_word_tokenize = None
    _UT_AVAILABLE = False


@lru_cache(maxsize=100_000)
def word_segment(text: str) -> str:
    if not text or not _UT_AVAILABLE:
        return text
    try:
        return _ut_word_tokenize(text, format="text")
    except Exception:
        return text


def word_segment_batch(texts: List[str]) -> List[str]:
    return [word_segment(t if isinstance(t, str) else str(t)) for t in texts]


def unsegment(text: str) -> str:
    if not text:
        return text
    return text.replace("_", " ")


def is_available() -> bool:
    return _UT_AVAILABLE
