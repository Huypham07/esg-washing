"""Word segmentation tieng Viet - BAT BUOC truoc khi tokenize PhoBERT."""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=500_000)
def word_segment(sentence: str) -> str:
    from underthesea import word_tokenize
    try:
        return word_tokenize(sentence, format="text")
    except Exception:
        return sentence


def word_segment_batch(sentences: list[str]) -> list[str]:
    return [word_segment(s) for s in sentences]


def sent_tokenize(text: str) -> list[str]:
    from underthesea import sent_tokenize as _st
    try:
        return _st(text)
    except Exception:
        return [text]
