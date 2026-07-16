from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

_MODEL_DIR = os.environ.get("VNCORENLP_DIR", str(Path.home() / "vncorenlp"))
_BASE = "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/"
_WSEG_FILES = (
    "VnCoreNLP-1.2.jar",
    "models/wordsegmenter/vi-vocab",
    "models/wordsegmenter/wordsegmenter.rdr",
)
_segmenter = None


def _ensure_model(d: Path) -> None:
    import urllib.request

    for rel in _WSEG_FILES:
        dst = d / rel
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_BASE + rel, dst)


def _get_segmenter():
    global _segmenter
    if _segmenter is None:
        import py_vncorenlp

        d = Path(_MODEL_DIR)
        _ensure_model(d)
        cwd = os.getcwd()
        try:
            _segmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=str(d))
        finally:
            os.chdir(cwd)
    return _segmenter


@lru_cache(maxsize=500_000)
def word_segment(sentence: str) -> str:
    out = _get_segmenter().word_segment(sentence)
    return " ".join(out)


def word_segment_batch(sentences: list[str]) -> list[str]:
    return [word_segment(s) for s in sentences]


_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?%…)\]}])")
_SPACE_AFTER_OPEN = re.compile(r"([(\[{])\s+")


def _desegment(s: str) -> str:
    s = s.replace("_", " ")
    s = _SPACE_BEFORE_PUNCT.sub(r"\1", s)
    s = _SPACE_AFTER_OPEN.sub(r"\1", s)
    return re.sub(r"\s{2,}", " ", s).strip()


def sent_tokenize(text: str) -> list[str]:
    return [d for s in _get_segmenter().word_segment(text) if (d := _desegment(s))]
