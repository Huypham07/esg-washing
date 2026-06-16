"""Unit test tach ngu nghia + goi token cap (fake embeddings, khong load model)."""
import numpy as np

from esgwash.corpus.semantic_split import (
    adjacent_cosine, segment_indices, pack_to_token_cap, semantic_units,
)


def test_adjacent_cosine_len():
    emb = np.array([[1, 0], [1, 0], [0, 1]], dtype=float)
    sims = adjacent_cosine(emb)
    assert len(sims) == 2
    assert sims[0] > 0.99 and sims[1] < 0.01


def test_segment_indices_breaks_on_low_sim():
    # 4 cau: ranh gioi y sau cau index 1 (sim[1] thap)
    sims = np.array([0.8, 0.2, 0.9])
    segs = segment_indices(sims, threshold=0.5)
    assert segs == [[0, 1], [2, 3]]


def test_pack_respects_token_cap_without_splitting_sentences():
    sents = ["a", "b", "c"]
    toks = [200, 100, 100]      # 200 | 100+100 -> 2 don vi
    packed = pack_to_token_cap(sents, toks, max_tokens=256)
    assert packed == [["a"], ["b", "c"]]


def test_single_oversize_sentence_kept_alone():
    packed = pack_to_token_cap(["big"], [999], max_tokens=256)
    assert packed == [["big"]]   # khong the cat cau -> giu nguyen 1 don vi


def test_semantic_units_end_to_end():
    sents = ["s0", "s1", "s2"]
    emb = np.array([[1, 0], [1, 0], [0, 1]], dtype=float)  # break sau s1
    toks = [10, 10, 10]
    units = semantic_units(sents, emb, toks, threshold=0.5, max_tokens=256)
    assert units == [["s0", "s1"], ["s2"]]
