"""Dang ky stage end-to-end (spec 00 #4). Moi stage: ham thuan,
doc artifact truoc -> ghi artifact sau; chay doc lap duoc.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from esgwash.config import load_config

GOLD_DIR = Path("data/processed/gold")
METRICS_DIR = Path("outputs/metrics")


def stage_build_corpus() -> None:
    from esgwash.data.corpus_builder import build_corpus, corpus_stats
    cfg = load_config("corpus")
    sents = build_corpus(cfg)
    stats = corpus_stats(sents)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "corpus_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"corpus: {stats['n_sentences']} cau / {stats['n_docs']} doc")


def stage_prepare_gold() -> None:
    from esgwash.data import claim_merge, topic_merge
    cfg_topic = load_config("topic")
    GOLD_DIR.mkdir(parents=True, exist_ok=True)

    topic = topic_merge.build_masked_table(lang="vi")
    topic = topic_merge.split_stratified(topic, seed=42)
    if cfg_topic.get("augment_env_claims_positives", False):
        topic = topic_merge.augment_env_claims(topic, lang="vi")
    topic.to_parquet(GOLD_DIR / "topic_masked.parquet", index=False)

    cfg_claim = load_config("claim")
    claim = claim_merge.build_claim_table(
        lang="vi",
        augment_action=cfg_claim.get("augment_action_500", True),
        aux_env_claims=cfg_claim.get("aux_head_env_claims", True),
        augment_ml_promise=cfg_claim.get("augment_ml_promise", False))
    claim.to_parquet(GOLD_DIR / "claim_table.parquet", index=False)

    stats = {"topic": topic_merge.label_stats(topic),
             "claim": claim_merge.label_stats(claim)}
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "gold_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False, default=str))


def stage_export_annotation() -> None:
    from esgwash.data.vn_eval_set import sample_for_annotation
    cfg = load_config("corpus")
    sents = pd.read_parquet(cfg["out_sentences"])
    years = cfg.get("analysis_scope", {}).get("years")
    if years:
        sents = sents[sents["year"].isin(years)]
    preds_path = Path("outputs/metrics/topic_preds_corpus.parquet")
    preds = pd.read_parquet(preds_path) if preds_path.exists() else None
    out = sample_for_annotation(sents, n=300, seed=42, preds=preds)
    out_path = Path("data/annotation/vn_eval_todo.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"da xuat {len(out)} cau -> {out_path}"
          + ("" if preds is not None else " (chua co preds topic - stratify theo bank)"))


def _todo(name: str):
    def _fn():
        raise NotImplementedError(f"stage '{name}' trien khai o phase sau "
                                  "(train dung scripts/, grounding Phase C)")
    return _fn


STAGE_FNS = {
    "build_corpus": stage_build_corpus,
    "prepare_gold": stage_prepare_gold,
    "export_annotation": stage_export_annotation,
    "classify": _todo("classify"),
    "ground": _todo("ground"),
    "index": _todo("index"),
}
STAGES = list(STAGE_FNS)


def run_stage(name: str) -> None:
    STAGE_FNS[name]()
