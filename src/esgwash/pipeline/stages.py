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
    GOLD_DIR.mkdir(parents=True, exist_ok=True)

    # Nguon topic (2k + env_claims) da merge xong vao topic_masked.parquet va
    # chuyen file goc vao unused/ (2026-06-12) -> chi rebuild duoc neu khoi phuc file.
    if Path("data/source_dataset/topic/environmental_2k.csv").exists():
        topic = topic_merge.build_topic_table(lang="vi", seed=42)
        topic.to_parquet(GOLD_DIR / "topic_masked.parquet", index=False)
    else:
        topic = pd.read_parquet(GOLD_DIR / "topic_masked.parquet")
        print("nguon topic da o unused/ — giu topic_masked.parquet hien co")

    cfg_claim = load_config("claim")
    claim = claim_merge.build_claim_table(
        lang="vi",
        augment_action=cfg_claim.get("augment_action_500", True),
        augment_ml_promise=cfg_claim.get("augment_ml_promise", True))
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


CLASSIFY_DIR = Path("outputs/classify")
GROUND_DIR = Path("outputs/grounding")
INDEX_DIR = Path("outputs/index")


def _scope_sentences(sents: pd.DataFrame) -> pd.DataFrame:
    """Loc theo analysis_scope (banks/years) trong corpus.yml neu co."""
    scope = load_config("corpus").get("analysis_scope", {})
    if scope.get("years"):
        sents = sents[sents["year"].isin(scope["years"])]
    if scope.get("banks"):
        sents = sents[sents["bank"].isin(scope["banks"])]
    return sents


def stage_classify() -> None:
    from esgwash.pipeline.inference import classify_sentences, load_trained_model
    sents = _scope_sentences(pd.read_parquet(load_config("corpus")["out_sentences"]))
    topic = load_trained_model("topic")
    claim = load_trained_model("claim")
    out = classify_sentences(sents, topic, claim)
    CLASSIFY_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CLASSIFY_DIR / "sentences_classified.parquet", index=False)
    print(f"classified {len(out)} cau | commitment={int(out['is_commitment'].sum())} "
          f"| ESG={int((out[['is_env','is_soc','is_gov']].sum(axis=1) > 0).sum())}")


def stage_ground() -> None:
    from esgwash.grounding.nli import NLIScorer
    from esgwash.grounding.retriever import EvidenceRetriever
    from esgwash.pipeline.inference import ground_claims
    cfg = load_config("grounding")
    classified = pd.read_parquet(CLASSIFY_DIR / "sentences_classified.parquet")
    grounded = ground_claims(classified, EvidenceRetriever(cfg), NLIScorer(cfg), cfg)
    GROUND_DIR.mkdir(parents=True, exist_ok=True)
    grounded.to_parquet(GROUND_DIR / "claims_grounded.parquet", index=False)
    print(f"grounded {len(grounded)} commitment | support>=0.7: "
          f"{int((grounded['support'] >= 0.7).sum())}")


def stage_index() -> None:
    from esgwash.indices.cti import build_cti_table
    from esgwash.indices.disclosure import pillar_shares
    from esgwash.pipeline.inference import attach_support, to_long
    cfg = load_config("index")
    gcfg = load_config("grounding")
    classified = pd.read_parquet(CLASSIFY_DIR / "sentences_classified.parquet")
    grounded = pd.read_parquet(GROUND_DIR / "claims_grounded.parquet")
    long = attach_support(to_long(classified), grounded)
    boot = cfg.get("bootstrap", {})
    cti = build_cti_table(long, thetas=tuple(gcfg.get("support_thresholds", [0.5, 0.7, 0.9])),
                          n_resamples=boot.get("n_resamples", 1000), ci=boot.get("ci", 0.95))
    shares = pillar_shares(to_long(classified))
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    cti.merge(shares[["bank", "year", "pillar", "share", "share_dev"]],
              on=["bank", "year", "pillar"], how="left").to_parquet(
        INDEX_DIR / "cti.parquet", index=False)
    shares.to_parquet(INDEX_DIR / "pillar_shares.parquet", index=False)
    print(f"index xong: {len(cti)} o (bank,year,pillar) -> {INDEX_DIR}/cti.parquet")


STAGE_FNS = {
    "build_corpus": stage_build_corpus,
    "prepare_gold": stage_prepare_gold,
    "export_annotation": stage_export_annotation,
    "classify": stage_classify,
    "ground": stage_ground,
    "index": stage_index,
}
STAGES = list(STAGE_FNS)


def run_stage(name: str) -> None:
    STAGE_FNS[name]()
