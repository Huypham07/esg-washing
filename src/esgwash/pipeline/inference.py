"""Luong inference end-to-end (spec 00 #4 tang P2-P4): classify -> long -> ground -> index.

Ham thuan tren DataFrame -> dung chung cho stage (toan corpus) lan script demo (1 bank/year).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from esgwash.grounding.evidence_pool import build_pool, candidate_mask
from esgwash.grounding.support import grounded_flags, support_score
from esgwash.models.claim_model import ClaimModel
from esgwash.models.topic_model import TopicModel

PILLARS = ("env", "soc", "gov")
HF_REPOS = {"topic": "huypham71/esg-topic", "claim": "huypham71/esg-claim"}


def load_trained_model(name: str, source: str | None = None):
    """Load TopicModel/ClaimModel tu local dir hoac HF repo (mac dinh HF_REPOS).
    ClaimModel giu lai cho ablation; luong inference dung commitment_hf + specificity_llm."""
    src = source or HF_REPOS[name]
    if Path(src).exists() and (Path(src) / "config.json").exists():
        d = src
    else:
        from huggingface_hub import snapshot_download
        d = snapshot_download(src)
    cfg = json.loads((Path(d) / "config.json").read_text(encoding="utf-8"))
    cls = {"topic": TopicModel, "claim": ClaimModel}[name]
    return cls(cfg).load(d)


def load_commitment_model(cfg: dict | None = None):
    from esgwash.models.commitment_hf import CommitmentHF
    cfg = cfg or {}
    return CommitmentHF(repo=cfg.get("model", "dqa2412/esg-washing-optimized"),
                        threshold=cfg.get("threshold", 0.5),
                        max_length=cfg.get("max_length", 256))


def load_specificity_model(cfg: dict | None = None):
    from esgwash.models.specificity_llm import SpecificityLLM
    return SpecificityLLM(cfg or {})


def classify_sentences(sents: pd.DataFrame, topic_model, commitment_model,
                       specificity_model, spec_on_commitment: bool = True) -> pd.DataFrame:
    """-> sents + topic (is_/p_ tru), pillar_top, commitment, specificity (+ rubric JSON).

    Specificity chay bang LLM nen chi cham tren cau commitment (mau so CTII) khi
    spec_on_commitment=True — vua tiet kiem vua dung ngu nghia (specific chi vao CTI
    qua cam ket). Cau khac: is_specific=0.
    """
    texts = sents["sentence"].astype(str).tolist()
    tp = topic_model.predict(texts)
    com = commitment_model.predict(texts)
    out = sents.reset_index(drop=True).copy()
    for p in PILLARS:
        out[f"p_{p}"] = tp[p].values
        out[f"is_{p}"] = tp[f"is_{p}"].values
    out["pillar_top"] = tp["pillar"].values
    out["p_commitment"] = com["p_commitment"].values
    out["is_commitment"] = com["is_commitment"].values

    out["p_specific"] = 0.0
    out["is_specific"] = 0
    out["spec_parse_ok"] = pd.NA
    out["spec_rubric"] = pd.NA
    mask = (out["is_commitment"] == 1) if spec_on_commitment else pd.Series(True, index=out.index)
    if specificity_model is not None and bool(mask.any()):
        sp = specificity_model.predict(out.loc[mask, "sentence"].astype(str).tolist())
        out.loc[mask, "p_specific"] = sp["p_specificity"].to_numpy()
        out.loc[mask, "is_specific"] = sp["is_specific"].to_numpy()
        out.loc[mask, "spec_parse_ok"] = sp["parse_ok"].to_numpy()
        out.loc[mask, "spec_rubric"] = sp["rubric"].to_numpy()
    return out


def to_long(classified: pd.DataFrame) -> pd.DataFrame:
    """1 dong / (cau ESG x pillar cau duong). Cau khong tru nao -> bi loai."""
    parts = []
    for p in PILLARS:
        sub = classified[classified[f"is_{p}"] == 1].copy()
        sub["pillar"] = p
        parts.append(sub)
    return pd.concat(parts, ignore_index=True) if parts else classified.iloc[:0].copy()


def ground_claims(classified: pd.DataFrame, retriever, nli, cfg: dict) -> pd.DataFrame:
    """Per doc: pool ung vien -> retrieve top-k claim -> NLI -> support = max P_entail."""
    thetas = cfg.get("support_thresholds", [0.5, 0.7, 0.9])
    rows = []

    def _base(c):
        return {"doc_id": c["doc_id"], "sent_id": c["sent_id"], "bank": c.get("bank"),
                "year": c.get("year"), "claim_text": c["sentence"]}

    for doc_id, doc in classified.groupby("doc_id"):
        claims = doc[doc["is_commitment"] == 1]
        if claims.empty:
            continue
        cand = doc[candidate_mask(doc, cfg)]
        if cand.empty:
            for _, c in claims.iterrows():
                rows.append({**_base(c), "support": 0.0, "n_evidence": 0,
                             "top_evidence_ids": [], **grounded_flags(0.0, thetas)})
            continue
        cand_mat = retriever.embed(cand["sentence"].tolist())
        claim_mat = retriever.embed(claims["sentence"].tolist())
        c_sid = cand["sent_id"].to_numpy()
        c_bid = cand["block_id"].to_numpy() if "block_id" in cand else np.full(len(cand), -1)
        c_txt = cand["sentence"].to_numpy()
        for i, (_, c) in enumerate(claims.iterrows()):
            bid = c.get("block_id", -2)
            valid = (c_sid != c["sent_id"]) & (c_bid != bid)
            vidx = np.where(valid)[0]
            if len(vidx) == 0:
                rows.append({**_base(c), "support": 0.0, "n_evidence": 0,
                             "top_evidence_ids": [], **grounded_flags(0.0, thetas)})
                continue
            keep, sims = retriever.topk(claim_mat[i], cand_mat[vidx])
            chosen = vidx[keep]
            if len(chosen) == 0:
                rows.append({**_base(c), "support": 0.0, "n_evidence": 0,
                             "top_evidence_ids": [], **grounded_flags(0.0, thetas)})
                continue
            pairs = [(str(c_txt[j]), str(c["sentence"])) for j in chosen]
            ent = nli.entail(nli.score_pairs(pairs))
            sup = support_score(ent)
            rows.append({**_base(c), "support": round(sup, 4), "n_evidence": int(len(chosen)),
                         "top_evidence_ids": [int(x) for x in c_sid[chosen]],
                         "top_sims": [round(float(s), 3) for s in sims],
                         "top_entail": [round(float(e), 3) for e in ent],
                         **grounded_flags(sup, thetas)})
    return pd.DataFrame(rows)


def attach_support(claims_long: pd.DataFrame, grounded: pd.DataFrame) -> pd.DataFrame:
    """Gan support vao long-format (theo doc_id+sent_id) cho gCTI."""
    if grounded.empty:
        claims_long = claims_long.copy()
        claims_long["support"] = 0.0
        return claims_long
    sup = grounded[["doc_id", "sent_id", "support"]]
    out = claims_long.merge(sup, on=["doc_id", "sent_id"], how="left")
    out["support"] = out["support"].fillna(0.0)
    return out
