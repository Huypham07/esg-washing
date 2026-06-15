"""Luong inference end-to-end (spec 00 #4 tang P2-P4): classify -> long -> ground -> index.

Ham thuan tren DataFrame -> dung chung cho stage (toan corpus) lan script demo (1 bank/year).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from esgwash.grounding.evidence_pool import NUMERIC_PATTERN
from esgwash.grounding.support import grounded_flags, support_score
from esgwash.models.claim_model import ClaimModel
from esgwash.models.specificity_llm import _digit_runs
from esgwash.models.topic_model import TopicModel

PILLARS = ("env", "soc", "gov")
HF_REPOS = {"topic": "huypham71/esg-topic", "claim": "huypham71/esg-claim"}

CHUNKS_PATH = "data/corpus_chunk/chunks.parquet"


def load_chunks(path: str = CHUNKS_PATH, bank: str | None = None,
                year: int | None = None) -> pd.DataFrame:
    """Doc chunks.parquet (don vi = CHUNK). Cot chuan dung xuyen suot luong inference:
    `content_text` (van ban chunk) + `chunk_index` (id duy nhat trong 1 doc; ground_claims
    group theo doc_id nen chunk_index khong dung do giua cac doc). Moi chunk la mot don vi
    doc lap => bang chung co the den tu BAT KY chunk khac cung doc (chi loai chinh no).
    """
    df = pd.read_parquet(path)
    if bank is not None:
        df = df[df["bank"] == bank]
    if year is not None:
        df = df[df["year"] == year]
    df = df.reset_index(drop=True).copy()
    df["content_text"] = df["content_text"].astype(str)
    df["chunk_index"] = df["chunk_index"].astype(int)
    return df


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


def classify_chunks(chunks: pd.DataFrame, topic_model, commitment_model,
                    specificity_model, spec_on_commitment: bool = True) -> pd.DataFrame:
    """-> chunks + topic (is_/p_ tru), pillar_top, commitment, specificity (+ rubric JSON).

    Specificity chay bang LLM nen chi cham tren chunk commitment (mau so CTI) khi
    spec_on_commitment=True — vua tiet kiem vua dung ngu nghia (specific chi vao CTI
    qua cam ket). Chunk khac: is_specific=0.
    """
    texts = chunks["content_text"].astype(str).tolist()
    tp = topic_model.predict(texts)
    com = commitment_model.predict(texts)
    out = chunks.reset_index(drop=True).copy()
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
    out["spec_raw"] = pd.NA  # raw response LLM truoc parse (trace / re-parse offline)
    mask = (out["is_commitment"] == 1) if spec_on_commitment else pd.Series(True, index=out.index)
    if specificity_model is not None and bool(mask.any()):
        sp = specificity_model.predict(out.loc[mask, "content_text"].astype(str).tolist())
        out.loc[mask, "p_specific"] = sp["p_specificity"].to_numpy()
        out.loc[mask, "is_specific"] = sp["is_specific"].to_numpy()
        out.loc[mask, "spec_parse_ok"] = sp["parse_ok"].to_numpy()
        out.loc[mask, "spec_rubric"] = sp["rubric"].to_numpy()
        out.loc[mask, "spec_raw"] = sp["raw"].to_numpy()
    return out


def to_long(classified: pd.DataFrame) -> pd.DataFrame:
    """1 dong / (chunk ESG x pillar duong cua no). Chunk khong thuoc tru nao -> bi loai."""
    parts = []
    for p in PILLARS:
        sub = classified[classified[f"is_{p}"] == 1].copy()
        sub["pillar"] = p
        parts.append(sub)
    return pd.concat(parts, ignore_index=True) if parts else classified.iloc[:0].copy()


_SENT_SPLIT = re.compile(r"\n+")


def split_chunk_sentences(text: str) -> list[str]:
    """Tach chunk -> cau. Chunk duoc build bang cach noi cau bang '\\n' (build_chunks),
    nen tach lai theo '\\n' la trung thiet ke, KHONG can VnCoreNLP/Java."""
    return [s.strip() for s in _SENT_SPLIT.split(str(text)) if s.strip()]


def _quantified_items(rubric_json) -> list[dict]:
    """Lay cac item DINH LUONG tu spec_rubric (da qua verify_rubric luc classify) ->
    [{claim, figs}]. claim = 'action_or_event figure' (mot menh de gon de NLI)."""
    if not rubric_json or str(rubric_json) in ("<NA>", "nan", "None"):
        return []
    try:
        rub = json.loads(rubric_json)
    except (json.JSONDecodeError, TypeError):
        return []
    out = []
    for it in (rub.get("items") or []):
        if not it.get("is_quantified"):
            continue
        action = str(it.get("action_or_event") or "").strip()
        figure = str(it.get("figure") or "").strip()
        claim = f"{action} {figure}".strip()
        if claim:
            out.append({"claim": claim, "figs": _digit_runs(figure)})
    return out


def ground_claims(classified: pd.DataFrame, retriever, nli, cfg: dict,
                  evidence_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """ITEM-LEVEL grounding (2026-06-15). Claim = tung item dinh luong (specificity phan ra);
    evidence = CAU (tach chunk theo '\\n') o BAT KY dau trong doc, KE CA cung chunk — chi loai
    dung cau goc chua figure cua item (tranh tu suy ra chinh minh). support(chunk) = max P_entail
    tren cac item cua chunk; gCTI dung support nay. Tra ve 1 dong / chunk-commitment + trace JSON.

    evidence_df: pool bang chung (mac dinh = classified). Truyen rieng khi classify subset
    nhung muon bang chung lay tu TOAN BO bao cao (vd demo/--limit, eval mau)."""
    thetas = cfg.get("support_thresholds", [0.5, 0.7, 0.9])
    top_k = int(cfg.get("top_k", 5))
    use_numeric = cfg.get("pool", {}).get("numeric_regex", True)
    ev_source = evidence_df if evidence_df is not None else classified
    rows = []

    def _row(c, support, n_items, n_ev, top_ids, trace):
        return {"doc_id": c["doc_id"], "chunk_index": int(c["chunk_index"]),
                "bank": c.get("bank"), "year": c.get("year"),
                "support": round(float(support), 4), "n_items": int(n_items),
                "n_evidence": int(n_ev), "top_evidence_ids": top_ids,
                "item_grounding": json.dumps(trace, ensure_ascii=False),
                **grounded_flags(support, thetas)}

    from tqdm.auto import tqdm
    groups = list(classified.groupby("doc_id"))
    for doc_id, doc in tqdm(groups, desc="grounding (docs)", unit="doc"):
        commit = doc[doc["is_commitment"] == 1]
        if commit.empty:
            continue
        # Pool bang chung = TAT CA cau cua doc (tach '\n'), giu cau co so lieu (neu bat numeric)
        ev_doc = ev_source[ev_source["doc_id"] == doc_id]
        ev_src, ev_txt = [], []
        for _, r in ev_doc.iterrows():
            for s in split_chunk_sentences(r["content_text"]):
                if use_numeric and not NUMERIC_PATTERN.search(s):
                    continue
                ev_src.append(int(r["chunk_index"]))
                ev_txt.append(s)
        ev_src = np.array(ev_src, dtype=int)
        ev_mat = retriever.embed(ev_txt) if ev_txt else np.empty((0, 1))

        for _, c in tqdm(list(commit.iterrows()), desc=f"  {doc_id} claims",
                         unit="claim", leave=False):
            items = _quantified_items(c.get("spec_rubric"))
            if not items or len(ev_txt) == 0:
                rows.append(_row(c, 0.0, len(items), 0, [], []))
                continue
            best_sup, best_ids, trace = 0.0, [], []
            for it in items:
                # loai cau goc: cung chunk claim & chua chu so cua figure (self-support)
                same = ev_src == int(c["chunk_index"])
                has_fig = np.array([any(f in _digit_runs(t) for f in it["figs"]) for t in ev_txt]) \
                    if it["figs"] else np.zeros(len(ev_txt), dtype=bool)
                vidx = np.where(~(same & has_fig))[0]
                if len(vidx) == 0:
                    trace.append({"claim": it["claim"], "support": 0.0, "evidence": []})
                    continue
                claim_vec = retriever.embed([it["claim"]])[0]
                keep, sims = retriever.topk(claim_vec, ev_mat[vidx], k=top_k)
                chosen = vidx[keep]
                if len(chosen) == 0:
                    trace.append({"claim": it["claim"], "support": 0.0, "evidence": []})
                    continue
                ent = nli.entail(nli.score_pairs([(ev_txt[j], it["claim"]) for j in chosen]))
                sup_i = support_score(ent)
                ev_list = [{"src_chunk": int(ev_src[j]), "text": ev_txt[j][:200],
                            "sim": round(float(s), 3), "entail": round(float(e), 3)}
                           for j, s, e in zip(chosen, sims, ent)]
                trace.append({"claim": it["claim"], "support": round(float(sup_i), 4),
                              "evidence": ev_list})
                if sup_i > best_sup:
                    best_sup = sup_i
                    best_ids = [int(ev_src[j]) for j in chosen]
            n_ev = sum(len(t["evidence"]) for t in trace)
            rows.append(_row(c, best_sup, len(items), n_ev, best_ids, trace))
    return pd.DataFrame(rows)


def attach_support(claims_long: pd.DataFrame, grounded: pd.DataFrame) -> pd.DataFrame:
    """Gan support vao long-format (theo doc_id+chunk_index) cho gCTI."""
    if grounded.empty:
        claims_long = claims_long.copy()
        claims_long["support"] = 0.0
        return claims_long
    sup = grounded[["doc_id", "chunk_index", "support"]]
    out = claims_long.merge(sup, on=["doc_id", "chunk_index"], how="left")
    out["support"] = out["support"].fillna(0.0)
    return out
