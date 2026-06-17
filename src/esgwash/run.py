"""Luồng inference P2-P4: classify -> long -> ground -> index.

Gồm các hàm thuần trên DataFrame và bộ chạy đầy đủ theo từng (bank, year):
  python -m esgwash.run --bank bidv --year 2023 | --all
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from esgwash.grounding.evidence_pool import NUMERIC_PATTERN, action_anchors, anchor_overlap
from esgwash.grounding.support import grounded_flags, support_score, support_score_l1
from esgwash.models.commitment_model import CommitmentModel
from esgwash.models.specificity_llm import _digit_runs
from esgwash.models.topic_model import TopicModel

PILLARS = ("env", "soc", "gov")
HF_REPOS = {"topic": "huypham71/esg-topic", "commitment": "huypham71/esg-commitment"}

CHUNKS_PATH = "data/chunks.parquet"


def load_chunks(path: str = CHUNKS_PATH, bank: str | None = None,
                year: int | None = None) -> pd.DataFrame:
    """Đọc chunks.parquet; cột chuẩn: content_text (văn bản chunk) + chunk_index
    (id duy nhất trong một doc). Lọc theo bank/year nếu truyền."""
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
    """Nạp TopicModel/CommitmentModel từ thư mục local hoặc HF repo (mặc định HF_REPOS)."""
    src = source or HF_REPOS[name]
    if Path(src).exists() and (Path(src) / "config.json").exists():
        d = src
    else:
        from huggingface_hub import snapshot_download
        d = snapshot_download(src)
    cfg = json.loads((Path(d) / "config.json").read_text(encoding="utf-8"))
    cls = {"topic": TopicModel, "commitment": CommitmentModel}[name]
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
    """-> chunks + topic (is_/p_ theo trụ), pillar_top, commitment, specificity (+ rubric JSON).

    Specificity dùng LLM nên chỉ chấm trên chunk commitment (mẫu số CTI) khi
    spec_on_commitment=True: vừa tiết kiệm vừa đúng ngữ nghĩa, vì specific chỉ vào CTI
    qua cam kết. Chunk khác để is_specific=0.
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
    out["spec_level"] = 0       # 0=mơ hồ, 1=cụ thể (hành động có tên), 2=định lượng
    out["is_specific"] = 0      # = (spec_level >= 1); CTI đếm Mức 0
    out["spec_parse_ok"] = pd.NA
    out["spec_rubric"] = pd.NA
    out["spec_raw"] = pd.NA     # phản hồi LLM gốc trước parse, để truy vết / parse lại offline
    mask = (out["is_commitment"] == 1) if spec_on_commitment else pd.Series(True, index=out.index)
    if specificity_model is not None and bool(mask.any()):
        sp = specificity_model.predict(out.loc[mask, "content_text"].astype(str).tolist())
        out.loc[mask, "p_specific"] = sp["p_specificity"].to_numpy()
        out.loc[mask, "spec_level"] = sp["spec_level"].to_numpy()
        out.loc[mask, "is_specific"] = sp["is_specific"].to_numpy()
        out.loc[mask, "spec_parse_ok"] = sp["parse_ok"].to_numpy()
        out.loc[mask, "spec_rubric"] = sp["rubric"].to_numpy()
        out.loc[mask, "spec_raw"] = sp["raw"].to_numpy()
    return out


def to_long(classified: pd.DataFrame) -> pd.DataFrame:
    """Mỗi dòng = (chunk ESG × trụ dương của nó). Chunk không thuộc trụ nào thì bị loại."""
    parts = []
    for p in PILLARS:
        sub = classified[classified[f"is_{p}"] == 1].copy()
        sub["pillar"] = p
        parts.append(sub)
    return pd.concat(parts, ignore_index=True) if parts else classified.iloc[:0].copy()


_SENT_SPLIT = re.compile(r"\n+")


def split_chunk_sentences(text: str) -> list[str]:
    """Tách chunk thành câu theo '\\n' (build_chunks nối câu bằng '\\n'), không cần VnCoreNLP/Java."""
    return [s.strip() for s in _SENT_SPLIT.split(str(text)) if s.strip()]


def _quantified_items(rubric_json) -> list[dict]:
    """Lấy các item định lượng từ spec_rubric (đã qua verify_rubric lúc classify) ->
    [{claim, figs}]; claim = 'action_or_event figure' (mệnh đề gọn để NLI)."""
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


def _action_items(rubric_json) -> list[dict]:
    """Item hành động Mức 1: is_concrete_action & attributable_to_actor & KHÔNG is_quantified
    (định lượng -> Mức 2 path). -> [{claim, anchors}]; claim = action_or_event (mệnh đề hành động)."""
    if not rubric_json or str(rubric_json) in ("<NA>", "nan", "None"):
        return []
    try:
        rub = json.loads(rubric_json)
    except (json.JSONDecodeError, TypeError):
        return []
    out = []
    for it in (rub.get("items") or []):
        if not (it.get("is_concrete_action") and it.get("attributable_to_actor")):
            continue
        if it.get("is_quantified"):
            continue
        action = str(it.get("action_or_event") or "").strip()
        if action:
            out.append({"claim": action, "anchors": action_anchors(action)})
    return out


_L1_HYP_DEFAULT = "Theo báo cáo, ngân hàng đã thực sự thực hiện {action}."


def _l1_tok(text: str) -> list[str]:
    """Token hoá đơn giản (lowercase + tách khoảng trắng = âm tiết VN) cho BM25 retrieval L1."""
    return str(text).lower().split()


def ground_claims(classified: pd.DataFrame, retriever, nli, cfg: dict,
                  evidence_df: pd.DataFrame | None = None,
                  ground_l1: bool = False) -> pd.DataFrame:
    """Grounding mức item: claim = từng item định lượng (specificity phân rã), evidence = câu
    (tách chunk theo '\\n') ở bất kỳ đâu trong doc, kể cả cùng chunk, chỉ loại câu gốc chứa
    figure của item để tránh tự suy ra. support(chunk) = max P_entail trên các item; mỗi dòng =
    chunk-commitment kèm trace JSON.

    evidence_df: pool bằng chứng (mặc định = classified); truyền riêng khi classify một phần
    nhưng muốn lấy bằng chứng từ toàn báo cáo."""
    thetas = cfg.get("support_thresholds", [0.5, 0.7, 0.9])
    top_k = int(cfg.get("top_k", 5))
    use_numeric = cfg.get("pool", {}).get("numeric_regex", True)
    l1_hyp = cfg.get("l1_hypothesis", _L1_HYP_DEFAULT)
    l1_lam = float(cfg.get("l1_lambda", 0.5))
    l1_retrieval = cfg.get("l1_retrieval", "overlap")
    l1_bm25_topn = int(cfg.get("l1_bm25_topn", 20))
    l1_sim_floor = float(cfg.get("l1_sim_floor", 0.25))  # san riêng L1, thấp hơn L2 (sim_floor 0.5)
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
        # Pool bằng chứng = mọi câu của doc (tách '\n'), giữ câu có số liệu nếu bật numeric
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

        # Pool L1 (Mức 1): MỌI câu, KHÔNG lọc số (audit pitfall) — chỉ dựng khi bật ground_l1
        l1_src, l1_txt, l1_mat, l1_bm25 = np.array([], dtype=int), [], np.empty((0, 1)), None
        if ground_l1:
            _ls, _lt = [], []
            for _, r in ev_doc.iterrows():
                for s in split_chunk_sentences(r["content_text"]):
                    _ls.append(int(r["chunk_index"]))
                    _lt.append(s)
            l1_src, l1_txt = np.array(_ls, dtype=int), _lt
            l1_mat = retriever.embed(l1_txt) if l1_txt else np.empty((0, 1))
            if l1_retrieval == "bm25" and l1_txt:
                from rank_bm25 import BM25Okapi
                l1_bm25 = BM25Okapi([_l1_tok(t) for t in l1_txt])

        for _, c in tqdm(list(commit.iterrows()), desc=f"  {doc_id} claims",
                         unit="claim", leave=False):
            items = _quantified_items(c.get("spec_rubric"))
            if not items or len(ev_txt) == 0:
                # Không claim định lượng / pool số rỗng. Thử grounding Mức 1 (audit C1/C3) nếu bật.
                actions = _action_items(c.get("spec_rubric")) if ground_l1 else []
                if (ground_l1 and int(c.get("spec_level", 0) or 0) == 1
                        and actions and len(l1_txt) > 0):
                    best_sup, best_ids, trace = 0.0, [], []
                    for a in actions:
                        # C3: loại UNCONDITIONAL câu cùng chunk. 3a=anchor overlap | 3b=BM25 top-N.
                        not_same = l1_src != int(c["chunk_index"])
                        if l1_retrieval == "bm25" and l1_bm25 is not None:
                            sc = l1_bm25.get_scores(_l1_tok(a["claim"]))
                            order = np.argsort(-sc)
                            cand = np.array([j for j in order
                                             if not_same[j] and sc[j] > 0][:l1_bm25_topn], dtype=int)
                        else:
                            cand = np.array(
                                [j for j in range(len(l1_txt))
                                 if not_same[j] and anchor_overlap(l1_txt[j], a["anchors"])], dtype=int)
                        if len(cand) == 0:
                            trace.append({"claim": a["claim"], "support": 0.0,
                                          "evidence": [], "level": 1})
                            continue
                        claim_vec = retriever.embed([a["claim"]])[0]
                        keep, sims = retriever.topk(claim_vec, l1_mat[cand], k=top_k, floor=l1_sim_floor)
                        chosen = cand[keep]
                        if len(chosen) == 0:
                            trace.append({"claim": a["claim"], "support": 0.0,
                                          "evidence": [], "level": 1})
                            continue
                        hyp = l1_hyp.format(action=a["claim"])
                        probs = nli.score_pairs([(l1_txt[j], hyp) for j in chosen])
                        ent, con = nli.entail(probs), nli.contra(probs)
                        sup_a = support_score_l1(ent, con, l1_lam)
                        ev_list = [{"src_chunk": int(l1_src[j]), "text": l1_txt[j][:200],
                                    "sim": round(float(s), 3), "entail": round(float(e), 3),
                                    "contra": round(float(cc), 3)}
                                   for j, s, e, cc in zip(chosen, sims, ent, con)]
                        trace.append({"claim": a["claim"], "support": round(float(sup_a), 4),
                                      "evidence": ev_list, "level": 1})
                        if sup_a > best_sup:
                            best_sup = sup_a
                            best_ids = [int(l1_src[j]) for j in chosen]
                    n_ev = sum(len(t["evidence"]) for t in trace)
                    rows.append(_row(c, best_sup, len(actions), n_ev, best_ids, trace))
                else:
                    rows.append(_row(c, 0.0, len(items), 0, [], []))
                continue
            best_sup, best_ids, trace = 0.0, [], []
            for it in items:
                # loại câu gốc: cùng chunk với claim & chứa chữ số của figure (tránh tự suy ra)
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
    """Gắn support vào bảng long (theo doc_id+chunk_index) để tính gCTI."""
    if grounded.empty:
        claims_long = claims_long.copy()
        claims_long["support"] = 0.0
        return claims_long
    sup = grounded[["doc_id", "chunk_index", "support"]]
    out = claims_long.merge(sup, on=["doc_id", "chunk_index"], how="left")
    out["support"] = out["support"].fillna(0.0)
    return out


# --- Chạy đầy đủ P2-P4 cho từng (bank, year) -> outputs/cti/<bank>/<year>/ ---

CTI_ROOT = Path("outputs/cti")


def load_models(nli_model: str | None = None, need_classify: bool = True) -> dict:
    """Nạp model dùng chung: retriever + NLI (+ topic/commitment/specificity nếu need_classify).

    nli_model: override nli_model trong grounding.yml (B1/B2 dùng FEVER 2mil7); None = config (B0).
    need_classify=False (khi --reuse-classified): bỏ qua model P2 để chạy grounding nhanh."""
    from esgwash.config import load_config
    from esgwash.grounding.nli import NLIScorer
    from esgwash.grounding.retriever import EvidenceRetriever
    gcfg = load_config("grounding")
    if nli_model:
        gcfg = {**gcfg, "nli_model": nli_model}
    models = {"retriever": EvidenceRetriever(gcfg), "nli": NLIScorer(gcfg), "gcfg": gcfg}
    if need_classify:
        models["topic"] = load_trained_model("topic")
        models["commitment"] = load_commitment_model(load_config("commitment"))
        models["specificity"] = load_specificity_model(load_config("specificity"))
    return models


def run_bank_year(bank: str, year: int, models: dict, limit: int = 0,
                  out_root: str | Path = CTI_ROOT, ground_l1: bool = False,
                  reuse_classified: str | Path | None = None) -> Path:
    """Classify -> ground -> index cho 1 (bank, year); ghi kết quả vào outputs/cti/<bank>/<year>/."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from esgwash.indices.cti import build_cti_table
    from esgwash.indices.disclosure import pillar_shares

    gcfg = models["gcfg"]
    thetas = tuple(gcfg.get("support_thresholds", [0.5, 0.7, 0.9]))
    out_dir = Path(out_root) / bank / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)

    full = load_chunks(bank=bank, year=year)
    if full.empty:
        raise SystemExit(f"Không có chunk cho {bank} {year} trong data/chunks.parquet")
    if reuse_classified:  # TÁI DÙNG classify (P2 đồng nhất giữa các arm) — bỏ qua Qwen3, nhanh
        clf = pd.read_parquet(Path(reuse_classified) / bank / str(year) / "classified.parquet")
        print(f"### {bank} {year}: reuse classified ({len(clf)} chunk) <- {reuse_classified}/")
    else:
        sub = full.head(limit).copy() if limit else full
        print(f"### {bank} {year}: {len(sub)} chunk "
              f"(token/chunk p50={sub['token_count'].median():.0f} max={sub['token_count'].max()})")
        clf = classify_chunks(sub, models["topic"], models["commitment"], models["specificity"])
    clf.to_parquet(out_dir / "classified.parquet", index=False)

    long = to_long(clf)
    n_esg = int(clf[[f"is_{p}" for p in PILLARS]].sum(axis=1).gt(0).sum())
    n_commit = int(clf["is_commitment"].sum())
    lv = clf[clf["is_commitment"] == 1]["spec_level"].value_counts().to_dict()
    print(f"ESG={n_esg} ({n_esg/len(clf)*100:.1f}%) | commitment={n_commit} | "
          f"spec_level: mơ_hồ(0)={lv.get(0,0)} cụ_thể(1)={lv.get(1,0)} định_lượng(2)={lv.get(2,0)}")

    # Khi classify subset (limit), pool bằng chứng vẫn lấy từ toàn báo cáo
    evidence_pool = full if limit else clf
    grounded = ground_claims(clf, models["retriever"], models["nli"], gcfg,
                             evidence_df=evidence_pool, ground_l1=ground_l1)
    grounded.to_parquet(out_dir / "claims_grounded.parquet", index=False)

    long_s = attach_support(long, grounded)
    cti = build_cti_table(long_s, thetas=thetas, theta_l1=gcfg.get("theta_l1"), n_resamples=1000)
    shares = pillar_shares(long)
    cti = cti.merge(shares[["bank", "year", "pillar", "share"]],
                    on=["bank", "year", "pillar"], how="left")
    cti.to_parquet(out_dir / "cti.parquet", index=False)
    print(cti[["pillar", "n_commit", "cti", "cti_lo", "cti_hi", "gcti@0.7", "share"]]
          .to_string(index=False))

    cm = clf[clf["is_commitment"] == 1].merge(
        grounded[["chunk_index", "support", "n_evidence"]], on="chunk_index", how="left")
    no_pool = int((cm["n_evidence"].fillna(0) == 0).sum())
    info = {
        "bank": bank, "year": year, "n_chunks": int(len(clf)),
        "n_non_esg_dropped": int(len(clf) - n_esg),
        "non_esg_share": round(float(1 - n_esg / len(clf)), 3),
        "n_commitment": n_commit,
        "commit_no_evidence_pool": no_pool,
        "commit_no_evidence_share": round(no_pool / max(n_commit, 1), 3),
        "spec_level_counts": {int(k): int(v) for k, v in
                              cm["spec_level"].value_counts().sort_index().items()},
        "tok_p95": int(clf["token_count"].quantile(0.95)), "tok_max": int(clf["token_count"].max()),
        "chunk_gt_256tok_truncation_risk": int((clf["token_count"] > 256).sum()),
        "spec_parse_fail": int((clf["spec_parse_ok"] == False).sum()),  # noqa: E712
        "any_nan_pillar": bool(clf[[f"p_{p}" for p in PILLARS]].isna().any().any()),
    }
    (out_dir / "info_check.json").write_text(json.dumps(info, indent=2, ensure_ascii=False),
                                             encoding="utf-8")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(len(cti))
    ax[0].bar(x - 0.2, cti["cti"], 0.4, label="CTI", color="tab:orange")
    ax[0].bar(x + 0.2, cti["gcti@0.7"], 0.4, label="grounded-CTI@0.7", color="tab:red")
    ax[0].errorbar(x - 0.2, cti["cti"], yerr=[cti["cti"] - cti["cti_lo"], cti["cti_hi"] - cti["cti"]],
                   fmt="none", ecolor="k", capsize=3)
    ax[0].set_xticks(x); ax[0].set_xticklabels(cti["pillar"]); ax[0].set_ylim(0, 1)
    ax[0].set_title(f"{bank} {year} — CTI vs grounded-CTI"); ax[0].legend()
    sup = grounded["support"].dropna()
    ax[1].hist(sup, bins=20, color="tab:blue", alpha=0.8)
    for th in thetas:
        ax[1].axvline(th, ls="--", lw=1, label=f"θ={th}")
    ax[1].set_title("Phân bố evidence-support (commitment)"); ax[1].legend()
    fig.tight_layout(); fig.savefig(out_dir / "summary.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out_dir}/")
    return out_dir


def run_all(limit: int = 0, nli_model: str | None = None,
            out_root: str | Path = CTI_ROOT, ground_l1: bool = False,
            reuse_classified: str | Path | None = None, need_classify: bool = True) -> None:
    """Chạy mọi (bank, year) trong analysis_scope (corpus.yml); nạp model một lần."""
    from esgwash.config import load_config
    models = load_models(nli_model=nli_model, need_classify=need_classify)
    scope = load_config("corpus").get("analysis_scope", {})
    chunks = load_chunks()
    pairs = [(b, int(y)) for b in scope.get("banks", chunks["bank"].unique())
             for y in scope.get("years", chunks["year"].unique())]
    for bank, year in pairs:
        if not load_chunks(bank=bank, year=year).empty:
            run_bank_year(bank, year, models, limit=limit, out_root=out_root,
                          ground_l1=ground_l1, reuse_classified=reuse_classified)


def main(argv=None):
    import argparse
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Inference CTI cho 1 hoặc mọi (bank, year)")
    ap.add_argument("--bank", default="bidv")
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--all", action="store_true", help="chạy toàn bộ scope trong corpus.yml")
    ap.add_argument("--limit", type=int, default=0, help="0=full; >0 = N chunk đầu (smoke test)")
    ap.add_argument("--nli-model", default=None,
                    help="override NLI model (B1/B2: FEVER 2mil7); để trống = config (B0)")
    ap.add_argument("--out-root", default="outputs/cti",
                    help="thư mục gốc output (đặt riêng mỗi arm vd outputs/cti_b1 — tránh đè B0 đã commit)")
    ap.add_argument("--ground-l1", action="store_true",
                    help="B2: bật grounding Mức 1 (hành động có tên, pool phi-số + AIS hypothesis)")
    ap.add_argument("--reuse-classified", default=None,
                    help="ROOT chứa <bank>/<year>/classified.parquet để TÁI DÙNG (bỏ classify; B1/B2 nhanh+sạch)")
    args = ap.parse_args(argv)
    need_classify = args.reuse_classified is None
    if args.all:
        run_all(limit=args.limit, nli_model=args.nli_model, out_root=args.out_root,
                ground_l1=args.ground_l1, reuse_classified=args.reuse_classified,
                need_classify=need_classify)
    else:
        run_bank_year(args.bank, args.year,
                      load_models(nli_model=args.nli_model, need_classify=need_classify),
                      limit=args.limit, out_root=args.out_root, ground_l1=args.ground_l1,
                      reuse_classified=args.reuse_classified)


if __name__ == "__main__":
    main()
