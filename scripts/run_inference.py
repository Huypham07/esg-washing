"""Demo end-to-end 1 bank/nam: classify -> ground -> CTI/gCTI + query + kiem mat thong tin.

  python scripts/run_inference.py --bank bidv --year 2023
"""
import argparse
import json
import sys
import textwrap
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):  # tranh crash print tieng Viet tren console cp1252 (Windows)
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from esgwash.config import load_config
from esgwash.grounding.nli import NLIScorer
from esgwash.grounding.retriever import EvidenceRetriever
from esgwash.indices.cti import build_cti_table
from esgwash.indices.disclosure import pillar_shares
from esgwash.pipeline.inference import (attach_support, classify_chunks,
                                        ground_claims, load_chunks, load_commitment_model,
                                        load_specificity_model, load_trained_model, to_long)

PILLARS = ["env", "soc", "gov"]


def _wrap(s, n=110):
    return textwrap.shorten(str(s).replace("\n", " "), width=n, placeholder="…")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="bidv")
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = full; >0 = chi N chunk dau (smoke test)")
    args = ap.parse_args(argv)

    out_dir = Path(f"outputs/demo/{args.bank}_{args.year}")
    out_dir.mkdir(parents=True, exist_ok=True)
    gcfg = load_config("grounding")
    thetas = tuple(gcfg.get("support_thresholds", [0.5, 0.7, 0.9]))

    sub = load_chunks(bank=args.bank, year=args.year)
    if sub.empty:
        raise SystemExit(f"Khong co chunk cho {args.bank} {args.year} trong chunks.parquet")
    if args.limit:
        sub = sub.head(args.limit).copy()
    print(f"### {args.bank} {args.year}: {len(sub)} chunk "
          f"(token/chunk p50={sub['token_count'].median():.0f} max={sub['token_count'].max()})")

    # P2 classify: topic (ta) + commitment (cong su, HF) + specificity (small-LLM rubric)
    topic = load_trained_model("topic")
    commitment = load_commitment_model(load_config("commitment"))
    specificity = load_specificity_model(load_config("specificity"))
    clf = classify_chunks(sub, topic, commitment, specificity)
    clf.to_parquet(out_dir / "classified.parquet", index=False)

    long = to_long(clf)
    n_esg = clf[[f"is_{p}" for p in PILLARS]].sum(axis=1).gt(0).sum()
    n_commit = int(clf["is_commitment"].sum())
    cmm = clf[clf["is_commitment"] == 1]
    lv = cmm["spec_level"].value_counts().to_dict()
    print(f"ESG={int(n_esg)} ({n_esg/len(clf)*100:.1f}%) | commitment={n_commit} "
          f"| spec_level: mơ_hồ(0)={lv.get(0,0)} cụ_thể(1)={lv.get(1,0)} định_lượng(2)={lv.get(2,0)} "
          f"| specific(≥1)={int(clf['is_specific'].sum())}")

    # P3 ground — pool bang chung = TOAN BO bao cao (du --limit chi classify subset)
    retr = EvidenceRetriever(gcfg)
    nli = NLIScorer(gcfg)
    evidence_pool = load_chunks(bank=args.bank, year=args.year) if args.limit else clf
    grounded = ground_claims(clf, retr, nli, gcfg, evidence_df=evidence_pool)
    grounded.to_parquet(out_dir / "claims_grounded.parquet", index=False)

    # P4 index
    long_s = attach_support(long, grounded)
    cti = build_cti_table(long_s, thetas=thetas, n_resamples=1000)
    shares = pillar_shares(long)
    cti = cti.merge(shares[["bank", "year", "pillar", "share"]], on=["bank", "year", "pillar"], how="left")
    cti.to_parquet(out_dir / "cti.parquet", index=False)

    print("\n### CTI / grounded-CTI per tru")
    print(cti[["pillar", "n_commit", "cti", "cti_lo", "cti_hi",
               "gcti@0.7", "gcti@0.7_lo", "gcti@0.7_hi", "share"]].to_string(index=False))

    # ===== QUERY thu =====
    cm = clf[clf["is_commitment"] == 1].merge(
        grounded[["chunk_index", "support", "n_evidence", "top_evidence_ids"]],
        on="chunk_index", how="left")
    id2txt = dict(zip(clf["chunk_index"], clf["content_text"]))

    print("\n### [Q1] 5 cam ket KHONG cu the (cheap talk thuan):")
    for _, r in cm[cm["is_specific"] == 0].head(5).iterrows():
        print(" -", _wrap(r["content_text"]))

    print("\n### [Q2] 5 cam ket CU THE co bang chung manh (support cao):")
    top = cm[(cm["is_specific"] == 1)].sort_values("support", ascending=False).head(5)
    for _, r in top.iterrows():
        print(f" - [sup={r['support']:.2f}] CLAIM: {_wrap(r['content_text'], 90)}")
        for eid in (r["top_evidence_ids"] or [])[:1]:
            print(f"     └ EVIDENCE: {_wrap(id2txt.get(eid, '?'), 90)}")

    print("\n### [Q3] 5 cam ket CU THE nhung KHONG co bang chung do (cheap talk an, support<0.5):")
    for _, r in cm[(cm["is_specific"] == 1) & (cm["support"] < 0.5)].head(5).iterrows():
        print(f" - [sup={r['support']:.2f}] {_wrap(r['content_text'], 100)}")

    # ===== KIEM MAT THONG TIN =====
    no_pool = int((cm["n_evidence"].fillna(0) == 0).sum())
    n_spec_fail = int((clf["spec_parse_ok"] == False).sum())  # noqa: E712 - so chunk LLM parse loi
    info = {
        "bank": args.bank, "year": args.year, "n_chunks": int(len(clf)),
        "n_non_esg_dropped": int(len(clf) - n_esg),
        "non_esg_share": round(float(1 - n_esg / len(clf)), 3),
        "n_commitment": n_commit,
        "commit_no_evidence_pool": no_pool,
        "commit_no_evidence_share": round(no_pool / max(n_commit, 1), 3),
        "spec_level_counts": {int(k): int(v) for k, v in
                              cm["spec_level"].value_counts().sort_index().items()},
        "tok_p95": int(clf["token_count"].quantile(0.95)), "tok_max": int(clf["token_count"].max()),
        "chunk_gt_256tok_truncation_risk": int((clf["token_count"] > 256).sum()),
        "spec_parse_fail": n_spec_fail,
        "any_nan_pillar": bool(clf[[f"p_{p}" for p in PILLARS]].isna().any().any()),
    }
    print("\n### KIEM MAT THONG TIN / SAI LECH")
    print(json.dumps(info, indent=2, ensure_ascii=False))
    (out_dir / "info_check.json").write_text(json.dumps(info, indent=2, ensure_ascii=False),
                                             encoding="utf-8")

    # ===== VISUALIZE =====
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(len(cti))
    ax[0].bar(x - 0.2, cti["cti"], 0.4, label="CTI", color="tab:orange")
    ax[0].bar(x + 0.2, cti["gcti@0.7"], 0.4, label="grounded-CTI@0.7", color="tab:red")
    ax[0].errorbar(x - 0.2, cti["cti"], yerr=[cti["cti"] - cti["cti_lo"], cti["cti_hi"] - cti["cti"]],
                   fmt="none", ecolor="k", capsize=3)
    ax[0].set_xticks(x); ax[0].set_xticklabels(cti["pillar"]); ax[0].set_ylim(0, 1)
    ax[0].set_title(f"{args.bank} {args.year} — CTI vs grounded-CTI"); ax[0].legend()
    sup = grounded["support"].dropna()
    ax[1].hist(sup, bins=20, color="tab:blue", alpha=0.8)
    for th in thetas:
        ax[1].axvline(th, ls="--", lw=1, label=f"θ={th}")
    ax[1].set_title("Phan bo evidence-support (commitment)"); ax[1].legend()
    fig.tight_layout(); fig.savefig(out_dir / "summary.png", dpi=120, bbox_inches="tight")
    print(f"\n-> {out_dir}/ (classified, grounded, cti, summary.png, info_check.json)")


if __name__ == "__main__":
    main()
