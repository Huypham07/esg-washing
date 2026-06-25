"""Pipeline ESG-washing end-to-end theo tung (bank, year): classify -> specificity -> index.

  classify    : topic E/S/G + commitment tren tung chunk semantic.
  specificity : LLM-rubric cham spec_level (0/1/2) tren chunk commitment.
  index       : CTI/NAR/QDR per (bank, year, tru) + selective disclosure.

Chay (toan bo nang, ke ca specificity LLM, nam trong 1 lenh - chay tren Kaggle GPU):
  python -m esgwash.run --bank bidv --year 2023
  python -m esgwash.run --all
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from esgwash.models.commitment_model import CommitmentModel
from esgwash.models.topic_model import TopicModel

PILLARS = ("env", "soc", "gov")
HF_REPOS = {"topic": "huypham71/esg-topic", "commitment": "huypham71/esg-commitment"}
ATOMIC_FLAGS = ("co_cam_ket", "co_hanh_dong_ten", "co_so_dinh_luong", "quy_ve_bank", "co_moc_tg")

CHUNKS_PATH = "data/chunks.parquet"
CTI_ROOT = Path("outputs/cti")


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


def load_models() -> dict:
    """Nap topic + commitment + specificity dung chung cho moi (bank, year).
    Grounding da bo (xem legacy/README.md)."""
    from esgwash.config import load_config
    return {"topic": load_trained_model("topic"),
            "commitment": load_commitment_model(load_config("commitment")),
            "specificity": load_specificity_model(load_config("specificity"))}


def classify_chunks(chunks: pd.DataFrame, topic_model, commitment_model,
                    specificity_model, spec_on_commitment: bool = True) -> pd.DataFrame:
    """-> chunks + topic (p_/is_ theo tru, pillar_top) + commitment + specificity (spec_level).

    Specificity dung LLM nen chi cham tren chunk commitment (mau so CTI) khi
    spec_on_commitment=True: vua tiet kiem vua dung ngu nghia. Chunk khac de spec_level=0.
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
    out["spec_level"] = 0       # 0=mo ho, 1=cu the (hanh dong co ten), 2=dinh luong
    out["is_specific"] = 0      # = (spec_level >= 1)
    out["spec_parse_ok"] = pd.NA
    out["spec_rubric"] = pd.NA
    out["spec_raw"] = pd.NA     # phan hoi LLM goc truoc parse, de truy vet / parse lai offline
    # 5 atomic flags + evidence — initialise to 0 / empty for ALL rows (no NaN dtype surprises)
    for flag in ATOMIC_FLAGS:
        out[flag] = 0
    out["evidence"] = ""

    mask = (out["is_commitment"] == 1) if spec_on_commitment else pd.Series(True, index=out.index)
    if specificity_model is not None and bool(mask.any()):
        sp = specificity_model.predict(out.loc[mask, "content_text"].astype(str).tolist())
        out.loc[mask, "p_specific"] = sp["p_specificity"].to_numpy()
        out.loc[mask, "spec_level"] = sp["spec_level"].to_numpy()
        out.loc[mask, "is_specific"] = sp["is_specific"].to_numpy()
        out.loc[mask, "spec_parse_ok"] = sp["parse_ok"].to_numpy()
        out.loc[mask, "spec_rubric"] = sp["rubric"].to_numpy()
        out.loc[mask, "spec_raw"] = sp["raw"].to_numpy()
        # propagate 5 atomic flags + evidence from LLM output
        for flag in ATOMIC_FLAGS:
            out.loc[mask, flag] = sp[flag].to_numpy()
        out.loc[mask, "evidence"] = sp["evidence"].to_numpy()

    # Final commit gate: co_cam_ket AND (is_env OR is_soc OR is_gov)
    # PhoBERT commitment model is kept as cheap pre-filter deciding which rows get LLM scoring;
    # the FINAL is_commitment written to output reflects atomic flag intent + ESG topic gate.
    is_esg = ((out["is_env"] == 1) | (out["is_soc"] == 1) | (out["is_gov"] == 1)).astype(int)
    out["is_commitment"] = (out["co_cam_ket"].astype(int) & is_esg).astype(int)
    return out


def to_long(classified: pd.DataFrame) -> pd.DataFrame:
    """Mỗi dòng = (chunk ESG × trụ dương của nó). Chunk không thuộc trụ nào thì bị loại
    -> gate denominator CTI = cam ket co gan tru ESG."""
    parts = []
    for p in PILLARS:
        sub = classified[classified[f"is_{p}"] == 1].copy()
        sub["pillar"] = p
        parts.append(sub)
    return pd.concat(parts, ignore_index=True) if parts else classified.iloc[:0].copy()


# --- Chay end-to-end cho tung (bank, year) -> outputs/cti/<bank>/<year>/ ---


def _scope_pairs(banks: list[str], year: int | None, do_all: bool) -> list[tuple[str, int]]:
    """Danh sach (bank, year) can chay.

    --all              : toan bo analysis_scope (corpus.yml)
    --bank A B         : A va B, moi ban toan bo year
    --bank A --year Y  : chi (A, Y)
    """
    from esgwash.config import load_config
    scope = load_config("corpus").get("analysis_scope", {})
    chunks = load_chunks()
    all_years = [int(y) for y in scope.get("years", sorted(chunks["year"].unique()))]

    if do_all:
        all_banks = scope.get("banks", sorted(chunks["bank"].unique()))
        return [(b, y) for b in all_banks for y in all_years]

    if year is not None:
        return [(b, year) for b in banks]
    return [(b, y) for b in banks for y in all_years]


def run_bank_year(bank: str, year: int, models: dict, limit: int = 0) -> Path:
    """Classify -> specificity -> index cho 1 (bank, year); ghi outputs/cti/<bank>/<year>/."""
    from esgwash.config import load_config
    from esgwash.indices.cti import build_index_table
    from esgwash.indices.disclosure import pillar_shares

    boot = load_config("index").get("bootstrap", {})
    out_dir = CTI_ROOT / bank / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)

    full = load_chunks(bank=bank, year=year)
    if full.empty:
        raise SystemExit(f"Khong co chunk cho {bank} {year} trong {CHUNKS_PATH}")
    sub = full.head(limit).copy() if limit else full
    print(f"### {bank} {year}: {len(sub)} chunk")

    clf = classify_chunks(sub, models["topic"], models["commitment"], models["specificity"])
    clf.to_parquet(out_dir / "classified.parquet", index=False)

    idx = build_index_table(clf, n_resamples=int(boot.get("n_resamples", 1000)),
                            ci=float(boot.get("ci", 0.95)))
    ci_cols = [c for c in idx.columns if c.endswith("_lo") or c.endswith("_hi")]
    idx.drop(columns=ci_cols, inplace=True)
    idx.to_parquet(out_dir / "cti.parquet", index=False)

    long = to_long(clf)
    shares = pillar_shares(long)
    shares[["bank", "year", "pillar", "n", "share"]].to_parquet(
        out_dir / "pillar_shares.parquet", index=False)

    from esgwash.indices.cti import INDEX_LEGEND
    (out_dir / "legend.json").write_text(
        json.dumps(INDEX_LEGEND, indent=2, ensure_ascii=False), encoding="utf-8")
    if not idx.empty:
        print(idx[["bank", "year", "n_commit", "cti", "nar", "qdr"]].to_string(index=False))
    else:
        print(f"  [no ESG commitment chunks found for {bank} {year}]")

    _write_info_check(clf, out_dir, bank, year)
    print(f"-> {out_dir}/")
    return out_dir


def _write_info_check(clf: pd.DataFrame, out_dir: Path, bank: str, year: int) -> None:
    """Chi so chan doan 1 doc: ti le non-ESG bi loai, phan bo spec_level, rui ro truncate."""
    n_esg = int(clf[[f"is_{p}" for p in PILLARS]].sum(axis=1).gt(0).sum())
    n_commit = int(clf["is_commitment"].sum())
    commit = clf[clf["is_commitment"] == 1]
    info = {
        "bank": bank, "year": year, "n_chunks": int(len(clf)),
        "non_esg_share": round(float(1 - n_esg / len(clf)), 3),
        "n_commitment": n_commit,
        "spec_level_counts": {int(k): int(v) for k, v in
                              commit["spec_level"].value_counts().sort_index().items()},
        "chunk_gt_256tok": int((clf["token_count"] > 256).sum()) if "token_count" in clf else 0,
        "spec_parse_fail": int((clf["spec_parse_ok"] == False).sum()),  # noqa: E712
    }
    (out_dir / "info_check.json").write_text(json.dumps(info, indent=2, ensure_ascii=False),
                                             encoding="utf-8")


def main(argv=None):
    import argparse
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Pipeline ESG-washing end-to-end (CTI/NAR/QDR)")
    ap.add_argument("--bank", nargs="+", default=["bidv"],
                    help="1 hoac nhieu bank (vd --bank bidv mbbank); ket hop --year de chi dinh nam")
    ap.add_argument("--year", type=int, default=None,
                    help="Nam cu the; bo trong = chay toan bo year cua tung bank")
    ap.add_argument("--all", action="store_true", help="chay toan bo analysis_scope (corpus.yml)")
    ap.add_argument("--limit", type=int, default=0, help="0=full; >0 = N chunk dau (smoke test)")
    args = ap.parse_args(argv)
    models = load_models()  # nap 1 lan, tai dung cho moi (bank, year) -> tranh OOM do reload
    for bank, year in _scope_pairs(args.bank, args.year, args.all):
        if not load_chunks(bank=bank, year=year).empty:
            run_bank_year(bank, year, models, limit=args.limit)
            _free_gpu()  # giai phong phan manh giua cac bank khi chay --all


def _free_gpu() -> None:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
