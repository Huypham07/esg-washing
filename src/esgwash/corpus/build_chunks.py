"""Orchestrator re-chunk 50 báo cáo -> chunks (đơn vị semantic cho topic/commitment + specificity).

Luồng: TEXT OCR SẠCH trong zip (tác giả làm sẵn) -> clean_extracted_text -> tách câu + lọc nhiễu
-> embed câu (bi-encoder) -> semantic_units (cắt ranh giới ý + gói ≤max_tokens) -> chunks.parquet.

⚠️ KHÔNG re-OCR docling: docling hiện tại giải mã hỏng font subset tiếng Việt (in /uniXXXX
   thay diacritic) -> hỏng 55% câu. Text zip cũ SẠCH + đầy đủ hơn.

Chạy:
  python -m esgwash.corpus.build_chunks                 # build full 50 + QA
  python -m esgwash.corpus.build_chunks --qa            # QA lại từ parquet đã có
  python -m esgwash.corpus.build_chunks --limit 3       # smoke 3 báo cáo đầu
"""
from __future__ import annotations

import argparse
import io
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd
import yaml

from esgwash.corpus.semantic_split import semantic_units

if hasattr(sys.stdout, "reconfigure"):  # guard: stdout Jupyter có thể không có reconfigure
    sys.stdout.reconfigure(encoding="utf-8")  # tránh crash print tiếng Việt trên cp1252 (Windows)

# Cho phép `from src...` chạy mọi nơi (corpus -> training -> src -> repo root = parents[3])
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# Key bắt buộc trong configs/chunk.yml (validate sớm để lỗi rõ ràng thay vì KeyError sâu)
_REQUIRED_KEYS = ("zip_path", "output_dir", "scope", "chunk")
_REQUIRED_CHUNK_KEYS = ("max_tokens", "tokenizer", "semantic")
_REQUIRED_SCOPE_KEYS = ("banks", "years")

# Bắt (bank, year) từ path trong zip: .../<bank>/<...><year>...
_DOC_RE = re.compile(r"/([a-z]+)/[^/]*?((?:19|20)\d{2})")


def load_chunk_config(path: str = "configs/chunk.yml") -> dict:
    """Đọc + validate config. Raise ValueError liệt kê key thiếu (giống style train.yml)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config không tồn tại: {p}")
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Config rỗng/sai định dạng (không phải mapping YAML): {p}")

    for keys, block, label in [
        (_REQUIRED_KEYS, cfg, "root"),
        (_REQUIRED_CHUNK_KEYS, cfg.get("chunk", {}), "chunk"),
        (_REQUIRED_SCOPE_KEYS, cfg.get("scope", {}), "scope"),
    ]:
        missing = [k for k in keys if k not in (block or {})]
        if missing:
            raise ValueError(f"Config '{label}' thiếu key: {missing}. Cần đủ: {list(keys)}")

    return cfg


def iter_zip_docs(cfg: dict, limit: int = 0):
    """Yield (doc_id, bank, year, raw_text) cho các báo cáo IN-SCOPE trong zip (theo doc_id)."""
    banks = set(cfg["scope"]["banks"])
    years = set(int(y) for y in cfg["scope"]["years"])
    zp = Path(cfg["zip_path"])
    if not zp.exists():
        raise FileNotFoundError(f"Zip không tồn tại: {zp}")

    found: dict[str, tuple[str, int, str]] = {}
    with zipfile.ZipFile(zp) as zf:
        for name in sorted(zf.namelist()):
            if "__MACOSX" in name or not name.endswith(".txt"):
                continue
            m = _DOC_RE.search(name)
            if not m:
                continue
            bank, year = m.group(1), int(m.group(2))
            if bank not in banks or year not in years:
                continue
            raw = io.TextIOWrapper(zf.open(name), encoding="utf-8", errors="replace").read()
            found[f"{bank}_{year}"] = (bank, year, raw)

    doc_ids = sorted(found)
    if limit:
        doc_ids = doc_ids[:limit]
    for doc_id in doc_ids:
        bank, year, raw = found[doc_id]
        yield doc_id, bank, year, raw


def make_tokenizer(cfg: dict):
    """Load tokenizer Qwen de dem token (cho cot token_count va tran max_tokens).

    Dung AutoTokenizer vi bi-encoder co tokenizer rieng; chi can dem so token
    Qwen cho output chunk (khong can splitter Rust nua).
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(cfg["chunk"]["tokenizer"])


# Tiền tố bullet để strip khi explode (đồng bộ prepare_clean_corpus._BULLET_PREFIX)
_BULLET_PREFIX = r"^\s*[-–—•»*+▪◦·]+\s*"


def _filter_noise_sentences(df: pd.DataFrame) -> pd.DataFrame:
    """Explode '\\n' + strip bullet + is_noise refilter — bỏ BẢNG/boilerplate/mảnh vụn, GIỮ thứ tự.

    Báo cáo ngân hàng ~57% là bảng số -> bước này (is_noise_sentence của esg-washing) loại chúng
    để chunk chỉ chứa VĂN XUÔI có ích cho specificity LLM (nếu bỏ: +2.3x token, 22% chunk = rác bảng).
    """
    from esgwash.corpus.build_corpus import is_noise_sentence

    df = df.copy()
    df["sentence"] = df["sentence"].astype(str).str.split("\n")
    df = df.explode("sentence")
    df["sentence"] = df["sentence"].str.replace(_BULLET_PREFIX, "", regex=True).str.strip()
    df = df[df["sentence"].str.len() > 0]
    noise = df.apply(
        lambda r: is_noise_sentence(str(r["sentence"]), section_title=str(r.get("section_title", ""))),
        axis=1,
    )
    return df[~noise].reset_index(drop=True)


def build_chunks(cfg: dict, limit: int = 0) -> pd.DataFrame:
    """zip -> chunks_df. LUong: loc nhieu cau -> embed bi-encoder -> tach theo cosine -> goi tran token.

    clean_extracted_text -> build_single_document (tach cau) -> _filter_noise_sentences (bo bang/boilerplate)
    -> bi-encoder embed -> semantic_units (cat ranh gioi y bang cosine ke nhau < threshold, goi <= max_tokens).
    Cols: chunk_id, doc_id, bank, year, chunk_index, content_text, char_count, token_count, n_sentences.
    """
    from esgwash.corpus.document_loader import clean_extracted_text
    from esgwash.corpus.build_corpus import build_single_document
    from esgwash.corpus.sentence_embedder import SentenceEmbedder

    tok = make_tokenizer(cfg)
    embedder = SentenceEmbedder(cfg["chunk"]["semantic"]["embedder"])
    rows: list[dict] = []

    docs = list(iter_zip_docs(cfg, limit=limit))
    print(f"[build_chunks] in-scope: {len(docs)} báo cáo | max_tokens={cfg['chunk']['max_tokens']}")
    for doc_id, bank, year, raw in docs:
        df = build_single_document(clean_extracted_text(raw), bank=bank, year=year, doc_id=doc_id)
        if df.empty:
            print(f"  {doc_id}: 0 câu (bỏ)")
            continue
        df = _filter_noise_sentences(df)
        if df.empty:
            print(f"  {doc_id}: 0 câu sau lọc (bỏ)")
            continue
        sents = df["sentence"].astype(str).tolist()
        emb = embedder.embed(sents)
        toks = [len(tok.encode(s, add_special_tokens=False)) for s in sents]
        max_tokens = int(cfg["chunk"]["max_tokens"])
        thr = float(cfg["chunk"]["semantic"]["threshold"])
        units = semantic_units(sents, emb, toks, threshold=thr, max_tokens=max_tokens)
        for i, unit_sents in enumerate(units):
            ch = " ".join(unit_sents)
            rows.append({
                "chunk_id": f"{doc_id}__c{i:04d}", "doc_id": doc_id, "bank": bank,
                "year": year, "chunk_index": i, "content_text": ch,
                "char_count": len(ch),
                "token_count": len(tok.encode(ch, add_special_tokens=False)),
                "n_sentences": len(unit_sents),
            })
        print(f"  {doc_id}: {len(units):,} units")

    return pd.DataFrame(rows)


def write_outputs(chunks_df: pd.DataFrame, cfg: dict) -> None:
    """Ghi chunks.parquet (-> data/) + reports (-> report_dir, mac dinh experiments/chunking)."""
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    rep = Path(cfg.get("report_dir", out))
    (rep / "reports").mkdir(parents=True, exist_ok=True)

    (chunks_df.groupby("doc_id").agg(n_chunks=("chunk_id", "size")).reset_index()
        .to_json(rep / "reports" / "_cleaning_summary.json",
                 orient="records", force_ascii=False, indent=2))

    chunks_df.to_parquet(out / "chunks.parquet", index=False)
    print(f"[build_chunks] saved -> {out/'chunks.parquet'}")


def run_qa(chunks_df: pd.DataFrame, cfg: dict) -> str:
    """QA trần token + glyph + thống kê -> _qa_report.md. Trả markdown."""
    rep = Path(cfg.get("report_dir", cfg["output_dir"]))
    rep.mkdir(parents=True, exist_ok=True)
    max_tokens = int(cfg["chunk"]["max_tokens"])

    over = int((chunks_df["token_count"] > max_tokens).sum())
    glyph = int(chunks_df["content_text"].str.contains(r"/uni[0-9A-Fa-f]{4}|/dslash", regex=True).sum())
    empty = int((chunks_df["content_text"].str.strip().str.len() == 0).sum())

    lines = [
        "# QA report — re-chunk (nguồn: text zip sạch · chunker: bi-encoder semantic split)", "",
        f"- doc: **{chunks_df['doc_id'].nunique()}** | chunks: **{len(chunks_df):,}** | max_tokens={max_tokens}",
        f"- chunk > max_tokens (cau don qua dai, encoder truncate): **{over}**",
        f"- glyph /uni|/dslash (kỳ vọng 0): **{glyph}**",
        f"- chunk rỗng (kỳ vọng 0): **{empty}**",
        f"- token/chunk p50={chunks_df['token_count'].median():.0f} · "
        f"p90={chunks_df['token_count'].quantile(0.9):.0f} · max={chunks_df['token_count'].max()}",
        f"- char/chunk p50={chunks_df['char_count'].median():.0f}",
    ]

    # Invariant packing: chunk vuot cap chi duoc la cau don (n_sentences==1); glyph + rong van phai 0
    over_multi = int(((chunks_df["token_count"] > max_tokens) & (chunks_df["n_sentences"] > 1)).sum()) if "n_sentences" in chunks_df.columns else 0
    assert over_multi == 0, f"QA FAIL: {over_multi} chunk da-cau vuot max_tokens (loi packing)"
    assert glyph == 0, f"QA FAIL: {glyph} chunk dính glyph /uni"
    assert empty == 0, f"QA FAIL: {empty} chunk rỗng"

    txt = "\n".join(lines)
    (rep / "_qa_report.md").write_text(txt, encoding="utf-8")
    return txt


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Re-chunk 50 báo cáo (text zip sạch) -> chunks cho specificity LLM (CTI chunk-level)"
    )
    p.add_argument("--config", default="configs/chunk.yml", help="Đường dẫn config YAML")
    p.add_argument("--qa", action="store_true", help="Chỉ QA lại từ chunks.parquet đã có")
    p.add_argument("--limit", type=int, default=0, help="0 = full 50; >0 = smoke N báo cáo đầu")
    return p.parse_args(args)


def main(args: list[str] | None = None) -> None:
    a = parse_args(args)
    cfg = load_chunk_config(a.config)
    out = Path(cfg["output_dir"])

    if a.qa:  # chỉ đọc parquet đã có rồi QA lại (không rebuild)
        print(run_qa(pd.read_parquet(out / "chunks.parquet"), cfg))
        return

    chunks_df = build_chunks(cfg, limit=a.limit)
    write_outputs(chunks_df, cfg)
    print("=" * 60)
    print(run_qa(chunks_df, cfg))


if __name__ == "__main__":
    main()