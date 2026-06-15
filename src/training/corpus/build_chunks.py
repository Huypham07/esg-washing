"""Orchestrator re-chunk 50 báo cáo -> chunks cho specificity LLM (CTI / grounded-CTI chunk-level).

Luồng: TEXT OCR SẠCH trong zip (tác giả làm sẵn) -> clean_extracted_text
-> semantic-text-splitter (tokenizer Qwen, trần max_tokens, ngắt ở ranh giới câu Unicode) -> chunks.parquet.

⚠️ KHÔNG re-OCR docling: docling hiện tại giải mã hỏng font subset tiếng Việt (in /uniXXXX
   thay diacritic) -> hỏng 55% câu. Text zip cũ SẠCH + đầy đủ hơn.
⚠️ KHÔNG sinh sentences/mapping: CTI & grounded-CTI chấm THUẦN mức chunk nên không cần map
   câu->chunk. Corpus câu cho classifier (E/S/G/commitment) đã có riêng ở data/corpus/sentences_clean.parquet.

Chạy:
  python -m src.training.corpus.build_chunks                 # build full 50 + QA
  python -m src.training.corpus.build_chunks --qa            # QA lại từ parquet đã có
  python -m src.training.corpus.build_chunks --limit 3       # smoke 3 báo cáo đầu
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

if hasattr(sys.stdout, "reconfigure"):  # guard: stdout Jupyter có thể không có reconfigure
    sys.stdout.reconfigure(encoding="utf-8")  # tránh crash print tiếng Việt trên cp1252 (Windows)

# Cho phép `from src...` chạy mọi nơi (corpus -> training -> src -> repo root = parents[3])
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# Key bắt buộc trong config/chunk.yml (validate sớm để lỗi rõ ràng thay vì KeyError sâu)
_REQUIRED_KEYS = ("zip_path", "output_dir", "scope", "chunk")
_REQUIRED_CHUNK_KEYS = ("max_tokens", "tokenizer")
_REQUIRED_SCOPE_KEYS = ("banks", "years")

# Bắt (bank, year) từ path trong zip: .../<bank>/<...><year>...
_DOC_RE = re.compile(r"/([a-z]+)/[^/]*?((?:19|20)\d{2})")


def load_chunk_config(path: str = "config/chunk.yml") -> dict:
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


def make_splitter(cfg: dict):
    """Dựng semantic-text-splitter từ tokenizer Qwen + hàm đếm token (cho cột token_count).

    Trần = max_tokens token; splitter ngắt ưu tiên ở ranh giới CÂU (Unicode), chỉ cắt khi 1 câu
    đơn > max_tokens. Lib Rust cần `tokenizers.Tokenizer` (= tok.backend_tokenizer); AutoTokenizer
    wrapper thiếu `.to_str` nên KHÔNG truyền trực tiếp được.
    """
    from semantic_text_splitter import TextSplitter
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["chunk"]["tokenizer"])
    splitter = TextSplitter.from_huggingface_tokenizer(
        tok.backend_tokenizer, int(cfg["chunk"]["max_tokens"])
    )
    return splitter, tok


# Tiền tố bullet để strip khi explode (đồng bộ prepare_clean_corpus._BULLET_PREFIX)
_BULLET_PREFIX = r"^\s*[-–—•»*+▪◦·]+\s*"


def _filter_noise_sentences(df: pd.DataFrame) -> pd.DataFrame:
    """Explode '\\n' + strip bullet + is_noise refilter — bỏ BẢNG/boilerplate/mảnh vụn, GIỮ thứ tự.

    Báo cáo ngân hàng ~57% là bảng số -> bước này (is_noise_sentence của esg-washing) loại chúng
    để chunk chỉ chứa VĂN XUÔI có ích cho specificity LLM (nếu bỏ: +2.3x token, 22% chunk = rác bảng).
    """
    from src.training.corpus.build_corpus import is_noise_sentence

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
    """zip -> chunks_df. LỌC NHIỄU (câu) trước, rồi semantic-text-splitter GÓI thành chunk.

    clean_extracted_text -> build_single_document (tách câu) -> _filter_noise_sentences (bỏ bảng/boilerplate)
    -> nối câu đã lọc bằng '\\n' -> semantic-text-splitter (gói ≤max_tokens, ưu tiên ngắt giữa các câu).
    Cols: chunk_id, doc_id, bank, year, chunk_index, content_text, char_count, token_count.
    """
    from src.pipeline.document_loader import clean_extracted_text
    from src.training.corpus.build_corpus import build_single_document

    splitter, tok = make_splitter(cfg)
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
        # Nối câu ĐÃ LỌC bằng '\n' -> splitter coi mỗi câu là 1 dòng, ưu tiên ngắt giữa các câu
        text = "\n".join(df["sentence"].astype(str))
        chunks = splitter.chunks(text)
        for i, ch in enumerate(chunks):
            rows.append({
                "chunk_id": f"{doc_id}__c{i:04d}", "doc_id": doc_id, "bank": bank, "year": year,
                "chunk_index": i, "content_text": ch, "char_count": len(ch),
                "token_count": len(tok.encode(ch, add_special_tokens=False)),
            })
        print(f"  {doc_id}: {len(chunks):,} chunks")

    return pd.DataFrame(rows)


def write_outputs(chunks_df: pd.DataFrame, cfg: dict) -> None:
    """Ghi chunks.parquet + reports/_cleaning_summary.json (số chunk/doc)."""
    out = Path(cfg["output_dir"])
    (out / "reports").mkdir(parents=True, exist_ok=True)

    (chunks_df.groupby("doc_id").agg(n_chunks=("chunk_id", "size")).reset_index()
        .to_json(out / "reports" / "_cleaning_summary.json",
                 orient="records", force_ascii=False, indent=2))

    chunks_df.to_parquet(out / "chunks.parquet", index=False)
    print(f"[build_chunks] saved -> {out/'chunks.parquet'}")


def run_qa(chunks_df: pd.DataFrame, cfg: dict) -> str:
    """QA trần token + glyph + thống kê -> _qa_report.md. Trả markdown."""
    out = Path(cfg["output_dir"])
    max_tokens = int(cfg["chunk"]["max_tokens"])

    over = int((chunks_df["token_count"] > max_tokens).sum())
    glyph = int(chunks_df["content_text"].str.contains(r"/uni[0-9A-Fa-f]{4}|/dslash", regex=True).sum())
    empty = int((chunks_df["content_text"].str.strip().str.len() == 0).sum())

    lines = [
        "# QA report — re-chunk (nguồn: text zip sạch · chunker: semantic-text-splitter)", "",
        f"- doc: **{chunks_df['doc_id'].nunique()}** | chunks: **{len(chunks_df):,}** | max_tokens={max_tokens}",
        f"- chunk > max_tokens (kỳ vọng 0): **{over}**",
        f"- glyph /uni|/dslash (kỳ vọng 0): **{glyph}**",
        f"- chunk rỗng (kỳ vọng 0): **{empty}**",
        f"- token/chunk p50={chunks_df['token_count'].median():.0f} · "
        f"p90={chunks_df['token_count'].quantile(0.9):.0f} · max={chunks_df['token_count'].max()}",
        f"- char/chunk p50={chunks_df['char_count'].median():.0f}",
    ]

    # Invariant cứng: splitter đảm bảo trần token + không sinh chunk rỗng/glyph
    assert over == 0, f"QA FAIL: {over} chunk vượt max_tokens"
    assert glyph == 0, f"QA FAIL: {glyph} chunk dính glyph /uni"
    assert empty == 0, f"QA FAIL: {empty} chunk rỗng"

    txt = "\n".join(lines)
    (out / "_qa_report.md").write_text(txt, encoding="utf-8")
    return txt


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Re-chunk 50 báo cáo (text zip sạch) -> chunks cho specificity LLM (CTI chunk-level)"
    )
    p.add_argument("--config", default="config/chunk.yml", help="Đường dẫn config YAML")
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
