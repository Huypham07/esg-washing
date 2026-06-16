# ESG-Washing Pipeline Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tái thiết kế pipeline đo ESG-washing: bỏ grounding/NLI/gCTI (vòng tròn), đổi index sang CTI/NAR/QDR từ thang specificity 3 mức, tách đơn vị phân tích theo ranh giới ý (≤256 token).

**Architecture:** Pipeline **end-to-end** một lệnh: semantic chunk → Topic E/S/G → Commitment (gate denominator) → Specificity (LLM-rubric 3 mức) → CTI/NAR/QDR + selective disclosure + bootstrap CI → validation. Bước nặng (specificity LLM/GPU) nằm trong luồng; user chạy cả repo trên Kaggle rồi mang `outputs/` về. Mọi chỉ số chỉ là tỉ lệ output classifier.

**Tech Stack:** Python, pandas, PhoBERT (transformers), `vietnamese-bi-encoder` (sentence-transformers) cho tách ngữ nghĩa, Qwen3 (specificity), pytest.

**Spec:** `docs/superpowers/specs/2026-06-16-esg-washing-pipeline-redesign-design.md`

**Ràng buộc code hygiene (toàn plan):** comment giải thích *vì sao*, cấm comment nhật ký sửa đổi; không đẻ file script/trung gian thừa; sửa tại chỗ theo pattern sẵn có; xoá code chết (grounding → `legacy/`); YAGNI. Tiếng Việt không dấu trong comment/docstring giữ đúng phong cách file hiện hữu.

**Lưu ý vận hành:** Pipeline end-to-end; **không** tách pha truyền file. "Chạy trên Kaggle" = user ném cả repo lên Kaggle, chạy `python -m esgwash.run --all` (gồm specificity GPU) ra `outputs/`, mang về kiểm tra. Xem memory `feedback-kaggle-end-to-end`.

---

## File Structure

| File | Trách nhiệm | Thao tác |
|---|---|---|
| `src/esgwash/indices/cti.py` | CTI/NAR/QDR từ `spec_level` (bỏ gCTI) | Rewrite |
| `tests/test_indices.py` | unit test index (đang là stub) | Rewrite |
| `src/esgwash/corpus/semantic_split.py` | hàm thuần: tách câu theo ranh giới ý + gói ≤token cap | Create |
| `tests/test_semantic_split.py` | unit test tách ý + token cap (fake embedder) | Create |
| `src/esgwash/corpus/sentence_embedder.py` | wrapper bi-encoder embed câu (word-seg + encode) | Create |
| `src/esgwash/corpus/build_chunks.py` | thay packer 256 bằng semantic_split | Modify |
| `configs/chunk.yml` | thêm `semantic.threshold` | Modify |
| `legacy/grounding/` + `legacy/README.md` | archive grounding (nli/support/evidence_pool/retriever) | Move |
| `legacy/grounding.yml` | archive config | Move |
| `tests/test_evidence_pool.py` | đi theo grounding | Move → `legacy/tests/` |
| `src/esgwash/run.py` | orchestrator **end-to-end** (classify→specificity→index); bỏ grounding; gate denominator | Modify |
| `tests/test_run_pipeline.py` | unit test gate (to_long) | Create |
| `src/esgwash/validation/runner.py` | gọi known-group/synthetic/sensitivity/digit-shortcut + audit harness | Create |
| `configs/index.yml` | bỏ `theta_main`/grounding | Modify |

---

## Task 1: Index CTI/NAR/QDR từ thang 3 mức

**Files:**
- Modify: `src/esgwash/indices/cti.py`
- Test: `tests/test_indices.py`

Thay `is_specific`-based CTI + gCTI bằng 3 share từ `spec_level` (0→CTI, 1→NAR, 2→QDR). Denominator = `is_commitment==1` trong mỗi ô (bank,year,pillar). Mỗi share kèm bootstrap CI (tái dùng `bootstrap_ci`).

- [ ] **Step 1: Viết test thất bại**

```python
# tests/test_indices.py
"""Unit test CTI/NAR/QDR tu spec_level tren du lieu nho tu tao."""
import pandas as pd

from esgwash.indices.cti import compute_specificity_shares, build_index_table


def _toy():
    # 1 o (b,y,env): 4 commitment, spec_level = [0,0,1,2] -> CTI .5 NAR .25 QDR .25
    return pd.DataFrame({
        "bank": ["b"] * 4, "year": [2023] * 4, "pillar": ["env"] * 4,
        "is_commitment": [1, 1, 1, 1], "spec_level": [0, 0, 1, 2],
    })


def test_shares_sum_to_one_and_match_counts():
    out = compute_specificity_shares(_toy(), n_resamples=200)
    row = out.iloc[0]
    assert row["n_commit"] == 4
    assert row["cti"] == 0.5 and row["nar"] == 0.25 and row["qdr"] == 0.25
    assert abs(row["cti"] + row["nar"] + row["qdr"] - 1.0) < 1e-9


def test_ci_brackets_point():
    out = compute_specificity_shares(_toy(), n_resamples=200)
    row = out.iloc[0]
    assert row["cti_lo"] <= row["cti"] <= row["cti_hi"]


def test_build_index_table_has_all_three():
    out = build_index_table(_toy(), n_resamples=200)
    for c in ["cti", "nar", "qdr", "cti_lo", "qdr_hi", "n_commit"]:
        assert c in out.columns
```

- [ ] **Step 2: Chạy test để xác nhận FAIL**

Run: `python -m pytest tests/test_indices.py -v`
Expected: FAIL (`ImportError: cannot import name 'compute_specificity_shares'`).

- [ ] **Step 3: Viết implementation tối thiểu**

```python
# src/esgwash/indices/cti.py  (REWRITE toan bo file)
"""CTI / NAR / QDR tu thang specificity 3 muc (spec 2026-06-16).

Tren moi o (bank, year, pillar), denominator = cam ket co gan tru ESG do:
  CTI = ti le spec_level 0 (mo ho / cheap talk)   -> truc washing
  NAR = ti le spec_level 1 (hanh dong co ten)      -> vung xam
  QDR = ti le spec_level 2 (dinh luong, quy ve chu the) -> substance
CTI+NAR+QDR = 1. Chi la ti le output classifier, khong tu dat trong so.
Bo hoan toan grounded-CTI (xem legacy/README.md).

Input `claims_long`: moi dong = (cam ket x tru no thuoc ve), cot
bank, year, pillar, is_commitment, spec_level.
"""
from __future__ import annotations

import pandas as pd

from esgwash.indices.bootstrap import bootstrap_ci

CELL = ["bank", "year", "pillar"]
_LEVEL_COL = {"cti": 0, "nar": 1, "qdr": 2}


def _share_of_level(level: int):
    def stat(v):
        return float((v == level).mean())
    return stat


def compute_specificity_shares(claims_long: pd.DataFrame, n_resamples: int = 1000,
                               ci: float = 0.95, seed: int = 42) -> pd.DataFrame:
    """Bang share 3 muc + bootstrap CI cho moi o (bank, year, pillar)."""
    commit = claims_long[claims_long["is_commitment"] == 1]
    rows = []
    for (b, y, p), g in commit.groupby(CELL):
        lv = g["spec_level"].to_numpy(dtype=float)
        rec = {"bank": b, "year": y, "pillar": p, "n_commit": len(g)}
        for name, level in _LEVEL_COL.items():
            point, lo, hi = bootstrap_ci(lv, _share_of_level(level), n_resamples, ci, seed)
            rec[name] = round(point, 4)
            rec[f"{name}_lo"] = round(lo, 4)
            rec[f"{name}_hi"] = round(hi, 4)
        rows.append(rec)
    return pd.DataFrame(rows)


def build_index_table(claims_long: pd.DataFrame, n_resamples: int = 1000,
                      ci: float = 0.95, seed: int = 42) -> pd.DataFrame:
    """Alias on dinh ten cho orchestrator; co the merge them cot mo ta sau."""
    return compute_specificity_shares(claims_long, n_resamples, ci, seed)
```

- [ ] **Step 4: Chạy test để xác nhận PASS**

Run: `python -m pytest tests/test_indices.py -v`
Expected: PASS (3 test).

- [ ] **Step 5: Commit**

```bash
git add src/esgwash/indices/cti.py tests/test_indices.py
git commit -m "feat(index): CTI/NAR/QDR tu thang specificity 3 muc, bo gCTI"
```

---

## Task 2: Hàm thuần tách ngữ nghĩa + token cap

**Files:**
- Create: `src/esgwash/corpus/semantic_split.py`
- Test: `tests/test_semantic_split.py`

Tách danh sách câu (đã theo thứ tự, trong 1 block) thành các đoạn cùng-ý dựa trên độ tương đồng embedding câu kề nhau; sau đó gói mỗi đoạn xuống ≤ token cap, không cắt ngang câu. Hàm thuần (nhận embeddings + token_counts có sẵn) để test không cần model.

- [ ] **Step 1: Viết test thất bại**

```python
# tests/test_semantic_split.py
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
```

- [ ] **Step 2: Chạy test để xác nhận FAIL**

Run: `python -m pytest tests/test_semantic_split.py -v`
Expected: FAIL (`ModuleNotFoundError: esgwash.corpus.semantic_split`).

- [ ] **Step 3: Viết implementation tối thiểu**

```python
# src/esgwash/corpus/semantic_split.py
"""Tach cau theo ranh gioi y (topic shift) roi goi xuong tran token.

Tieu chi CHINH = ranh gioi ngu nghia: cat giua 2 cau ke nhau khi cosine embedding
tut duoi nguong -> mot block da-y thanh nhieu don vi. Tran token chi la rao an toan
de bao ve encoder (PhoBERT 256): mot doan cung-y qua dai moi bi cat them tai ranh
gioi cau. Khong bao gio cat ngang 1 cau.

Ham thuan (nhan embeddings + token_counts dung san) -> test khong can load model.
"""
from __future__ import annotations

import numpy as np


def adjacent_cosine(embeddings: np.ndarray) -> np.ndarray:
    """Cosine giua cac cap cau ke nhau -> mang do dai n-1 (n = so cau)."""
    if len(embeddings) < 2:
        return np.empty(0)
    a = embeddings[:-1]
    b = embeddings[1:]
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    denom = np.where((na * nb) == 0, 1.0, na * nb)
    return np.sum(a * b, axis=1) / denom


def segment_indices(sims: np.ndarray, threshold: float) -> list[list[int]]:
    """Gom index cau lien tiep thanh doan; cat sau cau i khi sims[i] < threshold."""
    n = len(sims) + 1
    segs, cur = [], [0]
    for i, s in enumerate(sims):
        if s < threshold:
            segs.append(cur)
            cur = [i + 1]
        else:
            cur.append(i + 1)
    segs.append(cur)
    return segs


def pack_to_token_cap(sentences: list[str], token_counts: list[int],
                      max_tokens: int) -> list[list[str]]:
    """Goi cac cau (cung 1 doan y) thanh don vi <= max_tokens, khong cat ngang cau.
    Mot cau don > max_tokens duoc giu rieng (encoder se truncate cau do, chap nhan)."""
    units, cur, cur_tok = [], [], 0
    for s, t in zip(sentences, token_counts):
        if cur and cur_tok + t > max_tokens:
            units.append(cur)
            cur, cur_tok = [], 0
        cur.append(s)
        cur_tok += t
    if cur:
        units.append(cur)
    return units


def semantic_units(sentences: list[str], embeddings: np.ndarray,
                   token_counts: list[int], threshold: float,
                   max_tokens: int) -> list[list[str]]:
    """Tach y -> goi token cap. Tra ve list cac don vi (moi don vi = list cau)."""
    if not sentences:
        return []
    sims = adjacent_cosine(np.asarray(embeddings, dtype=float))
    out = []
    for seg in segment_indices(sims, threshold):
        seg_sents = [sentences[i] for i in seg]
        seg_toks = [token_counts[i] for i in seg]
        out.extend(pack_to_token_cap(seg_sents, seg_toks, max_tokens))
    return out
```

- [ ] **Step 4: Chạy test để xác nhận PASS**

Run: `python -m pytest tests/test_semantic_split.py -v`
Expected: PASS (5 test).

- [ ] **Step 5: Commit**

```bash
git add src/esgwash/corpus/semantic_split.py tests/test_semantic_split.py
git commit -m "feat(corpus): ham thuan tach ngu nghia + goi token cap"
```

---

## Task 3: Embedder câu + tích hợp semantic split vào build_chunks

**Files:**
- Create: `src/esgwash/corpus/sentence_embedder.py`
- Modify: `src/esgwash/corpus/build_chunks.py` (hàm `build_chunks`, `make_splitter`)
- Modify: `configs/chunk.yml`

Thay đoạn `text = "\n".join(...); chunks = splitter.chunks(text)` bằng: embed câu (bi-encoder) → `semantic_units` → mỗi unit nối câu bằng khoảng trắng. Token cap vẫn dùng tokenizer Qwen sẵn có để đếm. Câu đơn > max_tokens: cắt tiếp bằng `semantic_text_splitter` cũ (giữ làm fallback cho 1 câu siêu dài).

- [ ] **Step 1: Tạo embedder**

```python
# src/esgwash/corpus/sentence_embedder.py
"""Embed cau tieng Viet bang bi-encoder (cho tach ngu nghia khi build chunk).

bkai vietnamese-bi-encoder dua tren PhoBERT nen can word-seg dau vao
(dung chung segmentation.word_segment_batch). Tach rieng de build_chunks khong
phu thuoc package grounding (da chuyen legacy)."""
from __future__ import annotations

import numpy as np

from esgwash.nlp.segmentation import word_segment_batch

DEFAULT_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"


class SentenceEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        seg = word_segment_batch([str(t) for t in texts])
        return self.model.encode(seg, normalize_embeddings=True, show_progress_bar=False)
```

- [ ] **Step 2: Thêm config semantic vào `configs/chunk.yml`**

Thêm khối (giữ nguyên phần còn lại):

```yaml
chunk:
  max_tokens: 256
  tokenizer: Qwen/Qwen3-0.6B
  semantic:
    embedder: bkai-foundation-models/vietnamese-bi-encoder
    threshold: 0.5    # cosine cau ke nhau < threshold -> cat ranh gioi y
```

Cập nhật `_REQUIRED_CHUNK_KEYS` trong `build_chunks.py` thành `("max_tokens", "tokenizer", "semantic")`.

- [ ] **Step 3: Sửa `build_chunks` dùng semantic_units**

Trong `src/esgwash/corpus/build_chunks.py`, thay thân vòng lặp đóng gói chunk (đoạn `text = "\n".join(...)` → hết vòng `for i, ch in enumerate(chunks)`) bằng:

```python
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
```

Thêm import đầu file: `from esgwash.corpus.semantic_split import semantic_units` và khởi tạo embedder trong `build_chunks` (sau `splitter, tok = make_splitter(cfg)`):

```python
    from esgwash.corpus.sentence_embedder import SentenceEmbedder
    embedder = SentenceEmbedder(cfg["chunk"]["semantic"]["embedder"])
```

`make_splitter` giữ nguyên (vẫn cần `tok` để đếm token; `splitter` không còn dùng để đóng gói nhưng giữ cho fallback câu siêu dài — xem Step 4).

- [ ] **Step 4: Fallback câu đơn > max_tokens**

Trong `pack_to_token_cap` một câu đơn vượt cap được giữ nguyên (encoder truncate). Đó là hành vi chấp nhận được theo spec; KHÔNG thêm logic cắt câu. Cập nhật docstring `build_chunks` cho khớp luồng mới (semantic split thay vì packer). Gỡ `make_splitter`/`splitter` nếu sau rà soát không còn tham chiếu (xoá code chết).

- [ ] **Step 5: Smoke test 2 báo cáo**

Run: `python -m esgwash.corpus.build_chunks --limit 2`
Expected: in `N units` mỗi doc; `_qa_report.md` invariant pass (over=0, glyph=0, empty=0). Kiểm tra phân bố `n_sentences` (kỳ vọng nhiều unit > 1 câu, một số block đa-ý tách ra).

- [ ] **Step 6: Commit**

```bash
git add src/esgwash/corpus/sentence_embedder.py src/esgwash/corpus/build_chunks.py configs/chunk.yml
git commit -m "feat(corpus): chunk theo ranh gioi ngu nghia (bi-encoder) + cap 256 token"
```

---

## Task 4: Archive grounding sang legacy/

**Files:**
- Move: `src/esgwash/grounding/` → `legacy/grounding/`
- Move: `configs/grounding.yml` → `legacy/grounding.yml`
- Move: `tests/test_evidence_pool.py` → `legacy/tests/test_evidence_pool.py`
- Create: `legacy/README.md`

Gỡ grounding khỏi luồng chính nhưng giữ lại có chủ đích (phòng reviewer hỏi). Việc strip tham chiếu trong `run.py` làm ở Task 5.

- [ ] **Step 1: Di chuyển package + config + test**

```bash
mkdir -p legacy/tests
git mv src/esgwash/grounding legacy/grounding
git mv configs/grounding.yml legacy/grounding.yml
git mv tests/test_evidence_pool.py legacy/tests/test_evidence_pool.py
```

- [ ] **Step 2: Viết `legacy/README.md`**

```markdown
# legacy/ — code đã gỡ khỏi luồng chính

## grounding/ (gỡ 2026-06-16)
Grounding nội văn bản (retriever + NLI + evidence pool + grounded-CTI) bị loại vì
là construct **vòng tròn**: evidence cho một claim định lượng lấy từ chính câu của
ngân hàng trong cùng báo cáo → đo "nhất quán nội bộ", không đo *walk* thật; NLI
(XNLI) đo textual entailment chứ không đo tính xác thực. Chi tiết: spec
`docs/superpowers/specs/2026-06-16-esg-washing-pipeline-redesign-design.md` §1.1.

Giữ lại để tái hiện/đối chứng nếu reviewer yêu cầu. KHÔNG import từ luồng chính.
```

- [ ] **Step 3: Xác nhận không còn import grounding ngoài legacy**

Run: `grep -rn "esgwash.grounding" src/ tests/ || echo "clean"`
Expected: chỉ còn các dòng trong `src/esgwash/run.py` (sẽ xử lý ở Task 5). Ghi lại danh sách.

- [ ] **Step 4: Commit**

```bash
git add -A legacy/ && git commit -m "refactor: archive grounding sang legacy/ (construct vong tron)"
```

---

## Task 5: run.py end-to-end — classify → specificity → index (bỏ grounding, gate denominator)

> **Thiết kế end-to-end** (KHÔNG tách pha truyền file): một lệnh chạy trọn classify →
> specificity (LLM) → index. Bước nặng (specificity GPU) vẫn nằm trong luồng; user chạy cả
> repo trên Kaggle rồi mang `outputs/` về. Xem memory `feedback-kaggle-end-to-end`.

**Files:**
- Modify: `src/esgwash/run.py`
- Test: `tests/test_run_pipeline.py`

Nội dung:
- Gỡ mọi import + hàm grounding (`NUMERIC_PATTERN`, `support_score`, `ground_claims`,
  `attach_support`, `split_chunk_sentences`, `_quantified_items`), gỡ import `numpy`/`re`/
  `_digit_runs` không còn dùng.
- `classify_chunks(chunks, topic, commitment, specificity)`: giữ specificity inline (chấm
  `spec_level` 0/1/2 trên chunk commitment); sinh cột `spec_*`.
- `to_long`: gate denominator = chunk có gắn trụ ESG (loại chunk không trụ).
- `run_bank_year`: classify → `build_index_table` (CTI/NAR/QDR) → `pillar_shares` →
  ghi `classified.parquet` + `cti.parquet` + `info_check.json`. Bỏ summary.png (vẽ hình thuộc
  `experiments/analyse.py`).
- `_scope_pairs(bank, year, do_all)`: 1 cặp hoặc toàn bộ `analysis_scope` (corpus.yml).
- `main`: `--bank/--year/--all/--limit`; `load_models` = topic+commitment+specificity.
- `configs/index.yml`: bỏ `theta_main` (xử lý ở Task 9 dọn dẹp nếu chưa).
- Test `tests/test_run_pipeline.py`: gate `to_long` (chunk không trụ bị loại; spec_level đi
  kèm vào bảng long). Thuần, không model.

Done when: `python -c "import esgwash.run"` sạch; `pytest tests/test_run_pipeline.py` pass;
grep không còn tham chiếu grounding/old-index trong run.py.

> Ghi chú lịch sử: bản plan gốc tách Task 5/6/7 thành 3 pha export/import file cho Kaggle.
> User làm rõ muốn **end-to-end** (Kaggle = chạy cả repo lấy kết quả cuối), nên gộp lại
> thành Task 5 này; bỏ `export_for_specificity`/`merge_specificity`/`--phase`.

## Task 8: Validation runner + manual audit harness

**Files:**
- Create: `src/esgwash/validation/runner.py`
- Modify: `tests/test_digit_shortcut.py` (chỉ nếu đổi chữ ký; nếu không, bỏ qua)

Gom các kiểm định sẵn có (`known_group`, `synthetic`, `sensitivity`, `digit_shortcut`) thành một runner đọc `outputs/cti/*/*/classified.parquet` + `configs/validation.yml`, ghi `outputs/metrics/`. Thêm hàm tạo **mẫu audit tay**: lấy ngẫu nhiên n đơn vị/trụ ra CSV để gán nhãn tay rồi tính agreement.

- [ ] **Step 1: Viết test thất bại cho audit sampler (hàm thuần)**

```python
# tests/test_validation_runner.py
import pandas as pd
from esgwash.validation.runner import sample_for_audit, audit_agreement


def test_sample_balanced_per_pillar():
    df = pd.DataFrame({
        "bank": ["b"] * 6, "year": [2023] * 6,
        "pillar": ["env", "env", "soc", "soc", "gov", "gov"],
        "chunk_index": range(6), "content_text": [f"t{i}" for i in range(6)],
        "spec_level": [0, 1, 2, 0, 1, 2],
    })
    s = sample_for_audit(df, n_per_pillar=1, seed=0)
    assert set(s["pillar"]) == {"env", "soc", "gov"} and len(s) == 3
    assert "gold_level" in s.columns      # cot trong de gan tay


def test_agreement_simple():
    audited = pd.DataFrame({"spec_level": [0, 1, 2, 2], "gold_level": [0, 1, 2, 1]})
    acc = audit_agreement(audited)
    assert abs(acc - 0.75) < 1e-9
```

- [ ] **Step 2: Chạy test để xác nhận FAIL**

Run: `python -m pytest tests/test_validation_runner.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Viết `runner.py`**

```python
# src/esgwash/validation/runner.py
"""Gom kiem dinh chi so + lay mau audit tay (spec 2026-06-16 §5).

- sample_for_audit: rut n don vi/tru ra CSV (cot gold_level de gan tay).
- audit_agreement: accuracy giua spec_level (pipeline) va gold_level (nguoi).
- run_all: chay known_group/synthetic/sensitivity/digit_shortcut neu du input.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

AUDIT_COLS = ["bank", "year", "pillar", "chunk_index", "content_text", "spec_level"]


def sample_for_audit(long: pd.DataFrame, n_per_pillar: int = 100,
                     seed: int = 42) -> pd.DataFrame:
    """Mau can bang theo tru de gan tay Muc 0/1/2 (cot gold_level de trong)."""
    parts = [g.sample(min(n_per_pillar, len(g)), random_state=seed)
             for _, g in long.groupby("pillar")]
    out = pd.concat(parts, ignore_index=True)
    out = out[[c for c in AUDIT_COLS if c in out.columns]].copy()
    out["gold_level"] = ""        # nguoi gan tay 0/1/2
    return out.reset_index(drop=True)


def audit_agreement(audited: pd.DataFrame) -> float:
    """Accuracy spec_level == gold_level (chi tren dong da gan)."""
    a = audited.dropna(subset=["gold_level"])
    a = a[a["gold_level"].astype(str).str.len() > 0]
    if a.empty:
        return float("nan")
    return float((a["spec_level"].astype(int) == a["gold_level"].astype(int)).mean())
```

- [ ] **Step 4: Chạy test để xác nhận PASS**

Run: `python -m pytest tests/test_validation_runner.py -v`
Expected: PASS (2 test).

- [ ] **Step 5: Commit**

```bash
git add src/esgwash/validation/runner.py tests/test_validation_runner.py
git commit -m "feat(validation): audit sampler + agreement + runner gom kiem dinh"
```

---

## Task 9: Smoke end-to-end 1 bank + dọn dẹp + cập nhật README

**Files:**
- Modify: `README.md`
- Verify: toàn bộ test suite

Chạy thử trọn 3 pha trên BIDV với spec giả nhỏ (không cần Kaggle) để xác nhận luồng thông, rồi cập nhật README (đang còn nói EWRI/neuro-symbolic — lỗi thời).

- [ ] **Step 1: Chạy pha classify**

Run: `python -m esgwash.run --phase classify --bank bidv --year 2023 --limit 40`
Expected: có `classified_pre.parquet` + `commit_for_spec.parquet`.

- [ ] **Step 2: Tạo spec_results giả để smoke pha index** (KHÔNG commit file này)

Run:
```bash
python -c "
import pandas as pd, numpy as np
c = pd.read_parquet('outputs/cti/bidv/2023/commit_for_spec.parquet')
rng = np.random.default_rng(0)
c['spec_level'] = rng.integers(0, 3, len(c))
c['is_specific'] = (c['spec_level'] >= 1).astype(int)
c[['doc_id','chunk_index','spec_level','is_specific']].to_parquet('outputs/specificity/_smoke.parquet')
print('smoke spec rows', len(c))
"
```
(Tạo thư mục `outputs/specificity/` nếu chưa có.)

- [ ] **Step 3: Chạy pha index trên spec giả**

Run: `python -m esgwash.run --phase index --bank bidv --year 2023 --input outputs/specificity/_smoke.parquet`
Expected: in bảng `pillar | n_commit | cti | cti_lo | cti_hi | nar | qdr | share`; tạo `outputs/cti/bidv/2023/cti.parquet`. Xác nhận `cti+nar+qdr≈1` mỗi hàng.

- [ ] **Step 4: Xoá file smoke**

Run: `rm outputs/specificity/_smoke.parquet`

- [ ] **Step 5: Chạy full test suite**

Run: `python -m pytest tests/ -v`
Expected: tất cả PASS (index, semantic_split, run_pipeline, validation_runner, digit_shortcut, specificity_llm, topic_merge, tuning). Nếu test nào tham chiếu grounding/API cũ → sửa cho khớp (hoặc đã move sang `legacy/tests`).

- [ ] **Step 6: Cập nhật `README.md`**

Viết lại phần Tổng quan + cấu trúc + hướng dẫn chạy cho khớp pipeline 3 pha mới (semantic chunk → topic/commitment → specificity-Kaggle → CTI/NAR/QDR). Bỏ mọi nhắc tới EWRI, neuro-symbolic, action levels cũ. Nêu rõ lệnh:
```
python -m esgwash.corpus.build_chunks
python -m esgwash.run --phase classify --all
# (Kaggle) python -m esgwash.run --phase specificity --input commit_for_spec.parquet --output spec_results.parquet
python -m esgwash.run --phase index --all --input outputs/specificity/spec_results.parquet
```

- [ ] **Step 7: Commit**

```bash
git add README.md
git commit -m "docs: README khop pipeline 3 pha (CTI/NAR/QDR, specificity tren Kaggle)"
```

---

## Ghi chú: cập nhật `docs/de-cuong-nghien-cuu.md`

Theo spec §7, cập nhật đề cương (specificity = LLM-rubric 3 mức; CTI = P(Mức 0); bỏ grounding; semantic unit) làm **SAU** khi chạy full 10×5 + đánh giá hiệu quả thật (audit agreement, validation). KHÔNG làm trong plan này để tránh viết docs lệch với kết quả thực. Tạo task riêng khi có số liệu cuối.

---

## Self-Review

**Spec coverage:**
- §2 khung nghiên cứu (RQ/đóng góp) — phản ánh qua index + validation (Task 1, 7, 8); RQ4 (chống shortcut) đã có `test_digit_shortcut.py` + audit (Task 8). ✓
- §3 CTI/NAR/QDR bỏ grounding — Task 1, 7. ✓
- §3.3 selective disclosure — Task 7 (pillar_shares). ✓
- §4.2 semantic chunk theo ranh giới ý + cap 256 — Task 2, 3. ✓
- §4.3 cổng topic denominator — Task 5 (`to_long` + is_commitment gate). ✓
- §4 tách pha specificity cho Kaggle — Task 5 (export), 6 (CLI), 7 (merge/index). ✓
- §5 validation + audit — Task 8. ✓
- §7 archive grounding + README — Task 4, 9; de-cuong hoãn (ghi chú). ✓
- §8 code hygiene — nhấn ở header + Task 3/7 (xoá code chết) + Task 8 (gỡ nhánh thừa). ✓

**Attribution (§4.4):** cải thiện rubric chạy ở pha specificity (Kaggle) — thuộc `specificity_llm.py` đã có `attributable_to_actor` + `verify_rubric`; tinh chỉnh prompt KHÔNG nằm trong plan refactor này (đo bằng audit Task 8). Nếu audit cho thấy attribution kém → task tinh chỉnh prompt riêng sau.

**Placeholder scan:** không có TBD/TODO mở; mọi step có code/command cụ thể. Task 8 Step 3 cố ý chỉ ra đoạn dead để engineer viết gọn (kèm bản đúng).

**Type consistency:** `spec_level` (int 0/1/2), `is_commitment` (0/1), khoá merge `doc_id`+`chunk_index` nhất quán xuyên Task 1/5/6/7. `build_index_table` (Task 1) = tên gọi ở Task 7. `export_for_specificity`/`merge_specificity` chữ ký khớp Task 5↔6↔7. Cột spec output (`p_specific/spec_rubric/spec_raw/spec_parse_ok`) đặt ở Task 6 khớp `merge_specificity` Task 5. ✓
