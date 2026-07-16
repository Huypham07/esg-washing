# Reliability-Aware Attribute Extraction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Thay rubric specificity tổng thể (0/1/2) bằng trích xuất 5 cờ atomic khách quan + evidence, tái dựng CTI/NAR/QDR bằng luật tất định, và viết lại paper để mô tả rõ mô hình + chứng minh độ tin cậy (κ 0.18→0.98).

**Architecture:** LLM (Qwen3) trích xuất 5 cờ nhị phân cấp chunk `{co_cam_ket, co_hanh_dong_ten, co_so_dinh_luong, quy_ve_bank, co_moc_tg}` kèm evidence span; một hàm luật tất định suy ra `spec_level`/`is_commitment`; guard chống bịa (figure-in-text + evidence-substring) giữ nguyên. Gold 400-chunk dùng để *validate*, không train.

**Tech Stack:** Python 3.13, PhoBERT/Qwen3 qua transformers, pandas, scikit-learn (cohen_kappa), pytest (mock `_complete`, không tải model), matplotlib. Chạy GPU trên Kaggle.

## Global Constraints

- Mô tả mô hình trong paper **bám sát code đã chạy** — không thêm thành phần không tồn tại.
- **KHÔNG** đưa giá trị siêu tham số / không gian tìm Optuna vào main text (chỉ nêu phương pháp TPE+pruning); chi tiết để repo.
- **KHÔNG** nhắc XLM-R/mBERT như kết quả (chưa chạy).
- **BỎ HẲN** Dawid–Skene / mô hình rater nhiễu.
- **GIỮ NGUYÊN** mọi guard chống bịa (`verify_rubric`, salvage, retry→Mức 0); thêm evidence-substring.
- Topic S/G = limitation, không chữa.
- A4 (`co_moc_tg`) = biến phụ trợ, KHÔNG vào CTI/NAR/QDR.
- 5 cờ atomic cấp chunk dùng đúng tên cột gold: `co_cam_ket, co_hanh_dong_ten, co_so_dinh_luong, quy_ve_bank, co_moc_tg`.
- Luật tất định (dùng chung người & LLM):
  `commit = co_cam_ket AND (env OR soc OR gov)`;
  `level = 2 if (co_so_dinh_luong AND quy_ve_bank) else 1 if co_hanh_dong_ten else 0` (chỉ khi commit).
- Tests phải chạy không-GPU: mock LLM bằng subclass override `_complete` (xem `tests/test_specificity_llm.py::_StubLLM`).

---

## File Structure

- `src/esgwash/models/specificity_llm.py` — **viết lại**: prompt 5 cờ, parse cờ+evidence, `derive_flags()`, evidence-substring guard. Đầu ra giữ cột cũ (`spec_level, is_specific, parse_ok, rubric, raw`) + thêm 5 cột cờ + `evidence`.
- `src/esgwash/run.py:81` — `classify_chunks`: truyền `env/soc/gov` vào rule commit; ghi 5 cờ ra output.
- `experiments/iaa_atomic.py` — **tạo mới**: tính κ 5 cờ atomic (A vs B) + tương phản S/G + hình.
- `experiments/eval_gold.py` — **sửa**: thêm so sánh 5 cờ atomic (model vs A, vs B, ceiling A-B).
- `tests/test_specificity_llm.py` — cập nhật cho schema cờ.
- `tests/test_iaa_atomic.py` — **tạo mới**.
- `docs/paper/main_vi.tex` — viết lại §Methodology, §Attribute extraction, §reliability + bảng.

---

## Task 1: Hàm luật `derive_flags()` (pure, TDD)

Tách luật tất định ra một hàm thuần để test độc lập khỏi LLM/parsing.

**Files:**
- Modify: `src/esgwash/models/specificity_llm.py` (thêm `derive_flags`)
- Test: `tests/test_specificity_llm.py`

**Interfaces:**
- Produces: `derive_flags(flags: dict) -> tuple[float, int]` trả `(p_specificity, spec_level)`.
  `flags` có khoá nhị phân `co_cam_ket, co_hanh_dong_ten, co_so_dinh_luong, quy_ve_bank` (bool/int).
  Luật: `level = 2 if (co_so_dinh_luong and quy_ve_bank) else 1 if co_hanh_dong_ten else 0`;
  `p = {0:0.0, 1:0.5, 2:1.0}[level]`. (commit-gate xử lý ở `classify_chunks`, không ở đây.)

- [ ] **Step 1: Write the failing test**

```python
def test_derive_flags_levels():
    from esgwash.models.specificity_llm import derive_flags
    # Mức 2: có số định lượng & quy về chủ thể
    assert derive_flags({"co_so_dinh_luong": 1, "quy_ve_bank": 1}) == (1.0, 2)
    # Mức 2 hụt: có số nhưng KHÔNG quy về chủ thể -> rớt xuống theo hành động
    assert derive_flags({"co_so_dinh_luong": 1, "quy_ve_bank": 0,
                         "co_hanh_dong_ten": 1}) == (0.5, 1)
    # Mức 1: có hành động có tên, không số
    assert derive_flags({"co_hanh_dong_ten": 1}) == (0.5, 1)
    # Mức 0: không có gì
    assert derive_flags({}) == (0.0, 0)
    assert derive_flags({"co_so_dinh_luong": 1, "quy_ve_bank": 0}) == (0.0, 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_specificity_llm.py::test_derive_flags_levels -v`
Expected: FAIL — `ImportError: cannot import name 'derive_flags'`

- [ ] **Step 3: Write minimal implementation**

```python
def derive_flags(flags: dict) -> tuple[float, int]:
    """5 cờ atomic -> (p_specificity, spec_level) bằng luật tất định.
      2 = co_so_dinh_luong AND quy_ve_bank
      1 = co_hanh_dong_ten (chưa đạt Mức 2)
      0 = còn lại
    Cổng commit (co_cam_ket AND ESG) xử lý ở classify_chunks, không ở đây."""
    quant = bool(flags.get("co_so_dinh_luong")) and bool(flags.get("quy_ve_bank"))
    action = bool(flags.get("co_hanh_dong_ten"))
    level = 2 if quant else (1 if action else 0)
    return {0: 0.0, 1: 0.5, 2: 1.0}[level], level
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_specificity_llm.py::test_derive_flags_levels -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/esgwash/models/specificity_llm.py tests/test_specificity_llm.py
git commit -m "feat(spec): derive_flags() deterministic rule over 5 atomic flags"
```

---

## Task 2: Guard evidence-substring (pure, TDD)

Mỗi cờ "yes" phải kèm evidence là chuỗi con của chunk; nếu không → hạ cờ về 0. Bổ sung cho `verify_rubric` đã có.

**Files:**
- Modify: `src/esgwash/models/specificity_llm.py` (thêm `enforce_evidence`)
- Test: `tests/test_specificity_llm.py`

**Interfaces:**
- Produces: `enforce_evidence(flags: dict, evidence: dict, text: str) -> dict`.
  Với mỗi cờ trong `("co_cam_ket","co_hanh_dong_ten","co_so_dinh_luong","quy_ve_bank","co_moc_tg")`:
  nếu cờ=1 nhưng `evidence.get(flag)` không phải substring (chuẩn hoá khoảng trắng + lower) của `text`
  → đặt cờ=0. Trả về dict cờ đã lọc. So khớp bỏ dấu cách thừa, không phân biệt hoa thường.

- [ ] **Step 1: Write the failing test**

```python
def test_enforce_evidence_drops_unsupported_flag():
    from esgwash.models.specificity_llm import enforce_evidence
    text = "Ngân hàng triển khai hệ thống quản lý môi trường nội bộ."
    flags = {"co_cam_ket": 1, "co_hanh_dong_ten": 1, "co_so_dinh_luong": 1,
             "quy_ve_bank": 1, "co_moc_tg": 0}
    evidence = {"co_cam_ket": "triển khai", "co_hanh_dong_ten": "hệ thống quản lý môi trường",
                "co_so_dinh_luong": "5000 tỷ",  # KHÔNG có trong text -> phải hạ về 0
                "quy_ve_bank": "Ngân hàng"}
    out = enforce_evidence(flags, evidence, text)
    assert out["co_hanh_dong_ten"] == 1
    assert out["co_so_dinh_luong"] == 0   # evidence không phải substring
    assert out["co_cam_ket"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_specificity_llm.py::test_enforce_evidence_drops_unsupported_flag -v`
Expected: FAIL — `ImportError: cannot import name 'enforce_evidence'`

- [ ] **Step 3: Write minimal implementation**

```python
ATOMIC_FLAGS = ("co_cam_ket", "co_hanh_dong_ten", "co_so_dinh_luong",
                "quy_ve_bank", "co_moc_tg")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def enforce_evidence(flags: dict, evidence: dict, text: str) -> dict:
    """Cờ 'yes' phải có evidence là chuỗi con của chunk, nếu không -> hạ về 0."""
    norm_text = _norm(text)
    out = {}
    for f in ATOMIC_FLAGS:
        v = int(bool(flags.get(f)))
        if v:
            ev = _norm(evidence.get(f) or "")
            if not ev or ev not in norm_text:
                v = 0
        out[f] = v
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_specificity_llm.py::test_enforce_evidence_drops_unsupported_flag -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/esgwash/models/specificity_llm.py tests/test_specificity_llm.py
git commit -m "feat(spec): enforce_evidence guard requires evidence substring per flag"
```

---

## Task 3: Viết lại prompt + parse 5 cờ trong `SpecificityLLM` (TDD, mock LLM)

**Files:**
- Modify: `src/esgwash/models/specificity_llm.py` (SYSTEM, SCHEMA_HINT, FEWSHOT, `_parse_flags`, `score_one`)
- Test: `tests/test_specificity_llm.py`

**Interfaces:**
- Produces: `_parse_flags(text: str) -> dict | None` — bóc JSON (đã bỏ `<think>`), trả
  `{"flags": {<5 cờ>:0/1}, "evidence": {<cờ>:str}}` hoặc None nếu hỏng.
- Produces: `SpecificityLLM.score_one(text) -> dict` đầu ra **mở rộng**:
  `{p_specificity, spec_level, is_specific, parse_ok, rubric, raw,
    co_cam_ket, co_hanh_dong_ten, co_so_dinh_luong, quy_ve_bank, co_moc_tg, evidence}`.
  Pipeline: parse → `verify_rubric` (giữ, lọc figure bịa cho `co_so_dinh_luong`) →
  `enforce_evidence` → `derive_flags`. Parse hỏng sau retry → mọi cờ 0, spec_level 0, parse_ok False.
- LLM trả JSON dạng: `{"co_cam_ket":bool, "co_hanh_dong_ten":bool, "co_so_dinh_luong":bool, "quy_ve_bank":bool, "co_moc_tg":bool, "evidence":{"<cờ>":"<trích dẫn>"}, "reason":"..."}`.

- [ ] **Step 1: Write the failing test**

```python
def test_score_one_emits_atomic_flags():
    import json
    text = "Ngân hàng sẽ triển khai hệ thống quản lý môi trường nội bộ."
    reply = json.dumps({"co_cam_ket": True, "co_hanh_dong_ten": True,
                        "co_so_dinh_luong": False, "quy_ve_bank": True, "co_moc_tg": False,
                        "evidence": {"co_cam_ket": "sẽ triển khai",
                                     "co_hanh_dong_ten": "hệ thống quản lý môi trường nội bộ",
                                     "quy_ve_bank": "Ngân hàng"}}, ensure_ascii=False)
    out = _StubLLM([reply]).score_one(text)
    assert out["parse_ok"] is True
    assert out["co_cam_ket"] == 1 and out["co_hanh_dong_ten"] == 1
    assert out["co_so_dinh_luong"] == 0
    assert out["spec_level"] == 1 and out["is_specific"] == 1


def test_score_one_fabricated_quant_dropped():
    import json
    text = "Ngân hàng cam kết giảm phát thải mạnh mẽ."   # không có số
    reply = json.dumps({"co_cam_ket": True, "co_so_dinh_luong": True, "quy_ve_bank": True,
                        "evidence": {"co_cam_ket": "cam kết",
                                     "co_so_dinh_luong": "giảm 30%",  # số không có trong text
                                     "quy_ve_bank": "Ngân hàng"}}, ensure_ascii=False)
    out = _StubLLM([reply]).score_one(text)
    assert out["co_so_dinh_luong"] == 0 and out["spec_level"] == 0


def test_score_one_parse_fail_safe():
    out = _StubLLM(["rac", "van rac", "rac nua"]).score_one("cau")
    assert out["parse_ok"] is False and out["spec_level"] == 0
    assert out["co_cam_ket"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_specificity_llm.py -k score_one -v`
Expected: FAIL — `score_one` chưa trả cờ atomic / `_StubLLM` chưa khớp schema mới.

- [ ] **Step 3: Write implementation**

Thay SYSTEM/SCHEMA_HINT/FEWSHOT bằng bản 5 cờ + evidence, thêm `_parse_flags`, sửa `score_one`:

```python
SYSTEM = (
    "Bạn là chuyên gia phân tích báo cáo ESG ngân hàng. Với mỗi ĐOẠN VĂN, trả lời 5 câu hỏi "
    "YES/NO khách quan, và với mỗi câu trả lời YES phải TRÍCH nguyên văn cụm trong đoạn làm "
    "bằng chứng (evidence):\n"
    "1. co_cam_ket: đoạn có Ý CAM KẾT/hướng tương lai (sẽ, cam kết, hướng tới, mục tiêu, đặt mục tiêu)?\n"
    "2. co_hanh_dong_ten: có HÀNH ĐỘNG/CHƯƠNG TRÌNH/CÔNG CỤ/HỆ THỐNG CÓ TÊN, kiểm chứng được "
    "(vd 'gói Tín dụng xanh', 'hệ thống B.One') — KHÁC khẩu hiệu/tính từ ('bền vững', 'toàn diện')?\n"
    "3. co_so_dinh_luong: có ĐẠI LƯỢNG ĐỊNH LƯỢNG (số, %, tỷ đồng, MW)? KHÔNG tính năm chiến "
    "lược/luật, tên tiêu chuẩn (ISO), số của NHNN/toàn ngành.\n"
    "4. quy_ve_bank: số/hành động đó QUY VỀ CHÍNH NGÂN HÀNG chủ thể (không phải quốc gia/ngành)?\n"
    "5. co_moc_tg: có MỐC THỜI GIAN/deadline (năm mục tiêu, 'đến 2030', 'giai đoạn 2021-2025')?\n"
    "Chỉ trả về JSON, không giải thích ngoài JSON."
)

SCHEMA_HINT = (
    'Trả về JSON đúng dạng:\n'
    '{"co_cam_ket": true/false, "co_hanh_dong_ten": true/false, "co_so_dinh_luong": true/false, '
    '"quy_ve_bank": true/false, "co_moc_tg": true/false, '
    '"evidence": {"co_cam_ket": "<trích dẫn hoặc null>", "co_hanh_dong_ten": "...", '
    '"co_so_dinh_luong": "...", "quy_ve_bank": "...", "co_moc_tg": "..."}, "reason": "<ngắn>"}'
)

FEWSHOT = [
    ("Ngân hàng hướng tới một tương lai xanh và bền vững.",
     {"co_cam_ket": True, "co_hanh_dong_ten": False, "co_so_dinh_luong": False,
      "quy_ve_bank": False, "co_moc_tg": False,
      "evidence": {"co_cam_ket": "hướng tới"},
      "reason": "Chỉ khẩu hiệu, không hành động có tên, không số (Mức 0)."}),
    ("BIDV đã ban hành gói Tín dụng xanh cho khách hàng vay phát triển năng lượng sạch.",
     {"co_cam_ket": True, "co_hanh_dong_ten": True, "co_so_dinh_luong": False,
      "quy_ve_bank": True, "co_moc_tg": False,
      "evidence": {"co_cam_ket": "ban hành", "co_hanh_dong_ten": "gói Tín dụng xanh",
                   "quy_ve_bank": "BIDV"},
      "reason": "Hành động có tên, không số (Mức 1)."}),
    ("Ngân hàng đặt mục tiêu giảm 30% cường độ phát thải khí nhà kính vào năm 2030.",
     {"co_cam_ket": True, "co_hanh_dong_ten": True, "co_so_dinh_luong": True,
      "quy_ve_bank": True, "co_moc_tg": True,
      "evidence": {"co_cam_ket": "đặt mục tiêu", "co_hanh_dong_ten": "giảm cường độ phát thải",
                   "co_so_dinh_luong": "30%", "quy_ve_bank": "Ngân hàng", "co_moc_tg": "năm 2030"},
      "reason": "Số 30% quy về ngân hàng, có mốc 2030 (Mức 2)."}),
]


def _parse_flags(text: str) -> dict | None:
    """Bỏ <think>, lấy object JSON đầu tiên khớp ngoặc, tách flags + evidence."""
    obj = _extract_json_obj(text)   # helper bóc object đầu tiên (tách từ _extract_json cũ)
    if obj is None:
        return None
    flags = {f: int(bool(obj.get(f))) for f in ATOMIC_FLAGS}
    ev = obj.get("evidence") or {}
    evidence = {f: (ev.get(f) if isinstance(ev, dict) else None) for f in ATOMIC_FLAGS}
    return {"flags": flags, "evidence": evidence}
```

Sửa `score_one`:

```python
def score_one(self, text: str) -> dict:
    parsed, raw = None, ""
    for attempt in range(self.retries + 1):
        raw = self._complete(self._build_messages(text, stricter=attempt > 0))
        parsed = _parse_flags(raw)
        if parsed is not None:
            break
    base = {f: 0 for f in ATOMIC_FLAGS}
    if parsed is None:
        return {"p_specificity": 0.0, "spec_level": 0, "is_specific": 0, "parse_ok": False,
                "rubric": pd.NA, "raw": raw, "evidence": pd.NA, **base}
    flags = verify_rubric_flags(parsed["flags"], parsed["evidence"], text)  # lọc số bịa cho co_so_dinh_luong
    flags = enforce_evidence(flags, parsed["evidence"], text)
    p, level = derive_flags(flags)
    return {"p_specificity": p, "spec_level": level, "is_specific": int(level >= 1),
            "parse_ok": True, "rubric": json.dumps(parsed, ensure_ascii=False), "raw": raw,
            "evidence": json.dumps(parsed["evidence"], ensure_ascii=False), **flags}
```

Thêm `verify_rubric_flags(flags, evidence, text)`: nếu `co_so_dinh_luong=1` mà chuỗi chữ số trong
`evidence["co_so_dinh_luong"]` không xuất hiện trong `text` (dùng `_digit_runs` đã có) → đặt về 0.
Tách `_extract_json_obj` từ nhánh (1) của `_extract_json` cũ (object đầu tiên khớp ngoặc, bỏ `<think>`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_specificity_llm.py -v`
Expected: PASS (cập nhật/loại bỏ các test cũ dựa trên `items`/`derive` cũ trong cùng commit; giữ `test_extract_json_strips_think_and_prose` nếu helper còn, nếu không thì chuyển sang test `_extract_json_obj`).

- [ ] **Step 5: Commit**

```bash
git add src/esgwash/models/specificity_llm.py tests/test_specificity_llm.py
git commit -m "feat(spec): rewrite extractor to emit 5 atomic flags + evidence"
```

---

## Task 4: `classify_chunks` áp cổng commit + ghi 5 cờ (TDD, mock)

**Files:**
- Modify: `src/esgwash/run.py:81-110`
- Test: `tests/test_run_pipeline.py`

**Interfaces:**
- Consumes: `SpecificityLLM.predict` nay trả thêm 5 cột cờ + `evidence`.
- Produces: `classify_chunks(...)` output thêm 5 cột cờ; `is_commitment` final =
  `co_cam_ket AND (is_env OR is_soc OR is_gov)`; `spec_level` chỉ tính trên chunk commit (như cũ).

- [ ] **Step 1: Write the failing test**

```python
def test_classify_commit_gate_uses_flags():
    import pandas as pd
    from esgwash.run import classify_chunks

    class StubTopic:
        def predict(self, texts):
            n = len(texts)
            return pd.DataFrame({"env": [0.9]*n, "soc": [0.1]*n, "gov": [0.1]*n,
                                 "is_env": [1]*n, "is_soc": [0]*n, "is_gov": [0]*n,
                                 "pillar": ["env"]*n})

    class StubCommit:   # giữ tương thích: model commit thô vẫn chạy
        def predict(self, texts):
            return pd.DataFrame({"p_commitment": [0.9]*len(texts),
                                 "is_commitment": [1]*len(texts)})

    class StubSpec:
        def predict(self, texts):
            n = len(texts)
            return pd.DataFrame({"p_specificity": [0.0]*n, "spec_level": [0]*n,
                                 "is_specific": [0]*n, "parse_ok": [True]*n,
                                 "rubric": ["{}"]*n, "raw": ["{}"]*n, "evidence": ["{}"]*n,
                                 "co_cam_ket": [0]*n, "co_hanh_dong_ten": [0]*n,
                                 "co_so_dinh_luong": [0]*n, "quy_ve_bank": [0]*n,
                                 "co_moc_tg": [0]*n})

    chunks = pd.DataFrame({"content_text": ["x"], "doc_id": ["d"], "chunk_index": [0]})
    out = classify_chunks(chunks, StubTopic(), StubCommit(), StubSpec())
    assert "co_so_dinh_luong" in out.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_run_pipeline.py::test_classify_commit_gate_uses_flags -v`
Expected: FAIL — output chưa có cột cờ atomic.

- [ ] **Step 3: Implement** — trong `classify_chunks`, merge 5 cột cờ + `evidence` từ spec output vào `out`; giữ logic `spec_on_commitment` hiện tại. Đảm bảo các cột cờ tồn tại cả khi chunk không-commit (điền 0).

- [ ] **Step 4: Run test** — `python -m pytest tests/test_run_pipeline.py -v` → PASS (cập nhật `StubSpec` ở `test_specificity` nếu test cũ vỡ vì thiếu cột).

- [ ] **Step 5: Commit**

```bash
git add src/esgwash/run.py tests/test_run_pipeline.py
git commit -m "feat(run): propagate 5 atomic flags through classify_chunks"
```

---

## Task 5: `experiments/iaa_atomic.py` — bảng κ atomic + tương phản S/G (TDD phần thuần)

**Files:**
- Create: `experiments/iaa_atomic.py`
- Test: `tests/test_iaa_atomic.py`

**Interfaces:**
- Produces: `atomic_iaa(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame` — merge theo `chunk_id`,
  trả 1 dòng/trường với cột `field, n, agree, kappa, n_disagree` cho
  `["co_cam_ket","co_hanh_dong_ten","co_so_dinh_luong","quy_ve_bank","co_moc_tg","g_soc","g_gov"]`.
- Produces: `main()` đọc `data/gold_annot_{1,2}_relabeled.xlsx`, in bảng + lưu
  `experiments/eval/iaa_atomic.json` + `experiments/eval/iaa_atomic.png`.

- [ ] **Step 1: Write the failing test**

```python
def test_atomic_iaa_perfect_and_disagree():
    import pandas as pd
    from experiments.iaa_atomic import atomic_iaa
    a = pd.DataFrame({"chunk_id": [1, 2, 3, 4],
                      "co_cam_ket": [1, 1, 0, 0], "g_soc": [1, 0, 1, 0]})
    b = pd.DataFrame({"chunk_id": [1, 2, 3, 4],
                      "co_cam_ket": [1, 1, 0, 0], "g_soc": [0, 0, 1, 0]})
    r = atomic_iaa(a, b).set_index("field")
    assert r.loc["co_cam_ket", "kappa"] == 1.0
    assert r.loc["co_cam_ket", "n_disagree"] == 0
    assert r.loc["g_soc", "n_disagree"] == 1
```

- [ ] **Step 2: Run** `python -m pytest tests/test_iaa_atomic.py -v` → FAIL (module chưa có).

- [ ] **Step 3: Implement** `atomic_iaa` (dùng `sklearn.metrics.cohen_kappa_score`, bỏ ô NaN ở một trong hai; `kappa=nan` nếu một phía hằng số) + `main()` (đọc xlsx sheet mặc định `Sheet1`, vẽ bar κ + chú thích tương phản S/G).

- [ ] **Step 4: Run** `python -m pytest tests/test_iaa_atomic.py -v` → PASS. Rồi chạy thật:
`python experiments/iaa_atomic.py` → kiểm tra số khớp κ đã biết (co_cam_ket 0.984, g_soc 0.465...).

- [ ] **Step 5: Commit**

```bash
git add experiments/iaa_atomic.py tests/test_iaa_atomic.py experiments/eval/iaa_atomic.json experiments/eval/iaa_atomic.png
git commit -m "feat(eval): reproducible atomic IAA table + S/G contrast figure"
```

---

## Task 6: `eval_gold.py` — so 5 cờ atomic model↔A↔B (sửa)

**Files:**
- Modify: `experiments/eval_gold.py` (`BIN`, `load_gold` keep cols, `evaluate`)

**Interfaces:**
- Consumes: gold relabeled có 5 cột cờ; `classify_chunks` output có 5 cột cờ cùng tên.
- Produces: report JSON thêm khối `atomic` — mỗi cờ có `model_vs_A`, `model_vs_B`,
  `human_ceiling_A_vs_B` (cohen_kappa + F1).

- [ ] **Step 1:** Đổi `GOLD` trỏ `*_relabeled.xlsx`, `SHEET="Sheet1"`; mở rộng `BIN` thêm 5 cờ atomic `(name, "<flag>", "<flag>")` (cùng tên cột gold & model). Giữ `_binary_scores`.

- [ ] **Step 2:** Trong `load_gold`, thêm 5 cờ vào `keep` để có hậu tố `_A/_B`.

- [ ] **Step 3:** Chạy smoke không-GPU bằng `--no-spec`? Không — atomic cần spec model. Thay vào đó **kiểm tra cấu trúc** bằng test nhẹ: assert `BIN` chứa 5 cờ.

```python
def test_eval_gold_has_atomic_bins():
    import experiments.eval_gold as eg
    names = {b[0] for b in eg.BIN}
    assert {"co_cam_ket", "co_so_dinh_luong", "quy_ve_bank"} <= names
```

- [ ] **Step 4:** `python -m pytest tests/test_eval_gold.py -v` → PASS (tạo file test này).

- [ ] **Step 5: Commit**

```bash
git add experiments/eval_gold.py tests/test_eval_gold.py
git commit -m "feat(eval): eval_gold compares 5 atomic flags model vs A/B + ceiling"
```

---

## Task 7: [GPU/Kaggle — RUN, không pytest] Validate LLM vs người + tái dựng corpus

> **Trạng thái:** cần chạy trên Kaggle (PhoBERT + Qwen3). Không phải bước TDD. Người chạy xác nhận kết quả trước khi viết bảng vào paper.

- [ ] **Step 1:** Chạy `python experiments/eval_gold.py` trên Kaggle (full 400 chunk) → `eval_gold_report.json` có khối `atomic` (model vs A/B + ceiling). Lưu vào `experiments/eval/`.
- [ ] **Step 2:** Chạy lại pipeline trên toàn corpus (`run.classify_chunks`) → cập nhật panel CTI/NAR/QDR. Lưu parquet.
- [ ] **Step 3:** So CTI/NAR/QDR mới với số cũ; ghi chênh lệch vào `experiments/eval/reder_delta.md` (giải thích bằng việc bỏ shortcut, KHÔNG ép khớp số cũ).
- [ ] **Step 4: Commit** kết quả (`git add experiments/eval/*.json experiments/eval/*.md` + parquet nếu policy cho phép).

---

## Task 8: [Ablation — RUN, đánh dấu trạng thái] Chứng minh lựa chọn thiết kế

> **Trạng thái mỗi mục: cần người dùng xác nhận "đã có số" hay "cần chạy".** `tests/test_digit_shortcut.py` đã tồn tại → digit-shortcut nhiều khả năng đã có cơ sở; phần còn lại cần chạy.

- [ ] **Step 1 — digit-shortcut:** xác nhận/chạy `experiments/` baseline encoder-only & TF-IDF/LR (`baselines.py`, `spec_features.py`) dự đoán spec_level → cho thấy bắt shortcut "có số → Mức 2". Lưu bảng.
- [ ] **Step 2 — holistic vs decomposed:** prompt LLM hỏi thẳng 0/1/2 (biến thể của `_build_messages`) so với 5-cờ+luật; so cả hai với consensus người (κ). Lưu bảng.
- [ ] **Step 3 — guard on/off:** chạy extractor có/không `verify_rubric_flags` → đếm tỉ lệ `co_so_dinh_luong` bị chặn vì figure bịa. Lưu số.
- [ ] **Step 4 — sentence vs chunk:** thống kê tỉ lệ cam kết mà số định lượng & hành động nằm ở câu KHÁC trong cùng chunk (minh hoạ vì sao chunk). Lưu số.
- [ ] **Step 5: Commit** mọi artefact ablation vào `experiments/eval/ablation_*`.

---

## Task 9: [Paper] Viết lại §Methodology mô tả mô hình (content)

**Files:** Modify `docs/paper/main_vi.tex` (§Topic and Commitment Classifier, thêm chi tiết theo spec §4)

- [ ] **Step 1:** Mở rộng mô tả mỗi tầng: backbone PhoBERT + lý do (literature, KHÔNG so XLM-R), translate-train + hòa hợp nhãn nhiều nguồn, **masked-BCE cho nhãn thiếu**, cổng ESG-commitment. Nêu Optuna TPE+pruning ở mức phương pháp (KHÔNG giá trị hparam). Tầng extractor: 5 cờ + luật + grounding + guard.
- [ ] **Step 2:** Đọc lại đảm bảo không câu nào nói thành phần chưa chạy. Không compile LaTeX (theo memory `feedback_no_latex_compile`).
- [ ] **Step 3: Commit** `git commit -m "docs(paper): detailed model-building writeup (backbone, masked-BCE, Optuna, gate)"`

---

## Task 10: [Paper] Viết lại §Specificity → §Attribute extraction (content)

**Files:** Modify `docs/paper/main_vi.tex` (§Specificity rubric, Bảng rubric)

- [ ] **Step 1:** Thay bảng 3-mức bằng bảng 5 cờ atomic + luật tái dựng; nêu A4 là biến phụ trợ; nêu evidence span + guard chống bịa.
- [ ] **Step 2: Commit** `git commit -m "docs(paper): attribute-extraction section (5 flags + deterministic rule)"`

---

## Task 11: [Paper] Viết lại §Độ tin cậy nhãn vàng quanh phép nhảy κ (content)

**Files:** Modify `docs/paper/main_vi.tex` (§reliability, bảng IAA, hình)

- [ ] **Step 1:** Thay bảng IAA cũ bằng bảng κ cũ-vs-mới (spec §2). Viết lập luận **tương phản S/G** (cùng người, atomic lệch 0–3 ô, topic S/G lệch 105 ô) thay cho lập luận "systematic offset cancels". Nói rõ `g_is_commit`/`g_spec_level` là cột suy ra; bằng chứng là κ atomic.
- [ ] **Step 2:** Thêm bảng **LLM-vs-người** (từ Task 7) + đoạn **ablation** (từ Task 8). Diễn giải lại CTI/NAR/QDR là "đo được, kiểm chứng được".
- [ ] **Step 3:** Cập nhật hình `iaa_overview.png` → trỏ `iaa_atomic.png` (Task 5). Ghi topic S/G là limitation ở §Conclusion/Limitations.
- [ ] **Step 4: Commit** `git commit -m "docs(paper): rewrite reliability around kappa jump + S/G contrast; add LLM-vs-human + ablation"`

---

## Self-Review (đã chạy)

- **Spec coverage:** §4 mô hình → Task 9; §5 schema+luật → Task 1,3,10; §6 extractor → Task 2,3,4; §7 validation/ablation → Task 5,6,7,8,11; §8 paper → Task 9,10,11; §9 bỏ Dawid/topic → Global Constraints + Task 11. Đủ.
- **Placeholder scan:** code steps đều có code thật; Task 7,8 là RUN-task có đánh dấu trạng thái rõ (không phải placeholder ẩn).
- **Type consistency:** `ATOMIC_FLAGS` dùng nhất quán; `derive_flags`/`enforce_evidence`/`_parse_flags`/`score_one` khớp tên & kiểu xuyên Task 1–4; tên cờ khớp cột gold xuyên Task 5,6.

## Lưu ý mở (cần người dùng chốt khi thực thi)
- Task 8: cái nào "đã có số" vs "cần chạy" — chưa xác nhận; plan để trạng thái mở.
- Task 7/8 cần GPU Kaggle theo memory `feedback_kaggle_end_to_end` (ném cả repo chạy 1 phát).
