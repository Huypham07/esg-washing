# Phân tích nhánh `optimize-research` của đồng nghiệp — đối chiếu & học hỏi cho track của tôi

> Ngày: 2026-06-12 · Brainstorm-technical · Bối cảnh: **2 bản song song, sẽ chọn 1**; mục tiêu = hiểu nhánh đồng nghiệp để **học hỏi/cải tiến** track local (silver) của tôi.
> Remote: `origin/optimize-research @ 82acd88` (Huypham07) — local đứng sau 5 commit (fast-forward được).

---

## 0. TL;DR

- Đồng nghiệp **không update nhỏ** mà **tái kiến trúc toàn bộ** thành package `src/esgwash/` + 6 spec học thuật (`superpowers/specs/00-05`) + đề cương. Cùng hướng pivot đã chốt (**grounded-CTI + translate-train**), làm **rất bài bản, truy vết related-work**.
- **Độ chín đảo ngược nhau:**
  - *Đồng nghiệp:* data + training = **code thật**; grounding + index + validation = **STUB** (`raise NotImplementedError`). **Chưa có kết quả nào.**
  - *Tôi (local):* CTI + bootstrap + selective disclosure = **đã chạy, có số** (E.52/S.54/G.60); nhưng nhãn **silver LLM** → JSON của chính tôi ghi *"circular eval — KHÔNG phải accuracy"*; grounding/validation (phase-04/06/07) **đã hoãn**.
- **Cả 2 là 2 lần thực thi CÙNG 1 kế hoạch** (brainstorm 2026-06-11). Khác ở **thứ tự** (tôi demo-first silver; đồng nghiệp rigorous-first) và **vài tinh chỉnh phương pháp** đồng nghiệp thêm.
- **Để track của tôi thắng:** hấp thụ phần rigor (translate-train + VN human-eval + validation + grounding) vào pipeline **đã chạy được** của mình → "có kết quả + bài bản + đi trước ở Phase D". Yếu tố quyết định nhất = **VN human-eval set** + **translate-train** (phá tính vòng tròn của silver).

---

## 1. Đồng nghiệp đã push gì (5 commit `b381ecd..82acd88`)

Tái kiến trúc thành **package layered** `src/esgwash/` (data / nlp / models / grounding / indices / validation / pipeline) + `configs/` per-stage + `pyproject.toml` + `tests/` + `superpowers/specs/`.

### Spec — nguyên tắc cốt lõi (đáng học nhất về mặt "khung")
- **CTI = tỉ lệ output của classifier đã công bố** (Bingler 2022). *"Không tự bịa công thức/trọng số/thang đo."* Tham số tự do **duy nhất** = ngưỡng → xử lý bằng **sensitivity analysis**, không chọn cứng.
- **Walk = proxy nội văn bản** (specificity + grounding), framing *intra-report substantiation* — KHÔNG tuyên bố đo "hành vi thật" (vì VN không có walk-data ngoài).
- **Cấm dùng nhãn LLM cũ làm nhãn train**; corpus build lại 100% từ raw OCR.
- Mọi kết luận kèm **bootstrap CI** + **ngưỡng n tối thiểu**; bảng **limitations L1-L5** khai báo sẵn (vd specificity α≈0.17).

### Bản đồ độ chín (đếm dòng code thực)

| Tầng | Module | Trạng thái |
|---|---|---|
| A. Data | corpus_builder(128), gold_loader(92), topic_merge(84), claim_merge(49), cleaning(75), noise_filter(43), **esgbert_labels(93)** | ✅ code thật |
| B. Train | **trainer(221)**, tuning(86), baselines(35), topic_model(20), claim_model(27) | ✅ code thật, chuẩn spec |
| C. Grounding | evidence_pool, retriever, nli, support | ❌ STUB |
| D. Index+Val | **cti**, grounded_cti, bootstrap, disclosure, known_group, synthetic, sensitivity | ❌ STUB |

`trainer.py` chất lượng tốt: `MultiHeadClassifier` (PhoBERT + N đầu sigmoid) + `masked_bce_loss` (bỏ ô NaN) dùng chung cho M1 và M2. Engineer giỏi.

### Lưu ý phụ
- **README STALE** — vẫn mô tả thiết kế CŨ (EWRI + Neuro-Symbolic), chưa cập nhật sang grounded-CTI.
- Topic 2k CSV đã chuyển vào `unused/`; chỉ còn artifact `topic_masked.parquet` (gitignored) → **gap tái lập** nếu chưa khôi phục file gốc.

---

## 2. Hai track = hai lần thực thi cùng 1 kế hoạch

| Khía cạnh | **Track local (tôi)** | **Track đồng nghiệp** |
|---|---|---|
| Nhãn train | **Silver LLM** (vi_silver) | **Translate-train** (EN gold→VI MT) + **ESGBERT cross-label** |
| Eval | circular (silver), chưa human-eval | translate test **+ VN human-eval 300 câu gán tay** (test chính) |
| Topic model | **3 model nhị phân riêng** | **1 model multi-label, masked BCE** |
| Claim model | commitment + specificity **rời** | **multi-task 2 đầu** (1 encoder, vì 2 tập trùng 100% text) |
| Phase D (CTI/bootstrap/SD) | ✅ **đã code + chạy + figures** | ❌ stub |
| Tuning | ✅ **5 Optuna study xong** | tuning.py có, chưa chạy |
| Grounding → grounded-CTI | hoãn (phase-06 exploratory) | spec 03 sẵn sàng, code stub |
| Validation | hoãn (phase-07) | spec 05 sẵn sàng, code stub |
| min-n xếp hạng | `min_commit=5` (yếu) | `min_n=30` + pooled |
| Data S/G augment | — | **+ ML-Promise EN/FR/JA**, action_500 |

**Cùng tư duy** (kế hoạch của tôi đã có phase-04 gold VN, phase-06 grounding, phase-07 validation, phase-08 baselines). Đồng nghiệp **làm rigorous-first**; tôi **demo-first** để de-risk + lấy kết quả nhanh.

---

## 3. Phán xét thẳng (cho kịch bản "chọn 1")

- **Hôm nay: KHÔNG bản nào xong.** Tôi = *có kết quả nhưng vòng tròn*; đồng nghiệp = *bài bản nhưng rỗng ruột* (grounding/index/validation stub, chưa train).
- **Endgame luận văn:** bản **rigorous sẽ thắng hội đồng** — NẾU hoàn thành. Điểm yếu chí mạng của tôi: **silver = circular eval** (không thể claim accuracy trên nhãn do LLM sinh ra) + artifact OCR `bsc` + chưa có grounded-CTI (đóng góp mới).
- **NHƯNG tôi đang đi trước ở thực thi:** Phase D đã chạy, tuning xong, end-to-end đã thông, có figures. Đồng nghiệp phải code lại toàn bộ Phase C+D.
- **Đường thắng của tôi:** không thi xem ai bài bản hơn trên giấy — **nhồi rigor vào pipeline đã chạy được của mình**. "Có kết quả + bài bản + đi trước Phase D" > "bài bản nhưng chưa chạy". Quyết định bởi **2 thứ phá circularity: VN human-eval + translate-train**.

---

## 4. HỌC HỎI — những cải tiến nên "đánh cắp" vào track của tôi (ưu tiên)

### P0 — Bắt buộc, phá tính vòng tròn / sống còn cho luận văn
- **L1. VN human-eval set (~300 câu gán tay).** Tôi đã PLAN (phase-04) nhưng hoãn. Đây là test set CHÍNH; gold EN chỉ là upper-bound. **Không có nó → mọi số F1 silver vô nghĩa với hội đồng** (chính JSON của tôi thừa nhận "không phải accuracy"). Stratified theo bank×trụ; đo Cohen's κ. *Việc người làm, không code thay được.*
- **L2. Translate-train arm (tối thiểu để so với silver).** Spec đồng nghiệp lập luận silver = **confirmation bias** + khó bảo vệ. Tôi nên: (a) train lại bằng translate-train (EN gold→VI MT đã có sẵn `data/translate/`), (b) **so silver vs translate-train trên VN human-eval** → để hội đồng tự thấy cái nào tốt. Nếu silver thua → bỏ; nếu hòa → silver thành 1 ablation hợp lệ. **Đây là khúc quyết định "ai thắng".**

### P1 — Nâng cấp rigor mạnh
- **L3. Construct-validation (spec 05 = blueprint).** Lấp phase-07 của tôi bằng 3 kiểm định chuẩn: **known-group** (`bank_signals.csv`: assurance/GRI/VNSI/green-credit/intl → nhóm có-tín-hiệu phải CTI thấp hơn, Mann-Whitney U + rank-biserial); **synthetic** (trộn câu non-specific 10/30/50% → CTI tăng đơn điệu); **sensitivity** (quét θ → Kendall τ ổn định ranking). Không có lớp này, CTI **không có bằng chứng hợp lệ** vì không có ground-truth washing.
- **L4. Grounding → grounded-CTI (đóng góp MỚI).** Tôi hoãn phase-06 → hiện chỉ là "CTI thuần specificity" = **không novel**. Spec 03 đồng nghiệp sẵn sàng code: evidence pool thu hẹp (số liệu/table/specific-fact, loại self-block), bi-encoder `bkai` top-k, NLI `mDeBERTa-xnli` **giữ phân phối** (không argmax), `support=max P_entail`, `grounded@θ` quét {0.5,0.7,0.9}. Tôi đã có `nli_verifier.py`/`evidence_linker.py` cũ (EWRI-style) → **refactor theo spec 03** rẻ hơn viết từ đầu.
- **L5. ESGBERT cross-labeling (ý mới đồng nghiệp thêm 2026-06-12).** Thay vì tự train 3 model nhị phân (confirmation bias), dùng `ESGBERT/{Env,Social,Gov}BERT` đã công bố để **điền nhãn chéo các ô NaN** (chỉ ở split=train) → rẻ, EN-native, tránh tự-xác-nhận. Có thể bổ sung vào data-prep topic của tôi.
- **L6. min-n=30 + fix artifact `bsc`.** Tôi đang `min_commit=5` → ranking dễ nhiễu; nâng 30 + pooled (bank×pillar / bank×year). `bsc` (BIDV Securities, OCR mất dấu → CTI .76 ARTIFACT) phải fix OCR hoặc loại — **tuyệt đối đừng trình "worst-washer" này cho thầy**. Corpus rebuild của đồng nghiệp (minhash near-dup, block_type) có thể xử lý gốc.

### P2 — Tinh chỉnh kiến trúc/hiệu quả
- **L7. Multi-task claim model.** commitment+specificity dùng **đúng cùng text** (đồng nghiệp đã xác minh trùng 100%) → train 1 encoder 2 đầu, nhất quán hơn + ít tham số. Tôi đang train rời.
- **L8. Masked-BCE multi-label topic (1 model)** thay 3 model nhị phân — gọn, chia sẻ encoder. (Cân nhắc: 3 model riêng của tôi đã tune xong & cho E.94/S.90 mạnh → có thể giữ, chỉ học cách xử lý partial-label.)
- **L9. ML-Promise augment (EN/FR/JA ~1.200 mẫu)** vá khoảng trống S/G — đúng chỗ gov/commit/spec của tôi đang yếu (.73-.74). Đồng nghiệp đã tải + kiểm kê (spec 01 §7.1).
- **L10. Kỷ luật "không bịa công thức" + bảng limitations.** Audit lại CTI/SD của tôi xem có trọng số/thang tự đặt nào không; khai báo L1-L5 (α=0.17, climate→S/G, MT noise, no walk thật).
- **L11. Corpus rebuild từ raw** (`raw_ocr_annual_report.zip` phủ rộng hơn: thêm vib, 2016-2019) — tái lập 100%, metadata block/section tốt hơn cho grounding.

---

## 5. Khúc quyết định: silver vs translate-train

| | Silver LLM (tôi) | Translate-train (đồng nghiệp) |
|---|---|---|
| Nguồn nhãn | LLM gán trực tiếp câu VI | EN gold người-gán → dịch máy sang VI |
| Defensibility | **Yếu** — circular, confirmation bias | **Mạnh** — baseline chuẩn XNLI/XTREME, nhãn gốc người-gán |
| Nhiễu | LLM hallucination khó định lượng | MT noise → **lọc được** (QE COMET-Kiwi + Confident Learning) |
| Eval accuracy | **Không hợp lệ** nếu test cũng silver | Hợp lệ trên VN human-eval |
| Chi phí | đã làm xong | dữ liệu dịch đã có sẵn `data/translate/` |

→ **Khuyến nghị:** chuyển trục chính sang **translate-train**, giữ silver như **1 ablation** ("LLM-silver vs translate-train trên VN human-eval"). Cả 2 đánh trên **cùng 1 VN human-eval set** → kết quả khách quan, đồng thời biến công sức silver thành đóng góp so sánh thay vì vứt đi.

---

## 6. An toàn git (làm trước khi đụng nhánh kia)

Tôi có **nhiều thay đổi chưa commit** đè đúng các file đồng nghiệp đã sửa (`train_model.py`, `tune_hyperparams.py`, `config/train*.yml`, `requirements.txt`) → **pull/merge bây giờ = conflict / mất work**.
→ **Commit track local vào branch riêng `demo-silver-track` TRƯỚC**, rồi mới fetch/đối chiếu nhánh đồng nghiệp (hoặc dùng worktree riêng để đọc `src/esgwash`).

---

## 7. Câu hỏi chưa giải quyết

1. **Ai sẽ "chọn 1"** — thầy hướng dẫn, hay tự bạn & đồng nghiệp? Tiêu chí chọn là gì (độ bài bản / có kết quả / deadline)?
2. **Deadline báo cáo thầy** còn bao lâu? Quyết định nên "hoàn thiện track đang chạy" hay "đổi sang esgwash".
3. **VN human-eval set** đã có người gán chưa, hay vẫn 0 câu? Đây là nút thắt chung của CẢ 2 track.
4. **ML-Promise + bank_signals.csv** — bạn có quyền truy cập data đồng nghiệp đã tải không (hay nằm ngoài git)?
5. Đồng nghiệp có biết bạn đang phân tích để hợp nhất ý không — hay 2 bên thực sự cạnh tranh độc lập?
