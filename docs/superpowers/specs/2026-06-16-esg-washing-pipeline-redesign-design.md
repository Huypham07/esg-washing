# Spec — Tái thiết kế pipeline đo ESG-washing (ngân hàng VN)

> Ngày: 2026-06-16 · Nhánh: `optimize-research`
> Mục tiêu: làm rõ bài toán / câu hỏi nghiên cứu / đóng góp, và tối ưu pipeline thành một
> thiết kế **defensible trước reviewer** — bỏ phần đo lường vòng tròn, làm sạch đơn vị phân
> tích và mẫu số, giữ một hệ chỉ số trung thực.

---

## 1. Bối cảnh & động lực

Pipeline hiện tại (nhánh `optimize-research`) đã chạy được end-to-end trên BIDV 2023/2024:
chunk 256-token → Topic E/S/G (PhoBERT) → Commitment (PhoBERT nhị phân) → Specificity
(LLM-rubric 3 mức, Qwen3) → Grounding (bi-encoder retrieval + mDeBERTa NLI) →
CTI / grounded-CTI / selective disclosure per (bank, year, trụ) + bootstrap CI.

Dữ liệu corpus: **10 NHTM × 5 năm (2020–2024) = 50 panel, ~13,800 chunk.**

### 1.1 Ba vấn đề cốt lõi phát hiện khi mổ xẻ kết quả

Mổ thực tế output BIDV (`outputs/cti/bidv/{2023,2024}/classified.parquet`):

1. **Grounding/gCTI là một construct vòng tròn, không đo được "walk".**
   Claim = một item định lượng (vd *"dư nợ tín dụng xanh 74.177 tỷ"*); evidence = các câu
   khác *có số* trong **cùng báo cáo**; chấm NLI entail. Nhưng:
   - Một con số định lượng thường chỉ xuất hiện **một lần** (chính câu claim). Sau khi loại
     câu gốc → pool rỗng → support = 0 → claim **Mức 2** bị đếm "cheap" trong gCTI (sai).
   - Nếu số lặp ở bảng khác, NLI entail giữa hai câu *do chính ngân hàng viết* chỉ xác nhận
     "báo cáo nhất quán với chính nó", **không** xác nhận walk thật.
   - XNLI đo *textual entailment*, không đo *tính xác thực*. Cả premise lẫn hypothesis đều là
     "talk" của ngân hàng → không có "walk" độc lập để NLI bấu víu.
   - Mâu thuẫn gốc: đề cương nói "đo walk ngay trong văn bản", nhưng đo walk của một câu bằng
     các câu khác *cùng tác giả, cùng báo cáo* thì gần như vòng tròn.

2. **`CTI_strict` (đếm Mức 1 là cheap talk) thổi chỉ số lên mức vô lý (0.6–0.84).**
   Mức 1 = *hành động/công cụ có tên, kiểm chứng được* (vd "ban hành gói Tín dụng xanh",
   "hệ thống B.One"). Đó **không phải** cheap talk — chỉ là chưa định lượng. Gộp nó vào
   washing là sai bản chất.

3. **Mẫu số commitment bị nhiễm + đơn vị phân tích không semantic.**
   - 2023: **42%** (62/147) câu "commitment" **không gắn trụ ESG nào**; 2024: **49%**
     (80/164). Classifier commitment bắn quá rộng. Per-pillar CTI đã loại nhóm này nhờ cổng
     topic, nhưng cần ghi nhận và siết tường minh.
   - Chunk 256-token cắt ngang câu → rubric mất thực thể/số → lỗi "Mức 0 bỏ sót tên riêng"
     (đáng lẽ Mức 1) → **thổi cả CTI_loose**. Một chunk còn trộn nhiều vai trò (cam kết +
     kết quả tài chính + khẩu hiệu) nên tín hiệu phân loại mâu thuẫn trong cùng đơn vị.

### 1.2 Nguyên tắc thiết kế lại

- **Trung thực construct:** chỉ đo cái thật sự đo được từ văn bản (độ cụ thể của cam kết),
  không giả vờ đo "walk" bằng grounding nội văn bản.
- **Đơn vị phân tích mạch lạc:** mỗi đơn vị có *một* vai trò ngữ nghĩa, không vượt giới hạn
  encoder (256 token).
- **Mẫu số sạch:** denominator của CTI = cam kết *có gắn trụ ESG*.
- **Giữ thông tin để phân tích sâu:** thang specificity 3 mức ở lại; CTI chỉ lấy đáy, hai mức
  trên thành chỉ số mô tả độc lập.

---

## 2. Khung nghiên cứu

### 2.1 Bài toán
Đo **decoupling giữa *talk* và *substance*** trong báo cáo thường niên của NHTM Việt Nam, đủ
ba trụ E/S/G. Vì Việt Nam **không có walk-data bên thứ ba** (CDP/MSCI/RepRisk như nước
ngoài), ta đo *substance ngay trong văn bản* qua **độ cụ thể của cam kết** (specificity),
**không** đo bằng grounding nội văn bản (vòng tròn — xem §1.1).

- **Talk** = các câu *cam kết* ESG (Commitment classifier, gate bằng Topic).
- **Substance proxy (nội văn bản)** = cam kết đó *cụ thể* đến đâu: mơ hồ / có hành động tên /
  định lượng-quy-về-chủ-thể.

### 2.2 Câu hỏi nghiên cứu
- **RQ1 — Prevalence.** Cheap talk (cam kết Mức 0, mơ hồ) phổ biến đến đâu giữa các ngân
  hàng, trụ E/S/G, và năm 2020–2024?
- **RQ2 — Selective disclosure.** Các ngân hàng có *né trụ khó, dồn trụ dễ* (cherry-picking)
  không? Phân bố lượng cam kết/câu ESG theo trụ lệch ra sao?
- **RQ3 — Substance gap.** Mức *thực chất* (định lượng, Mức 2) có tụt lại so với lượng cam
  kết không — tức khoảng cách talk–substance nội văn bản theo thời gian?
- **RQ4 — Phương pháp.** Thang specificity LLM-rubric 3 mức, attribution-aware, có vượt
  binary specificity và **chống được shortcut "có chữ số → cụ thể"** của encoder không?

### 2.3 Đóng góp mục tiêu
- **C1.** Khung đo ESG-washing đầu tiên **full E/S/G cho ngân hàng Việt Nam** qua
  translate-train cross-lingual, không cần walk-data ngoài.
- **C2 (phương pháp).** **Specificity-rubric** thứ tự 3 mức: phân rã cam kết thành item,
  đánh giá *định lượng-quy-về-chủ-thể* và *hành động-có-tên* riêng biệt, chống bịa số
  (`verify_rubric`) và chống shortcut chữ số.
- **C3.** Bức tranh **panel 10×5**: CTI/NAR/QDR có bootstrap CI + selective disclosure +
  bộ validation (known-group, synthetic, sensitivity, audit tay).
- **C4.** Dataset translate-train VI công khai + pipeline tái lập.

---

## 3. Hệ chỉ số (đã bỏ grounding)

### 3.1 Thang specificity (giữ 3 mức)
Trên mỗi **đơn vị cam kết** (commitment unit), LLM-rubric phân rã thành item và suy ra mức:

| Mức | Tên | Định nghĩa | Vai trò |
|---|---|---|---|
| 0 | Mơ hồ | chỉ khẩu hiệu/tính từ, không kiểm chứng được | → CTI |
| 1 | Hành động có tên | có hành động/công cụ/chương trình **có tên**, quy về chủ thể, nhưng không định lượng | → NAR |
| 2 | Định lượng | có đại lượng **đo được, quy về chính chủ thể** (không tính số của NHNN/quốc gia/chuẩn) | → QDR |

`is_specific = (mức ≥ 1)` chỉ dùng nội bộ; **CTI không dùng `is_specific`** mà dùng trực tiếp
phân bố mức (xem §3.2).

### 3.2 Ba chỉ số trên mỗi ô (bank `b`, year `y`, trụ `p`)
Denominator `N` = số **đơn vị cam kết có gắn trụ `p`** (cổng topic):

```
CTI(b,y,p) = #{commitment ở Mức 0} / N      # cheap talk / mơ hồ   (trục washing)
NAR(b,y,p) = #{commitment ở Mức 1} / N      # named-action rate    (vùng xám)
QDR(b,y,p) = #{commitment ở Mức 2} / N      # quantified-disclosure rate (substance)
```

- CTI + NAR + QDR = 1 theo định nghĩa.
- Mỗi chỉ số kèm **bootstrap CI 95%** (đã có `indices/bootstrap.py`).
- **Bỏ** `CTI_strict`, `grounded-CTI`, và toàn bộ band [loose, strict].

### 3.3 Selective disclosure (cherry-picking)
Phân bố lượng câu/đơn vị ESG theo trụ E/S/G trên mỗi (bank, year). Lệch mạnh = dấu hiệu né
trụ khó. (Đã có `indices/disclosure.py`.)

### 3.4 Phân tích phái sinh (tận dụng 3 mức — trả lời "phân tích được gì")
- **Substance gap** = (NAR + QDR thấp) bên cạnh lượng cam kết cao → "nói nhiều, cụ thể ít".
- **Hồ sơ truyền thông ngân hàng:** CTI thấp nhưng QDR cũng thấp (toàn Mức 1) = washing tinh
  vi "nói việc cụ thể nhưng né con số" — chỉ số nhị phân bỏ sót, thang 3 mức bắt được.
- **Xu hướng thời gian:** quỹ đạo CTI↓ / QDR↑ qua 2020–2024 cho biết ngân hàng có "thực chất
  hoá" cam kết theo thời gian không.

---

## 4. Pipeline sửa lại

```
Corpus (semantic units, ≤256 tok)
  └─> Topic E/S/G (PhoBERT)
        └─> Commitment (PhoBERT nhị phân, GATE bằng topic → denominator sạch)
              └─> Specificity (LLM-rubric 3 mức, attribution-aware)
                    └─> Tổng hợp: CTI / NAR / QDR + selective disclosure + bootstrap CI
                          └─> Validation: known-group · synthetic · sensitivity · audit tay
```

### 4.1 Thay đổi so với hiện tại
| # | Hạng mục | Hiện tại | Sửa thành |
|---|---|---|---|
| 1 | Đơn vị phân tích | chunk 256-token cắt cứng | **semantic unit**: gom câu liền kề cùng section, flush khi thêm câu vượt ~256 token, **luôn snap ranh giới câu** |
| 2 | Specificity → index | `is_specific` nhị phân + band | CTI = P(Mức 0); NAR = P(Mức 1); QDR = P(Mức 2) |
| 3 | Grounding / NLI / gCTI / evidence pool | đang chạy | **gỡ khỏi luồng chính** (archive code, không xoá) |
| 4 | Denominator CTI | commitment (gồm 42–49% non-ESG) | commitment **gate bằng topic** (chỉ câu gắn trụ) |
| 5 | Validation | có module, chưa chạy hệ thống | chạy đủ + báo cáo |

### 4.2 Semantic chunking (chi tiết)
- Đầu vào: `data/processed/sentences.parquet` (đã có, cùng `blocks.parquet` giữ ranh giới
  block/section).
- Gom câu liền kề **trong cùng block/section**; tích lũy token (đếm bằng tokenizer PhoBERT);
  **flush đơn vị khi câu kế tiếp sẽ làm vượt ngưỡng (mặc định 256, đặt cấu hình)**.
- Không bao giờ cắt ngang câu → PhoBERT không truncate; specificity-LLM nhận đơn vị mạch lạc.
- Giữ `bank, year, doc_id, unit_index, content_text, token_count` + danh sách câu thành phần.
- Ghi kèm thống kê: phân bố token/đơn vị, %đơn vị > 256 (kỳ vọng ≈ 0).

### 4.3 Cổng topic cho denominator
- Một đơn vị vào denominator CTI của trụ `p` **chỉ khi** `is_commitment=1` **và** `is_p=1`.
- Đơn vị commitment không gắn trụ nào → loại khỏi mọi CTI (ghi log tỉ lệ để minh bạch).

### 4.4 Attribution (cải thiện chất lượng Mức 2, không phải bước riêng)
- Giữ `attributable_to_actor` trong rubric; bổ sung guard tường minh: số gắn với
  NHNN/Chính phủ/toàn ngành/quốc gia/chuẩn (ISO, Basel, VIETGAP) → **không** tính Mức 2.
- Đây là cải thiện chất lượng prompt/rubric + `verify_rubric`, đo bằng audit tay (§5).

---

## 5. Đánh giá & validation

| Loại | Mục đích | Hiện trạng |
|---|---|---|
| **Known-group** | CTI phân biệt nhóm kỳ vọng khác nhau (vd ngân hàng có/không báo cáo bền vững riêng) | `validation/known_group.py` |
| **Synthetic manipulation** | bơm cam kết mơ hồ / thay số → CTI phải phản ứng đúng chiều | `validation/synthetic.py` |
| **Sensitivity** | ranking ổn định khi đổi threshold commitment, ngưỡng token | `validation/sensitivity.py` |
| **Digit-shortcut** | chứng minh rubric **không** chỉ học "có số → cụ thể" | `validation/digit_shortcut.py` |
| **Manual audit** | lấy mẫu n≥100 đơn vị/trụ, gán tay Mức 0/1/2 + cờ attribution; báo cáo agreement với pipeline | **mới — cần làm** |
| **Upper-bound EN** | eval model topic/commitment trên gold EN làm trần | có trong `experiments/` |

**Tiêu chí "đủ tốt":** (i) CTI per-pillar nằm dải hợp lý và **giải thích được** bằng audit;
(ii) known-group có hướng đúng; (iii) synthetic phản ứng đúng chiều; (iv) ranking ổn định
dưới sensitivity; (v) agreement audit↔pipeline ở mức chấp nhận được (báo cáo κ/accuracy).

---

## 6. Phạm vi & hạn chế (ghi minh bạch trong paper)

- **Translate-train domain gap:** gold EN dịch sang VI; commitment/specificity gốc từ dữ liệu
  *climate* áp cho cả S/G → domain shift; định lượng bằng upper-bound EN + audit VI.
- **Commitment classifier bắn rộng** (42–49% non-ESG) → giảm bằng cổng topic; phần dư là nhiễu
  còn lại, báo cáo tỉ lệ.
- **Lỗi LLM specificity:** attribution sai, sót tên riêng (Mức 0↔1) → đo bằng audit, cải thiện
  rubric; không tuyên bố hoàn hảo.
- **Không có walk thật:** substance là **proxy nội văn bản** (độ cụ thể), KHÔNG phải bằng
  chứng thực thi ngoài — scope tường minh; đây là lý do bỏ grounding thay vì giả vờ đo walk.
- **Nhiễu OCR** trong corpus thô.

---

## 7. Việc dọn dẹp & cập nhật tài liệu

- **Archive (không xoá):** `grounding/` (retriever, nli, support, evidence_pool), nhánh
  grounding trong `run.py`, `configs/grounding.yml` → chuyển sang `legacy/` hoặc gắn cờ
  `--enable-grounding` mặc định tắt, kèm README giải thích vì sao loại khỏi luồng chính.
- **Cập nhật `docs/de-cuong-nghien-cuu.md`** cho khớp thiết kế mới (specificity = LLM-rubric
  3 mức chứ không phải PhoBERT nhị phân; CTI = P(Mức 0); bỏ grounding/gCTI; semantic unit).
  Làm **sau** khi triển khai + đánh giá hiệu quả (theo yêu cầu: "cập nhật docs chuẩn" ở cuối).
- **Cập nhật `README.md`** (đang còn nói EWRI/neuro-symbolic — đã lỗi thời hoàn toàn).

---

## 8. Out of scope (lần này)

- Train lại commitment thành 3-way pledge/action (đã cân nhắc reframe pledge→action nhưng
  chọn hướng specificity-band cho gọn & trung thực).
- Thu thập walk-data bên thứ ba cho VN.
- Bất kỳ chỉ số nào tự đặt thang/trọng số ngoài tỉ lệ output classifier.
