# Thiết kế: Khung đo ESG cheap-talk nhận-biết-độ-tin-cậy qua trích xuất thuộc tính

Ngày: 2026-06-24
Nhánh: washing-depth
Trạng thái: đã duyệt các quyết định lõi, chờ duyệt spec

---

## 1. Vấn đề cần giải quyết

Phản hồi về paper hiện tại:
1. **Phương pháp nông / đóng góp mờ** — pipeline bị nhìn nhận là "fine-tune 2 BERT + prompt 1 LLM",
   không nói rõ *xây mô hình như thế nào*, *cải tiến gì*.
2. **CTI dựa trên phán đoán chủ quan nhất** — nhãn "mơ hồ" (nền của CTI) có IAA κ=0.12 giữa người;
   nhãn "cam kết" κ=0.18. Người gán nhãn tự mâu thuẫn → nguy cơ đánh đổ kết luận ESG-washing cuối.
3. **Gate commitment dựng trên nhãn mâu thuẫn** — lo ngại train/đo trên nền không tin cậy.

Mục tiêu (người dùng chốt): giải quyết **đồng thời** cả độ sâu phương pháp lẫn tính hợp lệ kết quả,
bằng **một khung thống nhất**.

## 2. Bằng chứng quyết định (đã có, không phải giả thuyết)

Đã gán lại 400 chunk bởi **hai người độc lập, mù** (`data/gold_annot_{1,2}_relabeled.xlsx`) theo
schema atomic. Kết quả IAA:

| Trường | κ cũ (rubric tổng thể) | κ mới (atomic) | số ô A≠B /400 |
|---|---|---|---|
| Cam kết (`co_cam_ket`) | 0.18 | **0.984** | 3 |
| Có hành động (`co_hanh_dong_ten`) | — | **1.000** | 0 |
| Có định lượng (`co_so_dinh_luong`) | 0.53 | **0.994** | 1 |
| Quy về ngân hàng (`quy_ve_bank`) | — | **0.994** | 1 |
| Có mốc thời gian (`co_moc_tg`) | — | **1.000** | 0 |
| → Cam kết suy ra (`g_is_commit`) | 0.18 | 0.984 | (suy ra) |
| → Mức cụ thể suy ra (`g_spec_level`, QWK) | 0.42 | ~1.0 | (suy ra) |

**Hai phép soi đã chạy (giữ tính trung thực):**
- `g_is_commit` ≈ `co_cam_ket` (98%), `g_spec_level` khớp 100% luật từ atomic (n=238) →
  hai cột này là **suy ra, không phải bằng chứng độc lập**. Bằng chứng thật = κ của 5 cờ atomic.
- **Lập luận tương phản (đắt nhất):** cùng hai người đó, cùng 400 chunk, vẫn lệch **105 ô** ở
  topic Xã hội (κ=0.47). Atomic lệch 0–3 ô, topic lệch 105 ô → người gán **độc lập thật**, và
  câu hỏi atomic *bản chất* ít mơ hồ hơn. Trả lời thẳng "sao κ nhảy 0.18→0.98": không phải người
  giỏi lên, mà rubric cũ hỏi sai câu hỏi.

## 3. Đóng góp / tường thuật

> *Vấn đề không phải con người không đáng tin, mà rubric tổng thể hỏi sai câu hỏi.*
> Tháo phán đoán chủ quan ("có phải cam kết?", "có mơ hồ?") thành các câu hỏi yes/no **khách quan,
> có bằng chứng**, rồi **tái dựng chỉ số bằng luật tất định**. Khung đo trở nên **kiểm chứng được,
> tái lập được, và có IAA cao**.

Ba nỗi lo được gỡ:
- *Nông* → decomposition + tái dựng bằng luật + battery ablation là method có thật, được mô tả rõ (§4–§7).
- *CTI chủ quan* → "mơ hồ" = **vắng mặt** đặc trưng khách quan, có κ atomic chứng minh.
- *2 gold mâu thuẫn* → sau decomposition chỉ còn 0–3 ô lệch; mâu thuẫn về cơ bản biến mất.

## 4. Kiến trúc mô hình & cách xây (phần phải mô tả RÕ — vá chỗ "sơ sài")

Pipeline 4 tầng nối tiếp. Mỗi tầng nêu: backbone + lý do, dữ liệu + cách dựng nhãn, giao thức
huấn luyện, và **cải tiến** so với baseline ngây thơ.

### 4.1 Tầng Topic Classifier (E/S/G)
- **Backbone:** PhoBERT (pretrained tiếng Việt) — lý do: corpus đích là báo cáo tiếng Việt;
  PhoBERT > mBERT/XLM-R trên tác vụ tiếng Việt do tiền-huấn-luyện chuyên ngữ + tách từ.
- **Kiến trúc:** 3 đầu sigmoid độc lập (đa nhãn, không softmax) — một chunk có thể vừa E vừa S.
- **Nhãn & dữ liệu:** translate-train. Gom nhiều nguồn EN chuyên gia (ESGBERT,
  environmental-claims) → dịch máy sang VI → fine-tune. **Cải tiến dữ liệu:** hòa hợp taxonomy
  khác nhau giữa các nguồn về chung E/S/G; bổ sung nguồn để **tăng phủ trụ S/G** (vốn ít hơn E).
- **Cải tiến huấn luyện — masked BCE cho nhãn thiếu:** nguồn dữ liệu chỉ gán một phần nhãn
  (vd chỉ có nhãn E). Loss **chỉ tính trên cột không-NaN** (`trainer.masked_bce_loss`), nên tận
  dụng được dữ liệu gán-một-phần thay vì bỏ. Đây là điểm kỹ thuật cần nêu trong paper.

### 4.2 Tầng Commitment Classifier
- **Backbone:** PhoBERT, 1 đầu sigmoid nhị phân.
- **Nhãn & dữ liệu:** translate-train từ ClimateBERT commitments-actions, ESGBERT action-500,
  ML-Promise (đa ngữ) — ML-Promise thêm vào để **tăng phủ cam kết S/G**.
- **Lưu ý hợp lệ:** classifier này **không** train trên 2 gold của ta (gold chỉ để đánh giá).
  Nỗi lo "train trên nhãn mâu thuẫn" không áp dụng cho tầng này.

### 4.3 Cổng ESG-commitment (composition)
- Chunk chỉ giữ khi **đồng thời** là cam kết **và** thuộc ≥1 trụ E/S/G. Quyết định thiết kế
  pipeline để cô lập đúng đơn vị đo (loại "cam kết tăng lợi nhuận" và "câu ESG mô tả thuần").

### 4.4 Tầng Attribute Extractor (lõi mới — thay rubric 0/1/2)
- **Backbone:** small instruct-LLM (Qwen3) — lý do dùng LLM thay encoder: phán đoán "hành động có
  được định lượng và quy về chủ thể chưa" cần **suy luận ngữ cảnh nhiều câu**, khó học ổn định từ
  vài nghìn nhãn; và encoder dễ **bắt shortcut "có chữ số → specific"** (xác nhận ở §7 ablation).
- **Không train** — LLM trích xuất + **luật tất định** suy nhãn. Gold dùng để *validate*, không
  để fine-tune → tránh hẳn vòng lặp "train trên nhãn chủ quan".
- **Cải tiến chính (đóng góp phương pháp):**
  1. *Decomposition khách quan:* không hỏi LLM nhãn tổng thể; hỏi 5 câu yes/no atomic (§5).
  2. *Tái dựng bằng luật tường minh:* nhãn mức = hàm tất định của 5 cờ → cùng input ra cùng output,
     truy được về câu chữ.
  3. *Grounding bằng evidence span:* mỗi cờ "yes" phải kèm chuỗi con của chunk làm bằng chứng.
  4. *Chống bịa (GIỮ NGUYÊN, đã có `verify_rubric`):* huỷ figure không xuất hiện trong text; huỷ
     baseline/timeline nếu không có năm/"so với"; salvage JSON hỏng; retry → mặc định an toàn
     (Mức 0) thay vì thổi phồng. **Thêm:** evidence span phải là substring của chunk, nếu không
     → cờ về 0.

## 5. Schema atomic & luật tái dựng (dùng CHUNG cho người và LLM)

5 cờ nhị phân cấp chunk (khớp đúng cột gold đã gán):

| Mã | Cột gold | Câu hỏi yes/no |
|---|---|---|
| A0 | `co_cam_ket` | Có ý cam kết / hướng tương lai (sẽ, cam kết, hướng tới, mục tiêu)? |
| A1 | `co_hanh_dong_ten` | Có hành động/chương trình/công cụ **có tên, kiểm chứng được**? |
| A2 | `co_so_dinh_luong` | Có số/chỉ tiêu định lượng? |
| A3 | `quy_ve_bank` | Số/hành động quy về **chính ngân hàng**? |
| A4 | `co_moc_tg` | Có mốc thời gian/deadline? |

Luật tất định:
```
commit = A0 AND (env OR soc OR gov)
level  = 2 nếu (A2 AND A3);  1 nếu A1;  0 nếu còn lại     (chỉ tính khi commit)
CTI = n0/N;  NAR = n1/N;  QDR = n2/N
```
- **A4 = biến phụ trợ** (đo độ "đặt cọc" của cam kết), **không** vào CTI/NAR/QDR. [đã chốt]

## 6. Viết lại LLM extractor (`src/esgwash/models/specificity_llm.py`)

- Đổi đầu ra thành **5 cờ atomic cấp chunk + evidence span** (cho phép phân rã item nội bộ rồi
  gộp lên chunk-level theo "max" để giữ logic phân rã đã có).
- `derive()` đổi để ăn 5 cờ thay vì item-attrs cũ; ánh xạ thẳng sang luật §5.
- Giữ greedy decoding (tái lập), giữ toàn bộ guard chống bịa, thêm kiểm tra evidence-substring.
- Tương thích `eval_gold.py` và `run.classify_chunks` (cùng tên cột đầu ra: `spec_level`,
  `is_commitment` suy từ A0∧ESG).

## 7. Validation battery + ablation (đối-trị "sơ sài")

1. **IAA atomic** (✅ đã có số) — bảng §2, trung tâm mục reliability.
2. **LLM vs người** — chạy extractor trên 400 chunk gold (Kaggle), so từng cờ atomic với consensus
   người (κ, P/R/F1). Gold giờ sạch nên là thước thật.
3. **Ablation chứng minh lựa chọn thiết kế đáng giá** (mỗi cái 1 con số trong paper):
   - *Encoder digit-shortcut:* TF-IDF/LR & encoder-only dự đoán specificity → cho thấy bắt shortcut
     "có số → Mức 2", sai ở chunk có số nền/quốc gia (dùng `baselines.py`, `spec_features.py`).
   - *Holistic-LLM vs decomposed-LLM:* hỏi LLM thẳng 0/1/2 so với 5-cờ+luật → decomposed khớp người
     tốt hơn / ổn định hơn.
   - *Có vs không guard chống bịa:* đo tỉ lệ figure bịa bị `verify_rubric` chặn.
   - *Sentence-level vs chunk-level:* minh hoạ vì sao chọn chunk (specificity trải nhiều câu).
4. **Tái dựng corpus** — chạy toàn bộ → CTI/NAR/QDR mới; báo cáo chênh so với số cũ.
5. **Sensitivity nhẹ** — đổi ngưỡng commit (A0); cho thấy **thứ hạng ngân hàng** + phát hiện
   **gov-cherry-picking** ổn định. Giữ bootstrap CI hiện có.

## 8. Thay đổi trong paper (`docs/paper/main_vi.tex`)

- §Methodology: mở rộng mô tả từng tầng theo §4 (backbone+lý do, dữ liệu+nhãn, masked-BCE, Optuna
  TPE+pruning, tách tune/test, multi-seed). Liệt kê cải tiến rõ ràng.
- §Specificity rubric → §Attribute extraction: 5 cờ + luật + grounding + guard.
- §Độ tin cậy nhãn vàng: viết lại quanh **phép nhảy κ** + **lập luận tương phản S/G**; bỏ lập luận
  "systematic offset cancels" cũ.
- Thêm bảng **LLM-vs-người** và bảng/đoạn **ablation**.
- Diễn giải lại CTI/NAR/QDR: "đo được, kiểm chứng được" thay vì "chỉ so sánh tương đối".

## 9. Phạm vi (YAGNI)

**Trong:** căn schema 5 cờ; viết lại extractor + guard substring; validation §7 (1–5); viết lại
§Methodology + §reliability + §attribute extraction; bảng LLM-vs-người + ablation.

**Ngoài (lần này):**
- **Dawid–Skene / mô hình rater nhiễu — BỎ HẲN** [đã chốt] (chỉ còn 0–3 ô lệch, vô ích).
- **Chữa topic S/G — KHÔNG.** Thừa nhận **limitation**: κ_S=0.47, κ_G=0.53; nêu ảnh hưởng tới
  share-by-pillar; đề xuất hướng tương lai (decompose topic). [đã chốt]
- Train mô hình mới cho atomic (không cần — LLM + luật + gold-để-validate là đủ).

## 10. Rủi ro & giảm thiểu

- *LLM-vs-người thấp hơn kỳ vọng:* nếu LLM lệch người ở A1/A3, tinh chỉnh prompt/few-shot (đã có
  cơ chế); guard giữ mặc định an toàn. Báo cáo trung thực dù số xấu.
- *Tái dựng corpus đổi CTI nhiều so với số cũ:* coi là kết quả (số cũ kém tin cậy hơn), giải thích
  bằng việc bỏ shortcut; không ép khớp số cũ.
- *Kaggle thời lượng:* extractor chạy 400 chunk (validate) rẻ; full corpus theo mẻ như hiện tại.
