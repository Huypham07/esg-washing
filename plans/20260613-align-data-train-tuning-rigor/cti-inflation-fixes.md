# Khắc phục CTI inflation — chẩn đoán + menu cách sửa (research 2026-06-13)

> Nối tiếp `cti-evaluation-and-framing.md` + `results-train-classifiers.md`.
> Mục tiêu: giảm CTI thổi phồng sao cho **phòng thủ được trong luận văn** (citable, đo được).
> Research: quantification (Forman/ACC, Saerens SLD, QuaPy), calibration (Guo temperature scaling),
> domain-transfer (Bingler 2022, Gururangan DAPT, Louis&Nenkova specificity).

## 1. Chẩn đoán — 2 NGUỒN inflation tách bạch (đừng gộp)

CTI = #{commit=1 & spec=0} / #{commit=1}. Cao = washing. Số đã verify:
- specificity test (climate dịch): TP=81 FN=40 FP=14 TN=179 → **TPR(cụ thể)=0.67, FPR=0.073**, macro-F1 .808.
- corpus 5.319 commit, **18.0% gán cụ thể** (E 21.3% · S 17.1% · **G 6.9%**); p_spec median 0.142, mean 0.263.

**Source A — bias bất đối xứng lỗi classifier** (recall 0.67 < precision 0.85): sót 1/3 câu cụ thể → đếm dư
"mơ hồ" → CTI lệch CAO. Đây là bias *in-domain*, trên test climate ≈ +8pt (predicted vague 69.8% vs true 61.5%).
→ Sửa được bằng **quantification (ACC/PACC)** HOẶC **soft-CTI + calibration**.

**Source B — thất bại transfer domain/construct** (climate → bank VN, nặng nhất S/G): classifier học đặc trưng
cụ-thể-khí-hậu (tấn CO2, %, mốc năm), KHÔNG nhận ra cụ-thể-quản-trị (tên uỷ ban, số thành viên, tần suất kiểm
toán). G 6.9% = phần lớn artifact. Đây là *covariate/concept shift*, KHÔNG phải nhiễu nhãn.

### 1.1 Bằng chứng then chốt — vì sao Source B mới là thủ phạm
Ráp ACC `p_true=(p_obs−FPR)/(TPR−FPR)=1.676·p_obs−0.1215` (matrix climate) lên corpus:

| Trụ | p_obs (cụ thể) | ACC p_true (cụ thể) | Ghi chú |
|---|---|---|---|
| E | 0.213 | 0.235 | sửa nhẹ |
| S | 0.171 | 0.165 | ~không đổi |
| **G** | **0.069** | **< 0 (clip 0)** | 🔴 p_obs < FPR 0.073 → prevalence ÂM |
| Tổng | 0.18 | ~0.18 | trùng fixed point → KHÔNG đổi |

→ 2 kết luận đắt giá (đưa thẳng vào luận văn):
1. **G observed-cụ-thể (6.9%) ≤ FPR classifier (7.3%)** → tín hiệu "cụ thể" của G nằm TRONG biên nhiễu
   false-positive → **instrument không tạo signal dùng được cho Governance** (cần Phase 04 xác nhận FPR target).
2. ACC-với-matrix-climate **gần như không sửa** corpus → inflation KHÔNG do Source A. Là covariate shift
   (Source B) → **giả định "prior-shift-only" của ACC bị vi phạm** → BẮT BUỘC đo TPR/FPR trên domain đích.

**Hệ quả lớn:** không có fix "miễn phí". Mọi de-bias đáng tin đều cần **nhãn target per-pillar (Phase 04)**.
Việc này vừa xác nhận Phase 04 là cổng, vừa ĐỊNH LẠI Phase 04 phải thu thêm gì (xem F1.1).

## 2. Menu cách sửa (4 nhóm × 11 cách)

### Nhóm 1 — Đo lại trên domain đích (CỔNG — mở khoá mọi fix Nhóm 2/3)
- **F1.1 — Phase 04: per-pillar confusion matrix trên bank VN thật.** Không chỉ "verify F1" mà thu đủ để ước
  lượng **TPR/FPR specificity riêng từng trụ E/S/G**. 🔴 **Refinement bắt buộc:** *stratified/active sampling
  theo p_spec* — random 100 câu G chỉ ~7 câu cụ thể → KHÔNG ước lượng nổi TPR. Phải oversample câu p_spec cao
  + biên 0.3–0.7 để có đủ POSITIVE mỗi trụ (≥~30 cụ thể/trụ). Đây là input sống còn cho F2.1, F3.1.

### Nhóm 2 — De-bias chỉ số (thống kê, KHÔNG sửa model)
- **F2.1 — Quantification / ACC (+PACC) per pillar [PRIMARY].** Dùng TPR/FPR target từ F1.1, hiệu chỉnh
  prevalence vague mỗi trụ. Small-n (bank×pillar ~30): ước lượng matrix ở mức **pooled-per-pillar** (1 bộ
  TPR/FPR/trụ), bootstrap CI per group + resample matrix ~Beta(TP+1,FN+1)/Beta(FP+1,TN+1). PACC nếu p_spec
  đã calibrate (F2.2). Lib: QuaPy 0.2.0. **Lưu ý:** với G, ACC có thể ra out-of-range → chính nó = phát hiện
  "không đo được G", không phải lỗi code. Citable: Forman 2008, González/Sebastiani survey, Hopkins&King 2010.
- **F2.2 — Soft-CTI + temperature scaling [RẺ, LÀM NGAY].** Thay đếm-ngưỡng bằng `CTI=mean(1−p_spec_cal)`.
  Bỏ ngưỡng 0.5 tuỳ tiện, giữ thông tin xác suất. p_spec đã lưu sẵn trong enriched parquet. Temperature T fit
  trên val (149) bằng NLL/LBFGS; đo ECE+Brier trước/sau. Số minh hoạ: soft vague ≈ 0.74 vs hard 0.82 (kéo
  xuống ~8pt, bớt brittle). ⚠️ Chỉ chữa Source A + ngưỡng — **KHÔNG chữa transfer** (T fit trên climate-val).
  Lib: torch LBFGS + torchmetrics BinaryCalibrationError. Citable: Guo 2017.

### Nhóm 3 — Sửa classifier tận gốc (Source B; cần nhãn F1.1 trừ F3.3)
- **F3.1 — Per-pillar threshold / Platt recalibration [rẻ nhất].** Dịch ngưỡng/curve riêng từng trụ trên
  300 nhãn Phase 04. Không retrain. +~0.05–0.08 F1 S/G. An toàn small-n hơn fine-tune.
- **F3.2 — Few-shot fine-tune target (SetFit / tiếp tục fine-tune).** +~0.08–0.12 F1 S/G. n≈40/trụ là biên —
  cần protocol nghiêm (CV, freeze encoder phần lớn). Sửa được transfer thật sự nếu đủ nhãn.
- **F3.3 — TAPT/DAPT: tiếp tục MLM trên corpus bank VN trước fine-tune.** Sửa encoder tận gốc, **không cần
  nhãn** cho bước MLM (chỉ cần text bank — đã có 60k câu). +~0.04–0.08. Citable: Gururangan 2020.
- **F3.4 — Re-operationalize specificity PER PILLAR (định nghĩa/luật/lexicon riêng) [đóng góp method lớn nhất].**
  Cụ-thể-G = tên uỷ ban/số thành viên/tần suất họp; cụ-thể-S = nhóm hưởng lợi/% nhân sự/tên chương trình. Công
  nhất (curate lexicon + có thể gán lại). Hợp làm "đóng góp mới" của luận văn.
- **F3.5 — LLM-as-judge re-label/cross-check target (few-shot).** Gemini/GPT gán specificity trên bank VN làm
  silver mới HOẶC cross-check nhãn người (đo κ). Rẻ, dùng để mở rộng nhãn hoặc kiểm định, KHÔNG làm ground-truth chính.

### Nhóm 4 — Phạm vi & framing (gần như free, LUÔN làm)
- **F4.1 — Giới hạn claim chính ở trụ E**, báo S/G = *reference only* + khai báo measurement error. Có tiền lệ
  (Bingler scope climate-only). "Rõ phạm vi" > "chính xác giả". 
- **F4.2 — Báo cáo raw vs ACC-corrected vs soft cạnh nhau** + bootstrap CI + ranking tương đối (đã có). Minh bạch độ lệch.
- **F4.3 — Phase 06 validation:** synthetic monotonicity (trộn câu mơ hồ → CTI tăng đơn điệu; bắt buộc, code được)
  + known-group (gán `bank_signals.csv`; mạnh nhất; chờ bàn đồng nghiệp).

## 3. Khuyến nghị — lộ trình theo tầng

| Tier | Khi nào | Cách | Cần nhãn? | Payoff |
|---|---|---|---|---|
| **0** | NGAY (vài giờ) | F2.2 soft-CTI+temperature · F4.1 E-first · F4.2 báo cạnh raw | Không | sanity + bớt brittle ngưỡng, +1 sensitivity, framing an toàn |
| **1** | CỔNG (Phase 04) | F1.1 per-pillar confusion matrix, **stratified sampling** | 300 (stratified) | mở khoá de-bias thật; tách thật-vs-artifact |
| **2** | sau P04 (rẻ, payoff cao) | F2.1 ACC per pillar · F3.1 per-pillar recalibrate · F4.3 synthetic | dùng nhãn P04 | CTI de-biased + validate khoa học |
| **3** | nếu cần cứu S/G & có giờ | F3.2 few-shot FT *hoặc* F3.3 TAPT; F3.4 re-operationalize nếu muốn đóng góp method | thêm nhãn | sửa transfer tận gốc |

**Đường tối thiểu phòng-thủ-được:** Tier 0 + Tier 1 + (F2.1 ACC + F4.3 synthetic). Đủ để nói "đã de-bias +
validate, và trung thực về trụ G". Tier 3 chỉ khi muốn cứu S/G hoặc tăng đóng góp.

## 4. Câu hỏi cần bạn quyết
1. **Phạm vi claim:** chấp nhận "E là chính, S/G reference" (F4.1) hay BẮT BUỘC cứu G (kéo theo Tier 3 + nhiều nhãn)?
2. **Ngân sách nhãn Phase 04:** 300 đủ NẾU stratified per-pillar; nếu muốn ACC tin cậy cho cả 3 trụ, G cần
   oversample mạnh (G-cụ-thể hiếm). Chấp nhận stratified (lệch khỏi phân bố tự nhiên) để ước lượng được TPR không?
3. **Validation:** synthetic-only (đủ tối thiểu) hay + known-group (cần gán `bank_signals` — đang chờ đồng nghiệp)?
4. **Đóng góp method:** có muốn làm F3.4 (re-operationalize per-pillar) như đóng góp mới, hay giữ Bingler-faithful + de-bias là đủ?

## 5. Chưa giải quyết
- TPR/FPR specificity trên domain bank VN thật (per pillar) = ?? — chỉ Phase 04 trả lời; mọi số ACC ở §1.1 dùng
  matrix climate, chỉ minh hoạ tính bất khả thi, KHÔNG phải số cuối.
- commitment precision 0.64 có lệch theo trụ không (mẫu số CTI)? — chưa đo per-pillar.
- Soft-CTI có inherit transfer bias tới mức nào (T fit climate-val)? — cần ECE đo trên subset bank có nhãn (P04).
- mDeBERTa-xnli VN ~78% zero-shot → rủi ro Phase 07 grounded-CTI (NLI cross-lingual) — kiểm trước khi làm P07.
