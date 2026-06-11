# Spec 01 — Dữ liệu

## 1. Corpus mục tiêu (inference, end-to-end)

- Nguồn duy nhất: `data/extracted/raw_ocr_annual_report.zip` (59 file txt OCR, per bank/year;
  phủ rộng hơn corpus cũ — có thêm vib và các năm 2016–2019).
- **Không** dùng `data/legacy/corpus/*.parquet` cũ — build lại từ raw để pipeline tái lập 100%.
- Output chuẩn: `data/processed/sentences.parquet` với schema
  `doc_id, bank, year, section_id, block_id, sent_id, sentence, ctx_prev, ctx_next, block_type, section_title`.
- Tiền xử lý: NFC normalize, sửa lỗi OCR phổ biến, lọc câu rác (len < 10, header/footer,
  số trang, mục lục), khử trùng lặp exact + near-dup (minhash) **trong cùng doc**.
- Phạm vi phân tích chính của paper: 10 ngân hàng × 2020–2024 (đủ cả 10 bank);
  các năm/bank lẻ giữ lại làm phụ lục.

## 2. Gold EN → VI (translate-train) — đã có đủ trên đĩa

`data/source_dataset/{topic,subst}/` (EN gốc) và `data/translate/` (VI, đã dịch xong toàn bộ).

| Task | File | n | Nhãn | Vai trò |
|---|---|---|---|---|
| Topic E | environmental_2k | 2000 | env 0/1 (658+) | trụ E |
| Topic S | social_2k | 2000 | soc 0/1 (804+) | trụ S |
| Topic G | governance_2k | 2000 | gov 0/1 (538+) | trụ G |
| Commitment | commitments_actions train/test | 1000/320 | 0/1 (425+/98+) | mẫu số CTI |
| Specificity | specificity train/test | 1000/320 | 0/1 (399+/123+) | tử số CTI |
| Env claims | env_claims train/val/test | 2117/265/265 | 0/1 | auxiliary (E) |
| Action | action_500 | 500 | 0/1 (213+) | augment commitment cho S/G |

**Sự kiện then chốt đã xác minh:** `specificity` và `commitments_actions` chia sẻ
**đúng cùng tập văn bản** (1000 train / 320 test trùng 100%, không leak chéo split)
→ cho phép train **một mô hình multi-task 2 đầu** (commitment + specificity) trên cùng input,
nhất quán hơn 2 mô hình rời và đúng tinh thần Bingler 2022.

## 3. Chiến lược merge dữ liệu Topic (3 tập nhị phân → 1 mô hình multi-label)

Vấn đề: mỗi tập chỉ có nhãn cho **một** trụ; câu negative trong environmental_2k
có thể là positive của S/G (nhãn thiếu, không phải nhãn 0 cho trụ khác).

**Chiến lược chính — Masked BCE (partial-label learning):**
- Gộp 6000 dòng thành một bảng `text, env, soc, gov` trong đó 2 cột không có nhãn = NaN.
- Loss: BCE chỉ tính trên cột có nhãn (mask NaN). 3 đầu sigmoid độc lập trên PhoBERT.
- Đây là cách chuẩn cho multi-label với partial labels; không bịa nhãn.

**Chiến lược phụ — cross-pseudo-labeling (vòng 2, optional):**
1. Train 3 mô hình nhị phân per-pillar trên tập riêng.
2. Inference chéo: mô hình E dự đoán trên social_2k + governance_2k, v.v.
3. Chỉ nhận pseudo-label có confidence ≥ τ (mặc định 0.9) → điền bớt NaN → retrain masked BCE.
4. Báo cáo ablation: masked-only vs masked+pseudo.

**Augment positives cho E (optional):** câu positive của `env_claims` là E-positive
chắc chắn → thêm `env=1` (soc/gov NaN). Báo cáo ablation có/không augment.

> **Không dùng:** `netzero_reduction.csv` — đã loại khỏi thiết kế (quyết định 2026-06-10);
> giữ file trên đĩa nhưng không xuất hiện trong bất kỳ task/chỉ số nào.

## 4. Lọc nhiễu dịch máy

1. **QE filter:** chấm điểm cặp (EN, VI) bằng reference-free QE
   (`Unbabel/wmt22-cometkiwi-da` — lưu ý model gated, cần accept license HF;
   fallback: LaBSE cosine similarity giữa EN và VI). Bỏ đáy ~5–10% theo phân phối điểm.
2. **Confident Learning** (cleanlab): sau khi train sơ bộ, tìm dòng nghi sai nhãn
   (do dịch làm trôi nghĩa) → loại hoặc hạ trọng số.
3. Ablation bắt buộc: {không lọc, QE, QE+CL} × {zero-shot XLM-R, translate-train PhoBERT}.

## 5. VN human-eval set (BỔ SUNG so với đề cương — bắt buộc)

Mô hình train trên bản dịch nhưng **đánh giá cuối phải trên tiếng Việt thật**:
- Lấy mẫu phân tầng ~300 câu từ corpus ngân hàng (stratified theo bank × trụ dự đoán).
- Người gán (tác giả + 1 người nữa nếu có) gán 3 nhãn: topic E/S/G (multi),
  commitment 0/1, specificity 0/1. Đo Cohen's κ.
- Đây là test set chính cho claim về chất lượng transfer; gold EN test chỉ là upper-bound.
- Công cụ: file CSV + hướng dẫn gán nhãn trong `superpowers/specs/annotation-guideline.md` (viết ở T5).

## 6. Dữ liệu thu thập thêm (cho validation — spec 05)

`data/external/bank_signals.csv` — gán tay cho 10 bank × 2020–2024, các cột proxy walk:
- `has_assurance` — báo cáo có đảm bảo độc lập (third-party assurance) không.
- `gri_compliant` — lập theo chuẩn GRI không.
- `vnsi_member` — thuộc rổ VNSI (HOSE Sustainability Index) năm đó không.
- `green_credit_disclosed` — có công bố dư nợ tín dụng xanh (số liệu SBV/BCTN) không.
- `intl_initiative` — ký UNEP-FI / PCAF / Equator Principles không.
Nguồn: chính các BCTN + website HOSE/SBV; kỳ vọng known-group: nhóm "có tín hiệu thật"
có CTI thấp hơn nhóm không.

## 7. Dữ liệu bổ sung đã khảo sát (kết quả search 2026-06-10)

### 7.1 ML-Promise / SemEval-2025 Task 6 (PromiseEval) — ĐÃ TẢI, đã kiểm kê

Seki et al. 2024 (arXiv 2411.04473; EMNLP 2025). File: `data/source_dataset/ml_promise/Trainset_*.json`
(license CC BY-NC-SA 4.0 — paper xuất bản phải ghi nguồn, dataset đóng gói lại không được thương mại).
Lưu ý kỹ thuật: JSON có BOM → đọc bằng `utf-8-sig`.

**Kiểm kê thực tế (2026-06-10) — tổng 2.110 mẫu, nhưng KHÔNG phải tất cả dùng được:**

| Lang | n | Text? | Promise Y/N | Dùng được |
|---|---|---|---|---|
| English | 400 | `data` (đoạn văn, median ~717 ký tự) | 313/87 | ✅ 400 |
| French | 400 | `data` đủ 400 | 319/81 | ✅ 400 |
| Japanese | 400 | `data` đủ 400 + `promise_string`/`evidence_string` + **`ESG_type` E/S/G (134/145/121)** | 358/41 | ✅ 400 |
| Chinese | 410 | KHÔNG có `data`; chỉ `promise_string` khi promise=Yes | 146/264 | ⚠️ 146 (chỉ positive) |
| Korean | 500 | **KHÔNG có text** (chỉ metadata + nhãn + URL/trang PDF) | 379/121 | ❌ 0 |

→ **Dùng được có văn bản: ~1.346 mẫu** (EN 400 + FR 400 + JA 400 + ZH 146 positive).
Dịch hết sang VI → **~1.346 mẫu VI** (KO chỉ cứu được nếu re-extract PDF từ URL — không đáng công).
Khuyến nghị dịch: **EN + FR + JA (1.200 mẫu đủ nhãn 2 chiều)**; ZH 146 chỉ thêm positive,
làm lệch class balance → để tùy chọn.

**Nhãn & cách dùng:**
- `promise_status` Y/N → augment đầu **commitment** của M2 (full ESG — vá khoảng trống S/G).
  JA có sẵn `ESG_type` → eval commitment **tách trụ** + augment topic (400 dòng có nhãn trụ).
- `evidence_status` Y/N (EN: 221 Yes/179 No) → **gold đánh giá lớp grounding** (spec 03 §5):
  đây là nhãn "trong cùng trang/đoạn có bằng chứng đỡ không" — trùng khớp định nghĩa
  intra-report substantiation của ta.
- `evidence_quality` {Clear, Not Clear, Misleading}: **Misleading cực hiếm (EN: 4 mẫu)**
  → không train/eval riêng lớp Misleading; chỉ dùng Clear vs Not-Clear nếu cần.
- `verification_timeline`: giá trị KHÔNG đồng nhất giữa ngôn ngữ ('Already' vs 'within_2_years',
  có cả giá trị thừa khoảng trắng '2 to 5 years ') → cần bảng map chuẩn hoá khi load.
  Vai trò: descriptive cho case study, không vào chỉ số.

**Cảnh báo phân phối:** tỉ lệ promise lệch mạnh theo ngôn ngữ (EN 78% Yes, ZH 36% Yes)
do cách sampling khác nhau → khi gộp phải stratify theo nguồn ngôn ngữ; mẫu là
**đoạn văn** (paragraph-level) — cùng granularity với commitments_actions/specificity
của ClimateBERT, nên gộp hợp lệ.

**Tham chiếu hiệu năng (SemEval-2025 Task 6):** hệ tốt nhất của CLaC (2505.23538) chỉ đạt
~0.53 (private LB) với DeBERTa-v3-large + attention pooling; bài học rút ra:
(a) multitask promise+evidence chia sẻ biểu diễn giúp tăng điểm — củng cố thiết kế
multi-task M2 của ta; (b) encoder fine-tune trên 400 mẫu là đủ baseline nhưng
class imbalance là vấn đề dai dẳng → pos_weight (spec 02 §3) là bắt buộc;
(c) ESG-BERT domain-pretrained đã bắt được pattern — với VN ta không có tương đương,
chấp nhận PhoBERT general. Repo tham khảo: github.com/CLaC-Lab/SemEval-2025-Task6
(ESG-BERT multi-head + DeBERTa attention pooling).
Lưu ý: repo `kinit-sk/semeval_2025` user gửi là **task khác** (fact-checked claim retrieval) — bỏ qua.

### 7.2 Dùng gián tiếp / không dùng

| Nguồn | Bản chất | Quyết định |
|---|---|---|
| `ESGBERT/{environment,social,governance,base}_data` | corpus KHÔNG nhãn (1.8–13.8M câu) để pretrain ESGBERT | chỉ dùng nếu làm domain-adaptive pretraining (optional, ngoài scope chính) |
| `yiyanghkust/finbert-esg(-9-categories)` | model train trên 2k/14k câu gán tay (data không công bố) | dùng **model** làm weak cross-check phân phối topic trên EN; không có data để dịch |
| `cea-list-ia/ESG-classification-en`, `ESG-classification-fr-en` | gated | giữ trạng thái "không dùng" như README en_gold |
| `FinanceMTEB/ESG`, các bộ sentiment/NER ESG khác | khác task (retrieval/sentiment/NER) | không dùng |

Kết luận: ngoài ML-Promise, **không tồn tại** bộ commitment/specificity riêng cho trụ S/G
ở mức câu — khoảng trống này là thật và chính là đóng góp dataset của nghiên cứu
(translate-train + VN human-eval set). Chiến lược chính cho S/G gap vẫn là
`action_500` + ML-Promise augment + eval tách trụ (spec 02 §2).
