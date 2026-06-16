# Đề cương nghiên cứu — Đo lường ESG-washing trong báo cáo của ngân hàng thương mại Việt Nam

> Xây dựng hệ thống NLP đo *khoảng cách giữa lời nói và việc làm* (talk vs walk) trong báo cáo bền vững của 10 ngân hàng thương mại Việt Nam

---

## 1. Bài toán

**ESG** là viết tắt của ba trụ cột phát triển bền vững: Môi trường (Environmental), Xã hội (Social), Quản trị (Governance). Hằng năm các ngân hàng phát hành báo cáo với rất nhiều cam kết kiểu "hướng tới phát thải ròng bằng 0", "đẩy mạnh tín dụng xanh", "tăng cường bình đẳng giới".

**ESG-washing** là hiện tượng **nói nhiều hơn làm**: cam kết nghe hay nhưng **mơ hồ, không số liệu, không hành động cụ thể** — mang tính hình thức thay vì thực chất. Về mặt lý thuyết đây là **decoupling** giữa *talk* (điều được nói) và *walk* (điều thực sự cam kết và chứng minh được).

**Mục tiêu:** từ văn bản báo cáo, **tự động phát hiện và đo mức độ** cam kết suôn theo thang 3 mức độ cụ thể hoá.

---

## 2. Vấn đề nghiên cứu

1. Hầu hết nghiên cứu NLP về washing chỉ làm tiếng Anh, chỉ trụ Môi trường. Chưa có nghiên cứu **full E/S/G cho ngân hàng Việt Nam**.
2. **Không có "walk data" thật cho VN.** Không thể đối chiếu cam kết với phát thải/đánh giá bên thứ ba như ở nước ngoài → đo "walk" **ngay trong văn bản** qua thang specificity 3 mức, không dựa vào dữ liệu ngoài.
3. Grounding (retrieval + NLI xác nhận bằng chứng nội văn bản) là **circular construct**: bằng chứng lấy từ cùng tài liệu với cam kết nên không thể xác nhận "walk" thực tế → đã loại bỏ (xem `legacy/README.md`).

---

## 3. Dữ liệu

### 3.1 Corpus báo cáo ngân hàng VN

| Nguồn | Mô tả |
|---|---|
| `source_data/raw_ocr_annual_report.zip` | Văn bản OCR sạch 50 báo cáo thường niên, 10 ngân hàng × 5 năm (2020–2024) |

**10 ngân hàng:** Agribank, BIDV, BSC, MBBank, OCB, SHB, Techcombank, Vietcombank, Viettinbank, VPBank.

### 3.2 Nhãn huấn luyện (translate-train EN→VN)

| File | Dòng | Nhãn | Vai trò | Nguồn |
|---|---|---|---|---|
| `environmental_2k.csv` | 2 000 | E ∈{0,1} | topic | ESGBERT (Mehra 2022) |
| `social_2k.csv` | 2 000 | S ∈{0,1} | topic | ESGBERT |
| `governance_2k.csv` | 2 000 | G ∈{0,1} | topic | ESGBERT |
| `commitments_actions.{train,test}.parquet` | 1 000/320 | commitment ∈{0,1} | mẫu số CTI | ClimateBERT |

---

## 4. Phương pháp

### 4.1 Pipeline tổng quát

```
Báo cáo (text zip)
    │
    ▼
[Chunking] semantic-text-splitter, ≤256 token/chunk
    │   → data/chunks.parquet  (13 812 chunks · p50=228 tok)
    ▼
[Topic] PhoBERT fine-tune — multi-label E/S/G
    │   → is_env, is_soc, is_gov
    ▼
[Commitment] CommitmentHF — nhị phân
    │   → is_commitment  (mẫu số CTI)
    ▼
[Specificity] Qwen3 LLM rubric — 3 mức
    │   → spec_level 0/1/2  (tử số CTI/NAR/QDR)
    ▼
[Indices] CTI / NAR / QDR per (bank, year)
         + Pillar share (selective disclosure)
```

### 4.2 Các thành phần

| # | Thành phần | Mô hình | Vai trò |
|---|---|---|---|
| ① | **Topic E/S/G** | `vinai/phobert-base` fine-tune, 3 sigmoid | Gán trụ ESG cho chunk; gate denominator CTI |
| ② | **Commitment** | `dqa2412/esg-washing-optimized` (CommitmentHF) | Xác định chunk là cam kết ESG |
| ③ | **Specificity** | `Qwen/Qwen3-1.7B` LLM rubric | Chấm spec_level 0/1/2 cho từng cam kết |

### 4.3 Thang specificity 3 mức

| Level | Tên | Định nghĩa | Ví dụ |
|---|---|---|---|
| 0 | Vague | Cam kết mơ hồ, không hành động/công cụ cụ thể | "hướng tới phát triển bền vững" |
| 1 | Named action | Nêu hành động/công cụ có tên, nhưng chưa có số liệu | "triển khai hệ thống quản lý carbon" |
| 2 | Quantified | Có con số định lượng, quy về chủ thể ngân hàng | "giảm phát thải 23% so với 2023" |

---

## 5. Chỉ số đo lường

### 5.1 CTI / NAR / QDR

Cho mỗi **(bank, year)**, denominator = tập hợp **unique** chunk cam kết có ít nhất 1 trụ ESG dương:

```
CTI = #{spec_level = 0} / #{ESG commitment chunks}   ← washing signal
NAR = #{spec_level = 1} / #{ESG commitment chunks}   ← named but unquantified
QDR = #{spec_level = 2} / #{ESG commitment chunks}   ← substance
```

**CTI + NAR + QDR = 1.** Không phân theo trụ để tránh double-count (chunk đa trụ). Kết quả lưu tại `outputs/cti/<bank>/<year>/cti.parquet`.

### 5.2 Selective disclosure (cherry-picking)

```
share(pillar) = n_chunks_pillar / (n_env + n_soc + n_gov)
```

Đo ngân hàng tập trung viết về trụ nào — lệch mạnh = né chủ đề khó. Lưu tại `pillar_shares.parquet`.

### 5.3 Diễn giải

- **CTI cao** → nhiều cam kết suôn → dấu hiệu washing mạnh
- **QDR cao** → báo cáo cụ thể, có số liệu → substance tốt
- **share(gov) >> share(env)** → tập trung quản trị, né môi trường

---

## 6. Chạy pipeline

```python
# Kaggle / notebook
from src.esgwash.run import main as run_inference

# Toàn bộ corpus
run_inference(["--all"])

# 1 bank tất cả year
run_inference(["--bank", "bidv"])

# Nhiều bank
run_inference(["--bank", "bidv", "mbbank", "--year", "2024"])
```

Output mỗi (bank, year): `outputs/cti/<bank>/<year>/`
- `classified.parquet` — chunk + topic + commitment + spec_level
- `cti.parquet` — CTI / NAR / QDR
- `pillar_shares.parquet` — selective disclosure
- `legend.json` — giải thích từng cột (EN)
- `info_check.json` — chẩn đoán pipeline

---

## 7. Validation

Sau khi chạy full corpus, kiểm tra độ tin cậy của specificity LLM bằng audit mẫu:

```python
from esgwash.validation.runner import sample_for_audit, audit_agreement
# Lấy ~100 chunk/trụ để audit thủ công
audit_df = sample_for_audit(long_df, n_per_pillar=100)
audit_df.to_csv("audit_sample.csv", index=False)
# Sau khi điền gold_level: tính agreement
kappa = audit_agreement(audit_df)
```

---

## 8. Hạn chế

1. **Không có ground-truth washing thật**: không thể đo recall/precision của CTI trực tiếp.
2. **Commitment/Specificity train trên dữ liệu climate EN**: có thể không transfer hoàn hảo sang S/G tiếng Việt.
3. **Proxy nội văn bản**: specificity đo ngôn ngữ trong báo cáo, không đo hành động thực tế ngoài thực tế.
4. **Chunking ảnh hưởng CTI**: chunk nhỏ hơn → vague sentence không "ẩn" vào chunk cụ thể → CTI tự nhiên cao hơn so với chunking thô.
