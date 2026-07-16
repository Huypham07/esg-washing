# Phân tích chi tiết: 10 Kaggle Notebooks + 1 TXT về ESG Greenwashing
> Đọc và tổng hợp ngày 2026-06-19

---

## Dataset chung (notebooks 2–10 trừ #7)
**`esg_greenwashing_energy_utilities_industrials_2010_2024.csv`** — do alitaqishah tạo
- 450 rows × 19 columns; 30 công ty; 3 sectors: Energy, Utilities, Industrials; 9 quốc gia; 2010–2024
- Target: `greenwashing_flag` (binary, 0=Clean, 1=Greenwashing) — **pre-labeled**, tỷ lệ 85.6% / 14.4%
- Key columns:
  - `scope1/2/3_emissions_mt_co2e`, `total_s1_s2`, `yoy_scope1_change_pct`
  - `carbon_intensity_tco2e_per_musd`, `esg_score_0_100`
  - `cdp_climate_score` (letter: D–A), `net_zero_target_set`, `sbti_committed`
  - `emissions_disclosed`, `third_party_verified`

---

## 1. `corporate-greenwashing-scores-eda.ipynb` ⭐ HAY NHẤT VỀ EDA

**Dataset**: `Greenwashing_Score_Data.xlsx` — 595 rows, 215 công ty, 2011–2023, `GW_SCORE` (0–1 float)

**Phát hiện nổi bật:**
- Panel sparsity cực cao: chỉ **21.3%** ô company×year có data → không thể dùng panel regression đơn giản
- `GW_SCORE` **không phải continuous** — chỉ có 22 giá trị rời rạc (mostly sixteenths), entropy=3.76 bits → ML model coi là continuous sẽ bị sai
- Score 1.0 là giá trị phổ biến nhất (upper-tail concentration)
- Score thay đổi đột ngột (step change), không smooth → data được cập nhật theo lô, không liên tục

**Techniques hay:**
```python
# Panel density
panel_density = len(df) / (df["COMPANY_NAME"].nunique() * len(years))
# Score entropy
score_entropy_bits = -(score_dist["probability"] * np.log2(score_dist["probability"])).sum()
effective_score_states = 2 ** score_entropy_bits
```
- Manual PCA bằng NumPy `eigh` trên 5 company features (không dùng sklearn)
- `LineCollection` với delta colormap (đỏ=tăng score, teal=giảm) cho trajectory plots
- Heatmap company×year với outlines cho excluded rows

---

## 2. `esg-logistics-vs-random-forest.ipynb` = notebook 3 (trùng nội dung)

---

## 3. `esg-vs-reality-carbon-claims-vs-reality.ipynb` ⭐ BEST ML METHODOLOGY

**Dataset**: alitaqishah 450×19

**Feature engineering:**
- CDP letter → số: `{'D':1,'D-':0,'C':2,'C-':1.5,'B-':3,'B':4,'A-':5,'A':6}`
- Interaction: `esg_x_netzero = esg_score * net_zero_binary`
- `commitment_score = net_zero_binary + sbti_binary + verified_binary + disclosed_binary` (0–4)

**Imbalance handling**: upsample minority class → 308:308

**Results:**
| Model | CV AUC | Test AUC | GW Recall | GW Precision |
|---|---|---|---|---|
| Logistic Regression | 0.835 ± 0.154 | 0.956 | 0.92 | 0.46 |
| Random Forest | 0.968 ± 0.033 | **0.997** | **1.00** | 0.87 |

**Key findings:**
- ESG paradox: greenwashing companies **không nhất thiết có ESG score thấp**
- Top predictors: carbon_intensity + YoY Scope 1 change (real emissions > stated commitments)
- Net-zero pledge ≠ execution; CDP score mạnh hơn ESG score trong prediction
- `esg_x_netzero` interaction: mô hình hóa credibility gap giữa high ESG score + net-zero pledge

---

## 4. `greenwashing-detection-using-huggingface.ipynb` ⭐ MOST NOVEL NLP

**Dataset**: DAX ESG Media Dataset (Swisstext 2023) — ~11,000 EN documents về 40 DAX companies
- `internal=1`: báo cáo nội bộ công ty; `internal=0`: external media

**Approach: Internal vs External Embedding Alignment Gap**
```python
model = SentenceTransformer("flax-sentence-embeddings/all_datasets_v3_mpnet-base")

# Length-weighted company embedding
weights = text_lengths / text_lengths.sum()
company_embedding = np.average(doc_embeddings, axis=0, weights=weights)

# Greenwashing proxy = internal_sim - external_sim per SDG
relevance_gap = cosine_sim(internal_emb, sdg_emb) - cosine_sim(external_emb, sdg_emb)
```

- So sánh alignment với 17 UN SDGs (embeddings của 17 SDG descriptions)
- KMeans (k=5) clustering trên company embeddings
- Heatmap: 40 companies × 17 SDGs → cosine similarity matrix

**Key findings (BMW example):**
- SDG được nhấn mạnh nội bộ nhưng không được external media xác nhận = greenwashing signal
- Relevance gap lớn nhất: Decent Work, Quality Education, Clean Water (BMW tự claim nhiều nhất)
- SDG 7 (Clean Energy) và SDG 9 (Industry/Innovation) dominant toàn bộ DAX index
- SDG 5 (Gender Equality) lowest relevance score toàn index

---

## 5. `esg-greenwashing-detection-energy-utilities.ipynb`

**Dataset**: alitaqishah 450×19 — EDA-heavy, ít ML

**RF không xử lý imbalance:**
- Accuracy: 0.83; GW precision: 0.36; GW recall: **0.25** (kém hơn nhiều so với notebook 3)
- Lesson: không handle imbalance → recall rất thấp với minority class

---

## 6. `esg-greenwashing-detection-using-ml-shap.ipynb` ⚠️ SHAP TỐT NHƯNG BUG

**Features engineered:**
- `esg_emission_gap = esg_score_0_100 - (100 - carbon_intensity_normalized * 100)` → đo khoảng cách ESG score vs thực tế
- `scope3_opacity_risk = (scope3_ratio > median) & (third_party_verified == 0)` → Scope 3 opacity flag

**SHAP workflow:**
```python
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test, check_additivity=False)
shap.summary_plot(shap_values[1], X_test)
```

**⚠️ Bug nghiêm trọng**: LabelEncoder applied to company name + ticker → gần như unique ID → data leakage → Accuracy = 100%, AUC = 1.000 (overfitting giả tạo). Không nên dùng kết quả số liệu này.

**Useful**: pattern của `esg_emission_gap` và `scope3_opacity_risk` features, SHAP workflow code.

---

## 7. `sentiment-analysis-text-mining.ipynb` ❌ KHÔNG LIÊN QUAN

**Dataset**: Urban Dictionary — hoàn toàn không liên quan ESG. Chỉ có thể học AFINN lexicon-based sentiment pattern cho reference.

---

## 8. `esg-2010-2024.ipynb`

**Dataset**: alitaqishah 450×19 — pure EDA/visualization

**Key code:**
```python
z = np.polyfit(data["carbon_intensity_tco2e_per_musd"], data["esg_score_0_100"], 1)
p = np.poly1d(z)
```

**Findings:**
- Highest carbon intensity: Utilities (65.96 MT avg Scope 1)
- Lowest carbon intensity: Energy (counter-intuitive — high absolute but high revenue denominator)
- Overall greenwashing rate: **14.4%**
- Negative ESG-carbon intensity relationship, nhưng scatter cao → ESG không perfectly aligned với emissions

---

## 9. `esg-analysis-correlation-t-test-visulaization-eda.ipynb` ⭐ KEY STATS

**Statistical tests:**
- ANOVA: F=60.79, p=4.4e-24 → ESG scores khác biệt có ý nghĩa giữa 3 sectors
- Chi-square (greenwashing vs third-party verification): χ²=2.115, p=0.146 → **NOT significant** (surprisng)
- T-test: bug do không encode Yes/No trước khi filter

**Correlation với `greenwashing_flag`:**
| Feature | r |
|---|---|
| `yoy_scope1_change_pct` | **+0.419** ← strongest |
| `esg_score_0_100` | +0.182 |
| `carbon_intensity` | -0.048 |
| `scope3_emissions` | -0.009 |

→ YoY Scope 1 tăng là signal mạnh nhất cho greenwashing.

---

## 10. `carbon-esg-15-years-of-heavy-industry.ipynb` ⭐ MOST COMPREHENSIVE

**Dataset**: alitaqishah 450×19 — 12 rich visualizations + quantitative findings

**Key numbers:**
- Scope 1 change 2010→2024: Energy -16.3%, **Utilities -51.3%**, Industrials -18.9%
- Greenwashing by sector: Energy 15.3%, Utilities 17.3%, Industrials 10.7%; total **14.4%**
- **Repeat offenders**: E.ON (46.7%), Eni (46.7%), Enel (40%), Equinor (33.3%), TotalEnergies (33.3%)
- By 2024: 100% sectors có net-zero target; nhưng Industrials chỉ 40% SBTi (vs 100% Energy)

**Novel features:**
- `scope3_opacity_risk`: high Scope3 ratio + no third-party verification
- 2024 bubble chart: ESG vs Scope1, bubble size = revenue, cross marker = greenwashing flag
- Company×year greenwashing heatmap → trực quan repeat offenders

---

## 11. `greenwashing detector.txt` — Agent Architecture Reference

**Architecture**: 2 LLM agents + 1 scraper tool (Google ADK + Genai)
1. `GreenwashingDetectionTool` (Main Agent): nhận URL → gọi extraction agent → evaluate → output JSON {score, verdict, explanation}
2. `EnvironmentalContentExtractorTool`: scrape URL → lọc chỉ lấy environmental content
3. `scrap_page`: HTML scraper → visible text

**Planned enhancements** (chưa build):
- Multi-source: news, NGO, compliance DB, social media
- Cross-check: corporate claims vs third-party audits, regulatory filings
- Scoring rubric: claim specificity, measurable targets, certifications, historical controversies

---

## Cross-Notebook Summary: Top Predictors của Greenwashing

| Feature | Signal | Source |
|---|---|---|
| `yoy_scope1_change_pct` | r=+0.419 | notebook 9 |
| `carbon_intensity_tco2e_per_musd` | Top RF feature | notebooks 3,5,6 |
| `cdp_numeric` (letter grade) | Strong predictor | notebooks 3,5,6 |
| `esg_emission_gap` | Gap ESG vs actual | notebook 6 |
| `scope3_opacity_risk` | High Scope3 + no verification | notebooks 6,10 |
| `commitment_score` | Pledges without action | notebook 3 |
| Internal-external embedding gap | NLP proxy | notebook 4 |

---

## Áp dụng vào Pipeline Hiện Tại

### Ngay lập tức có thể làm:
1. **ClimateBERT / MPNet** thay generic encoder → notebook 4 đã validate
2. **Internal vs external alignment gap** → idea cho cross-source verification nếu có external text
3. **`esg_emission_gap` + `scope3_opacity_risk`** → features cho hybrid model (text + numerical)
4. **SHAP workflow** từ notebook 6 (bỏ bug LabelEncoder) → explainability cho chunk scores

### Lesson learned từ các bug:
- Không LabelEncoder company name/ticker (data leakage)
- Phải handle class imbalance (recall của minority class chênh nhau 0.25 vs 1.00)
- GW_SCORE rời rạc → không treat as continuous regression target

### Research gap đã confirmed (từ NLP survey):
- Vietnamese cross-lingual = unique contribution, không notebook nào ở trên làm
- Chunk-level detection phù hợp với best practices
