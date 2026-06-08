# English gold datasets (translate-train source — Plan B)

Tải từ HuggingFace cho pipeline translate-train EN→VN (xem `docs/research-plan-B.md` §2).
**KHÔNG dùng nhãn LLM/repo cũ làm nhãn train chính** — đây là nguồn nhãn chuyên gia đã công bố.

## topic/  (sentence-level, dùng cho cây ESG)
| file | rows | label col | source |
|---|---|---|---|
| environmental_2k.csv | 2000 | `env` 0/1 | ESGBERT/environmental_2k (Mehra et al. 2022) |
| social_2k.csv | 2000 | `soc` 0/1 | ESGBERT/social_2k |
| governance_2k.csv | 2000 | `gov` 0/1 | ESGBERT/governance_2k |
| nature_2200.csv | 2200 | Water/Forest/Biodiversity/Nature (multi-label) | ESGBERT/WaterForestBiodiversityNature_2200 — subtree của E |

## subst/  (substantiveness / claim / action)
| file | rows (train/test) | label | source |
|---|---|---|---|
| specificity.{train,test}.parquet | 1000/320 | specific 0/1 | climatebert/climate_specificity (Bingler 2022) |
| commitments_actions.{train,test}.parquet | 1000/320 | commitment/action 0/1 | climatebert/climate_commitments_actions |
| action_500.csv | 500 | `action` 0/1 | ESGBERT/action_500 |
| netzero_reduction.csv | 3441 | `target` ∈ {none,reduction,net-zero} | climatebert/netzero_reduction_data (Schimanski 2023) |
| env_claims.{train,val,test}.parquet | 2117/265/265 | claim 0/1 | climatebert/environmental_claims (Stammbach 2022) |
| detection.{train,test}.parquet | 1300/400 | climate-related 0/1 | climatebert/climate_detection |
| tcfd.{train,test}.parquet | 1300/400 | 5-class TCFD | climatebert/tcfd_recommendations |

## Chưa lấy được (gated/manual) — có phương án thay thế trong plan
- `cea-list-ia/ESG-classification-en` (taxonomy 16-lớp): gated, cần token HF + accept terms.
- `ML-Promise` (Chen 2025, EMNLP): tải từ PromiseEval / SemEval-2025 Task 6.

## Lưu ý load
- CSV có cột thừa `Unnamed: 0` → bỏ.
- Vài parquet chứa ký tự U+2212 (minus), U+FFFD → chuẩn hoá unicode NFC khi load.
