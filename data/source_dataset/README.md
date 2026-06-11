# English gold datasets (translate-train source — Plan B)

Tải từ HuggingFace cho pipeline translate-train EN→VN (xem `docs/research-plan-B.md` §2).
**KHÔNG dùng nhãn LLM/repo cũ làm nhãn train chính** — đây là nguồn nhãn chuyên gia đã công bố.

## topic/  (sentence-level, nhãn E/S/G phẳng)
| file | rows | label col | source |
|---|---|---|---|
| environmental_2k.csv | 2000 | `env` 0/1 | ESGBERT/environmental_2k (Mehra et al. 2022) |
| social_2k.csv | 2000 | `soc` 0/1 | ESGBERT/social_2k |
| governance_2k.csv | 2000 | `gov` 0/1 | ESGBERT/governance_2k |

## subst/  (substantiveness / claim)
| file | rows (train/test) | label | source | dùng |
|---|---|---|---|---|
| specificity.{train,test}.parquet | 1000/320 | specific 0/1 | climatebert/climate_specificity (Bingler 2022) | ✅ thang ordinal |
| commitments_actions.{train,test}.parquet | 1000/320 | commitment/action 0/1 | climatebert/climate_commitments_actions | ✅ gate claim |
| netzero_reduction.csv | 3441 | `target` ∈ {none,reduction,net-zero} | climatebert/netzero_reduction_data (Schimanski 2023) | ✅ thang ordinal |
| env_claims.{train,val,test}.parquet | 2117/265/265 | claim 0/1 | climatebert/environmental_claims (Stammbach 2022) | ✅ gate claim |
| action_500.csv | 500 | `action` 0/1 | ESGBERT/action_500 (Mehra 2022) | ⏳ chờ quyết (positive-augmentation?) |

## Đã gỡ (không nằm trong thiết kế)
- `nature_2200` (subtree Water/Forest/Biodiversity) — bỏ vì topic để phẳng E/S/G.
- `detection.*` (climate-related 0/1) — trùng topic E.
- `tcfd.*` (5-class TCFD) — đã bỏ phân cấp/disclosure-structure.

## Không dùng (gated)
- `cea-list-ia/ESG-classification-en`, `ML-Promise` — xem plan §2.

## Lưu ý load
- CSV có cột thừa `Unnamed: 0` → bỏ.
- Vài parquet chứa ký tự U+2212 (minus), U+FFFD → chuẩn hoá unicode NFC khi load.
