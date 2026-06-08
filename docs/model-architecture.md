# Kiến trúc hệ thống (inference forward path) — Phương án B

Figure kiểu paper: luồng tính toán **lúc chạy trực tiếp** trên một câu báo cáo tiếng Việt → đến điểm số washing ở mức (bank, year, trụ). Không bao gồm train/validation (xem `research-plan-B.md`).

```mermaid
flowchart LR
    X["Câu tiếng Việt sᵢ<br/>(+ ctx_prev, ctx_next)"]:::io

    subgraph ENC["Shared Encoder"]
        direction TB
        WS["Word Segmentation<br/>(underthesea)"]
        PB["PhoBERT<br/>contextual embedding<br/>h = [CLS]"]
        WS --> PB
    end

    subgraph HEADS["Task heads"]
        direction TB
        TH["Topic head<br/>3× sigmoid → P(E), P(S), P(G)"]:::head
        SH["Substantiveness head<br/>ordinal → L0|L1|L2|L3"]:::head
    end

    GATE{"là claim /<br/>commitment?<br/>(L ≥ L1)"}:::gate

    subgraph GND["Evidence Grounding Module"]
        direction TB
        RET["Retrieval span bằng chứng<br/>BM25 / dense, trong cùng (bank, year)"]
        SLOT["Slot detector<br/>KPI · baseline · target · assurance"]
        NLI["Claim–Evidence NLI<br/>P(entail), P(neutral), P(contradict)"]
        RET --> NLI
        SLOT --> NLI
    end

    subgraph IDX["Tổng hợp & chỉ số"]
        direction TB
        AGG["Tổng hợp theo (bank, year, trụ)<br/>T = mật độ câu ESG (từ Topic)<br/>S = f(substantiveness, grounding)"]
        REG["Hồi quy decoupling<br/>S = β·T + ε"]
        OUT["Washing scoreₖ = residual ε̂<br/>(z-score, bootstrap CI)"]:::io
        AGG --> REG --> OUT
    end

    X --> WS
    PB --> TH
    PB --> SH
    TH -->|"P(E/S/G)"| AGG
    SH --> GATE
    GATE -->|có| RET
    GATE -->|không| AGG
    NLI --> AGG
    SH -->|"ordinal L"| AGG

    classDef io fill:#1f2937,color:#fff,stroke:#111;
    classDef head fill:#e0f2fe,stroke:#0369a1;
    classDef gate fill:#fef9c3,stroke:#ca8a04;
```

## Diễn giải khối (theo thứ tự forward)
| Khối | Vào → Ra | Ghi chú |
|---|---|---|
| **Shared Encoder** | câu VN → vector `h` | PhoBERT, dùng chung cho cả 2 head; bắt buộc tách từ trước. |
| **Topic head** | `h` → P(E), P(S), P(G) | 3 sigmoid độc lập (multi-label, phẳng); cấp tín hiệu **Talk T**. |
| **Substantiveness head** | `h` → L0–L3 | đầu ordinal (CORN/CORAL); cấp mức thực chất câu. |
| **Claim gate** | L → {có/không} | chỉ câu claim/commitment mới đi tiếp sang grounding (tiết kiệm + đúng ngữ nghĩa). |
| **Evidence Grounding** | claim → P(entail/neutral/contradict) + slot | retrieval bằng chứng trong cùng báo cáo → NLI **giữ nguyên phân phối**, không nhị phân hoá. |
| **Tổng hợp & chỉ số** | (T, S) theo (bank, year, trụ) → washing score | **washing = phần dư hồi quy S~T** (decoupling-as-residual). |

> Đây là kiến trúc **runtime** đơn-câu rồi gộp lên mức tổ chức. Encoder PhoBERT là backbone dùng chung; 2 head + module grounding nhánh ra; chỉ số washing là residual ở cuối.
