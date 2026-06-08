# Kiến trúc pipeline — Phương án B (ESG-washing NH VN, cross-lingual)

Sơ đồ tổng thể: 4 khối — **(A)** train mô hình từ gold EN qua translate-train, **(B)** chuẩn bị corpus VN, **(C)** pipeline suy luận → chỉ số washing, **(D)** validation 4 tầng. (Đồng bộ với `research-plan-B.md`; đã bỏ neuro-symbolic.)

```mermaid
flowchart TB
    %% ============ A. TRAIN-TIME ============
    subgraph A["A. TRAIN-TIME — gold EN → mô hình VN (translate-train)"]
        direction TB
        G1["data/en_gold/topic<br/>env / soc / gov _2k (+nature)"]
        G2["data/en_gold/subst<br/>specificity, commitments,<br/>netzero, env_claims, action..."]
        MT["MT EN→VN<br/>google/translategemma-4b-it<br/><i>giữ nguyên nhãn</i>"]
        QE["QE + lọc nhiễu<br/>round-trip + COMET-QE<br/>Confident Learning / Co-teaching"]
        WS1["Word-segment (underthesea)"]
        TT["Fine-tune PhoBERT"]
        MTOPIC["① Topic classifier<br/>3 đầu sigmoid E/S/G (phẳng)"]
        MSUB["② Substantiveness ordinal<br/>L0→L3 (CORN/CORAL)"]

        G1 --> MT
        G2 --> MT
        MT --> QE --> WS1 --> TT
        TT --> MTOPIC
        TT --> MSUB
    end

    %% ============ B. CORPUS VN ============
    subgraph B["B. CORPUS VN (đối tượng phân tích — đã có)"]
        direction TB
        RAW["Báo cáo NH VN (OCR)<br/>10 bank × 2020–2024"]
        CLEAN["clean_corpus.py<br/>NFC + sửa OCR + khử trùng lặp<br/>+ word-segment"]
        SENT["sentences_clean.parquet<br/>~119k câu (+ctx_prev/next)"]
        RAW --> CLEAN --> SENT
    end

    %% ============ C. INFERENCE / INDEX ============
    subgraph C["C. PIPELINE SUY LUẬN → CHỈ SỐ WASHING"]
        direction TB
        PTOPIC["Gán topic E/S/G<br/>→ Talk (T)"]
        PCLAIM{"Câu là claim/<br/>commitment?"}
        PSUB["Substantiveness ordinal<br/>L0–L3"]
        GR["Grounding<br/>retrieval span (BM25/dense)<br/>+ slot (KPI/baseline/target/assurance)<br/>+ claim–evidence NLI (entail/neutral/contradict)"]
        AGG["Tổng hợp theo (bank, year, trụ)<br/>T = mật độ câu ESG<br/>S = substantiveness × grounding"]
        DEC["Decoupling<br/>washing = residual(S ~ T)<br/>decoupling.py"]
        REP["Báo cáo: ranking + case study<br/>high-talk / low-substantiation"]

        PTOPIC --> PCLAIM
        PCLAIM -->|có| PSUB --> GR --> AGG
        PCLAIM -->|không| AGG
        PTOPIC --> AGG
        AGG --> DEC --> REP
    end

    %% model -> inference
    MTOPIC ==>|model| PTOPIC
    MSUB ==>|model| PSUB
    SENT ==>|input| PTOPIC

    %% ============ D. VALIDATION ============
    subgraph D["D. VALIDATION 4 tầng (0 gán nhãn)"]
        direction TB
        V1["T1 · Gold EN<br/>Macro-F1, QWK"]
        V2["T2 · Transfer EN→VN<br/>back-translation + teacher-student"]
        V3["T3 · Synthetic manipulation<br/>+ Known-group (VNSI/GRI/assurance/green-credit)"]
        V4["T4 · Case study (face validity)"]
    end

    MTOPIC -.-> V1
    MSUB -.-> V1
    PTOPIC -.-> V2
    DEC -.-> V3
    REP -.-> V4
    BSIG["data/external/bank_signals.csv"] -.-> V3
```

## Đọc nhanh theo dòng dữ liệu
1. **Gold EN** → dịch sang VN bằng `translategemma-4b-it` (giữ nhãn) → lọc nhiễu (QE + Confident Learning) → fine-tune **PhoBERT** ra 2 mô hình: **Topic E/S/G** và **Substantiveness L0–L3**. Train **chỉ** trên gold chuyên gia đã dịch — không có nhãn LLM.
2. **Corpus VN** (OCR 10 bank × 5 năm) → làm sạch → ~119k câu sạch + tách từ; **chỉ dùng để inference**, không làm nhãn train.
3. **Suy luận:** mỗi câu → topic (đo **Talk T**); câu nào là claim → substantiveness + **grounding** (retrieval + slot + NLI giữ phân phối) → đo **Substantiation S**.
4. Tổng hợp theo (bank, year, trụ) → **washing = phần dư hồi quy S ~ T** → báo cáo & case study.
5. **Validation** bám 4 điểm: gold EN, consistency transfer, synthetic + known-group, case study.

> Ghi chú: nét `==>` = nạp mô hình đã train vào suy luận; nét `-.->` = luồng kiểm chứng/phụ trợ; nét liền = luồng dữ liệu chính.
