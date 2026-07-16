# legacy/ — code đã gỡ khỏi luồng chính

## grounding/ (gỡ 2026-06-16)
Grounding nội văn bản (retriever + NLI + evidence pool + grounded-CTI) bị loại vì
là construct **vòng tròn**: evidence cho một claim định lượng lấy từ chính câu của
ngân hàng trong cùng báo cáo → đo "nhất quán nội bộ", không đo *walk* thật; NLI
(XNLI) đo textual entailment chứ không đo tính xác thực. Chi tiết: spec
`docs/superpowers/specs/2026-06-16-esg-washing-pipeline-redesign-design.md` §1.1.

Giữ lại để tái hiện/đối chứng nếu reviewer yêu cầu. KHÔNG import từ luồng chính.
