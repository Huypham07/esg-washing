"""P3b — Parser + prompt cho dieu kien DIRECT prompting (C2/C3) cua ablation specificity.

C2/C3 hoi THANG LLM ra muc 0/1/2 (holistic), KHAC C1 (decompose 4 co + luat).
=> PHAI dung parser integer RIENG nay, KHONG dung _parse_flags/derive_flags cua C1
   (neu dung nham -> C2/C3 sap thanh C1-vs-C1).

Prompt grounded verbatim theo main.tex Sec 3.3 (dinh nghia 3 muc + 4 thuoc tinh) ->
khong bi cao buoc "prompt co tinh yeu". Fallback parse-fail -> muc 0, parse_ok False
(khop pipeline C1: parse fail -> Muc 0).
"""
from __future__ import annotations

import json
import re
import unicodedata

# ---- Prompt C2 (zero-shot direct), dinh nghia verbatim Sec 3.3 (main.tex:136,149) ----
SYSTEM_DIRECT = (
    "Bạn là chuyên gia phân tích báo cáo ESG ngân hàng. Với mỗi ĐOẠN VĂN (một cam kết ESG), "
    "hãy gán MỘT mức độ CỤ THỂ (specificity) theo thang 0/1/2:\n"
    "- Mức 2: cam kết ĐỊNH LƯỢNG QUY VỀ CHÍNH NGÂN HÀNG — có đại lượng định lượng (số, %, tỷ đồng, MW) "
    "VÀ đại lượng/hành động đó thuộc về CHÍNH NGÂN HÀNG chủ thể (không phải số của ngành/quốc gia/bên khác).\n"
    "- Mức 1: có HÀNH ĐỘNG CÓ TÊN, kiểm chứng được (chương trình/công cụ/hệ thống có tên — khác khẩu hiệu "
    "hay tính từ chung như 'bền vững', 'toàn diện'), NHƯNG không có định lượng quy về ngân hàng.\n"
    "- Mức 0: KHÔNG có cả hành động có tên lẫn định lượng quy về ngân hàng (chỉ khẩu hiệu/mơ hồ).\n"
    "Lưu ý: năm chiến lược/luật, tên tiêu chuẩn (ISO), số của NHNN/toàn ngành KHÔNG tính là định lượng "
    "quy về ngân hàng.\n"
    'Chỉ trả về JSON đúng dạng {"level": 0|1|2, "reason": "<ngắn>"}. Không thêm chữ nào ngoài JSON.'
)

# JSON path: lay MOI gia tri "level":N; neu bat dong -> ambiguous -> fail-safe.
_LEVEL_JSON_RE = re.compile(r'"level"\s*:\s*"?([0-2])"?')
# Rubric/scale echo ("thang 0/1/2", "0 to 2") -> blank truoc keyword/standalone de khoi bat nham.
_SCALE_ECHO_RE = re.compile(r'0\s*[/,]\s*1\s*[/,]\s*2|0\s*(?:to|den|đến|-)\s*2', re.IGNORECASE)
# Keyword path: 'level'/'muc'/'mức' + digit trong <=6 ky tu; lookahead (?![/\d]) loai scale remnant + so nhieu chu so.
_LEVEL_KW_RE = re.compile(r'(?:level|m[ứưữu]c)\D{0,6}?([0-2])(?![/\d])', re.IGNORECASE)
# Strict standalone: CA output chi la mot chu so 0/1/2 (giu case hop le '2', tu choi digit chon trong prose).
_FULL_DIGIT_RE = re.compile(r'^\s*(?:m[ứưữu]c|level)?\s*[:=]?\s*"?([0-2])"?\.?\s*$', re.IGNORECASE)


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", str(text), flags=re.DOTALL)


def parse_level(text: str) -> tuple[int, bool]:
    """Trich muc 0/1/2 tu generation direct-prompt. Tra (level, parse_ok).
    parse_ok=False -> fallback muc 0 (pipeline retry + dry-run gate bat).
    Uu tien: JSON (nhat quan) > keyword 'level/muc N' > CA-output-la-mot-chu-so > fail.
    HARDENED (audit doi khang): NFC-normalize, chan <think> ho, JSON bat dong -> fail-safe,
    blank scale-echo, KHONG bat chu so chon trong prose (CO2/Scope 2/'2 chuong trinh')."""
    t = unicodedata.normalize("NFC", str(text))
    if "<think>" in t and "</think>" not in t:
        return 0, False   # thinking bi cat -> khong tin so ro ri trong think
    t = _strip_think(t)
    jvals = _LEVEL_JSON_RE.findall(t)
    if jvals:
        if len(set(jvals)) == 1:
            return int(jvals[0]), True
        return 0, False   # nhieu "level" bat dong (self-correct/nested) -> fail-safe
    t2 = _SCALE_ECHO_RE.sub(" ", t)
    m = _LEVEL_KW_RE.search(t2)
    if m:
        return int(m.group(1)), True
    m = _FULL_DIGIT_RE.match(t.strip())
    if m:
        return int(m.group(1)), True
    return 0, False


def build_direct_messages(chunk: str, fewshot: list[tuple[str, int]] | None = None,
                          stricter: bool = False) -> list[dict]:
    """messages cho C2 (fewshot=None) hoac C3 (fewshot = list[(text, level)])."""
    msgs = [{"role": "system", "content": SYSTEM_DIRECT}]
    for ex_text, ex_level in (fewshot or []):
        msgs.append({"role": "user", "content": f"Đoạn: {ex_text}"})
        msgs.append({"role": "assistant",
                     "content": json.dumps({"level": int(ex_level), "reason": ""},
                                           ensure_ascii=False)})
    tail = ("\nCHỈ xuất JSON hợp lệ, không thêm chữ nào khác." if stricter else "")
    msgs.append({"role": "user", "content": f"Đoạn: {chunk}{tail}"})
    return msgs


def _smoke_test() -> None:
    cases = [
        # -- JSON tot --
        ('{"level": 2, "reason": "co 30% quy ve bank"}', (2, True)),
        ('{"level":0}', (0, True)),
        ('{"reason":"khong phai muc 2, gan muc 0 hon","level":1}', (1, True)),  # reason echo level khac
        ('{"level" : 0 }', (0, True)),
        ('```json\n{"level": 1}\n```', (1, True)),                              # markdown fence
        ('{"level": 2', (2, True)),                                            # truncate ho ngoac
        # -- keyword / bare --
        ('2', (2, True)),
        ('Mức 1', (1, True)),
        ('Level: 0', (0, True)),
        ('Đây là mức 2 vì có số liệu.', (2, True)),
        ('<think>co the la 2030...</think>{"level": 1}', (1, True)),           # closed think
        # -- DANGER cu (phai thanh fail-safe hoac dung) --
        ('Giam 1,89 trieu tan CO2 moi nam.', (0, False)),                     # CO2 KHONG -> muc 2
        ('Bao cao phat thai theo Scope 2.', (0, False)),                      # Scope 2 KHONG -> muc 2
        ('Ngan hang trien khai 2 chuong trinh tin dung xanh.', (0, False)),   # '2 chuong trinh'
        ('Theo thang muc 0/1/2, toi danh gia doan nay la muc 2.', (2, True)), # scale echo -> van ra 2
        ('{"level": 0}\n{"level": 2}', (0, False)),                          # JSON bat dong -> fail-safe
        ('{"scores":{"level":2},"level":0}', (0, False)),                    # nested bat dong -> fail-safe
        ('<think>ro rang la muc 2 vi co so lieu', (0, False)),                # think HO -> fail-safe
        # -- fail an toan --
        ('The bank targets 30% by 2030', (0, False)),
        ('khong co json gi ca', (0, False)),
        ('{"level": 3}', (0, False)),                                         # muc 3 invalid -> fail
    ]
    ok = 0
    for text, expect in cases:
        got = parse_level(text)
        flag = "OK " if got == expect else "FAIL"
        if got == expect:
            ok += 1
        print(f"[{flag}] parse_level({text[:42]!r}) -> {got}  (expect {expect})")
    print(f"\nsmoke-test: {ok}/{len(cases)} pass")
    assert ok == len(cases), "SMOKE-TEST FAILED — khong duoc len Kaggle"


if __name__ == "__main__":
    _smoke_test()
