"""Specificity scorer bang small instruct-LLM + rubric 5 co atomic.

LLM phan ra 5 co nhi phan (co_cam_ket, co_hanh_dong_ten, co_so_dinh_luong,
quy_ve_bank, co_moc_tg) + evidence trich dan nguyen van. Luat tuong minh (derive_flags):
  spec_level 2 = co_so_dinh_luong AND quy_ve_bank
             1 = co_hanh_dong_ten (chua dat Muc 2)
             0 = con lai
  is_specific = (spec_level >= 1). CTI = ti le Muc 0.
verify_rubric_flags huy co_so_dinh_luong neu chu so trong evidence khong co trong text.
enforce_evidence ha co ve 0 neu evidence khong phai substring cua text.
Retry khi parse loi, het retry -> moi co 0, spec_level 0, parse_ok False.
"""
from __future__ import annotations

import json
import re

import pandas as pd

SYSTEM = (
    "Bạn là chuyên gia phân tích báo cáo ESG ngân hàng. Với mỗi ĐOẠN VĂN, trả lời 5 câu hỏi "
    "YES/NO khách quan, và với mỗi câu trả lời YES phải TRÍCH nguyên văn cụm trong đoạn làm "
    "bằng chứng (evidence):\n"
    "1. co_cam_ket: đoạn có Ý CAM KẾT/hướng tương lai (sẽ, cam kết, hướng tới, mục tiêu, đặt mục tiêu)?\n"
    "2. co_hanh_dong_ten: có HÀNH ĐỘNG/CHƯƠNG TRÌNH/CÔNG CỤ/HỆ THỐNG CÓ TÊN, kiểm chứng được "
    "(vd 'gói Tín dụng xanh', 'hệ thống B.One') — KHÁC khẩu hiệu/tính từ ('bền vững', 'toàn diện')?\n"
    "3. co_so_dinh_luong: có ĐẠI LƯỢNG ĐỊNH LƯỢNG (số, %, tỷ đồng, MW)? KHÔNG tính năm chiến "
    "lược/luật, tên tiêu chuẩn (ISO), số của NHNN/toàn ngành.\n"
    "4. quy_ve_bank: số/hành động đó QUY VỀ CHÍNH NGÂN HÀNG chủ thể (không phải quốc gia/ngành)?\n"
    "5. co_moc_tg: có MỐC THỜI GIAN/deadline (năm mục tiêu, 'đến 2030', 'giai đoạn 2021-2025')?\n"
    "Chỉ trả về JSON, không giải thích ngoài JSON."
)

SCHEMA_HINT = (
    'Trả về JSON đúng dạng:\n'
    '{"co_cam_ket": true/false, "co_hanh_dong_ten": true/false, "co_so_dinh_luong": true/false, '
    '"quy_ve_bank": true/false, "co_moc_tg": true/false, '
    '"evidence": {"co_cam_ket": "<trích dẫn hoặc null>", "co_hanh_dong_ten": "...", '
    '"co_so_dinh_luong": "...", "quy_ve_bank": "...", "co_moc_tg": "..."}, "reason": "<ngắn>"}'
)

FEWSHOT = [
    ("Ngân hàng hướng tới một tương lai xanh và bền vững.",
     {"co_cam_ket": True, "co_hanh_dong_ten": False, "co_so_dinh_luong": False,
      "quy_ve_bank": False, "co_moc_tg": False,
      "evidence": {"co_cam_ket": "hướng tới"},
      "reason": "Chỉ khẩu hiệu, không hành động có tên, không số (Mức 0)."}),
    ("BIDV đã ban hành gói Tín dụng xanh cho khách hàng vay phát triển năng lượng sạch.",
     {"co_cam_ket": True, "co_hanh_dong_ten": True, "co_so_dinh_luong": False,
      "quy_ve_bank": True, "co_moc_tg": False,
      "evidence": {"co_cam_ket": "ban hành", "co_hanh_dong_ten": "gói Tín dụng xanh",
                   "quy_ve_bank": "BIDV"},
      "reason": "Hành động có tên, không số (Mức 1)."}),
    ("Ngân hàng đặt mục tiêu giảm 30% cường độ phát thải khí nhà kính vào năm 2030.",
     {"co_cam_ket": True, "co_hanh_dong_ten": True, "co_so_dinh_luong": True,
      "quy_ve_bank": True, "co_moc_tg": True,
      "evidence": {"co_cam_ket": "đặt mục tiêu", "co_hanh_dong_ten": "giảm cường độ phát thải",
                   "co_so_dinh_luong": "30%", "quy_ve_bank": "Ngân hàng", "co_moc_tg": "năm 2030"},
      "reason": "Số 30% quy về ngân hàng, có mốc 2030 (Mức 2)."}),
]

_DIGITS_RE = re.compile(r"\d+")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")

ATOMIC_FLAGS = ("co_cam_ket", "co_hanh_dong_ten", "co_so_dinh_luong",
                "quy_ve_bank", "co_moc_tg")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def _digit_runs(s: str) -> list[str]:
    """Cac chuoi chu so trong s, da bo dau phan cach (1,89 / 5.000 -> 189 / 5000)."""
    return _DIGITS_RE.findall(re.sub(r"(?<=\d)[.,\s](?=\d)", "", str(s)))


def _extract_json_obj(text: str) -> dict | None:
    """Bo <think>...</think> roi lay object JSON dau tien khop ngoac. Tra None neu hong."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start:i + 1])
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    pass
                return None
    return None


def _parse_flags(text: str) -> dict | None:
    """Boc JSON (da bo <think>), tach flags + evidence. Tra None neu hong."""
    obj = _extract_json_obj(text)
    if obj is None:
        return None
    flags = {f: int(bool(obj.get(f))) for f in ATOMIC_FLAGS}
    ev = obj.get("evidence") or {}
    evidence = {f: (ev.get(f) if isinstance(ev, dict) else None) for f in ATOMIC_FLAGS}
    return {"flags": flags, "evidence": evidence}


def verify_rubric_flags(flags: dict, evidence: dict, text: str) -> dict:
    """Chong bia so: neu co_so_dinh_luong=1 ma chu so trong evidence khong co trong text -> ha ve 0."""
    flags = dict(flags)
    if flags.get("co_so_dinh_luong"):
        ev_str = evidence.get("co_so_dinh_luong") or ""
        ev_digits = _digit_runs(ev_str)
        text_digits = set(_digit_runs(text))
        if not ev_digits or not any(d in text_digits for d in ev_digits):
            flags["co_so_dinh_luong"] = 0
    return flags


def enforce_evidence(flags: dict, evidence: dict, text: str) -> dict:
    """Co 'yes' phai co evidence la chuoi con cua chunk, neu khong -> ha ve 0."""
    norm_text = _norm(text)
    out = {}
    for f in ATOMIC_FLAGS:
        v = int(bool(flags.get(f)))
        if v:
            ev = _norm(evidence.get(f) or "")
            if not ev or ev not in norm_text:
                v = 0
        out[f] = v
    return out


def derive_flags(flags: dict) -> tuple[float, int]:
    """5 co atomic -> (p_specificity, spec_level) bang luat tat dinh.
      2 = co_so_dinh_luong AND quy_ve_bank
      1 = co_hanh_dong_ten (chua dat Muc 2)
      0 = con lai
    Cong commit (co_cam_ket AND ESG) xu ly o classify_chunks, khong o day."""
    quant = bool(flags.get("co_so_dinh_luong")) and bool(flags.get("quy_ve_bank"))
    action = bool(flags.get("co_hanh_dong_ten"))
    level = 2 if quant else (1 if action else 0)
    return {0: 0.0, 1: 0.5, 2: 1.0}[level], level


class SpecificityLLM:
    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.model_name = cfg.get("model", "Qwen/Qwen3-0.6B")
        self.max_new_tokens = int(cfg.get("max_new_tokens", 256))
        self.retries = int(cfg.get("retries", 2))
        self.enable_thinking = bool(cfg.get("enable_thinking", False))
        self._tok = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from esgwash.models.trainer import get_device
        self._device = get_device()
        # fp16 tren GPU de giam ~mot nua bo nho (1.7B fp32 ~6.8GB -> ~3.4GB); CPU giu fp32.
        dtype = torch.float16 if self._device.type == "cuda" else torch.float32
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, dtype=dtype).to(self._device).eval()

    def _build_messages(self, text: str, stricter: bool = False) -> list[dict]:
        msgs = [{"role": "system", "content": SYSTEM}]
        for ex_text, ex_json in FEWSHOT:
            msgs.append({"role": "user", "content": f"Đoạn: {ex_text}\n{SCHEMA_HINT}"})
            msgs.append({"role": "assistant", "content": json.dumps(ex_json, ensure_ascii=False)})
        hint = SCHEMA_HINT + ("\nCHÚ Ý: chỉ xuất JSON hợp lệ, không thêm chữ nào khác."
                              if stricter else "")
        msgs.append({"role": "user", "content": f"Đoạn: {text}\n{hint}"})
        return msgs

    def _complete(self, messages: list[dict]) -> str:
        """Sinh van ban tu LLM (greedy -> tai lap duoc). Tach ra de test mock duoc."""
        import torch
        self._load()
        try:
            prompt = self._tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=self.enable_thinking)
        except TypeError:  # tokenizer khong ho tro enable_thinking (vd Qwen2.5)
            prompt = self._tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        enc = self._tok(prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self._model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                       do_sample=False,
                                       pad_token_id=self._tok.eos_token_id)
        return self._tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def score_one(self, text: str) -> dict:
        parsed, raw = None, ""
        for attempt in range(self.retries + 1):
            raw = self._complete(self._build_messages(text, stricter=attempt > 0))
            parsed = _parse_flags(raw)
            if parsed is not None:
                break
        base = {f: 0 for f in ATOMIC_FLAGS}
        if parsed is None:
            return {"p_specificity": 0.0, "spec_level": 0, "is_specific": 0,
                    "parse_ok": False, "rubric": pd.NA, "raw": raw,
                    "evidence": pd.NA, **base}
        flags = verify_rubric_flags(parsed["flags"], parsed["evidence"], text)
        flags = enforce_evidence(flags, parsed["evidence"], text)
        p, level = derive_flags(flags)
        return {"p_specificity": p, "spec_level": level, "is_specific": int(level >= 1),
                "parse_ok": True, "rubric": json.dumps(parsed, ensure_ascii=False), "raw": raw,
                "evidence": json.dumps(parsed["evidence"], ensure_ascii=False), **flags}

    def predict(self, sentences: list[str]) -> pd.DataFrame:
        from tqdm.auto import tqdm
        rows = [self.score_one(str(t))
                for t in tqdm(sentences, desc="specificity-LLM", unit="chunk")]
        return pd.DataFrame(rows, columns=["p_specificity", "spec_level", "is_specific",
                                           "parse_ok", "rubric", "raw",
                                           "evidence", *ATOMIC_FLAGS])
