"""Specificity scorer bang small instruct-LLM + rubric.

Tranh shortcut "co chu so -> specific" cua encoder: LLM phan ra cau thanh cac item
{action_or_event, figure, is_quantified, is_concrete_action, attributable_to_actor} + has_baseline_or_timeline,
roi suy nhan bang luat tuong minh (derive):
  spec_level 2 = dinh luong & quy ve chu the | 1 = hanh dong co ten & quy ve chu the | 0 = mo ho.
  is_specific = (spec_level >= 1). CTI = ti le Muc 0.
verify_rubric huy figure bia (so khong co trong text). Retry khi parse loi, het retry -> Muc 0.
"""
from __future__ import annotations

import json
import re

import pandas as pd

SYSTEM = (
    "Bạn là chuyên gia phân tích báo cáo ESG ngân hàng. Với mỗi ĐOẠN VĂN, phân rã TỪNG "
    "hành động/sự kiện thành item. Với MỖI item, đánh giá ĐỘC LẬP 2 thuộc tính:\n"
    "(1) is_quantified + attributable_to_actor: có ĐẠI LƯỢNG ĐỊNH LƯỢNG đo được QUY VỀ "
    "CHÍNH CHỦ THỂ không (ví dụ: giảm 30% phát thải, dư nợ tín dụng xanh 5.000 tỷ, 100 MW "
    "điện mặt trời). KHÔNG tính: năm của chiến lược/luật quốc gia, tên tiêu chuẩn (ISO, "
    "VIETGAP), số liệu của NHNN/toàn ngành/quốc gia, điều kiện vay, từ mơ hồ ('hấp dẫn').\n"
    "(2) is_concrete_action: item có nêu HÀNH ĐỘNG/CÔNG CỤ/CHƯƠNG TRÌNH/HỆ THỐNG CỤ THỂ CÓ "
    "TÊN, KIỂM CHỨNG ĐƯỢC của chủ thể không (ví dụ: 'ban hành gói Tín dụng xanh', 'hệ thống "
    "B.One', 'Chatbot AI', 'trồng cây xanh') — KHÁC với khẩu hiệu/tính từ chung chung KHÔNG "
    "kiểm chứng được ('chuyển đổi toàn diện', 'nâng cao năng lực', 'hướng tới bền vững', "
    "'thực chất, bài bản'). Tính từ/khát vọng -> is_concrete_action=false.\n"
    "Chỉ trả về JSON, không giải thích ngoài JSON."
)

SCHEMA_HINT = (
    'Trả về JSON đúng dạng:\n'
    '{"items": [{"action_or_event": "<hành động/sự kiện>", "figure": "<số liệu gắn với '
    'nó hoặc null>", "is_quantified": true/false, "attributable_to_actor": true/false, '
    '"is_concrete_action": true/false}], '
    '"has_baseline_or_timeline": true/false, "reason": "<giải thích ngắn>"}'
)

# Few-shot day 3 muc: (a) cu the-khong-so (Muc 1), (b) dinh luong (Muc 2),
# (c) nhieu item dinh luong, (d) mo ho thuan (Muc 0).
FEWSHOT = [
    ("Hưởng ứng Chiến lược quốc gia về tăng trưởng xanh giai đoạn 2021-2030, tầm nhìn "
     "2050, BIDV đã ban hành gói Tín dụng xanh cho khách hàng cá nhân vay phát triển năng "
     "lượng sạch (điện mặt trời, điện gió) hoặc trồng trọt chăn nuôi theo VIETGAP, ISO với "
     "lãi suất hấp dẫn và ưu đãi hơn thông thường.",
     {"items": [{"action_or_event": "ban hành gói Tín dụng xanh cho vay năng lượng sạch",
                 "figure": None, "is_quantified": False, "attributable_to_actor": True,
                 "is_concrete_action": True}],
      "has_baseline_or_timeline": False,
      "reason": "Gói Tín dụng xanh là hành động cụ thể có tên, kiểm chứng được (Mức 1) nhưng "
                "không có số quy về BIDV; các số (2021-2030, ISO) là của chiến lược quốc gia."}),
    ("Ngân hàng đặt mục tiêu giảm 30% cường độ phát thải khí nhà kính vào năm 2030 so với "
     "mức năm 2020.",
     {"items": [{"action_or_event": "giảm cường độ phát thải khí nhà kính",
                 "figure": "30% vào 2030", "is_quantified": True,
                 "attributable_to_actor": True, "is_concrete_action": True}],
      "has_baseline_or_timeline": True,
      "reason": "Mục tiêu định lượng 30% có mốc 2020 làm baseline và mốc 2030, quy về chủ thể (Mức 2)."}),
    # Đoạn NHIỀU hành động -> NHIỀU item trong CÙNG một mảng "items" (KHÔNG tách mỗi item một mảng).
    ("Năm 2023, dư nợ tín dụng xanh của ngân hàng đạt 74.000 tỷ đồng, tăng 12% so với năm trước; "
     "đồng thời ngân hàng tài trợ 109,5 tỷ đồng cho lĩnh vực giáo dục và trồng 330.000 cây xanh.",
     {"items": [{"action_or_event": "dư nợ tín dụng xanh", "figure": "74.000 tỷ đồng",
                 "is_quantified": True, "attributable_to_actor": True, "is_concrete_action": True},
                {"action_or_event": "tài trợ lĩnh vực giáo dục", "figure": "109,5 tỷ đồng",
                 "is_quantified": True, "attributable_to_actor": True, "is_concrete_action": True},
                {"action_or_event": "trồng cây xanh", "figure": "330.000 cây",
                 "is_quantified": True, "attributable_to_actor": True, "is_concrete_action": True}],
      "has_baseline_or_timeline": True,
      "reason": "Ba đại lượng định lượng quy về ngân hàng, có mốc 2023 và so với năm trước (Mức 2)."}),
    ("BIDV sẽ chuyển đổi toàn diện, đồng bộ, thực chất, bài bản tất cả các hoạt động, "
     "chuyển đổi mạnh mẽ căn bản từ tư duy nhận thức, nâng cao năng lực quản trị điều hành, "
     "hướng tới phát triển xanh, bền vững.",
     {"items": [{"action_or_event": "chuyển đổi toàn diện, đồng bộ, thực chất, bài bản",
                 "figure": None, "is_quantified": False, "attributable_to_actor": True,
                 "is_concrete_action": False},
                {"action_or_event": "nâng cao năng lực quản trị điều hành", "figure": None,
                 "is_quantified": False, "attributable_to_actor": True,
                 "is_concrete_action": False}],
      "has_baseline_or_timeline": False,
      "reason": "Toàn khẩu hiệu/tính từ chung chung, không nêu công cụ/chương trình cụ thể "
                "nào kiểm chứng được, không có số (Mức 0 - mơ hồ)."}),
]

DEFAULT_WEIGHTS = {"quantified_attributable": 0.6,
                   "baseline_or_timeline": 0.2, "any_quantified": 0.2}


# Item object PHANG (khong ngoac long nhau) co khoa action_or_event — de salvage khi JSON hong.
_ITEM_RE = re.compile(r'\{[^{}]*?"action_or_event"[^{}]*?\}', re.DOTALL)
_HAS_BT_RE = re.compile(r'"has_baseline_or_timeline"\s*:\s*(true|false)')


def _extract_json(text: str) -> dict | None:
    """Bo <think>...</think> (Qwen3) roi lay rubric. Thu parse chuan truoc; neu hong thi
    SALVAGE: Qwen3-1.7B hay sinh sai ngoac ({"items":[o], [o], [o]}) hoac bi truncate ->
    vot moi item object phang bang regex + tim co has_baseline_or_timeline. Giu duoc ~3/4
    truong hop ma neu khong se fallback is_specific=0 (lam CTI thoi phong)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # 1) JSON dung chuan: lay object dau tien khop ngoac
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if isinstance(obj, dict) and isinstance(obj.get("items"), list):
                            return obj
                    except json.JSONDecodeError:
                        pass
                    break
    # 2) Salvage: gom moi item object phang con doc duoc (bo cai cuoi bi truncate)
    items = []
    for m in _ITEM_RE.finditer(text):
        try:
            o = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(o, dict) and "action_or_event" in o:
            items.append(o)
    if not items:
        return None
    hb = _HAS_BT_RE.search(text)
    return {"items": items, "has_baseline_or_timeline": bool(hb and hb.group(1) == "true")}


_DIGITS_RE = re.compile(r"\d+")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def _digit_runs(s: str) -> list[str]:
    """Cac chuoi chu so trong s, da bo dau phan cach (1,89 / 5.000 -> 189 / 5000)."""
    return _DIGITS_RE.findall(re.sub(r"(?<=\d)[.,\s](?=\d)", "", str(s)))


def verify_rubric(rubric: dict, text: str) -> dict:
    """Chong bia: model 0.6B hay copy figure tu few-shot / bia so khong co trong doan.
    - item.is_quantified chi giu True neu figure co chu so XUAT HIEN trong `text`.
    - has_baseline_or_timeline chi giu True neu doan co nam (19xx/20xx) hoac cum 'so voi'.
    Tra ve ban rubric da loc (khong sua tai cho)."""
    text_digits = set(_digit_runs(text))
    low = str(text).lower()
    items = []
    for it in (rubric.get("items") or []):
        it = dict(it)
        if it.get("is_quantified"):
            figs = _digit_runs(it.get("figure") or "")
            if not figs or not any(f in text_digits for f in figs):
                it["is_quantified"] = False  # figure khong co thuc trong doan -> huy
        items.append(it)
    has_bt = bool(rubric.get("has_baseline_or_timeline")) and (
        bool(_YEAR_RE.search(text)) or "so với" in low or "so voi" in low)
    return {**rubric, "items": items, "has_baseline_or_timeline": has_bt}


def derive(rubric: dict, weights: dict | None = None) -> tuple[float, int]:
    """rubric -> (p_specificity, spec_level) bang luat tuong minh. spec_level:
      2 = DINH LUONG: co item is_quantified & attributable_to_actor (so do duoc quy ve chu the)
      1 = CU THE   : co item is_concrete_action & attributable_to_actor (hanh dong/cong cu co ten,
                     kiem chung duoc) nhung khong dat Muc 2
      0 = MO HO    : chi khau hieu/tinh tu, khong kiem chung duoc
    is_specific = (spec_level >= 1) suy ra o ngoai. CTI = ti le Muc 0 (cheap talk that su)."""
    items = rubric.get("items") or []
    quant_attr = any(bool(it.get("is_quantified")) and bool(it.get("attributable_to_actor"))
                     for it in items)
    concrete = any(bool(it.get("is_concrete_action")) and bool(it.get("attributable_to_actor"))
                   for it in items)
    level = 2 if quant_attr else (1 if concrete else 0)
    return {0: 0.0, 1: 0.5, 2: 1.0}[level], level


class SpecificityLLM:
    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.model_name = cfg.get("model", "Qwen/Qwen3-0.6B")
        self.max_new_tokens = int(cfg.get("max_new_tokens", 256))
        self.retries = int(cfg.get("retries", 2))
        self.enable_thinking = bool(cfg.get("enable_thinking", False))
        self.weights = {**DEFAULT_WEIGHTS, **cfg.get("score_weights", {})}
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
        rubric, raw = None, ""
        for attempt in range(self.retries + 1):
            raw = self._complete(self._build_messages(text, stricter=attempt > 0))
            rubric = _extract_json(raw)
            if rubric is not None and "items" in rubric:
                break
        # LUON luu raw response (truoc parse) de trace / re-parse offline duoc.
        if rubric is None or "items" not in rubric:
            return {"p_specificity": 0.0, "spec_level": 0, "is_specific": 0,
                    "parse_ok": False, "rubric": pd.NA, "raw": raw}
        rubric = verify_rubric(rubric, text)  # huy figure bia / baseline khong co trong doan
        p, level = derive(rubric, self.weights)
        return {"p_specificity": p, "spec_level": level, "is_specific": int(level >= 1),
                "parse_ok": True, "rubric": json.dumps(rubric, ensure_ascii=False), "raw": raw}

    def predict(self, sentences: list[str]) -> pd.DataFrame:
        from tqdm.auto import tqdm
        rows = [self.score_one(str(t))
                for t in tqdm(sentences, desc="specificity-LLM", unit="chunk")]
        return pd.DataFrame(rows, columns=["p_specificity", "spec_level", "is_specific",
                                           "parse_ok", "rubric", "raw"])
