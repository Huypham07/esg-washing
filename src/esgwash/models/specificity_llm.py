"""Specificity scorer bang small instruct-LLM + rubric (thay M2-specificity, 2026-06-14).

Ly do (xem thao luan): encoder hay bat shortcut "co chu so -> specific". Vi du cau BIDV
day so (2021-2030, ISO, VIETGAP) nhung KHONG specific vi cac so do tro toi chien luoc
quoc gia / dieu kien vay, khong phai dai luong dinh luong QUY VE chu the. LLM voi rubric
co the phan ra "hanh dong/su kien -> so lieu" va suy luan dieu nay.

Output co cau truc (phan tich nguoc duoc):
  items: [{action_or_event, figure, is_quantified, attributable_to_actor}], has_baseline_or_timeline, reason
Nhan suy ra bang LUAT TUONG MINH (khong de model tu do quyet, khong bia thang do):
  is_specific = 1 <=> ton tai item vua is_quantified vua attributable_to_actor.
  p_specificity = tong co trong so checklist (trong so trong config -> sensitivity sweep duoc).
Retry khi parse JSON loi; het retry -> fallback is_specific=0 (bao thu, coi nhu cheap talk).
"""
from __future__ import annotations

import json
import re

import pandas as pd

SYSTEM = (
    "Bạn là chuyên gia phân tích báo cáo ESG ngân hàng. Với mỗi ĐOẠN VĂN, hãy xác định "
    "đoạn có chứa CAM KẾT/HÀNH ĐỘNG ĐỊNH LƯỢNG QUY VỀ CHÍNH CHỦ THỂ hay không "
    "(phân rã TỪNG hành động/sự kiện trong đoạn thành các item riêng). "
    "Một con số chỉ tính là 'định lượng quy về chủ thể' khi nó là mục tiêu/kết quả đo "
    "lường được của chủ thể (ví dụ: giảm 30% phát thải, dư nợ tín dụng xanh 5.000 tỷ, "
    "100 MW điện mặt trời). KHÔNG tính: năm của chiến lược/luật quốc gia, tên tiêu chuẩn "
    "(ISO, VIETGAP), điều kiện đủ để vay, từ mơ hồ ('hấp dẫn', 'ưu đãi hơn'). "
    "Chỉ trả về JSON, không giải thích ngoài JSON."
)

SCHEMA_HINT = (
    'Trả về JSON đúng dạng:\n'
    '{"items": [{"action_or_event": "<hành động/sự kiện>", "figure": "<số liệu gắn với '
    'nó hoặc null>", "is_quantified": true/false, "attributable_to_actor": true/false}], '
    '"has_baseline_or_timeline": true/false, "reason": "<giải thích ngắn>"}'
)

# Few-shot: 1 negative (BIDV - đầy số nhưng không quy về chủ thể) + 1 positive.
FEWSHOT = [
    ("Hưởng ứng Chiến lược quốc gia về tăng trưởng xanh giai đoạn 2021-2030, tầm nhìn "
     "2050, BIDV đã ban hành gói Tín dụng xanh cho khách hàng cá nhân vay phát triển năng "
     "lượng sạch (điện mặt trời, điện gió) hoặc trồng trọt chăn nuôi theo VIETGAP, ISO với "
     "lãi suất hấp dẫn và ưu đãi hơn thông thường.",
     {"items": [{"action_or_event": "ban hành gói Tín dụng xanh", "figure": None,
                 "is_quantified": False, "attributable_to_actor": True}],
      "has_baseline_or_timeline": False,
      "reason": "Các số (2021-2030, 2050, ISO, VIETGAP) trỏ tới chiến lược quốc gia và "
                "điều kiện vay, không phải đại lượng định lượng quy về BIDV; lãi suất mơ hồ."}),
    ("Ngân hàng đặt mục tiêu giảm 30% cường độ phát thải khí nhà kính vào năm 2030 so với "
     "mức năm 2020.",
     {"items": [{"action_or_event": "giảm cường độ phát thải khí nhà kính",
                 "figure": "30% vào 2030", "is_quantified": True,
                 "attributable_to_actor": True}],
      "has_baseline_or_timeline": True,
      "reason": "Mục tiêu định lượng 30% có mốc 2020 làm baseline và mốc 2030, quy về chủ thể."}),
    # Đoạn NHIỀU hành động -> NHIỀU item trong CÙNG một mảng "items" (KHÔNG tách mỗi item một mảng).
    ("Năm 2023, dư nợ tín dụng xanh của ngân hàng đạt 74.000 tỷ đồng, tăng 12% so với năm trước; "
     "đồng thời ngân hàng tài trợ 109,5 tỷ đồng cho lĩnh vực giáo dục và trồng 330.000 cây xanh.",
     {"items": [{"action_or_event": "dư nợ tín dụng xanh", "figure": "74.000 tỷ đồng",
                 "is_quantified": True, "attributable_to_actor": True},
                {"action_or_event": "tài trợ lĩnh vực giáo dục", "figure": "109,5 tỷ đồng",
                 "is_quantified": True, "attributable_to_actor": True},
                {"action_or_event": "trồng cây xanh", "figure": "330.000 cây",
                 "is_quantified": True, "attributable_to_actor": True}],
      "has_baseline_or_timeline": True,
      "reason": "Ba đại lượng định lượng quy về ngân hàng, có mốc 2023 và so với năm trước."}),
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
    """rubric -> (p_specificity, is_specific) bang luat tuong minh."""
    w = weights or DEFAULT_WEIGHTS
    items = rubric.get("items") or []
    any_quant = any(bool(it.get("is_quantified")) for it in items)
    quant_attr = any(bool(it.get("is_quantified")) and bool(it.get("attributable_to_actor"))
                     for it in items)
    has_bt = bool(rubric.get("has_baseline_or_timeline"))
    score = (w["quantified_attributable"] * quant_attr
             + w["baseline_or_timeline"] * has_bt
             + w["any_quantified"] * any_quant)
    return min(1.0, round(float(score), 4)), int(quant_attr)


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
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, dtype=torch.float32).to(self._device).eval()

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
            return {"p_specificity": 0.0, "is_specific": 0, "parse_ok": False,
                    "rubric": pd.NA, "raw": raw}
        rubric = verify_rubric(rubric, text)  # huy figure bia / baseline khong co trong doan
        p, is_spec = derive(rubric, self.weights)
        return {"p_specificity": p, "is_specific": is_spec, "parse_ok": True,
                "rubric": json.dumps(rubric, ensure_ascii=False), "raw": raw}

    def predict(self, sentences: list[str]) -> pd.DataFrame:
        from tqdm.auto import tqdm
        rows = [self.score_one(str(t))
                for t in tqdm(sentences, desc="specificity-LLM", unit="chunk")]
        return pd.DataFrame(rows, columns=["p_specificity", "is_specific",
                                           "parse_ok", "rubric", "raw"])
