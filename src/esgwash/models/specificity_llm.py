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
    "Ban la chuyen gia phan tich bao cao ESG ngan hang. Voi moi cau, hay xac dinh "
    "cau co chua CAM KET/HANH DONG DINH LUONG QUY VE CHINH CHU THE hay khong. "
    "Mot con so chi tinh la 'dinh luong quy ve chu the' khi no la muc tieu/ket qua do "
    "luong duoc cua chu the (vd: giam 30% phat thai, du no tin dung xanh 5.000 ty, "
    "100 MW dien mat troi). KHONG tinh: nam cua chien luoc/luat quoc gia, ten tieu chuan "
    "(ISO, VIETGAP), dieu kien du vay, tu mo ho ('hap dan', 'uu dai hon'). "
    "Chi tra ve JSON, khong giai thich ngoai JSON."
)

SCHEMA_HINT = (
    'Tra ve JSON dung dang:\n'
    '{"items": [{"action_or_event": "<hanh dong/su kien>", "figure": "<so lieu gan voi '
    'no hoac null>", "is_quantified": true/false, "attributable_to_actor": true/false}], '
    '"has_baseline_or_timeline": true/false, "reason": "<giai thich ngan>"}'
)

# Few-shot: 1 negative (BIDV - day so nhung khong quy ve chu the) + 1 positive.
FEWSHOT = [
    ("Huong ung Chien luoc quoc gia ve tang truong xanh giai doan 2021-2030, tam nhin "
     "2050, BIDV da ban hanh goi Tin dung xanh cho khach hang ca nhan vay phat trien nang "
     "luong sach (dien mat troi, dien gio) hoac trong trot chan nuoi theo VIETGAP, ISO voi "
     "lai suat hap dan va uu dai hon thong thuong.",
     {"items": [{"action_or_event": "ban hanh goi Tin dung xanh", "figure": None,
                 "is_quantified": False, "attributable_to_actor": True}],
      "has_baseline_or_timeline": False,
      "reason": "Cac so (2021-2030, 2050, ISO, VIETGAP) tro toi chien luoc quoc gia va "
                "dieu kien vay, khong phai dai luong dinh luong quy ve BIDV; lai suat mo ho."}),
    ("Ngan hang dat muc tieu giam 30% cuong do phat thai khi nha kinh vao nam 2030 so voi "
     "muc nam 2020.",
     {"items": [{"action_or_event": "giam cuong do phat thai khi nha kinh",
                 "figure": "30% vao 2030", "is_quantified": True,
                 "attributable_to_actor": True}],
      "has_baseline_or_timeline": True,
      "reason": "Muc tieu dinh luong 30% co m2020 lam baseline va moc 2030, quy ve chu the."}),
]

DEFAULT_WEIGHTS = {"quantified_attributable": 0.6,
                   "baseline_or_timeline": 0.2, "any_quantified": 0.2}


def _extract_json(text: str) -> dict | None:
    """Bo <think>...</think> (Qwen3) roi lay object JSON dau tien bang khop ngoac."""
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
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


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
            self.model_name, torch_dtype=torch.float32).to(self._device).eval()

    def _build_messages(self, text: str, stricter: bool = False) -> list[dict]:
        msgs = [{"role": "system", "content": SYSTEM}]
        for ex_text, ex_json in FEWSHOT:
            msgs.append({"role": "user", "content": f"Cau: {ex_text}\n{SCHEMA_HINT}"})
            msgs.append({"role": "assistant", "content": json.dumps(ex_json, ensure_ascii=False)})
        hint = SCHEMA_HINT + ("\nCHU Y: chi xuat JSON hop le, khong them chu nao khac."
                              if stricter else "")
        msgs.append({"role": "user", "content": f"Cau: {text}\n{hint}"})
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
        if rubric is None or "items" not in rubric:
            return {"p_specificity": 0.0, "is_specific": 0, "parse_ok": False,
                    "rubric": json.dumps({"raw": raw[:500]}, ensure_ascii=False)}
        p, is_spec = derive(rubric, self.weights)
        return {"p_specificity": p, "is_specific": is_spec, "parse_ok": True,
                "rubric": json.dumps(rubric, ensure_ascii=False)}

    def predict(self, sentences: list[str]) -> pd.DataFrame:
        rows = [self.score_one(str(t)) for t in sentences]
        return pd.DataFrame(rows, columns=["p_specificity", "is_specific",
                                           "parse_ok", "rubric"])
