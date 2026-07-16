"""P1.c: thong ke verifier chong bia tren panel 45 bao cao.

So co TRUOC verifier (spec_rubric, = parse LLM goc) vs co CUOI (cot flags sau verifier),
tach theo 2 co che THAT trong specificity_llm.py:
  - verify_rubric_flags (digit-check): huy co_so_dinh_luong neu chu so trong evidence khong co trong text
  - enforce_evidence (substring-check): ha co neu evidence khong phai substring cua text
Import ham that -> replicate chinh xac; doi chieu final_computed == cot luu (sanity).
  python experiments/verifier_stats.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from esgwash.models.specificity_llm import ATOMIC_FLAGS, enforce_evidence, verify_rubric_flags  # noqa: E402

df = pd.concat([pd.read_parquet(p) for p in sorted((ROOT / "outputs/cti").glob("*/*/classified.parquet"))],
               ignore_index=True)
com = df[df["is_commitment"] == 1].copy()

s = {"n_committed": int(len(com)), "n_parse_ok": 0, "n_any_downgrade": 0,
     "quant_claimed": 0, "quant_dropped_digit": 0,
     "flags_claimed_after_digit": 0, "flags_dropped_substring": 0, "evidence_pass": 0,
     "mismatch_vs_stored": 0}

for _, row in com.iterrows():
    r = row.get("spec_rubric")
    if not isinstance(r, str):
        continue
    try:
        obj = json.loads(r)
    except Exception:
        continue
    bflags = obj.get("flags")
    if not isinstance(bflags, dict):
        continue
    ev = obj.get("evidence") or {}
    before = {f: int(bool(bflags.get(f, 0))) for f in ATOMIC_FLAGS}
    text = str(row["content_text"])
    s["n_parse_ok"] += 1

    after_digit = verify_rubric_flags(before, ev, text)     # co che 1
    final = enforce_evidence(after_digit, ev, text)         # co che 2
    stored = {f: int(row[f]) for f in ATOMIC_FLAGS}
    if final != stored:
        s["mismatch_vs_stored"] += 1

    if any(before[f] == 1 and final[f] == 0 for f in ATOMIC_FLAGS):
        s["n_any_downgrade"] += 1
    if before["co_so_dinh_luong"] == 1:
        s["quant_claimed"] += 1
        if after_digit["co_so_dinh_luong"] == 0:
            s["quant_dropped_digit"] += 1
    for f in ATOMIC_FLAGS:
        if after_digit[f] == 1:
            s["flags_claimed_after_digit"] += 1
            if final[f] == 0:
                s["flags_dropped_substring"] += 1
            else:
                s["evidence_pass"] += 1


def pct(a, b):
    return round(100 * a / b, 1) if b else 0.0


s["pct_chunks_any_downgrade"] = pct(s["n_any_downgrade"], s["n_parse_ok"])
s["pct_quant_dropped_by_digit_check"] = pct(s["quant_dropped_digit"], s["quant_claimed"])
s["pct_flags_dropped_by_substring"] = pct(s["flags_dropped_substring"], s["flags_claimed_after_digit"])
s["pct_evidence_pass_substring"] = pct(s["evidence_pass"], s["flags_claimed_after_digit"])

print(json.dumps(s, indent=2, ensure_ascii=False))
(ROOT / "experiments/eval/verifier_stats.json").write_text(
    json.dumps(s, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\n-> experiments/eval/verifier_stats.json")
