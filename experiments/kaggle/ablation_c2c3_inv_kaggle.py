"""P3c — Kaggle GPU run: C1-live baseline + C2/C3 (direct prompting) + INV robustness.

CHAY DUY NHAT MOT LAN (quyet dinh #3). Kaggle-notebook script. GPU T4, ~2-4h.
Da qua review doi khang (13 findings) — vá het truoc khi bam.

=== PREREQS (upload lam Kaggle input) — QUAN TRONG ===
1. REPO code: src/esgwash/ + experiments/ablation_direct_parser.py (da commit ad6873e).
      Preflight assert (cell [2]) chan neu parser thieu du upload cach nao.
2. INPUTS (experiments/kaggle/): DERIVED, KHONG commit -> chay build_kaggle_inputs.py de tao truoc khi upload:
   c2c3_input.parquet (390: chunk_id, content_text, c1_gold_level) · inv_vague_seeds.json ·
   ablation_c3_fewshot.json · committed_chunk_ids.json
3. Kaggle: BAT Internet (Settings > Internet: On) — can cho pip + tai Qwen3-1.7B tu HF Hub.
=== OUTPUTS (/kaggle/working) ===
   c1_live.parquet, c2_direct.parquet, c3_fewshot.parquet (chunk_id, spec_level_pred*, parse_ok, raw)
   inv_c1_flip.json (flip is_specific + digit-shortcut isolate + level2) · run_provenance.json
   (*c1_live cot ten spec_level; C2/C3 cot spec_level_pred — GIU TEN KHAC de tranh merge-collision voi gold.)

Model: Qwen/Qwen3-1.7B, do_sample=False (greedy), enable_thinking=False, max_new_tokens=512 — MIRROR specificity_llm.py.
C1-live baseline: C1/C2/C3/INV cung MOT env -> "differ ONLY in decomposition"; log agreement vs stored gold.
"""
# %% [1] SETUP — pin transformers (>=4.56 cho dtype= kwarg cua Qwen3 load; EXACT de comparability).
import subprocess, sys
import importlib.metadata as _md
TRANSFORMERS_PIN = "4.57.6"   # >=4.56: specificity_llm.py:191 dung from_pretrained(dtype=...) (4.51.x se TypeError)
try:
    _have = _md.version("transformers")
except _md.PackageNotFoundError:
    _have = None
if _have != TRANSFORMERS_PIN:
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"transformers=={TRANSFORMERS_PIN}"])
    if r.returncode != 0:
        raise SystemExit(f"pip install transformers=={TRANSFORMERS_PIN} FAILED — bat Kaggle Internet (Settings>Internet:On).")

# %% [2] IMPORTS + PATHS + PREFLIGHT
import json, time
from pathlib import Path
import pandas as pd
import torch
import transformers

# --- AUTO-DETECT dataset root (khong hardcode slug/nesting) tu vi tri parser ---
import glob  # noqa: E402
_hit = glob.glob("/kaggle/input/**/experiments/ablation_direct_parser.py", recursive=True)
assert _hit, ("Khong thay ablation_direct_parser.py duoi /kaggle/input — "
              "da 'Add Input' dataset esgwash-ablation chua? (panel Input ben phai)")
DATA = Path(_hit[0]).parents[1]   # .../experiments/ablation_direct_parser.py -> thu muc chua src/ + experiments/ + 4 input
REPO_DIR = DATA
INPUT_DIR = DATA
OUT_DIR = Path("/kaggle/working")
print(f"DATA auto-detected = {DATA}")

assert transformers.__version__ == TRANSFORMERS_PIN, \
    f"transformers {transformers.__version__} != pin {TRANSFORMERS_PIN} — reinstall khong an, Restart kernel & chay lai."
_parser = REPO_DIR / "experiments" / "ablation_direct_parser.py"
assert _parser.exists(), f"THIEU {_parser} — file UNTRACKED, git archive bo sot. Copy file vao Kaggle dataset."
assert torch.cuda.is_available(), "Can GPU — bat GPU accelerator tren Kaggle."

sys.path.insert(0, str(REPO_DIR / "src"))
sys.path.insert(0, str(REPO_DIR / "experiments"))
from esgwash.models.specificity_llm import SpecificityLLM  # noqa: E402
from esgwash.validation.digit_shortcut import PERTURBATIONS               # noqa: E402
from ablation_direct_parser import build_direct_messages, parse_level     # noqa: E402
print(f"torch={torch.__version__} | transformers={transformers.__version__} | cuda={torch.cuda.is_available()}")

# %% [3] LOAD DATA + verify chunk-set == committed 390
df = pd.read_parquet(INPUT_DIR / "c2c3_input.parquet")
committed_ids = set(json.loads((INPUT_DIR / "committed_chunk_ids.json").read_text(encoding="utf-8")))
assert set(df["chunk_id"].astype(str)) == committed_ids and df["chunk_id"].is_unique and len(df) == 390, \
    "c2c3_input.parquet chunk-set != 390 committed — dung sample, rebuild input."
vague = json.loads((INPUT_DIR / "inv_vague_seeds.json").read_text(encoding="utf-8"))
vague_texts = [v["content_text"] for v in vague]
fewshot_json = json.loads((INPUT_DIR / "ablation_c3_fewshot.json").read_text(encoding="utf-8"))
FEWSHOT = [(x["text"], int(x["level"])) for x in fewshot_json]
print(f"loaded: {len(df)} chunks, {len(vague_texts)} vague seeds, {len(FEWSHOT)} fewshot")

# %% [4] MODEL — load 1 lan; INV + C1-live dung C1 pipeline that; C2/C3 tai dung cung model.
CFG = {"model": "Qwen/Qwen3-1.7B", "max_new_tokens": 512, "enable_thinking": False, "retries": 2}
spec = SpecificityLLM(CFG)
spec._load()
MAX_NEW, RETRIES = CFG["max_new_tokens"], CFG["retries"]


def direct_complete(messages: list[dict]) -> str:
    """Sinh direct-prompt — MIRROR specificity_llm._complete (greedy, enable_thinking=False). Self-guard _load."""
    spec._load()
    tok, model, device = spec._tok, spec._model, spec._device
    try:
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=spec.enable_thinking)
    except TypeError:
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


def score_direct(chunk: str, fewshot=None) -> dict:
    """C2 (fewshot=None) / C3 (fewshot=FEWSHOT): generate -> parse_level -> retry stricter -> fallback muc 0."""
    raw = ""
    for attempt in range(RETRIES + 1):
        raw = direct_complete(build_direct_messages(chunk, fewshot=fewshot, stricter=attempt > 0))
        level, ok = parse_level(raw)
        if ok:
            return {"spec_level_pred": level, "parse_ok": True, "raw": raw}
    return {"spec_level_pred": 0, "parse_ok": False, "raw": raw}


# %% [5] INV — do 3 tin hieu (KHONG sua digit_shortcut; dung PERTURBATIONS). Reviewer: is_specific bi
# nhiem named-action (level1 khong-so) -> them flip cua co_so_dinh_luong (isolate digit-shortcut) + level2.
def inv_flip_breakdown(base_texts: list[str], perturbations: dict) -> dict:
    out = {"n_base_nonspecific": len(base_texts), "per_perturbation": {}}
    if not base_texts:
        out["overall"] = None
        return out
    keys = {"flip_is_specific": [], "flip_quantity_shortcut": [], "flip_level2": []}
    for name, fn in perturbations.items():
        pred = spec.predict([fn(s) for s in base_texts])
        m = {"flip_is_specific": float((pred["is_specific"] == 1).mean()),
             "flip_quantity_shortcut": float((pred["co_so_dinh_luong"] == 1).mean()),  # digit-shortcut chuan
             "flip_level2": float((pred["spec_level"] == 2).mean())}
        out["per_perturbation"][name] = m
        for k in keys:
            keys[k].append(m[k])
    out["overall"] = {k: sum(v) / len(v) for k, v in keys.items()}
    return out


def inv_base_from(seed_texts: list[str]) -> list[str]:
    """Seeds ma C1 du doan NON-specific (khop semantics flip_rates)."""
    pred = spec.predict(seed_texts)
    return [s for s, isp in zip(seed_texts, pred["is_specific"]) if isp == 0]

# %% [6] DRY-RUN GATE — CA 2 arm C2 & C3 (C2 chay truoc, rui ro nhat) + INV smoke. FAIL -> ABORT.
print("\n=== DRY-RUN GATE (5 chunk, ca 2 arm) ===")
heads = df["content_text"].head(5).tolist()
dry_c2 = [score_direct(t) for t in heads]                    # C2 zero-shot
dry_c3 = [score_direct(t, fewshot=FEWSHOT) for t in heads]   # C3 few-shot
for name, arm in (("C2/0-shot", dry_c2), ("C3/few-shot", dry_c3)):
    for i, d in enumerate(arm):
        print(f"  [{name} {i}] level={d['spec_level_pred']} ok={d['parse_ok']} raw={d['raw'][:80]!r}")
dry_c2_ok = sum(d["parse_ok"] for d in dry_c2)
dry_c3_ok = sum(d["parse_ok"] for d in dry_c3)
print(f"  parse_ok: C2={dry_c2_ok}/5 C3={dry_c3_ok}/5 | levels C2={[d['spec_level_pred'] for d in dry_c2]} "
      f"C3={[d['spec_level_pred'] for d in dry_c3]}")
inv_smoke_base = inv_base_from(vague_texts[:5])
inv_smoke = inv_flip_breakdown(inv_smoke_base, {"national_strategy": PERTURBATIONS["national_strategy"]})
print(f"  INV smoke: n_base={inv_smoke['n_base_nonspecific']} flip_qty={inv_smoke.get('overall')}")
assert dry_c2_ok >= 4, f"DRY-RUN FAIL C2 zero-shot: {dry_c2_ok}/5 < 4 -> KHONG chay full, kiem prompt/parser."
assert dry_c3_ok >= 4, f"DRY-RUN FAIL C3 few-shot: {dry_c3_ok}/5 < 4 -> KHONG chay full."
assert inv_smoke["n_base_nonspecific"] >= 1, "INV SMOKE FAIL: C1 khong giu cau vague nao non-specific -> kiem C1/seeds."
print("DRY-RUN PASS (C2 + C3 + INV) -> chay full.\n")

# %% [7] FULL RUN
from tqdm.auto import tqdm
t0 = time.time()

print("=== C1-live (baseline cung env; full pipeline 4-co+verifier+luat tren 390) ===")
c1_out = spec.predict(list(df["content_text"]))
c1_live = pd.DataFrame({"chunk_id": df["chunk_id"].values, "spec_level": c1_out["spec_level"].values,
                        "is_specific": c1_out["is_specific"].values})
c1_live.to_parquet(OUT_DIR / "c1_live.parquet", index=False)
mismatch = int((c1_live["spec_level"].values != df["c1_gold_level"].values).sum())
print(f"C1-live vs stored gold: mismatch={mismatch}/390 ({mismatch/390:.1%}) -> "
      f"{'KHOP: C1-live tai lap gold -> so C2/C3 dung C1 gold/P3a (0.663)' if mismatch <= 20 else 'LECH: env confound -> dung C1-live lam baseline C2/C3'}")

print(f"=== C2 (direct zero-shot) === [t+{time.time()-t0:.0f}s]")
c2 = [{"chunk_id": cid, **score_direct(t)} for cid, t in tqdm(zip(df["chunk_id"], df["content_text"]), total=len(df))]
pd.DataFrame(c2).to_parquet(OUT_DIR / "c2_direct.parquet", index=False)
print(f"  C2 done [t+{time.time()-t0:.0f}s] — con C3 + INV (~tuong tu C2 + {len(vague_texts)}x4 INV). "
      f"Abort duoc o day neu ETA vuot quota.")

print("=== C3 (direct few-shot) ===")
c3 = [{"chunk_id": cid, **score_direct(t, fewshot=FEWSHOT)} for cid, t in
      tqdm(zip(df["chunk_id"], df["content_text"]), total=len(df))]
pd.DataFrame(c3).to_parquet(OUT_DIR / "c3_fewshot.parquet", index=False)

print("=== INV (C1 flip tren cau vague da chen so trang tri) ===")
# base = vague seeds ma C1-live du doan non-specific (tai dung c1_out, khoi predict lai 149)
c1_isp = dict(zip(df["chunk_id"].astype(str), c1_out["is_specific"].values))
inv_base = [v["content_text"] for v in vague if c1_isp.get(str(v["chunk_id"]), 1) == 0]
inv = inv_flip_breakdown(inv_base, PERTURBATIONS)
(OUT_DIR / "inv_c1_flip.json").write_text(json.dumps(inv, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"INV overall: {inv['overall']} (n_base={inv['n_base_nonspecific']})")

# %% [8] PROVENANCE
prov = {
    "model": CFG["model"], "do_sample": False, "enable_thinking": False,
    "max_new_tokens": MAX_NEW, "retries": RETRIES,
    "torch": torch.__version__, "transformers": transformers.__version__,
    "pred_column": {"c1_live": "spec_level", "c2_c3": "spec_level_pred"},
    "n_chunks": len(df), "n_vague_seeds": len(vague_texts), "n_fewshot": len(FEWSHOT),
    "c1_live_vs_gold_mismatch": f"{mismatch}/390",
    "dry_run_parse_ok": f"C2={dry_c2_ok}/5 C3={dry_c3_ok}/5",
    "c2_parse_fail": int((~pd.DataFrame(c2)["parse_ok"]).sum()),
    "c3_parse_fail": int((~pd.DataFrame(c3)["parse_ok"]).sum()),
    "inv_n_base": inv["n_base_nonspecific"], "inv_overall": inv["overall"],
    "elapsed_sec": round(time.time() - t0),
}
(OUT_DIR / "run_provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n=== DONE ===")
print(json.dumps(prov, ensure_ascii=False, indent=2))
print("Tai ve: c1_live, c2_direct, c3_fewshot (.parquet), inv_c1_flip.json, run_provenance.json")
