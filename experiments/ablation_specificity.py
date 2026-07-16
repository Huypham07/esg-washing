"""P3a — CPU baselines cho ablation tang specificity (KHONG GPU).

Tinh moi "cach cham" KHONG can LLM tren 390 chunk gold-committed:
  - C1 (full): decomposition + verifier + rule (spec_level da co)
  - C1-verifier: decomposition + rule, BO verifier (derive_flags tren flags PRE-verifier)
  - Majority-class: doan muc pho bien nhat (san cong bang)
  - Human-ceiling: QWK annotator A vs B (tran tren)
  - Bootstrap CI cho moi QWK
  - INV-CPU: flip-rate DigitPresenceScorer (nua CPU cua robustness test; nua LLM o P3c)

Recompute TU artifact — KHONG doc eval_gold_report.json (hong), KHONG hardcode.
Xuat: experiments/eval/ablation_metrics.json (v1, cac hang KHONG-LLM).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Import ham THAT tu pipeline (DRY + trung thanh voi luat; khong trigger torch — chi o _load)
from esgwash.models.specificity_llm import (  # noqa: E402
    ATOMIC_FLAGS, _parse_flags, derive_flags,
)
from esgwash.validation.digit_shortcut import (  # noqa: E402
    PERTURBATIONS, DigitPresenceScorer, flip_rates,
)

PARQUET = ROOT / "experiments/eval/gold_classified.parquet"
GOLD_A = ROOT / "data/gold_annot_1_relabeled.xlsx"
GOLD_B = ROOT / "data/gold_annot_2_relabeled.xlsx"
OUT = ROOT / "experiments/eval/ablation_metrics.json"

SEED = 42
N_BOOT = 2000
LEVELS = [0, 1, 2]


def qwk(pred, true) -> float:
    return float(cohen_kappa_score(pred, true, weights="quadratic"))


def bootstrap_ci(pred, true, n_boot=N_BOOT, seed=SEED) -> tuple[float, float]:
    """Percentile bootstrap 95% CI cho QWK (resample chunk co hoan lai)."""
    rng = np.random.default_rng(seed)
    pred, true = np.asarray(pred), np.asarray(true)
    n = len(pred)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        # bo resample suy bien (pred hoac true hang so -> kappa nan)
        if len(set(pred[idx])) < 2 and len(set(true[idx])) < 2:
            continue
        k = cohen_kappa_score(pred[idx], true[idx], weights="quadratic")
        if not np.isnan(k):
            stats.append(k)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


def paired_diff_ci(pred_a, pred_b, true, n_boot=N_BOOT, seed=SEED) -> dict:
    """Paired bootstrap cho QWK(a) - QWK(b) tren CUNG resample (a,b,true paired tren 390 chunk).
    Tra point + 95% CI + frac_gt0 (ti le resample co diff>0)."""
    rng = np.random.default_rng(seed)
    pa, pb, t = np.asarray(pred_a), np.asarray(pred_b), np.asarray(true)
    n = len(t)
    point = qwk(pa, t) - qwk(pb, t)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ti = t[idx]
        if len(set(ti)) < 2:
            continue
        try:
            d = (cohen_kappa_score(pa[idx], ti, weights="quadratic")
                 - cohen_kappa_score(pb[idx], ti, weights="quadratic"))
        except Exception:
            continue
        if not np.isnan(d):
            diffs.append(d)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"delta_qwk": round(float(point), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "frac_gt0": round(float(np.mean(np.asarray(diffs) > 0)), 4)}


def vdr_bias(pred, true) -> float:
    """rate_pred(level==0) - rate_true(level==0). Duong = over-predict vague."""
    return round(float(np.mean(np.asarray(pred) == 0) - np.mean(np.asarray(true) == 0)), 4)


def cond_metrics(pred, gA, gB) -> dict:
    pred = np.asarray(pred)
    lo, hi = bootstrap_ci(pred, gA)
    return {
        "qwk_A": round(qwk(pred, gA), 4),
        "qwk_A_ci95": [lo, hi],
        "qwk_B": round(qwk(pred, gB), 4),
        "acc_A": round(float(np.mean(pred == gA)), 4),
        "macro_f1_A": round(float(f1_score(gA, pred, labels=LEVELS, average="macro")), 4),
        "vdr_bias": vdr_bias(pred, gA),
        "level_dist": {int(k): int(v) for k, v in zip(*np.unique(pred, return_counts=True))},
    }


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        return "unknown"


def c1_noverif_from_rubric(row) -> int:
    """C1 BO verifier: derive_flags tren flags PRE-verifier luu trong spec_rubric."""
    if not bool(row["spec_parse_ok"]):
        return 0  # parse fail -> muc 0 (nhu C1 full)
    try:
        parsed = json.loads(row["spec_rubric"])
        return derive_flags(parsed["flags"])[1]
    except Exception:
        return 0


def c1_noverif_from_raw(row) -> int:
    """Cross-check: re-parse spec_raw (bo verify/enforce) roi derive."""
    if not bool(row["spec_parse_ok"]):
        return 0
    p = _parse_flags(str(row["spec_raw"]))
    return derive_flags(p["flags"])[1] if p is not None else 0


def main() -> None:
    df = pd.read_parquet(PARQUET)
    gA = (pd.read_excel(GOLD_A)[["chunk_id", "g_spec_level", "g_is_commit"]]
          .rename(columns={"g_spec_level": "gA", "g_is_commit": "commitA"}))
    gB = (pd.read_excel(GOLD_B)[["chunk_id", "g_spec_level", "g_is_commit"]]
          .rename(columns={"g_spec_level": "gB", "g_is_commit": "commitB"}))
    m = df.merge(gA, on="chunk_id", how="inner").merge(gB, on="chunk_id", how="inner")
    assert len(m) == len(df) == 400, f"merge loss: {len(m)} vs {len(df)}"

    com = m[m["commitA"] == 1].copy()
    n = len(com)
    print(f"[load] merged {len(m)}/400, committed(A)={n}")

    gA_v, gB_v = com["gA"].to_numpy(), com["gB"].to_numpy()

    # --- C1 full (stored) + sanity: derive_flags(post-verifier cols) == stored spec_level ---
    c1 = com["spec_level"].to_numpy()
    rederive = com.apply(lambda r: derive_flags({f: r[f] for f in ATOMIC_FLAGS})[1], axis=1).to_numpy()
    mism = int((rederive != c1).sum())
    print(f"[sanity] stored spec_level vs derive_flags(post-verifier cols): mismatch={mism}/{n}")

    # --- C1-verifier (2 duong, phai khop) ---
    nv1 = com.apply(c1_noverif_from_rubric, axis=1).to_numpy()
    nv2 = com.apply(c1_noverif_from_raw, axis=1).to_numpy()
    disagree = int((nv1 != nv2).sum())
    print(f"[C1-noverif] rubric-vs-raw disagree={disagree}/{n} (ky vong 0)")
    n_changed = int((nv1 != c1).sum())
    print(f"[C1-noverif] khac C1 full o {n_changed}/{n} chunk (dau chan verifier)")

    # --- majority ---
    maj_level = int(pd.Series(gA_v).mode().iloc[0])
    maj = np.full(n, maj_level)
    print(f"[majority] muc pho bien = {maj_level}")

    conditions = {
        "C1_full": cond_metrics(c1, gA_v, gB_v),
        "C1_minus_verifier": {**cond_metrics(nv1, gA_v, gB_v),
                              "n_changed_vs_full": n_changed},
        "majority_class": cond_metrics(maj, gA_v, gB_v),
    }

    # --- paired difference: verifier gain (C1_full vs C1_minus_verifier) ---
    verifier_gain = paired_diff_ci(c1, nv1, gA_v)
    print(f"[paired] verifier gain (full - noverif): delta={verifier_gain['delta_qwk']}, "
          f"CI={verifier_gain['ci95']}, frac_gt0={verifier_gain['frac_gt0']}")

    # --- human ceiling (A vs B) ---
    both = com[com["commitB"] == 1]
    ceiling = {
        "qwk_AB_committedA_n": [round(qwk(gA_v, gB_v), 4), n],
        "qwk_AB_bothCommitted_n": [round(qwk(both["gA"], both["gB"]), 4), int(len(both))],
    }
    print(f"[ceiling] A-vs-B: {ceiling['qwk_AB_committedA_n']} (committed-A), "
          f"{ceiling['qwk_AB_bothCommitted_n']} (both)")

    # --- INV-CPU: DigitPresenceScorer flip tren cau vague (gold level-0) ---
    vague_seeds = com.loc[com["gA"] == 0, "content_text"].astype(str).tolist()
    dp_flip = flip_rates(DigitPresenceScorer(), vague_seeds, PERTURBATIONS)
    print(f"[INV-CPU] DigitPresence: n_base_nonspecific={dp_flip['n_base_nonspecific']}, "
          f"overall_flip_rate={dp_flip['overall_flip_rate']}")

    # --- P3c Kaggle outputs: direct C2/C3 + C1-live agreement + INV C1-side (neu da tai ve experiments/eval/) ---
    KAG = ROOT / "experiments/eval"
    kaggle_paired, inv_c1, c1_live_mismatch, has_kaggle = {}, None, None, False
    if (KAG / "c2_direct.parquet").exists() and (KAG / "c3_fewshot.parquet").exists():
        has_kaggle = True

        def _load_direct(fname):
            d = pd.read_parquet(KAG / fname)[["chunk_id", "spec_level_pred", "parse_ok"]]
            mg = com[["chunk_id"]].merge(d, on="chunk_id", how="left")
            assert mg["spec_level_pred"].notna().all(), f"{fname}: thieu chunk trong 390 committed"
            return mg["spec_level_pred"].astype(int).to_numpy(), int((~mg["parse_ok"].astype(bool)).sum())

        c2_pred, c2_fail = _load_direct("c2_direct.parquet")
        c3_pred, c3_fail = _load_direct("c3_fewshot.parquet")
        conditions["C2_direct_0shot"] = {**cond_metrics(c2_pred, gA_v, gB_v), "parse_fail": c2_fail}
        conditions["C3_direct_fewshot"] = {**cond_metrics(c3_pred, gA_v, gB_v), "parse_fail": c3_fail}
        # decomposition value = C1_minus_verifier (no verifier) vs direct (also no verifier) — apples-to-apples
        kaggle_paired["decomp_c1noverif_vs_c2"] = paired_diff_ci(nv1, c2_pred, gA_v)
        kaggle_paired["decomp_c1noverif_vs_c3"] = paired_diff_ci(nv1, c3_pred, gA_v)
        kaggle_paired["fullmethod_c1_vs_c2"] = paired_diff_ci(c1, c2_pred, gA_v)
        kaggle_paired["fullmethod_c1_vs_c3"] = paired_diff_ci(c1, c3_pred, gA_v)
        print(f"[kaggle] C2/C3 loaded (parse_fail C2={c2_fail} C3={c3_fail})")
    if (KAG / "c1_live.parquet").exists():
        cl = pd.read_parquet(KAG / "c1_live.parquet")[["chunk_id", "spec_level"]].rename(
            columns={"spec_level": "c1_live"})
        mg = com[["chunk_id"]].merge(cl, on="chunk_id", how="left")
        c1_live_mismatch = int((mg["c1_live"].values != c1).sum())
        print(f"[kaggle] C1-live vs stored gold mismatch = {c1_live_mismatch}/{n}")
    if (KAG / "inv_c1_flip.json").exists():
        inv_c1 = json.loads((KAG / "inv_c1_flip.json").read_text(encoding="utf-8"))
        print(f"[kaggle] INV C1 overall: {inv_c1.get('overall')}")

    out = {
        "meta": {
            "phase": ("P3-full" if has_kaggle else "P3a-cpu-baselines"),
            "git_sha": git_sha(),
            "config": "configs/specificity.yml (Qwen3-1.7B, do_sample=False)",
            "n_committed_A": n,
            "seed": SEED, "n_boot": N_BOOT,
            "source": "gold_classified.parquet + gold_annot_{1,2}_relabeled.xlsx (recomputed; NOT eval_gold_report.json)",
            "sanity_mismatch_c1_stored_vs_rederive": mism,
            "c1_noverif_rubric_vs_raw_disagree": disagree,
        },
        "human_ceiling": ceiling,
        "paired_diffs": {"verifier_gain_full_minus_noverif": verifier_gain, **kaggle_paired},
        "conditions": conditions,
        "inv_cpu_digitpresence": dp_flip,
        "inv_c1_side": inv_c1,
        "c1_live_mismatch": c1_live_mismatch,
        "notes": [
            "C1_full = decomposition+verifier+rule (proposed method).",
            "C1_minus_verifier = derive_flags on PRE-verifier flags (spec_rubric); isolates verifier gain.",
            "C2/C3 = direct LLM (no decomposition). decomp value = C1_minus_verifier vs C2/C3 (both no verifier).",
            "INV: DigitPresence flip ~1.0 (CPU) vs C1 flip_quantity_shortcut (LLM); C1 thap = robust khong an digit-shortcut.",
        ],
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- bang tom tat ---
    tag = "FULL C1/C1-noverif/C2/C3/majority" if has_kaggle else "P3a CPU rows"
    print(f"\n=== ABLATION ({tag}) — n=390 committed ===")
    print(f"{'condition':<20}{'QWK_A':>8}{'  CI95':>16}{'QWK_B':>8}{'acc':>7}{'mF1':>7}{'VDRbias':>9}")
    for name, c in conditions.items():
        ci = f"[{c['qwk_A_ci95'][0]:.3f},{c['qwk_A_ci95'][1]:.3f}]"
        print(f"{name:<20}{c['qwk_A']:>8.3f}{ci:>16}{c['qwk_B']:>8.3f}"
              f"{c['acc_A']:>7.3f}{c['macro_f1_A']:>7.3f}{c['vdr_bias']:>+9.3f}")
    print(f"{'human_ceiling(A-B)':<20}{ceiling['qwk_AB_committedA_n'][0]:>8.3f}")
    if has_kaggle:
        print("\n-- decomposition/full-method value (paired delta QWK vs direct, CI95, frac>0) --")
        for k in ("decomp_c1noverif_vs_c2", "decomp_c1noverif_vs_c3", "fullmethod_c1_vs_c2", "fullmethod_c1_vs_c3"):
            d = kaggle_paired[k]
            print(f"  {k:<28} delta={d['delta_qwk']:+.3f} CI={d['ci95']} frac>0={d['frac_gt0']}")
        if inv_c1:
            print(f"\n-- INV robustness: DigitPresence flip={dp_flip['overall_flip_rate']} "
                  f"vs C1 flip_quantity_shortcut={inv_c1['overall']['flip_quantity_shortcut']:.3f} "
                  f"(is_specific={inv_c1['overall']['flip_is_specific']:.3f})")
    print(f"\n[out] {OUT}")


if __name__ == "__main__":
    main()
