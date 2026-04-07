"""
EWRI: ESG-Washing Risk Index
============================

Theoretical Foundation
---------------------
Based on the Substantive vs Symbolic ESG Disclosure framework:

1. Marquis, C., Toffel, M., & Zhou, Y. (2016). "Scrutiny, Norms, and
   Selective Disclosure." - ESG-washing = gap between symbolic & substantive.

2. Lyon, T. P., & Montgomery, A. W. (2015). "The Means and End of
   Greenwash." - Unsubstantiated claims as primary greenwash mechanism.

3. Delmas, M. A., & Burbano, V. C. (2011). "The Drivers of Greenwashing."
   CMR. - Typology: selective disclosure, vague claims, false labels.

4. GRI Standards (2021). - Evidence quality: KPIs, time-bound targets,
   third-party verification, standard references.


Core Formula: Sentence-Level Washing Risk Score (WRS)
-----------------------------------------------------
    WRS_i = P_action(y_i) × (1 - λ(y_i) × ES_i)

Where:
- P_action(y): Action Penalty — inherent risk of each claim type
    · Indeterminate = 0.70  (high: vague, but not all are claims;
                             Delmas & Burbano 2011 distinguish vague
                             claims from contextual/procedural text)
    · Planning     = 0.55  (moderate: forward-looking, uncommitted)
    · Implemented  = 0.25  (low base: concrete but self-reported)

    Rationale for P(Indet) < 1.0:
        In Vietnamese ESG reports, ~65% sentences are classified as
        Indeterminate. Empirical inspection shows ~30% are contextual
        (cross-references, table headers, procedural text), not actual
        ESG claims. Per Cho et al. (2010), only content sentences that
        make evaluative ESG claims should receive full penalty.

- λ(y): Evidence Sensitivity — how effectively evidence reduces risk
    · Implemented   = 0.90  (evidence strongly validates concrete actions)
    · Planning      = 0.70  (evidence moderately validates commitments)
    · Indeterminate = 0.50  (evidence partially substantiates vague claims;
                             higher than before because when evidence IS
                             found for a vague claim, it provides meaningful
                             grounding — Marquis et al. 2016)

- ES_i: Unified Evidence Strength ∈ [0, 1]
    · ES = w_qual × Q(types) + w_sim × sim_calibrated + w_nli × D(nli)
    · Q: GRI quality hierarchy (Third_party > KPI > Standard > Time_bound)
    · sim_calibrated: Relative similarity (per-document calibration)
    · D(nli): Directional NLI (entailment ↑, contradiction → WRS × 1.3)

Key Insight (Multiplicative Interaction):
    "Implemented + No Evidence" → 0.25 × 1.0 = 0.25 (suspicious)
    "Implemented + Full Evidence" → 0.25 × 0.10 = 0.025 (genuine)
    "Indeterminate + No Evidence" → 0.70 × 1.0 = 0.70 (high risk)
    "Indeterminate + Full Evidence" → 0.70 × 0.50 = 0.35 (moderated)

Bank-Year EWRI:
    EWRI = (1/N) × Σ_i WRS_i × 100
"""

import math
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List


# ============================================================
# CONFIGURATION
# ============================================================

VALID_EVIDENCE_TYPES = ["KPI", "Standard", "Time_bound", "Third_party"]

# Action Penalty P_action(y)
# Marquis et al. (2016); Delmas & Burbano (2011)
# P(Indet) = 0.55: ~30-40% of Indeterminate sentences are contextual/procedural
# text, not actual ESG claims (Cho et al., 2010). 55% base risk reflects
# that roughly half are potentially vague claims deserving scrutiny.
# P(Impl) = 0.08: Concrete, verifiable claims carry very low base risk
# because they make falsifiable commitments (Lyon & Montgomery, 2015).
# P(Plan) = 0.30: Forward-looking statements carry moderate risk (Marquis et al., 2016).
ACTION_PENALTY = {
    "Implemented": 0.08,
    "Planning": 0.30,
    "Indeterminate": 0.55,
}

# Evidence Sensitivity λ(y)
# Higher λ = evidence reduces risk more effectively.
# λ(Impl) = 0.80: Evidence strongly validates concrete claims
# λ(Plan) = 0.65: Evidence partially grounds forward-looking statements
# λ(Indet) = 0.50: Evidence for vague claims provides moderate grounding
# (Marquis et al., 2016; GRI, 2021)
EVIDENCE_SENSITIVITY = {
    "Implemented": 0.80,
    "Planning": 0.65,
    "Indeterminate": 0.50,
}

# Unified Evidence Strength weights (GRI Standards, 2021)
W_QUALITY = 0.35      # Evidence quality from GRI type hierarchy
W_SIMILARITY = 0.35   # Calibrated semantic similarity
W_NLI = 0.30          # NLI verification (directional)

# Evidence Quality Hierarchy (GRI, 2021; Cho et al., 2012)
EVIDENCE_QUALITY_WEIGHTS = {
    "Third_party": 0.35,  # Tier 1: External verification
    "KPI": 0.30,          # Tier 2: Quantitative, falsifiable
    "Standard": 0.20,     # Tier 3: Normative framework
    "Time_bound": 0.15,   # Tier 4: Temporal specificity
}

# Contradiction amplifier (Lyon & Montgomery, 2015)
# "contradicted claims indicate stronger greenwashing than unsubstantiated"
CONTRADICTION_AMPLIFIER = 1.3

# Risk level thresholds
# Calibrated to the recalibrated P/λ parameters so that risk levels
# distribute meaningfully across the Vietnamese banking sector.
#
# Theoretical anchors (with P(Indet)=0.55, P(Impl)=0.08):
#   "All Impl + full ES"     → EWRI ≈  1.6  → Low
#   "All Impl + no ES"      → EWRI ≈  8.0  → Low
#   "60% Indet + 35% Impl"  → EWRI ≈ 33    → Medium
#   "75% Indet + poor ES"   → EWRI ≈ 41    → High
#   "All Indet + no ES"     → EWRI ≈ 55    → Very High
RISK_THRESHOLDS = {"Low": 25, "Medium": 38, "High": 43}

# Old formula parameters (for ablation comparison)
OLD_ALPHA = 2.0
OLD_BETA = 1.0
OLD_GAMMA = 1.0

ESG_TOPICS = ["E", "S_labor", "S_community", "S_product", "G"]


# ============================================================
# DATA CLASS
# ============================================================

@dataclass
class EWRIScore:
    """EWRI score and breakdown for a bank-year."""
    bank: str
    year: int
    total_sentences: int

    implemented: int
    planning: int
    indeterminate: int

    with_evidence: int
    without_evidence: int

    implemented_ratio: float
    planning_ratio: float
    indeterminate_ratio: float
    evidence_ratio: float
    avg_evidence_strength: float

    ewri: float
    ewri_old: float
    risk_level: str

    # Decomposition
    contribution_implemented: float
    contribution_planning: float
    contribution_indeterminate: float

    topic_breakdown: dict = field(default_factory=dict)
    topic_entropy: float = 0.0
    topic_coverage_index: float = 0.0

    sentence_risks: List[dict] = field(default_factory=list)


# ============================================================
# EVIDENCE STRENGTH
# ============================================================

def compute_evidence_score(row: pd.Series) -> float:
    """
    Unified Evidence Strength (GRI, 2021; Cho et al., 2012):

        ES = w_qual × Q(types) + w_sim × sim_calibrated + w_nli × D(nli)

    Components:
        Q(types): Quality-weighted count of evidence types, normalized
                  by max possible score. Uses GRI disclosure hierarchy.
        sim_calibrated: Relative semantic similarity, calibrated against
                       per-document mean/std to address uniformly high
                       within-document similarity.
        D(nli): Directional NLI signal:
                - entailment → boost (claim supported by evidence)
                - contradiction → 0 (penalty applied via WRS amplifier)
                - neutral → 0 (no signal)
    """
    # Component 1: Evidence Quality (GRI hierarchy)
    evidence_types = row.get("evidence_types", [])
    if isinstance(evidence_types, np.ndarray):
        evidence_types = evidence_types.tolist()
    if not isinstance(evidence_types, list):
        evidence_types = []

    quality = 0.0
    max_quality = sum(EVIDENCE_QUALITY_WEIGHTS.values())
    for etype in evidence_types:
        quality += EVIDENCE_QUALITY_WEIGHTS.get(etype, 0.0)
    quality = quality / max_quality if max_quality > 0 else 0.0

    # Component 2: Calibrated Similarity
    sim_raw = float(row.get("similarity_score", 0.0) or 0.0)
    sim_mean = float(row.get("sim_doc_mean", 0.0) or 0.0)
    sim_std = float(row.get("sim_doc_std", 0.1) or 0.1)
    if sim_std > 0.01:
        sim_calibrated = max(0.0, min(1.0, (sim_raw - sim_mean) / (2 * sim_std)))
    else:
        sim_calibrated = max(0.0, min(1.0, sim_raw))

    # Component 3: Directional NLI
    nli_label = str(row.get("nli_label", "") or "")
    nli_score = float(row.get("nli_entailment_score", 0.0) or 0.0)
    nli_direction = nli_score if nli_label == "entailment" else 0.0

    # Unified ES
    es = W_QUALITY * quality + W_SIMILARITY * sim_calibrated + W_NLI * nli_direction
    return max(0.0, min(1.0, es))


# ============================================================
# WASHING RISK SCORE
# ============================================================

def compute_washing_risk(
    action_label: str, evidence_strength: float, nli_label: str = ""
) -> float:
    """
    WRS = P_action(y) × (1 - λ(y) × ES) × C(nli)

    Where C(nli) is the contradiction amplifier:
        - contradiction: 1.3× (Lyon & Montgomery, 2015 — contradicted
          claims indicate stronger greenwashing than unsubstantiated)
        - otherwise: 1.0
    """
    p = ACTION_PENALTY.get(action_label, 0.55)
    lam = EVIDENCE_SENSITIVITY.get(action_label, 0.50)
    es = max(0.0, min(1.0, evidence_strength))
    wrs = p * (1.0 - lam * es)

    if nli_label == "contradiction":
        wrs = min(1.0, wrs * CONTRADICTION_AMPLIFIER)

    return max(0.0, min(1.0, wrs))


def compute_ewri_old(df_group: pd.DataFrame) -> float:
    """
    Original ADDITIVE formula (baseline for comparison).
    Risk_i = α×I(Indet) + β×I(Plan) + γ×(1-ES)
    EWRI = Σ Risk_i / (N × (α+β+γ)) × 100
    """
    alpha, beta, gamma = OLD_ALPHA, OLD_BETA, OLD_GAMMA
    max_possible = alpha + beta + gamma
    total_risk = 0.0

    for _, row in df_group.iterrows():
        label = row.get("action_label", "Unknown")
        act = alpha if label == "Indeterminate" else (beta if label == "Planning" else 0.0)
        es = compute_evidence_score(row)
        total_risk += act + gamma * (1.0 - es)

    N = len(df_group)
    return (total_risk / (N * max_possible)) * 100 if N > 0 else 0.0


def get_risk_level(ewri: float) -> str:
    if ewri < RISK_THRESHOLDS["Low"]:
        return "Low"
    elif ewri < RISK_THRESHOLDS["Medium"]:
        return "Medium"
    elif ewri < RISK_THRESHOLDS["High"]:
        return "High"
    return "Very High"


# ============================================================
# TOPIC COVERAGE
# ============================================================

def calculate_topic_entropy(topic_counts: dict) -> float:
    """Normalized Shannon entropy ∈ [0,1]. 1 = uniform, 0 = single-topic."""
    total = sum(topic_counts.values())
    if total == 0:
        return 0.0
    probs = [c / total for c in topic_counts.values() if c > 0]
    if len(probs) <= 1:
        return 0.0
    entropy = -sum(p * math.log2(p) for p in probs)
    max_entropy = math.log2(len(probs))
    return entropy / max_entropy if max_entropy > 0 else 0.0


# ============================================================
# ENRICHMENT
# ============================================================

def enrich_with_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-sentence columns: es_combined, wrs, wrs_old.
    Includes per-document similarity calibration for the unified ES formula.
    """
    df = df.copy()

    # Per-document similarity statistics for calibration
    if "similarity_score" in df.columns and "bank" in df.columns:
        doc_sim_stats = (
            df.groupby(["bank", "year"])["similarity_score"]
            .agg(["mean", "std"])
            .reset_index()
        )
        doc_sim_stats.columns = ["bank", "year", "sim_doc_mean", "sim_doc_std"]
        df = df.merge(doc_sim_stats, on=["bank", "year"], how="left")
        df["sim_doc_std"] = df["sim_doc_std"].fillna(0.1)
        df["sim_doc_mean"] = df["sim_doc_mean"].fillna(0.0)
    else:
        df["sim_doc_mean"] = 0.0
        df["sim_doc_std"] = 0.1

    df["es_combined"] = df.apply(compute_evidence_score, axis=1)

    df["wrs"] = df.apply(
        lambda row: compute_washing_risk(
            row["action_label"], row["es_combined"],
            str(row.get("nli_label", "") or "")
        ),
        axis=1,
    )

    def _wrs_old(row):
        label = row["action_label"]
        es = row["es_combined"]
        act = OLD_ALPHA if label == "Indeterminate" else (OLD_BETA if label == "Planning" else 0.0)
        gap = OLD_GAMMA * (1.0 - es)
        return (act + gap) / (OLD_ALPHA + OLD_BETA + OLD_GAMMA)

    df["wrs_old"] = df.apply(_wrs_old, axis=1)
    return df


# ============================================================
# BANK-YEAR EWRI
# ============================================================

def _topic_breakdown(group: pd.DataFrame, topic_col: str) -> dict:
    breakdown = {}
    for topic in ESG_TOPICS:
        tdf = group[group[topic_col] == topic]
        n = len(tdf)
        if n < 3:
            breakdown[topic] = {"ewri": None, "n": n, "risk_level": "N/A"}
            continue
        t_ewri = tdf["wrs"].mean() * 100
        impl = int((tdf["action_label"] == "Implemented").sum())
        indet = int((tdf["action_label"] == "Indeterminate").sum())
        has_ev = int(tdf["has_evidence"].sum()) if "has_evidence" in tdf.columns else 0
        breakdown[topic] = {
            "ewri": round(t_ewri, 2),
            "n": n,
            "risk_level": get_risk_level(t_ewri),
            "implemented_pct": round(impl / n * 100, 1),
            "indeterminate_pct": round(indet / n * 100, 1),
            "evidence_rate": round(has_ev / n * 100, 1),
        }
    return breakdown


def calculate_bank_year_ewri(df: pd.DataFrame) -> list[EWRIScore]:
    """Calculate EWRI for each bank-year with full breakdown."""
    topic_col = "topic_label" if "topic_label" in df.columns else None

    if "wrs" not in df.columns:
        df = enrich_with_risk_scores(df)

    scores = []

    for (bank, year), group in df.groupby(["bank", "year"]):
        N = len(group)
        if N == 0:
            continue

        impl = int((group["action_label"] == "Implemented").sum())
        plan = int((group["action_label"] == "Planning").sum())
        indet = int((group["action_label"] == "Indeterminate").sum())

        with_ev = int(group["has_evidence"].sum()) if "has_evidence" in group.columns else 0

        ewri_new = group["wrs"].mean() * 100
        ewri_old = group["wrs_old"].mean() * 100
        risk_level = get_risk_level(ewri_new)

        # Decomposition
        impl_mask = group["action_label"] == "Implemented"
        plan_mask = group["action_label"] == "Planning"
        indet_mask = group["action_label"] == "Indeterminate"
        c_impl = group.loc[impl_mask, "wrs"].sum() / N * 100 if impl > 0 else 0.0
        c_plan = group.loc[plan_mask, "wrs"].sum() / N * 100 if plan > 0 else 0.0
        c_indet = group.loc[indet_mask, "wrs"].sum() / N * 100 if indet > 0 else 0.0

        avg_es = group["es_combined"].mean()

        # Topic
        topic_bd = {}
        topic_counts = {}
        if topic_col:
            topic_bd = _topic_breakdown(group, topic_col)
            for t in ESG_TOPICS:
                topic_counts[t] = int((group[topic_col] == t).sum())

        te = calculate_topic_entropy(topic_counts) if topic_counts else 0.0

        # Sentence traceability
        sent_risks = []
        for _, row in group.iterrows():
            sent_risks.append({
                "sent_id": row.get("sent_id", ""),
                "sentence": str(row.get("sentence", row.get("text", "")))[:250],
                "action_label": row.get("action_label", "Unknown"),
                "action_confidence": round(float(row.get("action_confidence", 0.0)), 3),
                "has_evidence": bool(row.get("has_evidence", False)),
                "evidence_strength": round(float(row.get("es_combined", 0.0)), 3),
                "washing_risk": round(float(row.get("wrs", 0.0)), 3),
                "topic": row.get(topic_col, "Unknown") if topic_col else "Unknown",
                "evidence_types": list(row.get("evidence_types", [])) if isinstance(row.get("evidence_types", []), (list, np.ndarray)) else [],
                "nli_label": row.get("nli_label", ""),
                "best_evidence": str(row.get("best_evidence", ""))[:200],
            })
        sent_risks.sort(key=lambda x: x["washing_risk"], reverse=True)

        scores.append(EWRIScore(
            bank=bank, year=year, total_sentences=N,
            implemented=impl, planning=plan, indeterminate=indet,
            with_evidence=with_ev, without_evidence=N - with_ev,
            implemented_ratio=round(impl / N, 3),
            planning_ratio=round(plan / N, 3),
            indeterminate_ratio=round(indet / N, 3),
            evidence_ratio=round(with_ev / N, 3),
            avg_evidence_strength=round(avg_es, 3),
            ewri=round(ewri_new, 2), ewri_old=round(ewri_old, 2),
            risk_level=risk_level,
            contribution_implemented=round(c_impl, 2),
            contribution_planning=round(c_plan, 2),
            contribution_indeterminate=round(c_indet, 2),
            topic_breakdown=topic_bd,
            topic_entropy=round(te, 3),
            topic_coverage_index=round(1.0 - te, 3),
            sentence_risks=sent_risks,
        ))

    return scores


def scores_to_dataframe(scores: list[EWRIScore]) -> pd.DataFrame:
    data = []
    for s in scores:
        data.append({
            "bank": s.bank, "year": s.year,
            "total_sentences": s.total_sentences,
            "implemented": s.implemented, "planning": s.planning,
            "indeterminate": s.indeterminate,
            "with_evidence": s.with_evidence, "without_evidence": s.without_evidence,
            "implemented_ratio": s.implemented_ratio,
            "planning_ratio": s.planning_ratio,
            "indeterminate_ratio": s.indeterminate_ratio,
            "evidence_ratio": s.evidence_ratio,
            "avg_evidence_strength": s.avg_evidence_strength,
            "ewri": s.ewri, "ewri_old": s.ewri_old,
            "risk_level": s.risk_level,
            "contrib_implemented": s.contribution_implemented,
            "contrib_planning": s.contribution_planning,
            "contrib_indeterminate": s.contribution_indeterminate,
            "topic_entropy": s.topic_entropy,
            "topic_coverage_index": s.topic_coverage_index,
        })
    return pd.DataFrame(data)


def print_ewri_summary(df_scores: pd.DataFrame, scores: list[EWRIScore]):
    """Print comprehensive EWRI summary."""
    print("\n" + "=" * 70)
    print("EWRI SUMMARY (Interaction-Based Formula)")
    print("=" * 70)

    print(f"\nTotal bank-years: {len(df_scores)}")
    print(f"EWRI Range: [{df_scores['ewri'].min():.2f}, {df_scores['ewri'].max():.2f}]")
    print(f"EWRI Mean ± Std: {df_scores['ewri'].mean():.2f} ± {df_scores['ewri'].std():.2f}")

    print(f"\nRisk Distribution:")
    for level in ["Low", "Medium", "High", "Very High"]:
        c = (df_scores["risk_level"] == level).sum()
        if c > 0:
            print(f"  {level:10s}: {c}")

    print(f"\nFormula Comparison (Old Additive vs New Interaction):")
    print(f"  Old:  Mean={df_scores['ewri_old'].mean():.2f}, "
          f"Std={df_scores['ewri_old'].std():.2f}, "
          f"Range=[{df_scores['ewri_old'].min():.2f}, {df_scores['ewri_old'].max():.2f}]")
    print(f"  New:  Mean={df_scores['ewri'].mean():.2f}, "
          f"Std={df_scores['ewri'].std():.2f}, "
          f"Range=[{df_scores['ewri'].min():.2f}, {df_scores['ewri'].max():.2f}]")
    corr = df_scores[["ewri", "ewri_old"]].corr().iloc[0, 1]
    print(f"  Rank Correlation: {corr:.4f}")

    ewri_mean = max(df_scores['ewri'].mean(), 1e-9)
    print(f"\nEWRI Decomposition (average contribution):")
    print(f"  Indeterminate: {df_scores['contrib_indeterminate'].mean():.2f} pts "
          f"({df_scores['contrib_indeterminate'].mean() / ewri_mean * 100:.1f}%)")
    print(f"  Planning:      {df_scores['contrib_planning'].mean():.2f} pts "
          f"({df_scores['contrib_planning'].mean() / ewri_mean * 100:.1f}%)")
    print(f"  Implemented:   {df_scores['contrib_implemented'].mean():.2f} pts "
          f"({df_scores['contrib_implemented'].mean() / ewri_mean * 100:.1f}%)")

    sorted_s = df_scores.sort_values("ewri", ascending=False)
    print(f"\nTop 5 Highest EWRI:")
    for _, r in sorted_s.head(5).iterrows():
        print(f"  {r['bank']:15s} {r['year']}  EWRI={r['ewri']:5.1f} ({r['risk_level']})")

    print(f"\nTop 5 Lowest EWRI:")
    for _, r in sorted_s.tail(5).iterrows():
        print(f"  {r['bank']:15s} {r['year']}  EWRI={r['ewri']:5.1f} ({r['risk_level']})")


if __name__ == "__main__":
    import sys
    input_path = sys.argv[1] if len(sys.argv) > 1 else "data/corpus/actionability_sentences.parquet"
    df = pd.read_parquet(input_path)
    df = enrich_with_risk_scores(df)
    scores = calculate_bank_year_ewri(df)
    df_scores = scores_to_dataframe(scores).sort_values("ewri", ascending=False)
    print_ewri_summary(df_scores, scores)
