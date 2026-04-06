"""
ESG-Washing Analysis Module
============================

Comprehensive analysis of ESG-washing patterns, providing:
1. EWRI Decomposition: WHERE does washing risk come from?
2. Action-Evidence Interaction: HOW do claim types and evidence interact?
3. Topic Analysis: WHICH ESG topics show most washing?
4. Temporal Trends: HOW does washing change over time?
5. Cross-Bank Comparison: WHO washes the most?
6. Correlation Analysis: WHAT factors drive EWRI?
7. Evidence Type Analysis: WHICH evidence types matter?
8. Formula Comparison: IS the new formula better?
9. Qualitative Samples: ARE the results valid?

Theoretical grounding:
- Delmas & Burbano (2011) "Drivers of Greenwashing" — factors analysis
- GRI Standards (2021) — evidence quality dimensions
- Shannon Entropy — topic coverage measurement
"""

import math
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from collections import Counter
from datetime import datetime

from src.pipeline.ewri import (
    ESG_TOPICS, VALID_EVIDENCE_TYPES,
    ACTION_PENALTY, EVIDENCE_SENSITIVITY,
    compute_evidence_score, compute_washing_risk,
    get_risk_level, calculate_topic_entropy,
    EWRIScore,
)


# ============================================================
# 1. EWRI DECOMPOSITION
# ============================================================

def analyze_ewri_decomposition(df_scores: pd.DataFrame) -> dict:
    """
    Decompose EWRI into contributions from each action type.

    Shows EXACTLY where washing risk originates:
    - What % of EWRI is from Indeterminate claims?
    - What % from Planning? From Implemented?
    """
    ewri_mean = df_scores["ewri"].mean()
    if ewri_mean < 1e-6:
        ewri_mean = 1.0

    decomp = {
        "ewri_mean": round(ewri_mean, 2),
        "ewri_std": round(float(df_scores["ewri"].std()), 2),
        "ewri_min": round(float(df_scores["ewri"].min()), 2),
        "ewri_max": round(float(df_scores["ewri"].max()), 2),
        "components": {},
    }

    for label_key, col in [
        ("Indeterminate", "contrib_indeterminate"),
        ("Planning", "contrib_planning"),
        ("Implemented", "contrib_implemented"),
    ]:
        mean_val = df_scores[col].mean()
        decomp["components"][label_key] = {
            "mean_contribution": round(mean_val, 2),
            "pct_of_ewri": round(mean_val / ewri_mean * 100, 1),
            "std": round(float(df_scores[col].std()), 2),
        }

    return decomp


# ============================================================
# 2. ACTION-EVIDENCE INTERACTION MATRIX
# ============================================================

def analyze_action_evidence_interaction(df: pd.DataFrame) -> dict:
    """
    Create the Action × Evidence interaction matrix.

    Bins evidence strength into levels and shows:
    - Count of sentences in each (action, evidence_level) cell
    - Average WRS for each cell
    - This reveals the KEY interaction driving EWRI
    """
    if "es_combined" not in df.columns:
        df = df.copy()
        df["es_combined"] = df.apply(compute_evidence_score, axis=1)
    if "wrs" not in df.columns:
        df = df.copy()
        df["wrs"] = df.apply(
            lambda r: compute_washing_risk(r["action_label"], r["es_combined"]), axis=1
        )

    # Bin evidence strength
    bins = [0.0, 0.1, 0.3, 0.6, 1.01]
    labels_bin = ["None (0)", "Low (0-0.3)", "Medium (0.3-0.6)", "High (0.6+)"]
    df = df.copy()
    df["es_level"] = pd.cut(df["es_combined"], bins=bins, labels=labels_bin, right=False)

    matrix = {}
    for action_label in ["Implemented", "Planning", "Indeterminate"]:
        matrix[action_label] = {}
        adf = df[df["action_label"] == action_label]
        for es_level in labels_bin:
            cell = adf[adf["es_level"] == es_level]
            matrix[action_label][es_level] = {
                "count": len(cell),
                "pct_of_total": round(len(cell) / max(len(df), 1) * 100, 2),
                "avg_wrs": round(float(cell["wrs"].mean()), 3) if len(cell) > 0 else None,
                "avg_es": round(float(cell["es_combined"].mean()), 3) if len(cell) > 0 else None,
            }

    # Also compute theoretical WRS for reference
    theoretical = {}
    for action in ["Implemented", "Planning", "Indeterminate"]:
        theoretical[action] = {}
        for es_val, level_name in [(0.0, "ES=0"), (0.25, "ES=0.25"),
                                    (0.5, "ES=0.5"), (0.75, "ES=0.75"), (1.0, "ES=1.0")]:
            theoretical[action][level_name] = round(compute_washing_risk(action, es_val), 3)

    return {"empirical_matrix": matrix, "theoretical_matrix": theoretical}


# ============================================================
# 3. TOPIC ANALYSIS
# ============================================================

def analyze_topics(
    df: pd.DataFrame,
    ewri_scores: list[EWRIScore],
) -> dict:
    """
    Comprehensive topic analysis:
    - Per-topic EWRI (aggregated across banks)
    - Topic distribution per bank
    - Topic coverage (Shannon entropy)
    - Topic × Action label distribution
    """
    topic_col = "topic_label" if "topic_label" in df.columns else None
    if topic_col is None:
        return {"error": "No topic column found"}

    if "wrs" not in df.columns:
        df = df.copy()
        df["es_combined"] = df.apply(compute_evidence_score, axis=1)
        df["wrs"] = df.apply(
            lambda r: compute_washing_risk(r["action_label"], r["es_combined"]), axis=1
        )

    # Per-topic aggregated EWRI
    topic_summary = {}
    for topic in ESG_TOPICS:
        tdf = df[df[topic_col] == topic]
        n = len(tdf)
        if n < 5:
            topic_summary[topic] = {"n": n, "ewri": None}
            continue

        ewri = tdf["wrs"].mean() * 100
        impl = (tdf["action_label"] == "Implemented").mean()
        plan = (tdf["action_label"] == "Planning").mean()
        indet = (tdf["action_label"] == "Indeterminate").mean()
        ev_rate = tdf["has_evidence"].mean() if "has_evidence" in tdf.columns else 0
        avg_es = tdf["es_combined"].mean() if "es_combined" in tdf.columns else 0

        topic_summary[topic] = {
            "n": n,
            "pct_of_corpus": round(n / len(df) * 100, 1),
            "ewri": round(ewri, 2),
            "risk_level": get_risk_level(ewri),
            "implemented_pct": round(impl * 100, 1),
            "planning_pct": round(plan * 100, 1),
            "indeterminate_pct": round(indet * 100, 1),
            "evidence_rate": round(ev_rate * 100, 1),
            "avg_evidence_strength": round(avg_es, 3),
        }

    # Per-bank topic distribution
    bank_topic_dist = {}
    for bank in df["bank"].unique():
        bdf = df[df["bank"] == bank]
        counts = {t: int((bdf[topic_col] == t).sum()) for t in ESG_TOPICS}
        total = sum(counts.values())
        bank_topic_dist[bank] = {
            "counts": counts,
            "entropy": round(calculate_topic_entropy(counts), 3),
            "dominant_topic": max(counts, key=counts.get) if total > 0 else "N/A",
        }

    # Topic × Action crosstab
    topic_action = {}
    for topic in ESG_TOPICS:
        tdf = df[df[topic_col] == topic]
        if len(tdf) == 0:
            continue
        n = len(tdf)
        topic_action[topic] = {
            al: round((tdf["action_label"] == al).mean() * 100, 1)
            for al in ["Implemented", "Planning", "Indeterminate"]
        }

    return {
        "topic_summary": topic_summary,
        "bank_topic_distribution": bank_topic_dist,
        "topic_action_distribution": topic_action,
    }


# ============================================================
# 4. TEMPORAL TRENDS
# ============================================================

def analyze_temporal_trends(df_scores: pd.DataFrame) -> dict:
    """
    Temporal analysis of EWRI:
    - EWRI trend by year (mean, std across banks)
    - Per-bank temporal trends
    - Evidence improvement over time
    - Action label distribution change
    """
    yearly = df_scores.groupby("year").agg({
        "ewri": ["mean", "std", "min", "max"],
        "ewri_old": "mean",
        "evidence_ratio": "mean",
        "avg_evidence_strength": "mean",
        "implemented_ratio": "mean",
        "indeterminate_ratio": "mean",
        "planning_ratio": "mean",
        "topic_entropy": "mean",
    }).reset_index()

    yearly.columns = [
        "year", "ewri_mean", "ewri_std", "ewri_min", "ewri_max",
        "ewri_old_mean", "evidence_ratio_mean", "avg_es_mean",
        "impl_ratio_mean", "indet_ratio_mean", "plan_ratio_mean",
        "topic_entropy_mean",
    ]

    yearly_trends = yearly.round(3).to_dict("records")

    # Per-bank trends
    bank_trends = {}
    for bank in df_scores["bank"].unique():
        bdf = df_scores[df_scores["bank"] == bank].sort_values("year")
        ewri_vals = bdf["ewri"].tolist()
        years = bdf["year"].tolist()

        # Simple trend direction
        if len(ewri_vals) >= 2:
            diff = ewri_vals[-1] - ewri_vals[0]
            trend = "improving" if diff < -2 else ("worsening" if diff > 2 else "stable")
        else:
            trend = "insufficient_data"

        bank_trends[bank] = {
            "years": years,
            "ewri_values": [round(v, 2) for v in ewri_vals],
            "trend": trend,
            "ewri_change": round(ewri_vals[-1] - ewri_vals[0], 2) if len(ewri_vals) >= 2 else 0,
        }

    return {"yearly_trends": yearly_trends, "bank_trends": bank_trends}


# ============================================================
# 5. CROSS-BANK COMPARISON
# ============================================================

def analyze_cross_bank(
    df_scores: pd.DataFrame,
    ewri_scores: list[EWRIScore],
) -> dict:
    """
    Cross-bank comparison:
    - Overall ranking
    - Bank profiles (strengths/weaknesses)
    - Peer comparison statistics
    """
    bank_agg = df_scores.groupby("bank").agg({
        "ewri": ["mean", "std", "min", "max"],
        "ewri_old": "mean",
        "total_sentences": "sum",
        "implemented_ratio": "mean",
        "planning_ratio": "mean",
        "indeterminate_ratio": "mean",
        "evidence_ratio": "mean",
        "avg_evidence_strength": "mean",
        "topic_entropy": "mean",
    }).reset_index()

    bank_agg.columns = [
        "bank", "ewri_mean", "ewri_std", "ewri_min", "ewri_max",
        "ewri_old_mean", "total_sentences",
        "impl_ratio", "plan_ratio", "indet_ratio",
        "ev_ratio", "avg_es", "topic_entropy",
    ]
    bank_agg = bank_agg.sort_values("ewri_mean")

    # Rankings
    bank_agg["rank"] = range(1, len(bank_agg) + 1)

    # Profiles
    profiles = {}
    for _, row in bank_agg.iterrows():
        bank = row["bank"]
        # Identify strengths/weaknesses relative to mean
        mean_ewri = bank_agg["ewri_mean"].mean()
        mean_ev = bank_agg["ev_ratio"].mean()
        mean_impl = bank_agg["impl_ratio"].mean()

        strengths = []
        weaknesses = []

        if row["ewri_mean"] < mean_ewri - 2:
            strengths.append("Lower-than-average washing risk")
        if row["ewri_mean"] > mean_ewri + 2:
            weaknesses.append("Higher-than-average washing risk")
        if row["ev_ratio"] > mean_ev + 0.05:
            strengths.append("Better evidence coverage")
        if row["ev_ratio"] < mean_ev - 0.05:
            weaknesses.append("Weak evidence coverage")
        if row["impl_ratio"] > mean_impl + 0.05:
            strengths.append("Higher implementation ratio")
        if row["impl_ratio"] < mean_impl - 0.05:
            weaknesses.append("Low implementation ratio")
        if row["indet_ratio"] > 0.65:
            weaknesses.append("High proportion of vague claims (>65%)")
        if row["topic_entropy"] > 0.85:
            strengths.append("Good topic coverage (diverse ESG reporting)")
        if row["topic_entropy"] < 0.65:
            weaknesses.append("Poor topic coverage (concentrated reporting)")

        profiles[bank] = {
            "rank": int(row["rank"]),
            "ewri_mean": round(row["ewri_mean"], 2),
            "risk_level": get_risk_level(row["ewri_mean"]),
            "total_sentences": int(row["total_sentences"]),
            "strengths": strengths,
            "weaknesses": weaknesses,
        }

    return {
        "ranking": bank_agg.round(3).to_dict("records"),
        "profiles": profiles,
    }


# ============================================================
# 6. CORRELATION ANALYSIS
# ============================================================

def analyze_correlations(df: pd.DataFrame, df_scores: pd.DataFrame) -> dict:
    """
    Correlation analysis between key features and EWRI.

    Measures:
    - Spearman rank correlations for continuous variables
    - Point-biserial for binary flags
    """
    # Bank-year level correlations
    numeric_cols = [
        "ewri", "ewri_old",
        "implemented_ratio", "planning_ratio", "indeterminate_ratio",
        "evidence_ratio", "avg_evidence_strength",
        "topic_entropy",
    ]
    available_cols = [c for c in numeric_cols if c in df_scores.columns]

    corr_matrix = df_scores[available_cols].corr(method="spearman").round(3)
    ewri_correlations = corr_matrix["ewri"].drop("ewri").to_dict()

    # Sentence-level correlations
    sent_correlations = {}
    if "wrs" in df.columns and "es_combined" in df.columns:
        sent_features = ["es_combined", "wrs"]
        if "has_evidence" in df.columns:
            sent_features.append("has_evidence")

        for feat in sent_features:
            if feat in df.columns and feat != "wrs":
                try:
                    corr_val = df[["wrs", feat]].corr(method="spearman").iloc[0, 1]
                    sent_correlations[f"wrs_vs_{feat}"] = round(corr_val, 4)
                except Exception:
                    pass

    # Action label effect sizes (Eta-squared equivalent)
    action_effect = {}
    if "wrs" in df.columns:
        overall_mean = df["wrs"].mean()
        ss_total = ((df["wrs"] - overall_mean) ** 2).sum()
        ss_between = 0
        for label in ["Implemented", "Planning", "Indeterminate"]:
            ldf = df[df["action_label"] == label]
            if len(ldf) > 0:
                ss_between += len(ldf) * (ldf["wrs"].mean() - overall_mean) ** 2
        if ss_total > 0:
            eta_squared = ss_between / ss_total
            action_effect["eta_squared_action_on_wrs"] = round(eta_squared, 4)

    return {
        "ewri_correlations": ewri_correlations,
        "correlation_matrix": corr_matrix.to_dict(),
        "sentence_level_correlations": sent_correlations,
        "action_label_effect": action_effect,
    }


# ============================================================
# 7. EVIDENCE TYPE ANALYSIS
# ============================================================

def analyze_evidence_types(df: pd.DataFrame) -> dict:
    """
    Analyze which evidence types are most important for reducing washing risk.

    For each evidence type (KPI, Standard, Time_bound, Third_party):
    - Frequency of occurrence
    - Average WRS when present vs absent
    - Risk reduction effect
    """
    if "evidence_types" not in df.columns:
        return {"error": "No evidence_types column"}

    if "wrs" not in df.columns:
        df = df.copy()
        df["es_combined"] = df.apply(compute_evidence_score, axis=1)
        df["wrs"] = df.apply(
            lambda r: compute_washing_risk(r["action_label"], r["es_combined"]), axis=1
        )

    overall_wrs = df["wrs"].mean()

    type_analysis = {}
    for etype in VALID_EVIDENCE_TYPES:
        # Check if this evidence type is present for each sentence
        has_type = df["evidence_types"].apply(
            lambda x: etype in x if isinstance(x, list) else False
        )

        n_with = has_type.sum()
        n_without = len(df) - n_with

        wrs_with = df.loc[has_type, "wrs"].mean() if n_with > 0 else None
        wrs_without = df.loc[~has_type, "wrs"].mean() if n_without > 0 else None

        risk_reduction = None
        if wrs_with is not None and wrs_without is not None:
            risk_reduction = round((wrs_without - wrs_with) / max(wrs_without, 0.001) * 100, 1)

        type_analysis[etype] = {
            "count": int(n_with),
            "frequency_pct": round(n_with / max(len(df), 1) * 100, 2),
            "avg_wrs_present": round(wrs_with, 3) if wrs_with is not None else None,
            "avg_wrs_absent": round(wrs_without, 3) if wrs_without is not None else None,
            "risk_reduction_pct": risk_reduction,
        }

    # Evidence type co-occurrence
    cooccurrence = {}
    for t1 in VALID_EVIDENCE_TYPES:
        for t2 in VALID_EVIDENCE_TYPES:
            if t1 >= t2:
                continue
            has_both = df["evidence_types"].apply(
                lambda x: t1 in x and t2 in x if isinstance(x, list) else False
            )
            cooccurrence[f"{t1}+{t2}"] = int(has_both.sum())

    return {
        "type_importance": type_analysis,
        "cooccurrence": cooccurrence,
        "overall_avg_wrs": round(overall_wrs, 3),
    }


# ============================================================
# 8. FORMULA COMPARISON
# ============================================================

def compare_formulas(df_scores: pd.DataFrame) -> dict:
    """
    Compare Old (additive) vs New (interaction) EWRI formula.

    Metrics:
    - Discrimination: std, IQR, range
    - Rank stability: Spearman correlation
    - Risk level distribution
    """
    old = df_scores["ewri_old"]
    new = df_scores["ewri"]

    old_stats = {
        "mean": round(float(old.mean()), 2),
        "std": round(float(old.std()), 2),
        "iqr": round(float(old.quantile(0.75) - old.quantile(0.25)), 2),
        "range": round(float(old.max() - old.min()), 2),
        "cv": round(float(old.std() / max(old.mean(), 0.01) * 100), 1),
    }
    new_stats = {
        "mean": round(float(new.mean()), 2),
        "std": round(float(new.std()), 2),
        "iqr": round(float(new.quantile(0.75) - new.quantile(0.25)), 2),
        "range": round(float(new.max() - new.min()), 2),
        "cv": round(float(new.std() / max(new.mean(), 0.01) * 100), 1),
    }

    rank_corr = df_scores[["ewri", "ewri_old"]].corr(method="spearman").iloc[0, 1]

    # Risk level distributions
    old_risk = {}
    new_risk = {}
    for _, row in df_scores.iterrows():
        ol = get_risk_level(row["ewri_old"])
        nl = row["risk_level"]
        old_risk[ol] = old_risk.get(ol, 0) + 1
        new_risk[nl] = new_risk.get(nl, 0) + 1

    # Discrimination improvement
    std_improvement = (new_stats["std"] - old_stats["std"]) / max(old_stats["std"], 0.01) * 100
    range_improvement = (new_stats["range"] - old_stats["range"]) / max(old_stats["range"], 0.01) * 100

    return {
        "old_formula": old_stats,
        "new_formula": new_stats,
        "rank_correlation": round(rank_corr, 4),
        "old_risk_distribution": old_risk,
        "new_risk_distribution": new_risk,
        "std_improvement_pct": round(std_improvement, 1),
        "range_improvement_pct": round(range_improvement, 1),
    }


# ============================================================
# 9. QUALITATIVE SAMPLES
# ============================================================

def get_qualitative_samples(
    df: pd.DataFrame,
    ewri_scores: list[EWRIScore],
    n_per_category: int = 5,
) -> dict:
    """
    Extract representative sentence samples for qualitative verification.

    Categories:
    - High risk: Indeterminate + No Evidence (pure washing)
    - Medium risk: Planning + Some Evidence
    - Low risk: Implemented + Strong Evidence (substantive)
    - Suspicious: Implemented + No Evidence (unsubstantiated claims!)
    """
    if "wrs" not in df.columns:
        df = df.copy()
        df["es_combined"] = df.apply(compute_evidence_score, axis=1)
        df["wrs"] = df.apply(
            lambda r: compute_washing_risk(r["action_label"], r["es_combined"]), axis=1
        )

    def _extract_samples(mask, label, n=n_per_category):
        sub = df[mask].nlargest(n * 3, "wrs") if label != "low_risk" else df[mask].nsmallest(n * 3, "wrs")
        # Filter out noise (too short, or looks like heading)
        sub = sub[sub["sentence"].astype(str).str.len() > 30]
        rows = []
        for _, r in sub.head(n).iterrows():
            rows.append({
                "sentence": str(r.get("sentence", r.get("text", "")))[:300],
                "bank": r.get("bank", ""),
                "year": r.get("year", ""),
                "action_label": r.get("action_label", ""),
                "evidence_strength": round(float(r.get("es_combined", 0)), 3),
                "washing_risk": round(float(r.get("wrs", 0)), 3),
                "has_evidence": bool(r.get("has_evidence", False)),
                "evidence_types": r.get("evidence_types", []),
                "topic": r.get("topic_label", r.get("topic", "")),
            })
        return rows

    has_ev = df.get("has_evidence", pd.Series(False, index=df.index)).astype(bool)
    es = df.get("es_combined", pd.Series(0.0, index=df.index))

    samples = {
        "high_risk_pure_washing": _extract_samples(
            (df["action_label"] == "Indeterminate") & (~has_ev),
            "high_risk",
        ),
        "suspicious_unsubstantiated": _extract_samples(
            (df["action_label"] == "Implemented") & (es < 0.15),
            "suspicious",
        ),
        "medium_risk_planning": _extract_samples(
            (df["action_label"] == "Planning") & (es > 0.1) & (es < 0.5),
            "medium_risk",
        ),
        "low_risk_substantive": _extract_samples(
            (df["action_label"] == "Implemented") & (es > 0.4),
            "low_risk",
        ),
    }

    # Quality check: top risk claims noise rate
    noise_count = 0
    total_checked = 0
    for score in ewri_scores[:10]:
        for claim in score.sentence_risks[:10]:
            total_checked += 1
            sent = str(claim.get("sentence", ""))
            if len(sent) < 30 or sent.count(" ") < 3:
                noise_count += 1

    samples["quality_check"] = {
        "top_claims_checked": total_checked,
        "noise_count": noise_count,
        "noise_rate_pct": round(noise_count / max(total_checked, 1) * 100, 1),
    }

    return samples


# ============================================================
# MAIN: Run Full Analysis
# ============================================================

def run_full_analysis(
    df: pd.DataFrame,
    ewri_scores: list[EWRIScore],
    df_scores: pd.DataFrame,
) -> dict:
    """
    Run ALL analyses and return comprehensive results dict.
    """
    print("\n" + "=" * 70)
    print("RUNNING COMPREHENSIVE ESG-WASHING ANALYSIS")
    print("=" * 70)

    results = {}

    print("  [1/9] EWRI Decomposition...")
    results["decomposition"] = analyze_ewri_decomposition(df_scores)

    print("  [2/9] Action-Evidence Interaction...")
    results["interaction_matrix"] = analyze_action_evidence_interaction(df)

    print("  [3/9] Topic Analysis...")
    results["topic_analysis"] = analyze_topics(df, ewri_scores)

    print("  [4/9] Temporal Trends...")
    results["temporal_trends"] = analyze_temporal_trends(df_scores)

    print("  [5/9] Cross-Bank Comparison...")
    results["cross_bank"] = analyze_cross_bank(df_scores, ewri_scores)

    print("  [6/9] Correlation Analysis...")
    results["correlations"] = analyze_correlations(df, df_scores)

    print("  [7/9] Evidence Type Analysis...")
    results["evidence_types"] = analyze_evidence_types(df)

    print("  [8/9] Formula Comparison...")
    results["formula_comparison"] = compare_formulas(df_scores)

    print("  [9/9] Qualitative Samples...")
    results["qualitative_samples"] = get_qualitative_samples(df, ewri_scores)

    results["metadata"] = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_sentences": len(df),
        "total_bank_years": len(df_scores),
    }

    print("  Analysis complete.")
    return results


# ============================================================
# SAVE ANALYSIS
# ============================================================

def save_analysis(results: dict, output_dir: str = "outputs/analysis"):
    """Save analysis results to files."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Full JSON
    with open(out / "analysis_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    # Markdown summary
    md = _generate_analysis_markdown(results)
    with open(out / "analysis_report.md", "w", encoding="utf-8") as f:
        f.write(md)

    # CSV exports
    _save_csv_exports(results, out)

    print(f"Analysis saved to {out}/")


def _save_csv_exports(results: dict, out: Path):
    """Export key tables as CSV."""
    # Formula comparison
    if "formula_comparison" in results:
        fc = results["formula_comparison"]
        rows = [
            {"metric": k, "old_formula": fc["old_formula"].get(k), "new_formula": fc["new_formula"].get(k)}
            for k in fc["old_formula"]
        ]
        pd.DataFrame(rows).to_csv(out / "formula_comparison.csv", index=False)

    # Cross-bank ranking
    if "cross_bank" in results and "ranking" in results["cross_bank"]:
        pd.DataFrame(results["cross_bank"]["ranking"]).to_csv(
            out / "bank_ranking.csv", index=False
        )

    # Topic summary
    if "topic_analysis" in results and "topic_summary" in results["topic_analysis"]:
        ts = results["topic_analysis"]["topic_summary"]
        rows = [{"topic": t, **info} for t, info in ts.items()]
        pd.DataFrame(rows).to_csv(out / "topic_summary.csv", index=False)

    # Evidence type importance
    if "evidence_types" in results and "type_importance" in results["evidence_types"]:
        ti = results["evidence_types"]["type_importance"]
        rows = [{"type": t, **info} for t, info in ti.items()]
        pd.DataFrame(rows).to_csv(out / "evidence_type_importance.csv", index=False)

    # Temporal trends
    if "temporal_trends" in results and "yearly_trends" in results["temporal_trends"]:
        pd.DataFrame(results["temporal_trends"]["yearly_trends"]).to_csv(
            out / "temporal_trends.csv", index=False
        )


def _generate_analysis_markdown(results: dict) -> str:
    """Generate human-readable analysis report in Markdown."""
    lines = []
    lines.append("# ESG-Washing Comprehensive Analysis Report")
    lines.append(f"\nGenerated: {results.get('metadata', {}).get('generated_at', 'N/A')}")
    lines.append(f"Total sentences: {results.get('metadata', {}).get('total_sentences', 0):,}")
    lines.append(f"Total bank-years: {results.get('metadata', {}).get('total_bank_years', 0)}")

    # 1. Decomposition
    if "decomposition" in results:
        d = results["decomposition"]
        lines.append("\n## 1. EWRI Decomposition")
        lines.append(f"\nOverall EWRI: **{d['ewri_mean']:.2f}** ± {d['ewri_std']:.2f}")
        lines.append(f"Range: [{d['ewri_min']:.2f}, {d['ewri_max']:.2f}]")
        lines.append("\n| Component | Contribution (pts) | % of EWRI |")
        lines.append("|-----------|-------------------|-----------|")
        for label, info in d["components"].items():
            lines.append(f"| {label} | {info['mean_contribution']:.2f} | {info['pct_of_ewri']:.1f}% |")

    # 2. Interaction Matrix
    if "interaction_matrix" in results:
        lines.append("\n## 2. Action × Evidence Interaction")
        th = results["interaction_matrix"].get("theoretical_matrix", {})
        if th:
            lines.append("\n**Theoretical WRS (from formula):**")
            lines.append("| Action / ES | ES=0 | ES=0.25 | ES=0.5 | ES=0.75 | ES=1.0 |")
            lines.append("|-------------|------|---------|--------|---------|--------|")
            for action in ["Implemented", "Planning", "Indeterminate"]:
                vals = th.get(action, {})
                lines.append(f"| {action} | {vals.get('ES=0','-')} | {vals.get('ES=0.25','-')} | "
                           f"{vals.get('ES=0.5','-')} | {vals.get('ES=0.75','-')} | {vals.get('ES=1.0','-')} |")

        em = results["interaction_matrix"].get("empirical_matrix", {})
        if em:
            lines.append("\n**Empirical Distribution (actual data):**")
            lines.append("| Action | Evidence Level | Count | % | Avg WRS |")
            lines.append("|--------|---------------|-------|---|---------|")
            for action in ["Implemented", "Planning", "Indeterminate"]:
                for level, info in em.get(action, {}).items():
                    wrs_str = f"{info['avg_wrs']:.3f}" if info['avg_wrs'] is not None else "N/A"
                    lines.append(f"| {action} | {level} | {info['count']} | "
                               f"{info['pct_of_total']}% | {wrs_str} |")

    # 3. Topic Analysis
    if "topic_analysis" in results:
        ts = results["topic_analysis"].get("topic_summary", {})
        if ts:
            lines.append("\n## 3. Topic Analysis")
            lines.append("\n| Topic | N | % | EWRI | Risk | Impl% | Indet% | EvRate% |")
            lines.append("|-------|---|---|------|------|-------|--------|---------|")
            for topic, info in ts.items():
                if info.get("ewri") is not None:
                    lines.append(f"| {topic} | {info['n']} | {info['pct_of_corpus']}% | "
                               f"{info['ewri']:.1f} | {info['risk_level']} | "
                               f"{info['implemented_pct']}% | {info['indeterminate_pct']}% | "
                               f"{info['evidence_rate']}% |")

    # 4. Temporal
    if "temporal_trends" in results:
        yt = results["temporal_trends"].get("yearly_trends", [])
        if yt:
            lines.append("\n## 4. Temporal Trends")
            lines.append("\n| Year | EWRI Mean | Std | Old EWRI | Ev Rate | Impl% | Indet% |")
            lines.append("|------|-----------|-----|----------|---------|-------|--------|")
            for row in yt:
                lines.append(f"| {row['year']} | {row['ewri_mean']:.2f} | {row['ewri_std']:.2f} | "
                           f"{row['ewri_old_mean']:.2f} | {row['evidence_ratio_mean']:.3f} | "
                           f"{row['impl_ratio_mean']:.3f} | {row['indet_ratio_mean']:.3f} |")

    # 5. Cross-bank
    if "cross_bank" in results:
        profiles = results["cross_bank"].get("profiles", {})
        if profiles:
            lines.append("\n## 5. Cross-Bank Comparison")
            lines.append("\n| Rank | Bank | Avg EWRI | Risk | Strengths | Weaknesses |")
            lines.append("|------|------|----------|------|-----------|------------|")
            sorted_profiles = sorted(profiles.items(), key=lambda x: x[1]["rank"])
            for bank, p in sorted_profiles:
                s = "; ".join(p["strengths"][:2]) if p["strengths"] else "-"
                w = "; ".join(p["weaknesses"][:2]) if p["weaknesses"] else "-"
                lines.append(f"| {p['rank']} | {bank} | {p['ewri_mean']:.2f} | "
                           f"{p['risk_level']} | {s} | {w} |")

    # 6. Correlations
    if "correlations" in results:
        ec = results["correlations"].get("ewri_correlations", {})
        if ec:
            lines.append("\n## 6. Correlation Analysis")
            lines.append("\n**EWRI Correlations (Spearman):**")
            lines.append("| Feature | ρ (vs EWRI) | Interpretation |")
            lines.append("|---------|-------------|----------------|")
            for feat, rho in sorted(ec.items(), key=lambda x: abs(x[1]), reverse=True):
                interp = "strong" if abs(rho) > 0.7 else ("moderate" if abs(rho) > 0.4 else "weak")
                direction = "positive" if rho > 0 else "negative"
                lines.append(f"| {feat} | {rho:.3f} | {interp} {direction} |")

        ae = results["correlations"].get("action_label_effect", {})
        if ae:
            eta = ae.get("eta_squared_action_on_wrs", 0)
            lines.append(f"\n**Action Label Effect Size**: η² = {eta:.4f} "
                       f"({'large' if eta > 0.14 else 'medium' if eta > 0.06 else 'small'} effect)")

    # 7. Evidence Types
    if "evidence_types" in results:
        ti = results["evidence_types"].get("type_importance", {})
        if ti:
            lines.append("\n## 7. Evidence Type Importance")
            lines.append("\n| Type | Freq% | Avg WRS (present) | Avg WRS (absent) | Risk Reduction |")
            lines.append("|------|-------|------------------|-----------------|----------------|")
            for etype, info in sorted(ti.items(),
                                      key=lambda x: -(x[1].get("risk_reduction_pct") or 0)):
                rr = f"{info['risk_reduction_pct']}%" if info["risk_reduction_pct"] is not None else "N/A"
                wp = f"{info['avg_wrs_present']:.3f}" if info['avg_wrs_present'] is not None else "N/A"
                wa = f"{info['avg_wrs_absent']:.3f}" if info['avg_wrs_absent'] is not None else "N/A"
                lines.append(f"| {etype} | {info['frequency_pct']}% | {wp} | {wa} | {rr} |")

    # 8. Formula Comparison
    if "formula_comparison" in results:
        fc = results["formula_comparison"]
        lines.append("\n## 8. Formula Comparison (Old Additive vs New Interaction)")
        lines.append("\n| Metric | Old Formula | New Formula | Change |")
        lines.append("|--------|------------|------------|--------|")
        for metric in ["mean", "std", "iqr", "range", "cv"]:
            old_v = fc["old_formula"].get(metric, 0)
            new_v = fc["new_formula"].get(metric, 0)
            change = new_v - old_v if isinstance(new_v, (int, float)) and isinstance(old_v, (int, float)) else ""
            lines.append(f"| {metric} | {old_v} | {new_v} | {change:+.2f} |" if change != "" else
                        f"| {metric} | {old_v} | {new_v} | |")

        lines.append(f"\nRank Correlation: {fc['rank_correlation']:.4f}")
        lines.append(f"Std Improvement: {fc['std_improvement_pct']:+.1f}%")
        lines.append(f"Range Improvement: {fc['range_improvement_pct']:+.1f}%")

    # 9. Quality Check
    if "qualitative_samples" in results:
        qc = results["qualitative_samples"].get("quality_check", {})
        if qc:
            lines.append("\n## 9. Quality Verification")
            lines.append(f"\nTop risk claims noise rate: **{qc.get('noise_rate_pct', 0)}%** "
                       f"({qc.get('noise_count', 0)}/{qc.get('top_claims_checked', 0)})")

        for cat_key, cat_name in [
            ("high_risk_pure_washing", "High Risk (Pure Washing)"),
            ("suspicious_unsubstantiated", "Suspicious (Implemented without Evidence)"),
            ("low_risk_substantive", "Low Risk (Substantive)"),
        ]:
            samples = results["qualitative_samples"].get(cat_key, [])
            if samples:
                lines.append(f"\n### {cat_name}")
                for i, s in enumerate(samples[:3], 1):
                    lines.append(f"\n{i}. **[{s['bank']} {s['year']}]** [{s['action_label']}] "
                               f"WRS={s['washing_risk']:.3f}")
                    lines.append(f"   > \"{s['sentence'][:150]}...\"")
                    if s.get("evidence_types"):
                        lines.append(f"   Evidence types: {', '.join(s['evidence_types'])}")

    lines.append("\n---")
    lines.append("*Analysis generated by ESG-Washing Detection Framework*")

    return "\n".join(lines)


def print_analysis_summary(results: dict):
    """Print key analysis findings to console."""
    print("\n" + "=" * 70)
    print("KEY ANALYSIS FINDINGS")
    print("=" * 70)

    # Decomposition
    d = results.get("decomposition", {})
    if d:
        print(f"\n1. EWRI Decomposition:")
        for label, info in d.get("components", {}).items():
            print(f"   {label:15s}: {info['mean_contribution']:.2f} pts ({info['pct_of_ewri']:.1f}% of EWRI)")

    # Formula comparison
    fc = results.get("formula_comparison", {})
    if fc:
        print(f"\n2. Formula Comparison:")
        print(f"   Old: Mean={fc['old_formula']['mean']}, Std={fc['old_formula']['std']}")
        print(f"   New: Mean={fc['new_formula']['mean']}, Std={fc['new_formula']['std']}")
        print(f"   Discrimination improvement (Std): {fc['std_improvement_pct']:+.1f}%")

    # Correlations
    ec = results.get("correlations", {}).get("ewri_correlations", {})
    if ec:
        print(f"\n3. Key Correlations with EWRI:")
        for feat, rho in sorted(ec.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
            print(f"   {feat:25s}: ρ = {rho:+.3f}")

    # Evidence types
    ti = results.get("evidence_types", {}).get("type_importance", {})
    if ti:
        print(f"\n4. Evidence Type Importance (risk reduction):")
        for etype, info in sorted(ti.items(),
                                  key=lambda x: -(x[1].get("risk_reduction_pct") or 0)):
            rr = info.get("risk_reduction_pct")
            if rr is not None:
                print(f"   {etype:15s}: {rr:+.1f}% risk reduction (freq={info['frequency_pct']:.1f}%)")

    # Quality
    qc = results.get("qualitative_samples", {}).get("quality_check", {})
    if qc:
        print(f"\n5. Quality Check:")
        print(f"   Top claims noise rate: {qc.get('noise_rate_pct', 0):.1f}%")
