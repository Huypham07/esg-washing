"""
ESG-Washing Report Generator.

Generates structured reports from EWRI analysis results, including:
- Overall EWRI scores by bank-year (new interaction formula + old for comparison)
- EWRI decomposition (contribution of each action type)
- Topic breakdown (per E/S/G)
- Top risk claims with explanations
- Evidence coverage statistics
- Qualitative verification samples
- Full traceability: EWRI → contributing sentences → document location
- Comprehensive analysis results (correlations, interactions, trends)
"""

import json
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List
from datetime import datetime


@dataclass
class ESGWashingReport:
    """Structured report for ESG-washing analysis."""

    # Metadata
    generated_at: str = ""
    total_banks: int = 0
    total_bank_years: int = 0
    total_esg_sentences: int = 0

    # Overall EWRI statistics
    avg_ewri: float = 0.0
    min_ewri: float = 0.0
    max_ewri: float = 0.0
    std_ewri: float = 0.0

    # Old formula comparison
    avg_ewri_old: float = 0.0

    # Risk distribution
    risk_distribution: dict = field(default_factory=dict)

    # Per bank-year scores
    bank_year_scores: List[dict] = field(default_factory=list)

    # EWRI Decomposition
    ewri_decomposition: dict = field(default_factory=dict)

    # Topic breakdown (aggregated)
    topic_summary: dict = field(default_factory=dict)

    # Top risk claims (across all banks)
    top_risk_claims: List[dict] = field(default_factory=list)

    # All sentence details for full traceability
    sentence_details: List[dict] = field(default_factory=list)

    # Evidence statistics
    evidence_stats: dict = field(default_factory=dict)

    # Qualitative samples for verification
    qualitative_samples: dict = field(default_factory=dict)

    # Full analysis results (from analysis module)
    analysis_results: dict = field(default_factory=dict)

    def save(self, output_dir: Path):
        """Save report to multiple formats."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # JSON report (full — exclude huge sentence_details for size)
        report_dict = asdict(self)
        # Trim sentence_details to top 500 for JSON
        if len(report_dict.get("sentence_details", [])) > 500:
            report_dict["sentence_details_note"] = (
                f"Truncated to 500 of {len(report_dict['sentence_details'])} total. "
                f"Full details in sentence_details_full.csv."
            )
            report_dict["sentence_details"] = report_dict["sentence_details"][:500]

        with open(output_dir / "esg_washing_report.json", "w", encoding="utf-8") as f:
            json.dump(report_dict, f, ensure_ascii=False, indent=2, default=str)

        # Summary text report
        with open(output_dir / "esg_washing_summary.txt", "w", encoding="utf-8") as f:
            f.write(self._generate_text_report())

        # CSV formats
        if self.bank_year_scores:
            pd.DataFrame(self.bank_year_scores).to_csv(
                output_dir / "bank_year_scores.csv", index=False
            )
        if self.top_risk_claims:
            pd.DataFrame(self.top_risk_claims).to_csv(
                output_dir / "top_risk_claims_global.csv", index=False
            )
        if self.sentence_details:
            pd.DataFrame(self.sentence_details).to_csv(
                output_dir / "sentence_details_full.csv", index=False
            )

        print(f"Reports (JSON, TXT, CSV) saved to {output_dir}")

    def _generate_text_report(self) -> str:
        """Generate human-readable text report."""
        lines = []
        lines.append("=" * 70)
        lines.append("ESG-WASHING RISK ASSESSMENT REPORT")
        lines.append(f"Generated: {self.generated_at}")
        lines.append("=" * 70)

        lines.append(f"\nOVERVIEW")
        lines.append(f"  Banks analyzed: {self.total_banks}")
        lines.append(f"  Bank-years: {self.total_bank_years}")
        lines.append(f"  ESG sentences: {self.total_esg_sentences:,}")
        lines.append(f"  Average EWRI (new): {self.avg_ewri:.2f} ± {self.std_ewri:.2f}")
        lines.append(f"  Average EWRI (old): {self.avg_ewri_old:.2f}")
        lines.append(f"  EWRI Range: [{self.min_ewri:.2f}, {self.max_ewri:.2f}]")

        lines.append(f"\nRISK DISTRIBUTION")
        for level, count in self.risk_distribution.items():
            lines.append(f"  {level}: {count}")

        # Decomposition
        if self.ewri_decomposition:
            lines.append(f"\nEWRI DECOMPOSITION (Where does risk come from?)")
            for label, info in self.ewri_decomposition.items():
                lines.append(f"  {label:15s}: {info['mean_contribution']:.2f} pts "
                           f"({info['pct_of_ewri']:.1f}% of total)")

        lines.append(f"\nBANK-YEAR SCORES")
        for score in self.bank_year_scores[:15]:
            lines.append(
                f"  {score['bank']:15s} {score['year']}  "
                f"EWRI={score['ewri']:5.1f} (old={score.get('ewri_old', 0):5.1f})  "
                f"{score['risk_level']}"
            )

        if self.topic_summary:
            lines.append(f"\nTOPIC ANALYSIS")
            for topic, stats in self.topic_summary.items():
                if stats.get("avg_ewri") is not None:
                    lines.append(
                        f"  {topic:15s}: Avg EWRI={stats['avg_ewri']:.1f}  "
                        f"N={stats['total_sentences']}  "
                        f"Impl={stats.get('implemented_pct', 0):.1f}%"
                    )

        if self.top_risk_claims:
            lines.append(f"\nTOP RISK CLAIMS")
            for i, claim in enumerate(self.top_risk_claims[:10], 1):
                lines.append(
                    f"  {i}. [{claim.get('washing_risk', 0):.3f}] [{claim['action_label']}] "
                    f"{claim['sentence'][:80]}..."
                )

        if self.evidence_stats:
            lines.append(f"\nEVIDENCE STATISTICS")
            for key, val in self.evidence_stats.items():
                lines.append(f"  {key}: {val}")

        # Qualitative verification
        if self.qualitative_samples:
            lines.append(f"\nQUALITATIVE VERIFICATION SAMPLES")
            for cat_key, cat_name in [
                ("high_risk_pure_washing", "Pure Washing (Indeterminate + No Evidence)"),
                ("suspicious_unsubstantiated", "Suspicious (Implemented + No Evidence)"),
                ("low_risk_substantive", "Substantive (Implemented + Evidence)"),
            ]:
                samples = self.qualitative_samples.get(cat_key, [])
                if samples:
                    lines.append(f"\n  {cat_name}:")
                    for s in samples[:3]:
                        lines.append(
                            f"    [{s['bank']} {s['year']}] WRS={s['washing_risk']:.3f} "
                            f"\"{s['sentence'][:100]}...\""
                        )

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)


def generate_report(
    df: pd.DataFrame,
    ewri_scores: list,
    df_scores: pd.DataFrame,
    analysis_results: Optional[dict] = None,
) -> "ESGWashingReport":
    """
    Generate a structured ESG-washing report from pipeline results.
    """
    report = ESGWashingReport()
    report.generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report.total_banks = df["bank"].nunique() if "bank" in df.columns else 0
    report.total_bank_years = len(df_scores)
    report.total_esg_sentences = len(df)

    # EWRI statistics
    if len(df_scores) > 0:
        report.avg_ewri = round(float(df_scores["ewri"].mean()), 2)
        report.min_ewri = round(float(df_scores["ewri"].min()), 2)
        report.max_ewri = round(float(df_scores["ewri"].max()), 2)
        report.std_ewri = round(float(df_scores["ewri"].std()), 2)
        if "ewri_old" in df_scores.columns:
            report.avg_ewri_old = round(float(df_scores["ewri_old"].mean()), 2)

    # Risk distribution
    if "risk_level" in df_scores.columns:
        report.risk_distribution = df_scores["risk_level"].value_counts().to_dict()

    # Bank-year scores
    report.bank_year_scores = df_scores.to_dict("records")

    # EWRI Decomposition
    if all(c in df_scores.columns for c in ["contrib_implemented", "contrib_planning", "contrib_indeterminate"]):
        ewri_mean = max(df_scores["ewri"].mean(), 1e-9)
        report.ewri_decomposition = {
            "Indeterminate": {
                "mean_contribution": round(float(df_scores["contrib_indeterminate"].mean()), 2),
                "pct_of_ewri": round(float(df_scores["contrib_indeterminate"].mean()) / ewri_mean * 100, 1),
            },
            "Planning": {
                "mean_contribution": round(float(df_scores["contrib_planning"].mean()), 2),
                "pct_of_ewri": round(float(df_scores["contrib_planning"].mean()) / ewri_mean * 100, 1),
            },
            "Implemented": {
                "mean_contribution": round(float(df_scores["contrib_implemented"].mean()), 2),
                "pct_of_ewri": round(float(df_scores["contrib_implemented"].mean()) / ewri_mean * 100, 1),
            },
        }

    # Topic summary
    topic_col = "topic_label" if "topic_label" in df.columns else None

    if topic_col:
        esg_topics = ["E", "S_labor", "S_community", "S_product", "G"]
        for topic in esg_topics:
            topic_df = df[df[topic_col] == topic]
            if len(topic_df) == 0:
                continue

            topic_ewris = []
            for score in ewri_scores:
                if hasattr(score, "topic_breakdown") and topic in score.topic_breakdown:
                    tb = score.topic_breakdown[topic]
                    if tb.get("ewri") is not None:
                        topic_ewris.append(tb["ewri"])

            action_col = "action_label" if "action_label" in topic_df.columns else None
            impl_pct = 0
            if action_col:
                impl_pct = round((topic_df[action_col] == "Implemented").mean() * 100, 1)

            report.topic_summary[topic] = {
                "total_sentences": len(topic_df),
                "avg_ewri": round(sum(topic_ewris) / len(topic_ewris), 2) if topic_ewris else None,
                "implemented_pct": impl_pct,
            }

    # Sentence-level traceability (from EWRIScore objects)
    all_claims = []
    for score in ewri_scores:
        if hasattr(score, "sentence_risks"):
            for claim in score.sentence_risks:
                claim["bank"] = score.bank
                claim["year"] = score.year
                all_claims.append(claim)

    all_claims.sort(key=lambda x: x.get("washing_risk", 0), reverse=True)
    report.sentence_details = all_claims
    report.top_risk_claims = all_claims[:30]

    # Evidence statistics
    if "has_evidence" in df.columns:
        has_ev = df["has_evidence"].sum()
        report.evidence_stats = {
            "sentences_with_evidence": int(has_ev),
            "evidence_rate": f"{100 * has_ev / len(df):.1f}%",
        }
        if "es_combined" in df.columns:
            report.evidence_stats["mean_evidence_strength"] = round(
                float(df["es_combined"].mean()), 3
            )
        elif "evidence_strength" in df.columns:
            report.evidence_stats["mean_evidence_strength"] = round(
                float(df["evidence_strength"].mean()), 3
            )

    # Analysis results
    if analysis_results:
        # Store analysis but exclude huge data
        report.analysis_results = {
            k: v for k, v in analysis_results.items()
            if k not in ["qualitative_samples"]
        }
        # Store qualitative samples separately
        if "qualitative_samples" in analysis_results:
            report.qualitative_samples = analysis_results["qualitative_samples"]

    return report


if __name__ == "__main__":
    print("ESG-Washing Report Generator")
    print("Usage: from report import generate_report")
