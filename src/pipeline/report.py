"""
ESG-Washing Report Generator.

Generates structured reports from EWRI analysis results, including:
- Overall EWRI scores by bank-year
- Topic breakdown (per E/S/G)
- Top risk claims with explanations
- Evidence coverage statistics
- Traceability: EWRI → contributing sentences → document location
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
    
    # Risk distribution
    risk_distribution: dict = field(default_factory=dict)
    
    # Per bank-year scores
    bank_year_scores: List[dict] = field(default_factory=list)
    
    # Topic breakdown (aggregated)
    topic_summary: dict = field(default_factory=dict)
    
    # Top risk claims (across all banks, max top N)
    top_risk_claims: List[dict] = field(default_factory=list)
    
    # All sentence details for full traceability
    sentence_details: List[dict] = field(default_factory=list)
    
    # Evidence statistics
    evidence_stats: dict = field(default_factory=dict)
    
    def save(self, output_dir: Path):
        """Save report to multiple formats."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # JSON report (full)
        report_dict = asdict(self)
        with open(output_dir / "esg_washing_report.json", "w", encoding="utf-8") as f:
            json.dump(report_dict, f, ensure_ascii=False, indent=2, default=str)
        
        # Summary text report
        with open(output_dir / "esg_washing_summary.txt", "w", encoding="utf-8") as f:
            f.write(self._generate_text_report())
            
        # CSV formats
        if self.bank_year_scores:
            pd.DataFrame(self.bank_year_scores).to_csv(output_dir / "bank_year_scores.csv", index=False)
        if self.top_risk_claims:
            pd.DataFrame(self.top_risk_claims).to_csv(output_dir / "top_risk_claims_global.csv", index=False)
        if self.sentence_details:
            pd.DataFrame(self.sentence_details).to_csv(output_dir / "sentence_details_full.csv", index=False)
            
        print(f"Reports (JSON, TXT, CSV) saved to {output_dir}")
    
    def _generate_text_report(self) -> str:
        """Generate human-readable text report."""
        lines = []
        lines.append("=" * 60)
        lines.append("ESG-WASHING RISK ASSESSMENT REPORT")
        lines.append(f"Generated: {self.generated_at}")
        lines.append("=" * 60)
        
        lines.append(f"\n📊 OVERVIEW")
        lines.append(f"  Banks analyzed: {self.total_banks}")
        lines.append(f"  Bank-years: {self.total_bank_years}")
        lines.append(f"  ESG sentences: {self.total_esg_sentences:,}")
        lines.append(f"  Average EWRI: {self.avg_ewri:.2f}")
        lines.append(f"  EWRI Range: [{self.min_ewri:.2f}, {self.max_ewri:.2f}]")
        
        lines.append(f"\n📈 RISK DISTRIBUTION")
        for level, count in self.risk_distribution.items():
            lines.append(f"  {level}: {count}")
        
        lines.append(f"\n🏦 BANK-YEAR SCORES (Top 10)")
        for score in self.bank_year_scores[:10]:
            lines.append(f"  {score['bank']:15s} {score['year']}  "
                        f"EWRI={score['ewri']:5.1f}  {score['risk_level']}")
        
        if self.topic_summary:
            lines.append(f"\n📋 TOPIC ANALYSIS")
            for topic, stats in self.topic_summary.items():
                if stats.get("avg_ewri") is not None:
                    lines.append(f"  {topic:15s}: Avg EWRI={stats['avg_ewri']:.1f}  "
                                f"N={stats['total_sentences']}  "
                                f"Risk={stats.get('avg_risk_level', 'N/A')}")
        
        if self.top_risk_claims:
            lines.append(f"\n⚠️ TOP RISK CLAIMS")
            for i, claim in enumerate(self.top_risk_claims[:10], 1):
                lines.append(f"  {i}. [{claim['risk_score']:.1f}] [{claim['action_pred']}] "
                            f"{claim['sentence'][:80]}...")
                if claim.get("explanation"):
                    lines.append(f"     → {claim['explanation'][:100]}")
        
        if self.evidence_stats:
            lines.append(f"\n📎 EVIDENCE STATISTICS")
            for key, val in self.evidence_stats.items():
                lines.append(f"  {key}: {val}")
        
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


def generate_report(
    df: pd.DataFrame,
    ewri_scores: list,
    df_scores: pd.DataFrame,
) -> ESGWashingReport:
    """
    Generate a structured ESG-washing report from pipeline results.
    
    Args:
        df: ESG sentences DataFrame with evidence scores
        ewri_scores: List of EWRIScore objects
        df_scores: EWRI scores DataFrame
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
    
    # Risk distribution
    if "risk_level" in df_scores.columns:
        report.risk_distribution = df_scores["risk_level"].value_counts().to_dict()
    
    # Bank-year scores
    report.bank_year_scores = df_scores.to_dict("records")
    
    # Topic summary (aggregate across all bank-years)
    topic_col = None
    for col in ["topic_label", "topic", "topic_pred"]:
        if col in df.columns:
            topic_col = col
            break
    
    if topic_col:
        esg_topics = ["E", "S_labor", "S_community", "S_product", "G"]
        for topic in esg_topics:
            topic_df = df[df[topic_col] == topic]
            if len(topic_df) == 0:
                continue
            
            # Collect per-topic EWRI from scores
            topic_ewris = []
            for score in ewri_scores:
                if hasattr(score, "topic_breakdown") and topic in score.topic_breakdown:
                    tb = score.topic_breakdown[topic]
                    if tb.get("ewri") is not None:
                        topic_ewris.append(tb["ewri"])
            
            action_col = "action_pred" if "action_pred" in topic_df.columns else None
            impl_pct = 0
            if action_col:
                impl_pct = round((topic_df[action_col] == "Implemented").mean() * 100, 1)
            
            report.topic_summary[topic] = {
                "total_sentences": len(topic_df),
                "avg_ewri": round(sum(topic_ewris) / len(topic_ewris), 2) if topic_ewris else None,
                "implemented_pct": impl_pct,
            }
    
    # Top risk claims (collect from all bank-years, sort globally)
    all_claims = []
    for score in ewri_scores:
        if hasattr(score, "top_risk_claims"):
            for claim in score.top_risk_claims:
                claim["bank"] = score.bank
                claim["year"] = score.year
                all_claims.append(claim)
    
    all_claims.sort(key=lambda x: x.get("risk_score", 0), reverse=True)
    report.sentence_details = all_claims
    report.top_risk_claims = all_claims[:20]  # Top 20 globally
    
    # Evidence statistics
    if "has_evidence" in df.columns:
        has_ev = df["has_evidence"].sum()
        report.evidence_stats = {
            "sentences_with_evidence": int(has_ev),
            "evidence_rate": f"{100*has_ev/len(df):.1f}%",
        }
        if "evidence_strength" in df.columns:
            report.evidence_stats["mean_evidence_strength"] = round(
                float(df["evidence_strength"].mean()), 3
            )
    
    return report


if __name__ == "__main__":
    print("ESG-Washing Report Generator")
    print("Usage: from report import generate_report")
