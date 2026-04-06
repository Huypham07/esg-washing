import sys
import yaml
import pandas as pd
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.corpus.build_corpus import build as build_corpus
from src.pipeline.evidence_experiments import apply_evidence_variant, run_experiments
from src.pipeline.ewri import (
    calculate_bank_year_ewri, scores_to_dataframe, EWRIScore,
    enrich_with_risk_scores, print_ewri_summary,
)
from src.training.neuro_symbolic import SymbolicReasoner, create_constrained_inference
from src.pipeline.report import ESGWashingReport, generate_report
from src.pipeline.analysis import run_full_analysis, save_analysis, print_analysis_summary


def load_pipeline_config(config_path: str = "config/pipeline.yml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class ESGWashingPipeline:
    def __init__(self, config_path: str = "config/pipeline.yml"):
        self.config = load_pipeline_config(config_path)
        self.reasoner = SymbolicReasoner(
            min_confidence=self.config.get("neuro_symbolic", {}).get("min_rule_confidence", 0.3)
        )
        self._topic_model = None
        self._action_model = None
        self._linker = None

    def _resolve_model_path(self, task: str) -> str:
        model_cfg = self.config["model"][task]
        configured_path = Path(model_cfg["path"])

        candidates: list[Path] = [configured_path, configured_path / "final"]
        configured_str = str(configured_path)

        if "outputs/" in configured_str:
            alt = Path(configured_str.replace("outputs/", "output/"))
            candidates.extend([alt, alt / "final"])
        if "output/" in configured_str:
            alt = Path(configured_str.replace("output/", "outputs/"))
            candidates.extend([alt, alt / "final"])

        seen: set[str] = set()
        dedup_candidates = []
        for candidate in candidates:
            key = str(candidate)
            if key not in seen:
                dedup_candidates.append(candidate)
                seen.add(key)

        for candidate in dedup_candidates:
            if candidate.exists():
                if candidate != configured_path:
                    print(f"[Pipeline] Model fallback for '{task}': {candidate}")
                return str(candidate)

        tried = ", ".join(str(p) for p in dedup_candidates)
        raise FileNotFoundError(
            f"Model for task '{task}' not found. Tried: {tried}. "
            f"Please train/export model artifacts first."
        )
    
    def _load_topic_model(self):
        """Lazy load topic classification model."""
        if self._topic_model is not None:
            return
        
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        model_path = self._resolve_model_path("topic")
        print(f"[Pipeline] Loading topic model: {model_path}")
        
        self._topic_tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self._topic_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self._topic_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._topic_model.to(self._topic_device)
        self._topic_model.eval()
        
        self._topic_predictor = create_constrained_inference(
            task="topic",
            alpha=self.config.get("neuro_symbolic", {}).get("rule_alpha", 0.3),
            labels=self.config["model"]["topic"].get("labels"),
            config=self.config,
        )
    
    def _load_action_model(self):
        """Lazy load actionability classification model."""
        if self._action_model is not None:
            return
        
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        model_path = self._resolve_model_path("actionability")
        print(f"[Pipeline] Loading actionability model: {model_path}")
        
        self._action_tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self._action_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self._action_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._action_model.to(self._action_device)
        self._action_model.eval()
        
        self._action_predictor = create_constrained_inference(
            task="action",
            alpha=self.config.get("neuro_symbolic", {}).get("rule_alpha", 0.3),
            labels=self.config["model"]["actionability"].get("labels"),
            config=self.config,
        )
    
    def build_corpus(
        self,
        raw_txt_path: Optional[str] = None
    ) -> pd.DataFrame:
        print("\n" + "=" * 60)
        print("Build Corpus")
        print("=" * 60)
        build_corpus(
            input_path=raw_txt_path,
            output_blocks=self.config["paths"]["blocks"],
            output_sentences=self.config["paths"]["sentences"]
        )
        
        sentences_path = Path(self.config["paths"]["sentences"])
        df = pd.read_parquet(sentences_path)
        print(f"Corpus built: {len(df):,} sentences")
        return df
    
    def topic_classification(self, df: pd.DataFrame, batch_size: int = 64) -> pd.DataFrame:
        print("\n" + "="*60)
        print("Topic Classification")
        print("="*60)
        
        import torch
        from tqdm.auto import tqdm
        
        self._load_topic_model()
        
        sentences = df["sentence"].tolist()
        all_labels = []
        all_probs = []
        all_explanations = []
        
        for i in tqdm(range(0, len(sentences), batch_size), desc="Topic classification"):
            batch = sentences[i:i + batch_size]
            
            inputs = self._topic_tokenizer(
                batch, return_tensors="pt", truncation=True,
                padding=True, max_length=self.config["model"]["topic"]["max_length"],
            )
            inputs = {k: v.to(self._topic_device) for k, v in inputs.items()}
            
            with torch.no_grad():
                logits = self._topic_model(**inputs).logits
            
            augmented = self._topic_predictor.predict(logits, batch)
            for aug in augmented:
                all_labels.append(aug.label)
                all_probs.append(aug.confidence)
        
        df = df.copy()
        df["topic_label"] = all_labels
        df["topic_confidence"] = all_probs
        
        print(f"\nTopic distribution:")
        print(df["topic_label"].value_counts())
        
        return df
    
    def actionability_classification(self, df: pd.DataFrame, batch_size: int = 64) -> pd.DataFrame:
        print("\n" + "="*60)
        print("Actionability Classification")
        print("="*60)
        
        import torch
        from tqdm.auto import tqdm
        
        # Filter ESG-only
        esg_df = df[df["topic_label"] != "Non_ESG"].copy()
        print(f"ESG sentences: {len(esg_df):,} / {len(df):,}")
        
        self._load_action_model()
        
        sentences = esg_df["sentence"].tolist()
        all_labels = []
        all_probs = []
        all_explanations = []
        
        for i in tqdm(range(0, len(sentences), batch_size), desc="Actionability"):
            batch = sentences[i:i + batch_size]
            
            inputs = self._action_tokenizer(
                batch, return_tensors="pt", truncation=True,
                padding=True, max_length=self.config["model"]["actionability"]["max_length"],
            )
            inputs = {k: v.to(self._action_device) for k, v in inputs.items()}
            
            with torch.no_grad():
                logits = self._action_model(**inputs).logits
            
            augmented = self._action_predictor.predict(logits, batch)
            for aug in augmented:
                all_labels.append(aug.label)
                all_probs.append(aug.confidence)
                all_explanations.append("; ".join(aug.explanations[:2]))
        
        esg_df["action_label"] = all_labels
        esg_df["action_confidence"] = all_probs
        
        print(f"\nActionability distribution:")
        print(esg_df["action_label"].value_counts())
        
        return esg_df
    
    def evidence_extr(self, df: pd.DataFrame, evidence_variant: str = "nli") -> pd.DataFrame:
        print("\n" + "="*60)
        print(f"Evidence Detection + Linking [{evidence_variant}]")
        print("="*60)

        df = apply_evidence_variant(df, variant=evidence_variant)
        has_ev = int(df["has_evidence"].sum()) if "has_evidence" in df.columns else 0
        print(f"Sentences with evidence: {has_ev:,} / {len(df):,} ({100*has_ev/max(len(df),1):.1f}%)")
        return df

    def ewri(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[EWRIScore]]:
        print("\n" + "="*60)
        print("EWRI Calculation")
        print("="*60)

        # Enrich with per-sentence risk scores
        df = enrich_with_risk_scores(df)

        # Calculate bank-year EWRI
        scores = calculate_bank_year_ewri(df)
        df_scores = scores_to_dataframe(scores)
        df_scores = df_scores.sort_values("ewri", ascending=False)

        # Print summary
        print_ewri_summary(df_scores, scores)

        return df, df_scores, scores

    def analysis(
        self, df: pd.DataFrame, ewri_scores: list[EWRIScore], df_scores: pd.DataFrame
    ) -> dict:
        """Run comprehensive ESG-washing analysis."""
        print("\n" + "="*60)
        print("Comprehensive Analysis")
        print("="*60)

        analysis_results = run_full_analysis(df, ewri_scores, df_scores)
        print_analysis_summary(analysis_results)

        # Save analysis outputs
        output_dir = self.config["paths"].get("analysis_dir", "outputs/analysis")
        save_analysis(analysis_results, output_dir)

        return analysis_results

    def report(
        self, df: pd.DataFrame, ewri_scores: list[EWRIScore], df_scores: pd.DataFrame,
        analysis_results: Optional[dict] = None,
    ) -> ESGWashingReport:
        """Generate structured report with analysis."""
        print("\n" + "="*60)
        print("Report Generation")
        print("="*60)

        report = generate_report(df, ewri_scores, df_scores, analysis_results)

        output_dir = Path(self.config["paths"].get("report_dir", "outputs/reports"))
        output_dir.mkdir(parents=True, exist_ok=True)
        report.save(output_dir)
        print(f"Report saved to: {output_dir}")

        return report

    def run(
        self,
        skip_corpus: bool = False,
        raw_txt_path: Optional[str] = None,
        evidence_variant: Optional[str] = "nli",
        run_evidence_experiments: bool = False,
        evidence_experiment_variants: Optional[list[str]] = None,
        skip_to_ewri: bool = False,
        evidence_input: Optional[str] = None,
    ) -> dict:
        block_path = Path(self.config["paths"]["blocks"])
        sentences_path = Path(self.config["paths"]["sentences"])
        esg_sentences_path = Path(self.config["paths"]["esg_sentences"])
        actionability_sentences_path = Path(self.config["paths"]["actionability_sentences"])

        df = None

        # ── Stage: Load or Build Data ──────────────────────────────
        if skip_to_ewri and evidence_input:
            # Load pre-computed evidence data directly
            print(f"\n[Pipeline] Loading pre-computed evidence data: {evidence_input}")
            df = pd.read_parquet(evidence_input)
            print(f"[Pipeline] Loaded {len(df):,} rows")

        elif skip_corpus:
            if not actionability_sentences_path.exists():
                raise FileNotFoundError(
                    f"File not found: {actionability_sentences_path}. Cannot skip corpus build."
                )
            df = pd.read_parquet(actionability_sentences_path)
            print(f"[Pipeline] Loaded classified sentences: {len(df):,} rows")

            # ── Evidence ─────────────────────────────────────────
            if run_evidence_experiments:
                exp_output_dir = self.config["paths"].get(
                    "evidence_experiments_dir", "outputs/experiments/evidence"
                )
                print("\n" + "=" * 60)
                print("Running Evidence Experiments (Ablation)")
                print("=" * 60)
                summary_df = run_experiments(
                    input_path=str(actionability_sentences_path),
                    output_dir=exp_output_dir,
                    variants=evidence_experiment_variants,
                )
                print(summary_df.to_string(index=False))

                # Use primary variant for downstream
                evidence_path = Path(exp_output_dir) / f"evidence_{evidence_variant}.parquet"
                if evidence_path.exists():
                    df = pd.read_parquet(evidence_path)
                else:
                    df = self.evidence_extr(df, evidence_variant=evidence_variant)
            else:
                df = self.evidence_extr(df, evidence_variant=evidence_variant)
        else:
            # Full pipeline: build corpus → classify → evidence
            corpus_df = self.build_corpus(raw_txt_path=raw_txt_path)

            sentences_df = pd.read_parquet(sentences_path)
            print(f"Loaded sentences: {len(sentences_df):,} rows")

            topic_df = self.topic_classification(sentences_df)
            esg_df = topic_df[topic_df["topic_label"] != "Non_ESG"].copy()
            esg_sentences_path.parent.mkdir(parents=True, exist_ok=True)
            esg_df.to_parquet(esg_sentences_path, index=False)

            df = self.actionability_classification(topic_df)
            actionability_sentences_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(actionability_sentences_path, index=False)

            df = self.evidence_extr(df, evidence_variant=evidence_variant)

        # ── EWRI ──────────────────────────────────────────────
        df, df_scores, ewri_scores = self.ewri(df)

        # ── Analysis ──────────────────────────────────────────
        analysis_results = self.analysis(df, ewri_scores, df_scores)

        # ── Report ────────────────────────────────────────────
        report = self.report(df, ewri_scores, df_scores, analysis_results)

        # ── Save enriched data ────────────────────────────────
        output_data_path = Path(
            self.config["paths"].get("es_scored", "data/corpus/esg_sentences_es_scored.parquet")
        )
        output_data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_data_path, index=False)
        print(f"\nSaved enriched data: {output_data_path}")

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)

        return {
            "esg_df": df,
            "ewri_scores": ewri_scores,
            "df_scores": df_scores,
            "report": report,
            "analysis": analysis_results,
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ESG-Washing Detection Pipeline")
    parser.add_argument("--skip-corpus", action="store_true", help="Skip corpus building")
    parser.add_argument(
        "--raw-txt-path",
        type=str,
        default=None,
        help="Path to raw txt file or directory for corpus building",
    )
    parser.add_argument(
        "--evidence-variant",
        type=str,
        default="nli",
        help="Evidence variant: nli, window, no_nli",
    )
    parser.add_argument(
        "--run-evidence-experiments",
        action="store_true",
        help="Run all/selected evidence variants for ablation before main evidence step",
    )
    parser.add_argument(
        "--evidence-experiment-variants",
        type=str,
        default=None,
        help="Comma-separated variants for experiment runner",
    )
    parser.add_argument(
        "--skip-to-ewri",
        action="store_true",
        help="Skip to EWRI stage; requires --evidence-input",
    )
    parser.add_argument(
        "--evidence-input",
        type=str,
        default=None,
        help="Path to pre-computed evidence parquet file (used with --skip-to-ewri)",
    )
    args = parser.parse_args()
    experiment_variants = (
        [v.strip() for v in args.evidence_experiment_variants.split(",") if v.strip()]
        if args.evidence_experiment_variants
        else None
    )
    
    pipeline = ESGWashingPipeline()
    results = pipeline.run(
        skip_corpus=args.skip_corpus,
        raw_txt_path=args.raw_txt_path,
        evidence_variant=args.evidence_variant,
        run_evidence_experiments=args.run_evidence_experiments,
        evidence_experiment_variants=experiment_variants,
        skip_to_ewri=args.skip_to_ewri,
        evidence_input=args.evidence_input,
    )
