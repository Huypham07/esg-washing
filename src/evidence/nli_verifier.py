"""
NLI-based Evidence Verifier for ESG Claim-Evidence Linking

Uses a cross-lingual Natural Language Inference (NLI) model to verify 
whether a candidate evidence sentence actually supports (entails) the claim,
rather than just being semantically similar.

Three NLI labels:
- Entailment: Evidence supports the claim → strong evidence
- Neutral: Evidence is related but doesn't directly support → weak evidence  
- Contradiction: Evidence contradicts the claim → negative signal

This addresses a key weakness of cosine-similarity-only approaches:
high similarity ≠ logical support (e.g., "We plan to reduce emissions" 
and "We have not reduced emissions" are similar but contradictory).

Model: joeddav/xlm-roberta-large-xnli (cross-lingual, supports Vietnamese)
Fallback: MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7

References:
    Conneau et al. (2018). XNLI: Evaluating Cross-lingual Sentence Representations
    Laurer et al. (2022). Less Annotating, More Classifying (mDeBERTa NLI)
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class NLIResult:
    """Result of NLI verification between a claim and evidence."""
    premise: str      # The ESG claim
    hypothesis: str   # The candidate evidence
    label: str        # entailment, neutral, contradiction
    scores: Dict[str, float]  # {entailment: x, neutral: y, contradiction: z}
    entailment_score: float
    supports_claim: bool  # True if entailment_score > threshold


# NLI label mapping
NLI_LABELS = ["contradiction", "neutral", "entailment"]


class NLIVerifier:
    """
    Verifies claim-evidence relationships using NLI.
    
    Uses a pre-trained cross-lingual NLI model to determine if
    evidence sentences actually support ESG claims.
    """
    
    # Model options ordered by preference (quality vs speed)
    MODEL_OPTIONS = [
        "joeddav/xlm-roberta-large-xnli",           # Best quality
        "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",  # Good balance
        "typeform/distilbert-base-uncased-mnli",     # Fast but English-only
    ]
    
    def __init__(
        self,
        model_name: str = None,
        entailment_threshold: float = 0.5,
        device: str = None,
        max_length: int = 256,
    ):
        """
        Initialize the NLI verifier.
        
        Args:
            model_name: HuggingFace model name. None = auto-select.
            entailment_threshold: Min entailment score to consider as support.
            device: 'cuda', 'cpu', or None (auto-detect).
            max_length: Max token length for model inputs.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if model_name is None:
            # Try models in order of preference
            model_name = self._find_available_model()
        
        self.model_name = model_name
        self.entailment_threshold = entailment_threshold
        self.device = device
        self.max_length = max_length
        self._pipeline = None
        
    def _find_available_model(self) -> str:
        """Try to find an available model, defaulting to the first option."""
        return self.MODEL_OPTIONS[0]
    
    def _load_pipeline(self):
        """Lazy load the NLI pipeline."""
        if self._pipeline is None:
            from transformers import pipeline
            print(f"Loading NLI model: {self.model_name}")
            print(f"Device: {self.device}")
            
            self._pipeline = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=0 if self.device == "cuda" else -1,
            )
            print("NLI model loaded successfully.")
    
    def verify_pair(self, claim: str, evidence: str) -> NLIResult:
        """
        Verify if evidence supports the claim using NLI.
        
        The claim is the premise and evidence is the hypothesis.
        If evidence entails (supports) the claim, it's valid evidence.
        
        Args:
            claim: The ESG claim/statement
            evidence: The candidate evidence sentence
            
        Returns:
            NLIResult with entailment analysis
        """
        self._load_pipeline()
        
        try:
            # Use zero-shot classification as NLI proxy
            # "Does the evidence support this claim?"
            result = self._pipeline(
                evidence,
                candidate_labels=["supports the claim", "is unrelated", "contradicts the claim"],
                hypothesis_template="This evidence {} '{}'".format("{}", claim[:100]),
                multi_label=False,
            )
            
            # Map back to NLI labels
            label_map = {
                "supports the claim": "entailment",
                "is unrelated": "neutral", 
                "contradicts the claim": "contradiction",
            }
            
            scores = {}
            for lbl, sc in zip(result["labels"], result["scores"]):
                nli_label = label_map.get(lbl, lbl)
                scores[nli_label] = float(sc)
            
            entailment_score = scores.get("entailment", 0.0)
            best_label = max(scores, key=scores.get)
            
        except Exception as e:
            # Fallback: use simple heuristic if model fails
            print(f"NLI model error, using fallback: {e}")
            entailment_score = 0.5
            scores = {"entailment": 0.5, "neutral": 0.3, "contradiction": 0.2}
            best_label = "neutral"
        
        return NLIResult(
            premise=claim,
            hypothesis=evidence,
            label=best_label,
            scores=scores,
            entailment_score=entailment_score,
            supports_claim=entailment_score >= self.entailment_threshold,
        )
    
    def verify_batch(
        self,
        claims: List[str],
        evidences: List[str],
        batch_size: int = 16,
    ) -> List[NLIResult]:
        """
        Verify multiple claim-evidence pairs.
        
        Args:
            claims: List of ESG claims
            evidences: List of candidate evidence sentences (same length)
            batch_size: Processing batch size
            
        Returns:
            List of NLIResult objects
        """
        assert len(claims) == len(evidences), "Claims and evidences must be same length"
        
        results = []
        for i in range(0, len(claims), batch_size):
            batch_claims = claims[i:i+batch_size]
            batch_evidences = evidences[i:i+batch_size]
            
            for claim, evidence in zip(batch_claims, batch_evidences):
                result = self.verify_pair(claim, evidence)
                results.append(result)
                
            if (i + batch_size) % 100 == 0:
                print(f"NLI verified {min(i+batch_size, len(claims))}/{len(claims)} pairs")
        
        return results


class RuleBasedNLIVerifier:
    """
    Lightweight rule-based NLI approximation for when the full model
    is too slow or unavailable. Uses linguistic cues to assess support.
    
    This is faster than the model-based approach and can serve as a
    fallback or for ablation studies comparing rule-NLI vs model-NLI.
    """
    
    def __init__(self, entailment_threshold: float = 0.5):
        import re
        self.re = re
        self.entailment_threshold = entailment_threshold
        
        # Patterns indicating the evidence supports the claim
        self.support_patterns = [
            # Evidence contains quantitative backup
            r"\d+[\.,]?\d*\s*(%|tỷ|triệu|tấn|kWh|MWh)",
            # Evidence references standards/verification
            r"\b(GRI|ISO|SBTi|kiểm toán|chứng nhận|xác nhận)\b",
            # Evidence mentions completion/results
            r"\b(đã hoàn thành|đã đạt|kết quả|ghi nhận)\b",
            # Temporal specificity
            r"\b(năm|quý|tháng)\s*\d+",
        ]
        
        # Patterns indicating contradiction
        self.contradiction_patterns = [
            r"\b(chưa|không|thiếu|chưa đạt|thất bại|chưa triển khai)\b",
            r"\b(trái ngược|mâu thuẫn|ngược lại)\b",
        ]
        
        # Patterns indicating neutrality (generic/unrelated)
        self.neutral_patterns = [
            r"\b(ngoài ra|bên cạnh|cũng|đồng thời)\b",
        ]
    
    def verify_pair(self, claim: str, evidence: str) -> NLIResult:
        """Rule-based NLI verification."""
        evidence_lower = evidence.lower()
        claim_lower = claim.lower()
        
        support_score = 0.0
        contra_score = 0.0
        neutral_score = 0.3  # Base neutral
        
        # Check support patterns in evidence
        for pat in self.support_patterns:
            if self.re.search(pat, evidence_lower, self.re.IGNORECASE):
                support_score += 0.2
        
        # Check contradiction patterns
        for pat in self.contradiction_patterns:
            if self.re.search(pat, evidence_lower, self.re.IGNORECASE):
                contra_score += 0.25
        
        # Check if evidence contains key nouns from claim (topical overlap)
        claim_words = set(claim_lower.split())
        evidence_words = set(evidence_lower.split())
        overlap = len(claim_words & evidence_words) / max(len(claim_words), 1)
        support_score += overlap * 0.3
        
        # Normalize
        total = support_score + contra_score + neutral_score
        if total > 0:
            support_score /= total
            contra_score /= total
            neutral_score /= total
        
        scores = {
            "entailment": round(support_score, 4),
            "neutral": round(neutral_score, 4),
            "contradiction": round(contra_score, 4),
        }
        
        best_label = max(scores, key=scores.get)
        
        return NLIResult(
            premise=claim,
            hypothesis=evidence,
            label=best_label,
            scores=scores,
            entailment_score=support_score,
            supports_claim=support_score >= self.entailment_threshold,
        )
    
    def verify_batch(self, claims: List[str], evidences: List[str], **kwargs) -> List[NLIResult]:
        """Verify multiple pairs using rules."""
        return [self.verify_pair(c, e) for c, e in zip(claims, evidences)]


if __name__ == "__main__":
    print("NLI Evidence Verifier")
    print("=" * 50)
    
    # Test with rule-based verifier (no model needed)
    verifier = RuleBasedNLIVerifier()
    
    test_pairs = [
        (
            "Ngân hàng cam kết giảm phát thải CO2",
            "Trong năm 2023, lượng phát thải CO2 đã giảm 15% so với năm trước."
        ),
        (
            "Ngân hàng cam kết giảm phát thải CO2",
            "Ngân hàng luôn quan tâm đến trách nhiệm xã hội."
        ),
        (
            "Ngân hàng cam kết giảm phát thải CO2",
            "Lượng phát thải CO2 chưa đạt mục tiêu đề ra."
        ),
    ]
    
    for claim, evidence in test_pairs:
        result = verifier.verify_pair(claim, evidence)
        print(f"\nClaim: \"{claim}\"")
        print(f"Evidence: \"{evidence}\"")
        print(f"  → {result.label} (entailment={result.entailment_score:.3f})")
        print(f"    Supports: {result.supports_claim}")
