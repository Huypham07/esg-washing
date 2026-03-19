"""
Enhanced Claim-Evidence Linking Module (v2)

Improvements over the original claim_evidence_linker.py:

1. **Document-level search with TF-IDF pre-filtering**:
   - Instead of fixed ±5 window, searches ALL sentences in the same document
   - Uses TF-IDF to pre-filter top-K candidates, then semantic similarity
   - Captures long-range evidence (e.g., claim in intro, evidence in appendix)

2. **Multi-evidence aggregation**:
   - Links each claim to TOP-K supporting evidence (not just best match)
   - Aggregates evidence strength from multiple sources
   - More robust than single-match approach

3. **NLI verification (optional)**:
   - Uses NLI model/rules to verify entailment between claim and evidence
   - Filters out high-similarity but non-supporting sentences
   - Adds entailment_score to evidence strength formula

Enhanced Evidence Strength formula:
    ES_v2 = w_sim × max_sim + w_R × R(e) + w_nli × NLI(claim, evidence)
    
Where:
    - w_sim = 0.35 (semantic similarity weight)
    - w_R   = 0.35 (rule-based evidence weight)
    - w_nli = 0.30 (NLI entailment weight)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import warnings

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine


@dataclass
class EnhancedClaimEvidenceLink:
    """Enhanced link supporting multiple evidence sentences."""
    claim_id: str
    claim_text: str
    actionability: str
    
    # Primary (best) evidence
    evidence_found: bool
    best_evidence: Optional[str]
    best_evidence_idx: Optional[int]
    similarity_score: float
    
    # Multi-evidence
    all_evidence: List[Dict] = field(default_factory=list)
    num_evidence: int = 0
    avg_similarity: float = 0.0
    
    # NLI scores
    nli_entailment_score: float = 0.0
    nli_label: str = "neutral"
    
    # Evidence metadata
    evidence_types: List[str] = field(default_factory=list)
    search_method: str = "window"  # "window" or "document"


class EnhancedClaimEvidenceLinker:
    """
    Enhanced evidence linker with document-level search, multi-evidence,
    and optional NLI verification.
    """
    
    # Weight configurations for enhanced evidence strength
    WEIGHT_CONFIGS = {
        "default": {"w_sim": 0.35, "w_R": 0.35, "w_nli": 0.30},
        "no_nli": {"w_sim": 0.50, "w_R": 0.50, "w_nli": 0.00},
        "nli_heavy": {"w_sim": 0.25, "w_R": 0.25, "w_nli": 0.50},
        "sim_heavy": {"w_sim": 0.50, "w_R": 0.30, "w_nli": 0.20},
    }
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        window_size: int = 5,
        document_level: bool = True,
        tfidf_top_k: int = 20,
        similarity_threshold: float = 0.5,
        top_k_evidence: int = 3,
        use_nli: bool = True,
        nli_mode: str = "rule",  # "rule" or "model"
        weight_config: str = "default",
        device: str = None,
    ):
        """
        Initialize enhanced linker.
        
        Args:
            model_name: Sentence transformer model
            window_size: Sentence window for local search (still used as boost)
            document_level: Enable document-level search with TF-IDF pre-filtering
            tfidf_top_k: Number of TF-IDF candidates to consider per document
            similarity_threshold: Minimum cosine similarity for valid link
            top_k_evidence: Number of evidence sentences to keep per claim
            use_nli: Whether to use NLI verification
            nli_mode: "rule" (fast, no model) or "model" (accurate, heavy)
            weight_config: Evidence strength weight configuration
            device: 'cuda', 'cpu', or None
        """
        import torch
        from sentence_transformers import SentenceTransformer
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"[EnhancedLinker] Loading: {model_name}")
        print(f"[EnhancedLinker] Device: {device}")
        print(f"[EnhancedLinker] Document-level: {document_level}, TF-IDF top-K: {tfidf_top_k}")
        print(f"[EnhancedLinker] NLI: {use_nli} (mode={nli_mode})")
        print(f"[EnhancedLinker] Top-K evidence: {top_k_evidence}")
        
        self.model = SentenceTransformer(model_name, device=device)
        self.window_size = window_size
        self.document_level = document_level
        self.tfidf_top_k = tfidf_top_k
        self.similarity_threshold = similarity_threshold
        self.top_k_evidence = top_k_evidence
        self.use_nli = use_nli
        self.nli_mode = nli_mode
        self.device = device
        self.weights = self.WEIGHT_CONFIGS.get(weight_config, self.WEIGHT_CONFIGS["default"])
        
        # NLI verifier
        self._nli_verifier = None
        if use_nli:
            self._init_nli(nli_mode)
        
        # Caches
        self._embeddings_cache = None
        self._tfidf_cache = {}
    
    def _init_nli(self, mode: str):
        """Initialize NLI verifier."""
        from evidence.nli_verifier import RuleBasedNLIVerifier, NLIVerifier
        
        if mode == "rule":
            self._nli_verifier = RuleBasedNLIVerifier(entailment_threshold=0.4)
            print("[EnhancedLinker] Using rule-based NLI verifier")
        elif mode == "model":
            self._nli_verifier = NLIVerifier(entailment_threshold=0.5, device=self.device)
            print("[EnhancedLinker] Using model-based NLI verifier")
    
    def embed_sentences(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Embed sentences using sentence transformer."""
        print(f"[EnhancedLinker] Embedding {len(texts)} sentences...")
        embeddings = self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True
        )
        return embeddings
    
    def _build_tfidf_index(self, texts: List[str], doc_key: str) -> np.ndarray:
        """Build TF-IDF matrix for a document (bank-year), with caching."""
        if doc_key in self._tfidf_cache:
            return self._tfidf_cache[doc_key]
        
        vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        self._tfidf_cache[doc_key] = tfidf_matrix
        return tfidf_matrix
    
    def find_evidence_candidates(
        self,
        claim_idx: int,
        df: pd.DataFrame,
        embeddings: np.ndarray = None,
        text_column: str = "text",
    ) -> List[Tuple[int, float]]:
        """
        Find candidate evidence sentences using hybrid approach:
        1. Window-based (±N) for local context
        2. Document-level TF-IDF pre-filtering for global context
        
        Returns:
            List of (candidate_idx, relevance_boost) tuples
        """
        claim_row = df.iloc[claim_idx]
        bank = claim_row["bank"]
        year = claim_row["year"]
        
        # Filter to same document (bank + year)
        same_doc_mask = (df["bank"] == bank) & (df["year"] == year)
        same_doc_idxs = df[same_doc_mask].index.tolist()
        
        candidates_with_boost = []
        seen = set()
        
        # --- Phase 1: Window-based (local context, higher boost) ---
        for idx in same_doc_idxs:
            distance = abs(idx - claim_idx)
            if 0 < distance <= self.window_size:
                # Closer sentences get higher boost
                proximity_boost = 1.0 - (distance / (self.window_size + 1)) * 0.3
                candidates_with_boost.append((idx, proximity_boost))
                seen.add(idx)
        
        # --- Phase 2: Document-level TF-IDF pre-filtering ---
        if self.document_level and len(same_doc_idxs) > self.window_size * 2:
            doc_key = f"{bank}_{year}"
            doc_texts = df.loc[same_doc_idxs, text_column].tolist()
            
            try:
                tfidf_matrix = self._build_tfidf_index(doc_texts, doc_key)
                
                # Find claim position in document
                claim_doc_pos = same_doc_idxs.index(claim_idx)
                claim_tfidf = tfidf_matrix[claim_doc_pos]
                
                # Compute TF-IDF similarity
                tfidf_sims = sklearn_cosine(claim_tfidf, tfidf_matrix).flatten()
                
                # Get top-K candidates by TF-IDF
                top_k_idxs = np.argsort(tfidf_sims)[::-1][:self.tfidf_top_k + 1]
                
                for doc_pos in top_k_idxs:
                    actual_idx = same_doc_idxs[doc_pos]
                    if actual_idx != claim_idx and actual_idx not in seen:
                        # Lower boost for document-level (further away)
                        tfidf_boost = 0.7 + tfidf_sims[doc_pos] * 0.3
                        candidates_with_boost.append((actual_idx, tfidf_boost))
                        seen.add(actual_idx)
            except Exception as e:
                # TF-IDF can fail on very small documents
                pass
        
        return candidates_with_boost
    
    def link_claim_to_evidence(
        self,
        claim_idx: int,
        df: pd.DataFrame,
        embeddings: np.ndarray,
        text_column: str = "text",
    ) -> EnhancedClaimEvidenceLink:
        """
        Link a claim to its top-K supporting evidence, with NLI verification.
        """
        claim_row = df.iloc[claim_idx]
        claim_text = claim_row[text_column]
        claim_emb = embeddings[claim_idx]
        
        # Find candidates (window + document-level)
        candidates_with_boost = self.find_evidence_candidates(
            claim_idx, df, embeddings, text_column
        )
        
        if not candidates_with_boost:
            return EnhancedClaimEvidenceLink(
                claim_id=f"{claim_row['bank']}_{claim_row['year']}_{claim_idx}",
                claim_text=claim_text,
                actionability=claim_row.get("actionability", "Unknown"),
                evidence_found=False,
                best_evidence=None,
                best_evidence_idx=None,
                similarity_score=0.0,
                search_method="none",
            )
        
        candidate_idxs = [c[0] for c in candidates_with_boost]
        candidate_boosts = [c[1] for c in candidates_with_boost]
        
        # Compute semantic similarities
        candidate_embs = embeddings[candidate_idxs]
        raw_similarities = sklearn_cosine(
            claim_emb.reshape(1, -1), candidate_embs
        )[0]
        
        # Apply proximity boosts
        boosted_similarities = raw_similarities * np.array(candidate_boosts)
        
        # Get top-K candidates above threshold
        sorted_indices = np.argsort(boosted_similarities)[::-1]
        
        all_evidence = []
        for rank, sort_idx in enumerate(sorted_indices[:self.top_k_evidence * 2]):
            sim = float(boosted_similarities[sort_idx])
            raw_sim = float(raw_similarities[sort_idx])
            candidate_idx = candidate_idxs[sort_idx]
            
            if raw_sim < self.similarity_threshold * 0.8:
                continue
            
            evidence_row = df.iloc[candidate_idx]
            evidence_text = evidence_row[text_column]
            evidence_types = evidence_row.get("evidence_types", [])
            if not isinstance(evidence_types, list):
                evidence_types = []
            
            # NLI verification
            nli_score = 0.5  # default neutral
            nli_label = "neutral"
            
            if self.use_nli and self._nli_verifier is not None:
                nli_result = self._nli_verifier.verify_pair(claim_text, evidence_text)
                nli_score = nli_result.entailment_score
                nli_label = nli_result.label
                
                # Skip contradicting evidence
                if nli_label == "contradiction" and nli_score < 0.2:
                    continue
            
            all_evidence.append({
                "evidence_idx": candidate_idx,
                "evidence_text": evidence_text,
                "raw_similarity": round(raw_sim, 4),
                "boosted_similarity": round(sim, 4),
                "nli_score": round(nli_score, 4),
                "nli_label": nli_label,
                "evidence_types": evidence_types,
                "rank": rank,
                "is_local": abs(candidate_idx - claim_idx) <= self.window_size,
            })
            
            if len(all_evidence) >= self.top_k_evidence:
                break
        
        # Build result
        if not all_evidence:
            return EnhancedClaimEvidenceLink(
                claim_id=f"{claim_row['bank']}_{claim_row['year']}_{claim_idx}",
                claim_text=claim_text,
                actionability=claim_row.get("actionability", "Unknown"),
                evidence_found=False,
                best_evidence=None,
                best_evidence_idx=None,
                similarity_score=float(max(raw_similarities)) if len(raw_similarities) > 0 else 0.0,
                search_method="document" if self.document_level else "window",
            )
        
        best = all_evidence[0]
        
        # Determine search method
        has_global = any(not e["is_local"] for e in all_evidence)
        search_method = "hybrid" if has_global else "window"
        
        # Collect all evidence types
        all_ev_types = set()
        for e in all_evidence:
            all_ev_types.update(e["evidence_types"])
        
        return EnhancedClaimEvidenceLink(
            claim_id=f"{claim_row['bank']}_{claim_row['year']}_{claim_idx}",
            claim_text=claim_text,
            actionability=claim_row.get("actionability", "Unknown"),
            evidence_found=True,
            best_evidence=best["evidence_text"],
            best_evidence_idx=best["evidence_idx"],
            similarity_score=best["raw_similarity"],
            all_evidence=all_evidence,
            num_evidence=len(all_evidence),
            avg_similarity=round(np.mean([e["raw_similarity"] for e in all_evidence]), 4),
            nli_entailment_score=best["nli_score"],
            nli_label=best["nli_label"],
            evidence_types=list(all_ev_types),
            search_method=search_method,
        )
    
    def compute_enhanced_evidence_strength(self, link: EnhancedClaimEvidenceLink) -> float:
        """
        Compute enhanced evidence strength (ES_v2) with multi-source scoring.
        
        ES_v2 = w_sim × max_sim + w_R × R(e) + w_nli × NLI(claim, evidence)
        """
        w_sim = self.weights["w_sim"]
        w_R = self.weights["w_R"]
        w_nli = self.weights["w_nli"]
        
        if not link.evidence_found:
            return 0.0
        
        # Component 1: Semantic similarity (best match)
        sim_component = w_sim * min(max(link.similarity_score, 0.0), 1.0)
        
        # Component 2: Rule-based evidence (count of evidence types)
        valid_types = ["KPI", "Standard", "Time_bound", "Third_party"]
        type_count = sum(1 for t in link.evidence_types if t in valid_types)
        rule_component = w_R * min(type_count / 4.0, 1.0)
        
        # Component 3: NLI entailment
        if self.use_nli:
            nli_component = w_nli * min(max(link.nli_entailment_score, 0.0), 1.0)
        else:
            # Redistribute NLI weight
            sim_component += w_nli * 0.5 * min(max(link.similarity_score, 0.0), 1.0)
            rule_component += w_nli * 0.5 * min(type_count / 4.0, 1.0)
            nli_component = 0.0
        
        # Multi-evidence bonus: more evidence = more confidence
        multi_bonus = 0.0
        if link.num_evidence > 1:
            multi_bonus = min(0.05 * (link.num_evidence - 1), 0.1)
        
        strength = sim_component + rule_component + nli_component + multi_bonus
        return min(strength, 1.0)
    
    def link_corpus(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        save_embeddings: bool = True,
    ) -> pd.DataFrame:
        """
        Link all ESG claims in corpus to their evidence.
        
        Returns:
            DataFrame with enhanced linking results.
        """
        print(f"[EnhancedLinker] Processing {len(df)} sentences...")
        
        # Embed all sentences
        texts = df[text_column].tolist()
        embeddings = self.embed_sentences(texts)
        
        if save_embeddings:
            self._embeddings_cache = embeddings
        
        # Link each claim
        results = []
        for idx in range(len(df)):
            if idx % 500 == 0:
                print(f"  Linking {idx}/{len(df)}...")
            
            link = self.link_claim_to_evidence(idx, df, embeddings, text_column)
            es_v2 = self.compute_enhanced_evidence_strength(link)
            
            results.append({
                "claim_id": link.claim_id,
                "claim_idx": idx,
                "claim_text": link.claim_text,
                "actionability": link.actionability,
                "evidence_found": link.evidence_found,
                "best_evidence": link.best_evidence,
                "best_evidence_idx": link.best_evidence_idx,
                "similarity_score": link.similarity_score,
                "num_evidence": link.num_evidence,
                "avg_similarity": link.avg_similarity,
                "nli_entailment_score": link.nli_entailment_score,
                "nli_label": link.nli_label,
                "evidence_types": link.evidence_types,
                "search_method": link.search_method,
                "evidence_strength_v2": es_v2,
            })
        
        result_df = pd.DataFrame(results)
        
        # Summary
        found = result_df["evidence_found"].sum()
        print(f"\n[EnhancedLinker] Evidence found: {found}/{len(result_df)} ({100*found/len(result_df):.1f}%)")
        print(f"[EnhancedLinker] Avg similarity: {result_df['similarity_score'].mean():.3f}")
        print(f"[EnhancedLinker] Avg ES_v2: {result_df['evidence_strength_v2'].mean():.3f}")
        
        if self.use_nli:
            entails = (result_df["nli_label"] == "entailment").sum()
            print(f"[EnhancedLinker] NLI entailment: {entails}/{found} ({100*entails/max(found,1):.1f}%)")
        
        # Search method breakdown
        if self.document_level:
            methods = result_df["search_method"].value_counts()
            print(f"[EnhancedLinker] Search methods: {dict(methods)}")
        
        return result_df


def compare_v1_v2(df: pd.DataFrame, text_column: str = "text") -> Dict:
    """
    Run both V1 (window-only) and V2 (enhanced) linkers and compare.
    Useful for ablation study in the thesis.
    """
    from evidence.claim_evidence_linker import ClaimEvidenceLinker, analyze_linking_quality
    
    print("=" * 60)
    print("COMPARISON: V1 (Window) vs V2 (Enhanced)")
    print("=" * 60)
    
    # V1: Original (window only)
    print("\n--- Running V1 (Window-only, ±5) ---")
    v1_linker = ClaimEvidenceLinker(window_size=5, similarity_threshold=0.5)
    v1_results = v1_linker.link_corpus(df, text_column=text_column)
    v1_stats = analyze_linking_quality(v1_results)
    
    # V2: Enhanced (document-level + NLI)
    print("\n--- Running V2 (Document-level + NLI) ---")
    v2_linker = EnhancedClaimEvidenceLinker(
        window_size=5,
        document_level=True,
        tfidf_top_k=20,
        top_k_evidence=3,
        use_nli=True,
        nli_mode="rule",
    )
    v2_results = v2_linker.link_corpus(df, text_column=text_column)
    
    # Compare
    v1_rate = v1_stats["evidence_rate"]
    v2_rate = v2_results["evidence_found"].mean()
    v1_sim = v1_stats["avg_similarity"]
    v2_sim = v2_results["similarity_score"].mean()
    
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"{'Metric':<30} {'V1 (Window)':>12} {'V2 (Enhanced)':>14}")
    print("-" * 60)
    print(f"{'Evidence Rate':<30} {v1_rate:>11.3f} {v2_rate:>13.3f}")
    print(f"{'Avg Similarity':<30} {v1_sim:>11.3f} {v2_sim:>13.3f}")
    print(f"{'Avg Evidence Strength':<30} {'N/A':>12} {v2_results['evidence_strength_v2'].mean():>13.3f}")
    
    if "nli_entailment_score" in v2_results.columns:
        avg_nli = v2_results[v2_results["evidence_found"]]["nli_entailment_score"].mean()
        print(f"{'Avg NLI Entailment':<30} {'N/A':>12} {avg_nli:>13.3f}")
    
    return {
        "v1_stats": v1_stats,
        "v2_rate": v2_rate,
        "v2_avg_sim": v2_sim,
        "v2_avg_strength": v2_results["evidence_strength_v2"].mean(),
        "improvement_rate": v2_rate - v1_rate,
    }


if __name__ == "__main__":
    print("Enhanced Claim-Evidence Linker (v2)")
    print("=" * 50)
    print("Usage:")
    print("  from evidence.enhanced_linker import EnhancedClaimEvidenceLinker")
    print("  linker = EnhancedClaimEvidenceLinker()")
    print("  results = linker.link_corpus(df)")
