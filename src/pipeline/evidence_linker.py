import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import warnings
from tqdm.auto import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine


EVIDENCE_VARIANTS = {
    # Document + window retrieval + model-based NLI verification
    "nli": {
        "window_size": 5,
        "document_level": True,
        "tfidf_top_k": 20,
        "similarity_threshold": 0.5,
        "top_k_evidence": 3,
        "use_nli": True,
    },
    # Local window retrieval only (no doc-level, no NLI)
    "window": {
        "window_size": 5,
        "document_level": False,
        "tfidf_top_k": 0,
        "similarity_threshold": 0.5,
        "top_k_evidence": 1,
        "use_nli": False,
    },
    # Document + window retrieval without NLI
    "no_nli": {
        "window_size": 5,
        "document_level": True,
        "tfidf_top_k": 20,
        "similarity_threshold": 0.5,
        "top_k_evidence": 3,
        "use_nli": False,
    },
}


@dataclass
class ClaimEvidenceLink:
    claim_id: str
    claim_text: str
    action_label: str
    
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


class ClaimEvidenceLinker:
    """
    Link ESG claims to supporting evidence sentences using:
    1. Window-based local retrieval (proximity context)
    2. Document-level TF-IDF pre-filtering (global context)
    3. Semantic similarity ranking (sentence-transformers)
    4. Optional NLI verification (mDeBERTa-xnli)

    Outputs raw linking signals (similarity, NLI, evidence types).
    The unified Evidence Strength (ES) is computed downstream by the
    EWRI module, following GRI (2021) and Cho et al. (2012).
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        window_size: int = 5,
        document_level: bool = True,
        tfidf_top_k: int = 20,
        similarity_threshold: float = 0.5,
        top_k_evidence: int = 3,
        use_nli: bool = True,
        device: str = None,
    ):
        import torch
        from sentence_transformers import SentenceTransformer
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"[EvidenceLinker] Loading: {model_name}")
        print(f"[EvidenceLinker] Device: {device}")
        print(f"[EvidenceLinker] Document-level: {document_level}, TF-IDF top-K: {tfidf_top_k}")
        print(f"[EvidenceLinker] NLI: {use_nli}")
        print(f"[EvidenceLinker] Top-K evidence: {top_k_evidence}")
        
        self.model = SentenceTransformer(model_name, device=device)
        self.window_size = window_size
        self.document_level = document_level
        self.tfidf_top_k = tfidf_top_k
        self.similarity_threshold = similarity_threshold
        self.top_k_evidence = top_k_evidence
        self.use_nli = use_nli
        self.device = device
        
        # NLI verifier
        self._nli_verifier = None
        if use_nli:
            self._init_nli()
        
        # Caches
        self._embeddings_cache = None
        self._tfidf_cache = {}
    
    def _init_nli(self):
        """Initialize model-based NLI verifier (mDeBERTa-xnli)."""
        try:
            from src.pipeline.nli_verifier import NLIVerifier
        except Exception:
            from pipeline.nli_verifier import NLIVerifier

        self._nli_verifier = NLIVerifier(entailment_threshold=0.5, device=self.device)
        print("[EvidenceLinker] NLI verifier initialized")
    
    def embed_sentences(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Embed sentences using sentence transformer."""
        print(f"[EvidenceLinker] Embedding {len(texts)} sentences...")
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
        Find candidate evidence sentences using combined approach:
        1. Window-based (+-N) for local context
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
    ) -> ClaimEvidenceLink:
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
            return ClaimEvidenceLink(
                claim_id=f"{claim_row['bank']}_{claim_row['year']}_{claim_idx}",
                claim_text=claim_text,
                action_label=claim_row.get("action_label", "Unknown"),
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
        pre_candidates = []
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

            pre_candidates.append({
                "rank": rank,
                "evidence_idx": candidate_idx,
                "evidence_text": evidence_text,
                "raw_similarity": round(raw_sim, 4),
                "boosted_similarity": round(sim, 4),
                "evidence_types": evidence_types,
                "is_local": abs(candidate_idx - claim_idx) <= self.window_size,
            })

        nli_by_rank = {}
        if self.use_nli and self._nli_verifier is not None and pre_candidates:
            claims_batch = [claim_text] * len(pre_candidates)
            evidences_batch = [item["evidence_text"] for item in pre_candidates]
            nli_results = self._nli_verifier.verify_batch(
                claims_batch,
                evidences_batch,
                batch_size=16,
                show_progress=False,
            )
            for item, nli_result in zip(pre_candidates, nli_results):
                nli_by_rank[item["rank"]] = (
                    float(nli_result.entailment_score),
                    str(nli_result.label),
                )

        for item in pre_candidates:
            nli_score, nli_label = nli_by_rank.get(item["rank"], (0.5, "neutral"))

            if nli_label == "contradiction" and nli_score < 0.2:
                continue

            all_evidence.append({
                "evidence_idx": item["evidence_idx"],
                "evidence_text": item["evidence_text"],
                "raw_similarity": item["raw_similarity"],
                "boosted_similarity": item["boosted_similarity"],
                "nli_score": round(nli_score, 4),
                "nli_label": nli_label,
                "evidence_types": item["evidence_types"],
                "rank": item["rank"],
                "is_local": item["is_local"],
            })
            
            if len(all_evidence) >= self.top_k_evidence:
                break
        
        # Build result
        if not all_evidence:
            return ClaimEvidenceLink(
                claim_id=f"{claim_row['bank']}_{claim_row['year']}_{claim_idx}",
                claim_text=claim_text,
                action_label=claim_row.get("action_label", "Unknown"),
                evidence_found=False,
                best_evidence=None,
                best_evidence_idx=None,
                similarity_score=float(max(raw_similarities)) if len(raw_similarities) > 0 else 0.0,
                search_method="document" if self.document_level else "window",
            )
        
        best = all_evidence[0]
        
        # Determine search method
        has_global = any(not e["is_local"] for e in all_evidence)
        search_method = "document_window" if has_global else "window"
        
        # Collect all evidence types
        all_ev_types = set()
        for e in all_evidence:
            all_ev_types.update(e["evidence_types"])
        
        return ClaimEvidenceLink(
            claim_id=f"{claim_row['bank']}_{claim_row['year']}_{claim_idx}",
            claim_text=claim_text,
            action_label=claim_row.get("action_label", "Unknown"),
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
    
    def link_corpus(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        save_embeddings: bool = True,
    ) -> pd.DataFrame:
        """
        Link all ESG claims in corpus to their evidence.
        
        Returns:
            DataFrame with linking results.
        """
        print(f"[EvidenceLinker] Processing {len(df)} sentences...")
        
        # Embed all sentences
        texts = df[text_column].tolist()
        embeddings = self.embed_sentences(texts)
        
        if save_embeddings:
            self._embeddings_cache = embeddings
        
        # Link each claim
        results = []
        for idx in tqdm(range(len(df)), desc="Linking claims"):
            link = self.link_claim_to_evidence(idx, df, embeddings, text_column)

            results.append({
                "claim_id": link.claim_id,
                "claim_idx": idx,
                "claim_text": link.claim_text,
                "action_label": link.action_label,
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
            })

        result_df = pd.DataFrame(results)

        # Summary
        found = result_df["evidence_found"].sum()
        print(f"\n[EvidenceLinker] Evidence found: {found}/{len(result_df)} ({100*found/len(result_df):.1f}%)")
        print(f"[EvidenceLinker] Avg similarity: {result_df['similarity_score'].mean():.3f}")
        
        if self.use_nli:
            entails = (result_df["nli_label"] == "entailment").sum()
            print(f"[EvidenceLinker] NLI entailment: {entails}/{found} ({100*entails/max(found,1):.1f}%)")
        
        # Search method breakdown
        if self.document_level:
            methods = result_df["search_method"].value_counts()
            print(f"[EvidenceLinker] Search methods: {dict(methods)}")
        
        return result_df


def run_linking_variant(
    df: pd.DataFrame,
    variant: str = "nli",
    text_column: str = "sentence",
) -> pd.DataFrame:
    if variant not in EVIDENCE_VARIANTS:
        available = ", ".join(sorted(EVIDENCE_VARIANTS.keys()))
        raise ValueError(f"Unknown evidence variant '{variant}'. Available: {available}")

    cfg = EVIDENCE_VARIANTS[variant]
    linker = ClaimEvidenceLinker(**cfg)

    if text_column not in df.columns:
        if "sentence" in df.columns:
            text_column = "sentence"
        elif "text" in df.columns:
            text_column = "text"
        else:
            raise ValueError("Input dataframe must contain either 'sentence' or 'text' column")

    return linker.link_corpus(df, text_column=text_column)
