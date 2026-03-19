"""
Claim-Evidence Linking Module

This module implements semantic similarity-based linking between ESG claims
and supporting evidence sentences using sentence embeddings.

Key improvement over regex-only approach:
- Uses NLP (embeddings) to find semantically related evidence
- Computes continuous similarity scores instead of binary matching
- Enables explicit claim→evidence mapping for academic contribution
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class ClaimEvidenceLink:
    """Represents a link between a claim and its supporting evidence."""
    claim_id: str
    claim_text: str
    actionability: str
    evidence_found: bool
    best_evidence: Optional[str]
    best_evidence_idx: Optional[int]
    similarity_score: float
    evidence_types: List[str]
    

class ClaimEvidenceLinker:
    """
    Links ESG claims to supporting evidence using semantic similarity.
    
    Approach:
    1. Embed all sentences using multilingual sentence transformer
    2. For each claim, find candidate evidence in ±N window
    3. Compute cosine similarity between claim and candidates
    4. Select best match if similarity > threshold
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        window_size: int = 5,
        similarity_threshold: float = 0.5,
        device: str = None
    ):
        """
        Initialize the linker.
        
        Args:
            model_name: Sentence transformer model (multilingual for Vietnamese)
            window_size: Number of sentences before/after to search for evidence
            similarity_threshold: Minimum similarity to consider as valid link
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        print(f"Loading sentence transformer: {model_name}")
        print(f"Device: {device}")
        
        self.model = SentenceTransformer(model_name, device=device)
        self.window_size = window_size
        self.similarity_threshold = similarity_threshold
        self.device = device
        
        # Cache for embeddings
        self._embeddings_cache = None
        
    def embed_sentences(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed a list of sentences.
        
        Args:
            texts: List of sentences to embed
            batch_size: Batch size for encoding
            
        Returns:
            Embeddings as numpy array (n_sentences, embedding_dim)
        """
        print(f"Embedding {len(texts)} sentences...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings
    
    def find_evidence_candidates(
        self,
        claim_idx: int,
        df: pd.DataFrame
    ) -> List[int]:
        """
        Find candidate evidence sentences in a window around the claim.
        
        Args:
            claim_idx: Index of the claim sentence
            df: DataFrame with all sentences
            
        Returns:
            List of candidate indices
        """
        # Get sentences in ±window from same document/bank/year
        claim_row = df.iloc[claim_idx]
        
        # Filter to same context (bank, year, document)
        same_context = df[
            (df['bank'] == claim_row['bank']) &
            (df['year'] == claim_row['year'])
        ]
        
        # Find indices within window
        candidates = []
        for idx in same_context.index:
            distance = abs(idx - claim_idx)
            if 0 < distance <= self.window_size:  # Exclude self
                candidates.append(idx)
                
        return candidates
    
    def compute_similarity(
        self,
        claim_emb: np.ndarray,
        evidence_embs: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between claim and evidence embeddings.
        
        Args:
            claim_emb: Claim embedding (1, embedding_dim)
            evidence_embs: Evidence embeddings (n_candidates, embedding_dim)
            
        Returns:
            Similarity scores (n_candidates,)
        """
        if len(evidence_embs) == 0:
            return np.array([])
            
        similarities = cosine_similarity(
            claim_emb.reshape(1, -1),
            evidence_embs
        )[0]
        
        return similarities
    
    def link_claim_to_evidence(
        self,
        claim_idx: int,
        df: pd.DataFrame,
        embeddings: np.ndarray,
        text_column: str = 'text'
    ) -> ClaimEvidenceLink:
        """
        Link a single claim to its best supporting evidence.
        
        Args:
            claim_idx: Index of the claim
            df: DataFrame with all sentences
            embeddings: Pre-computed embeddings for all sentences
            text_column: Column name for text
            
        Returns:
            ClaimEvidenceLink object
        """
        claim_row = df.iloc[claim_idx]
        claim_emb = embeddings[claim_idx]
        
        # Find candidates
        candidate_idxs = self.find_evidence_candidates(claim_idx, df)
        
        if not candidate_idxs:
            # No candidates found
            return ClaimEvidenceLink(
                claim_id=f"{claim_row['bank']}_{claim_row['year']}_{claim_idx}",
                claim_text=claim_row[text_column],
                actionability=claim_row.get('actionability', 'Unknown'),
                evidence_found=False,
                best_evidence=None,
                best_evidence_idx=None,
                similarity_score=0.0,
                evidence_types=[]
            )
        
        # Compute similarities
        candidate_embs = embeddings[candidate_idxs]
        similarities = self.compute_similarity(claim_emb, candidate_embs)
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        best_candidate_idx = candidate_idxs[best_idx]
        
        # Check threshold
        if best_similarity >= self.similarity_threshold:
            best_evidence_row = df.iloc[best_candidate_idx]
            evidence_types = best_evidence_row.get('evidence_types', [])
            
            return ClaimEvidenceLink(
                claim_id=f"{claim_row['bank']}_{claim_row['year']}_{claim_idx}",
                claim_text=claim_row[text_column],
                actionability=claim_row.get('actionability', 'Unknown'),
                evidence_found=True,
                best_evidence=best_evidence_row[text_column],
                best_evidence_idx=best_candidate_idx,
                similarity_score=float(best_similarity),
                evidence_types=evidence_types if isinstance(evidence_types, list) else []
            )
        else:
            # Below threshold
            return ClaimEvidenceLink(
                claim_id=f"{claim_row['bank']}_{claim_row['year']}_{claim_idx}",
                claim_text=claim_row[text_column],
                actionability=claim_row.get('actionability', 'Unknown'),
                evidence_found=False,
                best_evidence=None,
                best_evidence_idx=None,
                similarity_score=float(best_similarity),
                evidence_types=[]
            )
    
    def link_corpus(
        self,
        df: pd.DataFrame,
        text_column: str = 'text',
        save_embeddings: bool = True
    ) -> pd.DataFrame:
        """
        Link all ESG claims in the corpus to their best evidence.
        
        Args:
            df: DataFrame with ESG sentences
            text_column: Column name containing text
            save_embeddings: Whether to cache embeddings
            
        Returns:
            DataFrame with linking results
        """
        print(f"Processing {len(df)} sentences...")
        
        # Embed all sentences
        texts = df[text_column].tolist()
        embeddings = self.embed_sentences(texts)
        
        if save_embeddings:
            self._embeddings_cache = embeddings
        
        # Link each claim
        results = []
        for idx in range(len(df)):
            if idx % 1000 == 0:
                print(f"Linking claim {idx}/{len(df)}...")
                
            link = self.link_claim_to_evidence(idx, df, embeddings, text_column=text_column)
            results.append({
                'claim_id': link.claim_id,
                'claim_idx': idx,
                'claim_text': link.claim_text,
                'actionability': link.actionability,
                'evidence_found': link.evidence_found,
                'best_evidence': link.best_evidence,
                'best_evidence_idx': link.best_evidence_idx,
                'similarity_score': link.similarity_score,
                'evidence_types': link.evidence_types
            })
        
        return pd.DataFrame(results)


def analyze_linking_quality(links_df: pd.DataFrame) -> Dict:
    """
    Analyze the quality of claim-evidence linking.
    
    Args:
        links_df: DataFrame with linking results
        
    Returns:
        Dictionary with quality metrics
    """
    total = len(links_df)
    found = links_df['evidence_found'].sum()
    
    # Similarity distribution
    avg_similarity = links_df['similarity_score'].mean()
    high_sim = (links_df['similarity_score'] >= 0.7).sum()
    medium_sim = ((links_df['similarity_score'] >= 0.5) & (links_df['similarity_score'] < 0.7)).sum()
    low_sim = (links_df['similarity_score'] < 0.5).sum()
    
    # By actionability
    by_action = links_df.groupby('actionability').agg({
        'evidence_found': 'mean',
        'similarity_score': 'mean'
    }).round(3)
    
    return {
        'total_claims': total,
        'evidence_found': found,
        'evidence_rate': round(found / total, 3),
        'avg_similarity': round(avg_similarity, 3),
        'high_similarity_count': high_sim,
        'medium_similarity_count': medium_sim,
        'low_similarity_count': low_sim,
        'by_actionability': by_action.to_dict()
    }


if __name__ == "__main__":
    # Example usage
    print("Claim-Evidence Linker Module")
    print("=" * 50)
    print("This module links ESG claims to supporting evidence")
    print("using semantic similarity (sentence embeddings).")
    print()
    print("Usage:")
    print("  from evidence.claim_evidence_linker import ClaimEvidenceLinker")
    print("  linker = ClaimEvidenceLinker()")
    print("  links_df = linker.link_corpus(df)")
