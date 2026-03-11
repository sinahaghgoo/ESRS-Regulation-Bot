"""
Retriever Module
================
Implements three retrieval strategies:
  1. Dense retrieval (vector similarity via VectorIndex)
  2. Sparse retrieval (BM25 keyword matching)
  3. Hybrid retrieval (dense + sparse fused via Reciprocal Rank Fusion)

Optional Cohere reranking can be applied as a post-retrieval stage.
All retrievers return results with full chunk metadata including page references.
"""

import json
import math
import re
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from collections import Counter
import numpy as np
import requests

from rag.config import (
    OPENAI_API_KEY, COHERE_API_KEY, ARTIFACTS_DIR,
    EMBEDDING_MODEL, EMBEDDING_DIMENSIONS,
)
from rag.embedder import embed_query
from rag.indexer import VectorIndex
from rag.chunker import Chunk


# ---------------------------------------------------------------------------
# BM25 Implementation (lightweight, no external dependency)
# ---------------------------------------------------------------------------

class BM25Index:
    """
    BM25 sparse retrieval index.
    Implements the Okapi BM25 scoring algorithm with parameters k1=1.5, b=0.75.
    Supports serialization to disk for persistence.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs: Dict[str, int] = {}       # term -> number of docs containing it
        self.doc_lens: List[int] = []               # length of each document
        self.avg_doc_len: float = 0.0
        self.n_docs: int = 0
        self.term_freqs: List[Dict[str, int]] = []  # per-doc term frequencies
        self.chunk_metadata: List[Dict] = []         # metadata for each indexed doc
        self._built = False
    
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace + punctuation tokenizer with lowercasing."""
        # Remove HTML tags for cleaner tokenization
        text = re.sub(r'<[^>]+>', ' ', text)
        # Lowercase and split on non-alphanumeric
        tokens = re.findall(r'[a-z0-9]+', text.lower())
        return tokens
    
    def build(self, chunks: List[Chunk]):
        """
        Build the BM25 index from a list of chunks.
        
        Args:
            chunks: List of Chunk objects to index
        """
        self.n_docs = len(chunks)
        self.doc_freqs = {}
        self.doc_lens = []
        self.term_freqs = []
        self.chunk_metadata = []
        
        for chunk in chunks:
            tokens = self._tokenize(chunk.text)
            self.doc_lens.append(len(tokens))
            
            # Count term frequencies for this document
            tf = Counter(tokens)
            self.term_freqs.append(dict(tf))
            
            # Update document frequencies
            for term in set(tokens):
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1
            
            # Store metadata
            self.chunk_metadata.append(chunk.to_dict())
        
        self.avg_doc_len = sum(self.doc_lens) / max(self.n_docs, 1)
        self._built = True
    
    def search(
        self,
        query: str,
        top_k: int = 15,
        company_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search the BM25 index for relevant chunks.
        
        Args:
            query: Natural language query
            top_k: Number of results to return
            company_filter: Optional company name filter
        
        Returns:
            List of result dicts with score and chunk metadata
        """
        if not self._built:
            return []
        
        query_tokens = self._tokenize(query)
        scores = np.zeros(self.n_docs)
        
        for token in query_tokens:
            if token not in self.doc_freqs:
                continue
            
            df = self.doc_freqs[token]
            # IDF component
            idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)
            
            for i in range(self.n_docs):
                tf = self.term_freqs[i].get(token, 0)
                if tf == 0:
                    continue
                
                # BM25 score component
                dl = self.doc_lens[i]
                tf_norm = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / self.avg_doc_len)
                )
                scores[i] += idf * tf_norm
        
        # Apply company filter
        if company_filter:
            for i in range(self.n_docs):
                if self.chunk_metadata[i].get("company", "") != company_filter:
                    scores[i] = -1e10
        
        # Get top-k
        k = min(top_k, self.n_docs)
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            item = dict(self.chunk_metadata[idx])
            item["score"] = float(scores[idx])
            results.append(item)
        
        return results
    
    def save(self, path: Path):
        """Persist BM25 index to disk."""
        data = {
            "k1": self.k1,
            "b": self.b,
            "doc_freqs": self.doc_freqs,
            "doc_lens": self.doc_lens,
            "avg_doc_len": self.avg_doc_len,
            "n_docs": self.n_docs,
            "term_freqs": self.term_freqs,
            "chunk_metadata": self.chunk_metadata,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)
    
    def load(self, path: Path) -> bool:
        """Load BM25 index from disk. Returns True if successful."""
        if not path.exists():
            return False
        with open(path, "r") as f:
            data = json.load(f)
        self.k1 = data["k1"]
        self.b = data["b"]
        self.doc_freqs = data["doc_freqs"]
        self.doc_lens = data["doc_lens"]
        self.avg_doc_len = data["avg_doc_len"]
        self.n_docs = data["n_docs"]
        self.term_freqs = data["term_freqs"]
        self.chunk_metadata = data["chunk_metadata"]
        self._built = True
        return True


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """
    Merge multiple ranked result lists using Reciprocal Rank Fusion.
    
    RRF score for document d = sum over all lists L of: 1 / (k + rank_L(d))
    where k is a constant (default 60) and rank is 1-based.
    
    Args:
        ranked_lists: List of ranked result lists (each sorted by score desc)
        k: RRF constant (default 60 as per README specification)
    
    Returns:
        Merged and re-ranked list of results with RRF scores
    """
    # Map chunk_id -> (rrf_score, best_metadata)
    rrf_scores: Dict[str, float] = {}
    best_meta: Dict[str, Dict] = {}
    
    for ranked_list in ranked_lists:
        for rank_0based, item in enumerate(ranked_list):
            cid = item.get("chunk_id", "")
            if not cid:
                # Generate a fallback ID from text hash
                cid = hashlib.md5(item.get("text", "")[:200].encode()).hexdigest()
            
            rrf_score = 1.0 / (k + rank_0based + 1)  # 1-based rank
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + rrf_score
            
            # Keep the metadata from the highest-scoring appearance
            if cid not in best_meta or item.get("score", 0) > best_meta[cid].get("score", 0):
                best_meta[cid] = item
    
    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    
    results = []
    for cid in sorted_ids:
        item = dict(best_meta[cid])
        item["rrf_score"] = rrf_scores[cid]
        item["score"] = rrf_scores[cid]  # override score with RRF score
        results.append(item)
    
    return results


# ---------------------------------------------------------------------------
# Cohere Reranker (optional, with caching)
# ---------------------------------------------------------------------------

RERANK_CACHE_DIR = ARTIFACTS_DIR / "rerank_cache"
RERANK_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def cohere_rerank(
    query: str,
    documents: List[Dict[str, Any]],
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """
    Rerank documents using the Cohere reranking API.
    Results are cached to avoid redundant API calls.
    
    Args:
        query: The search query
        documents: List of result dicts (must have 'text' key)
        top_n: Number of top results to return after reranking
    
    Returns:
        Reranked list of result dicts
    """
    if not COHERE_API_KEY:
        print("    Warning: No Cohere API key, skipping reranking")
        return documents[:top_n]
    
    # Build cache key from query + document texts
    doc_texts = [d.get("text", "")[:500] for d in documents]
    cache_raw = f"{query}::{'||'.join(doc_texts)}"
    cache_key = hashlib.sha256(cache_raw.encode()).hexdigest()
    cache_file = RERANK_CACHE_DIR / f"{cache_key}.json"
    
    # Check cache
    if cache_file.exists():
        with open(cache_file, "r") as f:
            cached_indices = json.load(f)
        reranked = []
        for idx, score in cached_indices:
            if idx < len(documents):
                item = dict(documents[idx])
                item["rerank_score"] = score
                item["score"] = score
                reranked.append(item)
        return reranked[:top_n]
    
    # Call Cohere API
    try:
        resp = requests.post(
            "https://api.cohere.ai/v1/rerank",
            headers={
                "Authorization": f"Bearer {COHERE_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "rerank-english-v3.0",
                "query": query,
                "documents": doc_texts[:25],  # Keep candidate set tight
                "top_n": min(top_n, len(doc_texts)),
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        
        # Parse results
        reranked = []
        cache_data = []
        for r in data.get("results", []):
            idx = r["index"]
            score = r["relevance_score"]
            cache_data.append((idx, score))
            if idx < len(documents):
                item = dict(documents[idx])
                item["rerank_score"] = score
                item["score"] = score
                reranked.append(item)
        
        # Save to cache
        with open(cache_file, "w") as f:
            json.dump(cache_data, f)
        
        return reranked[:top_n]
        
    except Exception as e:
        print(f"    Cohere rerank error: {e}, returning original order")
        return documents[:top_n]


# ---------------------------------------------------------------------------
# Unified Retriever
# ---------------------------------------------------------------------------

class Retriever:
    """
    Unified retriever supporting dense, sparse, and hybrid search.
    
    Manages both the dense (vector) index and the sparse (BM25) index,
    and applies optional Cohere reranking.
    """
    
    def __init__(
        self,
        vector_index: VectorIndex,
        bm25_index: BM25Index,
        config: Dict[str, Any],
    ):
        """
        Args:
            vector_index: VectorIndex instance for dense retrieval
            bm25_index: BM25Index instance for sparse retrieval
            config: Pipeline configuration dict
        """
        self.vector_index = vector_index
        self.bm25_index = bm25_index
        self.config = config
    
    def retrieve(
        self,
        query: str,
        company: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query using the configured strategy.
        
        Args:
            query: Natural language query
            company: Optional company name filter
            top_k: Override for number of results
        
        Returns:
            List of result dicts with text, metadata, scores, and page refs
        """
        method = self.config.get("retrieval_method", "hybrid")
        k = top_k or self.config.get("top_k", 15)
        rrf_k = self.config.get("rrf_k", 60)
        use_reranker = self.config.get("use_reranker", False)
        rerank_top_n = self.config.get("rerank_top_n", 10)
        
        # --- Dense retrieval ---
        if method == "dense":
            query_vec = embed_query(query)
            results = self.vector_index.search(query_vec, top_k=k, company_filter=company)
        
        # --- Sparse retrieval (BM25) ---
        elif method == "sparse":
            results = self.bm25_index.search(query, top_k=k, company_filter=company)
        
        # --- Hybrid retrieval (dense + sparse + RRF) ---
        elif method == "hybrid":
            query_vec = embed_query(query)
            dense_results = self.vector_index.search(
                query_vec, top_k=k * 2, company_filter=company
            )
            sparse_results = self.bm25_index.search(
                query, top_k=k * 2, company_filter=company
            )
            results = reciprocal_rank_fusion(
                [dense_results, sparse_results], k=rrf_k
            )
            results = results[:k]
        
        else:
            raise ValueError(f"Unknown retrieval method: {method}")
        
        # --- Optional reranking ---
        if use_reranker and results:
            results = cohere_rerank(query, results, top_n=rerank_top_n)
        
        return results
