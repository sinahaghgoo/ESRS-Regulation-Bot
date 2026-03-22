"""
Dense, sparse, hybrid-RRF, and reranked retrieval methods for RQ1.

Six retrieval methods are compared in the RQ1 factorial experiment:

1. **Dense** — Embedding-based vector search via Qdrant (cosine similarity).
   Uses OpenAI text-embedding-3-large (3072 dimensions).

2. **Dense + Rerank** — Dense retrieval followed by cross-encoder reranking.
   Retrieves 3x candidates and re-scores with a cross-encoder model.

3. **Sparse (BM25)** — Keyword matching with Okapi BM25 (k1=1.5, b=0.75).
   IDF statistics are computed globally across all companies to provide
   stable vocabulary weights for ESRS-specific terminology.

4. **Sparse + Rerank** — BM25 followed by cross-encoder reranking.

5. **Hybrid (RRF)** — Reciprocal Rank Fusion combining dense and sparse
   results with k=60 (standard RRF parameter from Cormack et al., 2009).

6. **Hybrid + Rerank** — RRF fusion followed by cross-encoder reranking.

The cross-encoder reranker uses ``cross-encoder/ms-marco-MiniLM-L-6-v2``
(22 M params, ~80 MB), the most widely cited lightweight reranker in IR
benchmarks. It runs ~30x faster on CPU than larger alternatives like
BAAI/bge-reranker-v2-m3 (568 M params).
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from rag.config.settings import (
    ARTIFACTS_DIR,
    PipelineConfig,
)
from rag.indexing.embedder import EmbeddingService
from rag.indexing.qdrant_index import QdrantChunkIndex
from rag.types import ChunkRecord, RetrievalHit

_TOKEN_RE = re.compile(r"\w+")
_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton cross-encoder shared across all Retriever instances
# ---------------------------------------------------------------------------
_SHARED_CROSS_ENCODER = None
_SHARED_CROSS_ENCODER_NAME: str | None = None


def _get_shared_cross_encoder(model_name: str):
    """Lazily load the cross-encoder once and reuse across retrievers.

    Using a module-level singleton avoids loading the ~80 MB model
    repeatedly when evaluating multiple retrieval methods in sequence.
    """
    global _SHARED_CROSS_ENCODER, _SHARED_CROSS_ENCODER_NAME
    if _SHARED_CROSS_ENCODER is not None and _SHARED_CROSS_ENCODER_NAME == model_name:
        return _SHARED_CROSS_ENCODER

    from sentence_transformers import CrossEncoder

    _logger.info("Loading shared cross-encoder model: %s", model_name)
    _SHARED_CROSS_ENCODER = CrossEncoder(model_name)
    _SHARED_CROSS_ENCODER_NAME = model_name
    _logger.info(
        "Cross-encoder loaded on device: %s",
        next(_SHARED_CROSS_ENCODER.model.parameters()).device,
    )
    gc.collect()
    return _SHARED_CROSS_ENCODER


# ---------------------------------------------------------------------------
# BM25 implementation
# ---------------------------------------------------------------------------

class _BM25:
    """Okapi BM25 scoring (k1=1.5, b=0.75).

    IDF statistics are computed over all chunks (all companies) to provide a
    stable vocabulary baseline. Company filtering is applied after scoring so
    that rare ESRS terms keep the same IDF weight regardless of which company
    is being queried. This design matches standard cross-collection BM25 and
    is documented as a controlled variable in the thesis.

    The BM25 formula follows Robertson et al. (1994):
        score(q, d) = sum_{t in q} IDF(t) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * |d| / avgdl))
    where IDF(t) = log(1 + (N - df + 0.5) / (df + 0.5)).
    """

    def __init__(self, chunks: list[ChunkRecord], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunks = chunks
        self.doc_tokens = [self._tok(c.text) for c in chunks]
        self.doc_lens = [len(t) for t in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / max(len(self.doc_lens), 1)
        self.tf = [Counter(tokens) for tokens in self.doc_tokens]

        # Document frequency: number of documents containing each term.
        self.df: Counter[str] = Counter()
        for tfs in self.tf:
            self.df.update(tfs.keys())
        self.n = len(chunks)

    @staticmethod
    def _tok(text: str) -> list[str]:
        """Simple word tokenization (alphanumeric sequences, lowercased)."""
        return _TOKEN_RE.findall(text.lower())

    def score(self, query: str) -> list[tuple[ChunkRecord, float]]:
        """Score all chunks against the query and return sorted by score descending."""
        q_terms = self._tok(query)
        out: list[tuple[ChunkRecord, float]] = []

        for idx, chunk in enumerate(self.chunks):
            score = 0.0
            tf_doc = self.tf[idx]
            dl = self.doc_lens[idx] or 1

            for term in q_terms:
                tf = tf_doc.get(term, 0)
                if tf <= 0:
                    continue
                df = self.df.get(term, 0)
                idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
                denom = tf + self.k1 * (1 - self.b + self.b * (dl / max(self.avgdl, 1e-6)))
                score += idf * ((tf * (self.k1 + 1)) / max(denom, 1e-6))

            out.append((chunk, score))

        out.sort(key=lambda x: x[1], reverse=True)
        return out


# ---------------------------------------------------------------------------
# Unified Retriever
# ---------------------------------------------------------------------------

class Retriever:
    """Unified retrieval interface for all six RQ1 methods.

    Supports: dense, dense_rerank, sparse, sparse_rerank, hybrid_rrf, hybrid_rerank.

    The retriever is initialised with a set of chunks and the corresponding
    Qdrant index. Dense retrieval uses the pre-built vector index; sparse
    retrieval uses an in-memory BM25 index built at initialisation time.

    Reranking uses a cross-encoder model specified in the pipeline config.
    Rerank scores are cached to disk to avoid redundant model inference
    across repeated evaluations.
    """

    def __init__(
        self,
        chunks: list[ChunkRecord],
        config: PipelineConfig,
        embedder: EmbeddingService,
        qdrant: QdrantChunkIndex,
    ) -> None:
        self.chunks = chunks
        self.config = config
        self.embedder = embedder
        self.qdrant = qdrant
        self.chunk_by_id = {c.chunk_id: c for c in chunks}
        self.collection_name = qdrant.collection_name(config.chunk_strategy, config.chunk_size)

        # Build BM25 index over all chunks (IDF computed globally).
        self.bm25 = _BM25(chunks)

        self._cross_encoder = None
        self._rerank_cache_path = ARTIFACTS_DIR / "rerank_cache.json"
        self._rerank_cache: dict[str, list[dict]] = {}
        self._load_rerank_cache()

    # ------------------------------------------------------------------
    # Rerank cache helpers
    # ------------------------------------------------------------------

    def _load_rerank_cache(self) -> None:
        """Load cached rerank scores from disk."""
        if self._rerank_cache_path.exists():
            try:
                self._rerank_cache = json.loads(
                    self._rerank_cache_path.read_text(encoding="utf-8")
                )
            except Exception:
                self._rerank_cache = {}

    def _persist_rerank_cache(self) -> None:
        """Write rerank cache to disk for reproducibility."""
        self._rerank_cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._rerank_cache_path.write_text(
            json.dumps(self._rerank_cache), encoding="utf-8"
        )

    @staticmethod
    def _rerank_cache_key(query: str, hits: list[RetrievalHit], top_k: int) -> str:
        """Deterministic cache key from query + candidate list."""
        signature = "||".join(
            f"{h.chunk.chunk_id}:{h.rank}:{h.score:.6f}" for h in hits
        )
        raw = f"{query}@@{top_k}@@{signature}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Cross-encoder reranking
    # ------------------------------------------------------------------

    def _ensure_cross_encoder(self) -> None:
        """Lazy-load the shared cross-encoder model on first use."""
        if self._cross_encoder is not None:
            return
        self._cross_encoder = _get_shared_cross_encoder(self.config.rerank_model)

    def _rerank(
        self,
        query: str,
        hits: list[RetrievalHit],
        top_k: int,
        method_label: str = "hybrid_rerank",
    ) -> list[RetrievalHit]:
        """Rerank candidates using the configured cross-encoder.

        Parameters
        ----------
        query : str
            The user query.
        hits : list[RetrievalHit]
            Candidate hits to rerank.
        top_k : int
            Number of results to return after reranking.
        method_label : str
            Retrieval method name for the returned RetrievalHit objects.
        """
        if not hits:
            return []

        # Check cache first — reject entries with NaN scores.
        cache_key = self._rerank_cache_key(query, hits, top_k)
        if cache_key in self._rerank_cache:
            cached = self._rerank_cache[cache_key]
            has_nan = any(
                math.isnan(item.get("score", float("nan"))) for item in cached
            )
            if has_nan:
                _logger.warning(
                    "Rerank cache entry has NaN scores — discarding and re-scoring."
                )
                del self._rerank_cache[cache_key]
            else:
                out: list[RetrievalHit] = []
                for rank, item in enumerate(cached, start=1):
                    base = self.chunk_by_id.get(item["chunk_id"])
                    if base is None:
                        continue
                    out.append(
                        RetrievalHit(
                            chunk=base,
                            score=float(item["score"]),
                            rank=rank,
                            method=method_label,
                            aux={
                                "pre_rerank_rank": int(item["pre_rank"]),
                                "pre_rerank_score": float(item["pre_score"]),
                                "cached": True,
                            },
                        )
                    )
                return out

        # Score all candidates with the cross-encoder.
        self._ensure_cross_encoder()
        pairs = [[query, h.chunk.text] for h in hits]
        try:
            scores = self._cross_encoder.predict(
                pairs, batch_size=16, show_progress_bar=False
            )
        except Exception as exc:
            _logger.warning(
                "Cross-encoder predict failed (%s), retrying with batch_size=4", exc
            )
            gc.collect()
            scores = self._cross_encoder.predict(
                pairs, batch_size=4, show_progress_bar=False
            )

        scores_array = np.asarray(scores, dtype=np.float64).ravel()
        if np.any(np.isnan(scores_array)):
            raise RuntimeError(
                f"Cross-encoder returned NaN scores for {len(pairs)} pairs. "
                f"Model: {self.config.rerank_model}. Check model compatibility."
            )

        # Sort candidates by cross-encoder score descending.
        scored = sorted(
            zip(hits, scores_array), key=lambda x: float(x[1]), reverse=True
        )

        out: list[RetrievalHit] = []
        cache_payload: list[dict] = []
        for rank, (base, score) in enumerate(scored[:top_k], start=1):
            cache_payload.append({
                "chunk_id": base.chunk.chunk_id,
                "score": float(score),
                "pre_rank": base.rank,
                "pre_score": base.score,
            })
            out.append(
                RetrievalHit(
                    chunk=base.chunk,
                    score=float(score),
                    rank=rank,
                    method=method_label,
                    aux={
                        "pre_rerank_rank": base.rank,
                        "pre_rerank_score": base.score,
                    },
                )
            )
        self._rerank_cache[cache_key] = cache_payload
        self._persist_rerank_cache()
        return out

    # ------------------------------------------------------------------
    # Core retrieval methods
    # ------------------------------------------------------------------

    def _dense(self, query: str, top_k: int, company: str | None) -> list[RetrievalHit]:
        """Dense retrieval via Qdrant vector search (cosine similarity)."""
        qvec = self.embedder.embed_text(query)
        rows = self.qdrant.search(self.collection_name, qvec, top_k=top_k, company=company)
        hits: list[RetrievalHit] = []
        for rank, (cid, score, payload) in enumerate(rows, start=1):
            chunk = self.chunk_by_id.get(cid)
            if chunk is None:
                continue
            hits.append(RetrievalHit(chunk=chunk, score=score, rank=rank, method="dense"))
        return hits

    def _sparse(self, query: str, top_k: int, company: str | None) -> list[RetrievalHit]:
        """Sparse retrieval via BM25 scoring with optional company filter."""
        ranked = self.bm25.score(query)
        hits: list[RetrievalHit] = []
        rank = 0
        for chunk, score in ranked:
            if company and chunk.company != company:
                continue
            rank += 1
            hits.append(RetrievalHit(chunk=chunk, score=score, rank=rank, method="sparse"))
            if rank >= top_k:
                break
        return hits

    def _rrf_fuse(
        self,
        dense_hits: list[RetrievalHit],
        sparse_hits: list[RetrievalHit],
        top_k: int,
    ) -> list[RetrievalHit]:
        """Fuse dense and sparse results via Reciprocal Rank Fusion.

        RRF score = sum(1 / (k + rank)) where k=60 (Cormack et al., 2009).
        This is a rank-level fusion method that does not require score
        normalisation between the two retrieval systems.
        """
        accum: dict[str, float] = defaultdict(float)
        provenance: dict[str, dict] = defaultdict(dict)

        for hits, tag in ((dense_hits, "dense"), (sparse_hits, "sparse")):
            for hit in hits:
                accum[hit.chunk.chunk_id] += 1.0 / (self.config.rrf_k + hit.rank)
                provenance[hit.chunk.chunk_id][tag] = {
                    "rank": hit.rank,
                    "score": hit.score,
                }

        ranked = sorted(accum.items(), key=lambda x: x[1], reverse=True)[:top_k]
        out: list[RetrievalHit] = []
        for rank, (cid, score) in enumerate(ranked, start=1):
            chunk = self.chunk_by_id[cid]
            out.append(
                RetrievalHit(
                    chunk=chunk,
                    score=score,
                    rank=rank,
                    method="hybrid_rrf",
                    aux=provenance[cid],
                )
            )
        return out

    # ------------------------------------------------------------------
    # Unified retrieval entry point
    # ------------------------------------------------------------------

    def retrieve(
        self, query: str, company: str | None = None, top_k: int | None = None
    ) -> list[RetrievalHit]:
        """Run the retrieval method specified in the pipeline configuration.

        Parameters
        ----------
        query : str
            The search query.
        company : str | None
            If given, restrict results to chunks from this company.
        top_k : int | None
            Number of results to return. Defaults to config.top_k.

        Returns
        -------
        list[RetrievalHit]
            Top-k results sorted by relevance score descending.
        """
        k = top_k or self.config.top_k
        method = self.config.retrieval_method

        # Normalise "hybrid" alias.
        if method == "hybrid":
            method = "hybrid_rrf"

        # Candidate pool multiplier for reranking: retrieve more candidates
        # than needed so the cross-encoder has a richer pool to rerank from.
        rerank_pool = 3

        if method == "dense":
            return self._dense(query, top_k=k, company=company)

        if method == "dense_rerank":
            candidates = self._dense(query, top_k=k * rerank_pool, company=company)
            return self._rerank(query, candidates, top_k=k, method_label="dense_rerank")

        if method == "sparse":
            return self._sparse(query, top_k=k, company=company)

        if method == "sparse_rerank":
            candidates = self._sparse(query, top_k=k * rerank_pool, company=company)
            return self._rerank(query, candidates, top_k=k, method_label="sparse_rerank")

        if method == "hybrid_rrf":
            dense_hits = self._dense(query, top_k=k * 2, company=company)
            sparse_hits = self._sparse(query, top_k=k * 2, company=company)
            return self._rrf_fuse(dense_hits, sparse_hits, top_k=k)

        if method == "hybrid_rerank":
            dense_hits = self._dense(query, top_k=k * rerank_pool, company=company)
            sparse_hits = self._sparse(query, top_k=k * rerank_pool, company=company)
            fused = self._rrf_fuse(dense_hits, sparse_hits, top_k=k * rerank_pool)
            return self._rerank(query, fused, top_k=k, method_label="hybrid_rerank")

        raise ValueError(f"Unknown retrieval method: {method}")
