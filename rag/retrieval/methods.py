"""Dense, dense+rerank, sparse, sparse+rerank, hybrid-RRF, and hybrid+rerank retrieval for RQ1."""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

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
# Singleton cross-encoder shared across all Retriever instances.
# ---------------------------------------------------------------------------
_SHARED_CROSS_ENCODER = None
_SHARED_CROSS_ENCODER_NAME: str | None = None


def _get_shared_cross_encoder(model_name: str):
    """Lazily load the cross-encoder once and reuse across retrievers."""
    global _SHARED_CROSS_ENCODER, _SHARED_CROSS_ENCODER_NAME
    if _SHARED_CROSS_ENCODER is not None and _SHARED_CROSS_ENCODER_NAME == model_name:
        return _SHARED_CROSS_ENCODER

    import os
    import torch
    from sentence_transformers import CrossEncoder

    # Force CPU on Apple Silicon — MPS can cause OOM / compatibility errors
    # with large cross-encoder models.
    device = "cpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _logger.info("Apple Silicon detected — forcing cross-encoder to CPU for stability.")
        # Prevent PyTorch from auto-selecting MPS.
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    elif torch.cuda.is_available():
        device = "cuda"

    _logger.info("Loading shared BGE reranker model: %s (device=%s)", model_name, device)
    try:
        _SHARED_CROSS_ENCODER = CrossEncoder(model_name, device=device)
    except TypeError:
        # Fallback for sentence-transformers versions where 'device' is
        # not a direct constructor kwarg.
        _logger.info("Retrying CrossEncoder without device kwarg …")
        _SHARED_CROSS_ENCODER = CrossEncoder(model_name)
        # Move to correct device after init.
        if hasattr(_SHARED_CROSS_ENCODER, "model"):
            _SHARED_CROSS_ENCODER.model = _SHARED_CROSS_ENCODER.model.to(torch.device(device))
    _SHARED_CROSS_ENCODER_NAME = model_name
    gc.collect()
    return _SHARED_CROSS_ENCODER


class _BM25:
    """Small BM25 implementation (Okapi k1=1.5, b=0.75)."""

    def __init__(self, chunks: list[ChunkRecord], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunks = chunks
        self.doc_tokens = [self._tok(c.text) for c in chunks]
        self.doc_lens = [len(t) for t in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / max(len(self.doc_lens), 1)
        self.tf = [Counter(tokens) for tokens in self.doc_tokens]
        self.df: Counter[str] = Counter()
        for tfs in self.tf:
            self.df.update(tfs.keys())
        self.n = len(chunks)

    @staticmethod
    def _tok(text: str) -> list[str]:
        return _TOKEN_RE.findall(text.lower())

    def score(self, query: str) -> list[tuple[ChunkRecord, float]]:
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


class Retriever:
    """Unified retrieval interface for all six RQ1 methods.

    Supports: dense, dense_rerank, sparse, sparse_rerank, hybrid_rrf, hybrid_rerank.
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
        self.bm25 = _BM25(chunks)
        self._cross_encoder = None
        self._rerank_cache_path = ARTIFACTS_DIR / "rerank_cache.json"
        self._rerank_cache: dict[str, list[dict]] = {}
        self._load_rerank_cache()

    # ------------------------------------------------------------------
    # Rerank cache helpers
    # ------------------------------------------------------------------

    def _load_rerank_cache(self) -> None:
        if self._rerank_cache_path.exists():
            try:
                self._rerank_cache = json.loads(self._rerank_cache_path.read_text(encoding="utf-8"))
            except Exception:
                self._rerank_cache = {}

    def _persist_rerank_cache(self) -> None:
        self._rerank_cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._rerank_cache_path.write_text(json.dumps(self._rerank_cache), encoding="utf-8")

    @staticmethod
    def _rerank_cache_key(query: str, hits: list[RetrievalHit], top_k: int) -> str:
        signature = "||".join(f"{h.chunk.chunk_id}:{h.rank}:{h.score:.6f}" for h in hits)
        raw = f"{query}@@{top_k}@@{signature}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Cross-encoder reranker (model from config.rerank_model)
    # ------------------------------------------------------------------

    def _ensure_cross_encoder(self) -> None:
        """Lazy-load the shared cross-encoder model on first use."""
        if self._cross_encoder is not None:
            return
        self._cross_encoder = _get_shared_cross_encoder(self.config.rerank_model)

    def _bge_rerank(
        self,
        query: str,
        hits: list[RetrievalHit],
        top_k: int,
        method_label: str = "hybrid_rerank",
    ) -> list[RetrievalHit]:
        """Rerank candidates using the configured local cross-encoder.

        Parameters
        ----------
        method_label : str
            The retrieval method name to stamp on the returned
            ``RetrievalHit`` objects (e.g. ``"dense_rerank"``,
            ``"sparse_rerank"``, ``"hybrid_rerank"``).
        """
        if not hits:
            return []

        # Check cache first.
        cache_key = self._rerank_cache_key(query, hits, top_k)
        if cache_key in self._rerank_cache:
            cached = self._rerank_cache[cache_key]
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
        # Use small batch size (16) to avoid memory pressure on Apple Silicon.
        self._ensure_cross_encoder()
        pairs = [[query, h.chunk.text] for h in hits]
        try:
            scores = self._cross_encoder.predict(pairs, batch_size=16, show_progress_bar=False)
        except Exception as exc:
            _logger.warning("Cross-encoder predict failed (%s), retrying with batch_size=4", exc)
            gc.collect()
            scores = self._cross_encoder.predict(pairs, batch_size=4, show_progress_bar=False)

        # Pair hits with scores and sort descending.
        scored = sorted(zip(hits, scores), key=lambda x: float(x[1]), reverse=True)

        out: list[RetrievalHit] = []
        cache_payload: list[dict] = []
        for rank, (base, score) in enumerate(scored[:top_k], start=1):
            cache_payload.append(
                {
                    "chunk_id": base.chunk.chunk_id,
                    "score": float(score),
                    "pre_rank": base.rank,
                    "pre_score": base.score,
                }
            )
            out.append(
                RetrievalHit(
                    chunk=base.chunk,
                    score=float(score),
                    rank=rank,
                    method=method_label,
                    aux={"pre_rerank_rank": base.rank, "pre_rerank_score": base.score},
                )
            )
        self._rerank_cache[cache_key] = cache_payload
        self._persist_rerank_cache()
        return out

    # ------------------------------------------------------------------
    # Core retrieval methods
    # ------------------------------------------------------------------

    def _dense(self, query: str, top_k: int, company: str | None) -> list[RetrievalHit]:
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
        accum: dict[str, float] = defaultdict(float)
        provenance: dict[str, dict] = defaultdict(dict)
        for hits, tag in ((dense_hits, "dense"), (sparse_hits, "sparse")):
            for hit in hits:
                accum[hit.chunk.chunk_id] += 1.0 / (self.config.rrf_k + hit.rank)
                provenance[hit.chunk.chunk_id][tag] = {"rank": hit.rank, "score": hit.score}
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

    def retrieve(self, query: str, company: str | None = None, top_k: int | None = None) -> list[RetrievalHit]:
        """Run retrieval method from current config.

        Supported methods
        -----------------
        dense            – embedding-based vector search
        dense_rerank     – dense retrieval followed by cross-encoder reranking
        sparse           – BM25 keyword retrieval
        sparse_rerank    – sparse retrieval followed by cross-encoder reranking
        hybrid_rrf       – dense + sparse fused via Reciprocal Rank Fusion
        hybrid_rerank    – hybrid RRF followed by cross-encoder reranking
        """
        k = top_k or self.config.top_k
        method = self.config.retrieval_method
        if method == "hybrid":
            method = "hybrid_rrf"

        # --- Dense retrieval ---
        if method == "dense":
            return self._dense(query, top_k=k, company=company)
        # --- Dense retrieval + cross-encoder reranking ---
        if method == "dense_rerank":
            candidates = self._dense(query, top_k=k * 3, company=company)
            return self._bge_rerank(query, candidates, top_k=k, method_label="dense_rerank")
        # --- Sparse (BM25) retrieval ---
        if method == "sparse":
            return self._sparse(query, top_k=k, company=company)
        # --- Sparse retrieval + cross-encoder reranking ---
        if method == "sparse_rerank":
            candidates = self._sparse(query, top_k=k * 3, company=company)
            return self._bge_rerank(query, candidates, top_k=k, method_label="sparse_rerank")
        # --- Hybrid RRF (dense + sparse fused via Reciprocal Rank Fusion) ---
        if method == "hybrid_rrf":
            dense_hits = self._dense(query, top_k=k * 2, company=company)
            sparse_hits = self._sparse(query, top_k=k * 2, company=company)
            return self._rrf_fuse(dense_hits, sparse_hits, top_k=k)
        # --- Hybrid RRF + cross-encoder reranking ---
        if method == "hybrid_rerank":
            dense_hits = self._dense(query, top_k=k * 3, company=company)
            sparse_hits = self._sparse(query, top_k=k * 3, company=company)
            fused = self._rrf_fuse(dense_hits, sparse_hits, top_k=k * 3)
            return self._bge_rerank(query, fused, top_k=k, method_label="hybrid_rerank")
        raise ValueError(f"Unknown retrieval method: {method}")
