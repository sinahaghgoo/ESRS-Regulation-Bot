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
    RERANK_CACHE_VERSION,
    configure_local_torch_runtime,
    get_rerank_batch_size,
)
from rag.data.text_utils import sentence_split
from rag.indexing.embedder import EmbeddingService
from rag.indexing.qdrant_index import QdrantChunkIndex
from rag.types import ChunkRecord, RetrievalHit

_TOKEN_RE = re.compile(r"\w+")
_logger = logging.getLogger(__name__)
_RERANK_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "for", "from",
    "has", "have", "how", "in", "is", "it", "of", "on", "or", "that",
    "the", "their", "this", "to", "what", "which", "with",
}

# ---------------------------------------------------------------------------
# ESRS-specific query expansion for dense retrieval
# ---------------------------------------------------------------------------

# Map ESRS standard identifiers to domain-specific expansion terms.
# These terms are prepended to the query before embedding to help the
# dense retriever bridge the vocabulary gap between natural language
# questions and the formal, terminology-heavy ESRS report text.
_ESRS_EXPANSION_TERMS: dict[str, str] = {
    "esrs 1": "ESRS 1 General Requirements double materiality assessment sustainability matters",
    "esrs 2": "ESRS 2 General Disclosures governance remuneration sustainability incentive",
    "esrs e1": "ESRS E1 Climate Change transition plan greenhouse gas emissions GHG Scope 1 2 3",
    "esrs e2": "ESRS E2 Pollution pollutant emissions air water soil substances of concern",
    "esrs e3": "ESRS E3 Water and Marine Resources water consumption water-stressed areas",
    "esrs e4": "ESRS E4 Biodiversity and Ecosystems biodiversity-sensitive areas species habitats",
    "esrs e5": "ESRS E5 Resource Use and Circular Economy waste generation recycling incineration",
    "esrs s1": "ESRS S1 Own Workforce workforce composition gender diversity employees",
    "esrs s2": "ESRS S2 Workers in Value Chain supply chain value chain workers due diligence",
    "esrs s3": "ESRS S3 Affected Communities community engagement grievance mechanisms",
    "esrs s4": "ESRS S4 Consumers and End-Users consumer safety health data privacy",
    "esrs g1": "ESRS G1 Business Conduct anti-corruption anti-bribery whistleblowing governance",
}


def _expand_query_for_dense(query: str) -> str:
    """Expand a query with ESRS domain-specific terms for dense retrieval.

    Detects which ESRS standard the query relates to (by scanning for
    standard identifiers like "ESRS E1" or common topic keywords) and
    prepends relevant expansion terms. This helps the embedding model
    capture the regulatory context that BM25 handles via exact keyword
    matching.

    The expansion is only applied to dense retrieval (not sparse/BM25),
    as BM25 already excels at keyword matching and expansion would
    add noise to the term-frequency statistics.
    """
    query_lower = query.lower()

    # Direct ESRS standard mention detection.
    for std_key, expansion in _ESRS_EXPANSION_TERMS.items():
        if std_key in query_lower:
            return f"{expansion}. {query}"

    # Keyword-based detection for queries that don't mention the standard.
    _keyword_map = {
        "materiality": "esrs 1",
        "double materiality": "esrs 1",
        "remuneration": "esrs 2",
        "incentive scheme": "esrs 2",
        "governance": "esrs 2",
        "climate": "esrs e1",
        "transition plan": "esrs e1",
        "greenhouse gas": "esrs e1",
        "emissions": "esrs e1",
        "ghg": "esrs e1",
        "pollution": "esrs e2",
        "pollutant": "esrs e2",
        "water consumption": "esrs e3",
        "water-stressed": "esrs e3",
        "marine": "esrs e3",
        "biodiversity": "esrs e4",
        "ecosystem": "esrs e4",
        "species": "esrs e4",
        "waste": "esrs e5",
        "circular economy": "esrs e5",
        "recycling": "esrs e5",
        "workforce": "esrs s1",
        "employee": "esrs s1",
        "gender": "esrs s1",
        "value chain": "esrs s2",
        "supply chain": "esrs s2",
        "communit": "esrs s3",
        "grievance": "esrs s3",
        "consumer": "esrs s4",
        "privacy": "esrs s4",
        "corruption": "esrs g1",
        "bribery": "esrs g1",
        "whistleblow": "esrs g1",
    }
    for keyword, std_key in _keyword_map.items():
        if keyword in query_lower:
            expansion = _ESRS_EXPANSION_TERMS[std_key]
            return f"{expansion}. {query}"

    # No match — return query unchanged.
    return query


def _minmax_scale(values: np.ndarray) -> np.ndarray:
    """Scale a score vector into [0, 1] without changing its ordering."""
    if values.size == 0:
        return values
    lo = float(values.min())
    hi = float(values.max())
    if hi - lo < 1e-12:
        return np.ones_like(values, dtype=np.float64)
    return (values - lo) / (hi - lo)


def _query_chunk_overlap(query: str, chunk_text: str) -> float:
    """Fraction of unique query tokens present in the chunk text."""
    query_terms = set(_TOKEN_RE.findall(query.lower()))
    if not query_terms:
        return 0.0
    chunk_terms = set(_TOKEN_RE.findall(chunk_text.lower()))
    return len(query_terms & chunk_terms) / len(query_terms)


def _extract_query_terms(query: str) -> list[str]:
    """Return informative query terms for snippet selection."""
    terms: list[str] = []
    for token in _TOKEN_RE.findall(query.lower()):
        if token.isdigit():
            terms.append(token)
            continue
        if len(token) >= 3 and token not in _RERANK_STOPWORDS:
            terms.append(token)
    return terms


def _split_rerank_units(text: str) -> list[str]:
    """Split chunk text into compact ranking units.

    Paragraph-first splitting handles list and table heavy reports better than
    sentence splitting alone. Each paragraph is then split into sentences when
    possible to let the reranker focus on the most query-aligned evidence span.
    """
    units: list[str] = []
    for paragraph in re.split(r"\n{2,}", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if len(lines) > 1 and any(len(line.split()) <= 18 for line in lines):
            units.extend(lines)
            continue
        sentences = sentence_split(paragraph)
        units.extend(sentences or [paragraph])
    return units or [text.strip()]


def _truncate_words(text: str, max_words: int) -> str:
    """Truncate *text* to approximately *max_words* words."""
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip()


def _centered_word_window(text: str, query_terms: list[str], max_words: int) -> str:
    """Return a word window centred around the earliest matching query term."""
    words = text.split()
    if len(words) <= max_words:
        return text.strip()

    query_terms_set = set(query_terms)
    anchor_idx = 0
    for idx, word in enumerate(words):
        token = re.sub(r"\W+", "", word.lower())
        if token in query_terms_set:
            anchor_idx = idx
            break

    start = max(0, anchor_idx - max_words // 4)
    end = min(len(words), start + max_words)
    start = max(0, end - max_words)
    return " ".join(words[start:end]).strip()


def _unit_score(unit: str, query_terms: list[str]) -> tuple[float, float]:
    """Score a text unit by query-term coverage and frequency."""
    if not query_terms:
        return (0.0, 0.0)

    unit_tokens = _TOKEN_RE.findall(unit.lower())
    if not unit_tokens:
        return (0.0, 0.0)

    token_set = set(unit_tokens)
    overlap = len(token_set & set(query_terms))
    density = sum(unit_tokens.count(term) for term in query_terms)
    return (
        overlap / max(len(set(query_terms)), 1),
        density / max(len(unit_tokens), 1),
    )


def _build_rerank_text(query: str, chunk: ChunkRecord, max_words: int = 220) -> str:
    """Build a query-focused rerank snippet from a chunk.

    Long 1024-token chunks often place the relevant passage far from the
    beginning. Passing the full chunk to a compact cross-encoder biases the
    model toward the first few hundred tokens. We therefore extract the most
    query-aligned local region while keeping the section header context.
    """
    header = " > ".join(part.strip() for part in chunk.header_hierarchy if part.strip())
    body = chunk.text.strip()
    if not body:
        return chunk.retrieval_text

    query_terms = _extract_query_terms(query)
    if len(body.split()) <= max_words:
        excerpt = body
    else:
        units = _split_rerank_units(body)
        ranked_units = [(_unit_score(unit, query_terms), idx, unit) for idx, unit in enumerate(units)]
        ranked_units.sort(key=lambda item: (item[0][0], item[0][1]), reverse=True)

        if not ranked_units or ranked_units[0][0] == (0.0, 0.0):
            excerpt = _centered_word_window(body, query_terms, max_words)
        else:
            best_idx = ranked_units[0][1]
            selected = {best_idx}
            total_words = len(units[best_idx].split())
            left = best_idx - 1
            right = best_idx + 1

            while total_words < max_words and (left >= 0 or right < len(units)):
                candidates: list[tuple[tuple[float, float], int]] = []
                if left >= 0:
                    candidates.append((_unit_score(units[left], query_terms), left))
                if right < len(units):
                    candidates.append((_unit_score(units[right], query_terms), right))
                candidates.sort(key=lambda item: (item[0][0], item[0][1], -abs(item[1] - best_idx)), reverse=True)
                if not candidates:
                    break
                _, chosen_idx = candidates[0]
                unit_words = len(units[chosen_idx].split())
                if total_words + unit_words > int(max_words * 1.15) and total_words >= int(max_words * 0.75):
                    break
                selected.add(chosen_idx)
                total_words += unit_words
                if chosen_idx == left:
                    left -= 1
                elif chosen_idx == right:
                    right += 1
                else:
                    if chosen_idx < best_idx:
                        left = chosen_idx - 1
                    else:
                        right = chosen_idx + 1

            excerpt = " ".join(units[idx] for idx in sorted(selected)).strip()
            if len(excerpt.split()) > max_words:
                excerpt = _centered_word_window(excerpt, query_terms, max_words)

        if len(excerpt.split()) < max_words // 3:
            excerpt = _centered_word_window(body, query_terms, max_words)

    if header:
        return f"[Section: {header}] {excerpt}"
    return excerpt

# ---------------------------------------------------------------------------
# Singleton cross-encoder shared across all Retriever instances
# ---------------------------------------------------------------------------
_SHARED_CROSS_ENCODER = None
_SHARED_CROSS_ENCODER_NAME: str | None = None


def _rerank_blend_weights(method_label: str, chunk_size: int) -> dict[str, float]:
    """Blend cross-encoder and first-stage scores for reranking.

    Long 1024-token chunks are already strong recall carriers. The reranker
    only sees a focused excerpt, so its score should be slightly less
    dominant there to avoid demoting a good first-stage candidate because the
    snippet missed supporting context elsewhere in the chunk.
    """
    weights = {
        "dense_rerank": {"cross": 0.55, "base": 0.20, "rank": 0.15, "overlap": 0.10},
        "sparse_rerank": {"cross": 0.45, "base": 0.25, "rank": 0.15, "overlap": 0.15},
        "hybrid_rerank": {"cross": 0.35, "base": 0.35, "rank": 0.15, "overlap": 0.15},
    }.get(
        method_label,
        {"cross": 0.50, "base": 0.20, "rank": 0.20, "overlap": 0.10},
    )

    if chunk_size >= 1024 and method_label.endswith("_rerank"):
        weights = dict(weights)
        weights["cross"] = max(0.15, weights["cross"] - 0.10)
        weights["base"] += 0.10

    return weights


def _get_shared_cross_encoder(model_name: str):
    """Lazily load the cross-encoder once and reuse across retrievers.

    Using a module-level singleton avoids loading the ~80 MB model
    repeatedly when evaluating multiple retrieval methods in sequence.
    """
    global _SHARED_CROSS_ENCODER, _SHARED_CROSS_ENCODER_NAME
    if _SHARED_CROSS_ENCODER is not None and _SHARED_CROSS_ENCODER_NAME == model_name:
        return _SHARED_CROSS_ENCODER

    from sentence_transformers import CrossEncoder

    configure_local_torch_runtime()
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
        self.doc_tokens = [self._tok(c.retrieval_text) for c in chunks]
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
        raw = f"{RERANK_CACHE_VERSION}@@{query}@@{top_k}@@{signature}"
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
        # The reranker sees the same structure-enriched chunk representation
        # that is used for dense indexing and BM25 scoring.
        self._ensure_cross_encoder()
        rerank_query = (
            _expand_query_for_dense(query)
            if method_label in {"dense_rerank", "hybrid_rerank"}
            else query
        )
        pairs = [[rerank_query, _build_rerank_text(rerank_query, h.chunk)] for h in hits]
        batch_size = get_rerank_batch_size()
        try:
            scores = self._cross_encoder.predict(
                pairs, batch_size=batch_size, show_progress_bar=False
            )
        except Exception as exc:
            _logger.warning(
                "Cross-encoder predict failed (%s), retrying with batch_size=%s",
                exc,
                max(1, batch_size // 2),
            )
            gc.collect()
            scores = self._cross_encoder.predict(
                pairs, batch_size=max(1, batch_size // 2), show_progress_bar=False
            )

        scores_array = np.asarray(scores, dtype=np.float64).ravel()
        if np.any(np.isnan(scores_array)):
            raise RuntimeError(
                f"Cross-encoder returned NaN scores for {len(pairs)} pairs. "
                f"Model: {self.config.rerank_model}. Check model compatibility."
            )

        cross_scaled = _minmax_scale(scores_array)
        base_scaled = _minmax_scale(
            np.asarray([float(hit.score) for hit in hits], dtype=np.float64)
        )
        rank_scaled = _minmax_scale(
            np.asarray([1.0 / (self.config.rrf_k + hit.rank) for hit in hits], dtype=np.float64)
        )
        overlap_scaled = np.asarray(
            [_query_chunk_overlap(query, hit.chunk.retrieval_text) for hit in hits],
            dtype=np.float64,
        )

        blend_weights = _rerank_blend_weights(method_label, self.config.chunk_size)

        blended_scores = (
            blend_weights["cross"] * cross_scaled
            + blend_weights["base"] * base_scaled
            + blend_weights["rank"] * rank_scaled
            + blend_weights["overlap"] * overlap_scaled
        )

        # Sort candidates by the blended score rather than by the raw cross-
        # encoder score alone. This preserves strong first-stage rankings
        # while still allowing the cross-encoder to reorder ambiguous cases.
        scored = sorted(
            zip(hits, scores_array, blended_scores),
            key=lambda item: float(item[2]),
            reverse=True,
        )

        out: list[RetrievalHit] = []
        cache_payload: list[dict] = []
        for rank, (base, cross_score, blended_score) in enumerate(scored[:top_k], start=1):
            cache_payload.append({
                "chunk_id": base.chunk.chunk_id,
                "score": float(blended_score),
                "cross_score": float(cross_score),
                "pre_rank": base.rank,
                "pre_score": base.score,
            })
            out.append(
                RetrievalHit(
                    chunk=base.chunk,
                    score=float(blended_score),
                    rank=rank,
                    method=method_label,
                    aux={
                        "pre_rerank_rank": base.rank,
                        "pre_rerank_score": base.score,
                        "cross_encoder_score": float(cross_score),
                        "blended_score": float(blended_score),
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
        """Dense retrieval via Qdrant vector search (cosine similarity).

        Applies ESRS-specific query expansion before embedding to bridge
        the vocabulary gap between natural language questions and formal
        regulatory report text. This improves recall for domain-specific
        queries where general-purpose embeddings may miss terminology.
        """
        expanded = _expand_query_for_dense(query)
        qvec = self.embedder.embed_text(expanded)
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
        # Increased from 3x to 5x to give the cross-encoder a deeper pool,
        # which is especially important when the base retriever's ranking
        # may miss relevant chunks in the initial top-k.
        rerank_pool = 5

        # Fusion pool multiplier for hybrid RRF: how many candidates each
        # sub-system (dense, sparse) contributes to the fusion. Increased
        # from 2x to 3x to capture more candidates from both systems.
        fusion_pool = 3

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
            dense_hits = self._dense(query, top_k=k * fusion_pool, company=company)
            sparse_hits = self._sparse(query, top_k=k * fusion_pool, company=company)
            return self._rrf_fuse(dense_hits, sparse_hits, top_k=k)

        if method == "hybrid_rerank":
            dense_hits = self._dense(query, top_k=k * rerank_pool, company=company)
            sparse_hits = self._sparse(query, top_k=k * rerank_pool, company=company)
            fused = self._rrf_fuse(dense_hits, sparse_hits, top_k=k * rerank_pool)
            return self._rerank(query, fused, top_k=k, method_label="hybrid_rerank")

        raise ValueError(f"Unknown retrieval method: {method}")
