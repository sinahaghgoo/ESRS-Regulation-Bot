"""
RQ1 metric computation with explicit relevance alignment rules.

Relevance Matching Strategy (Thesis Justification)
===================================================
A retrieved chunk is deemed relevant to a gold evidence passage when the
textual overlap between the two exceeds a configurable threshold (default
50%) of the gold passage length. This approach is based on two principles:

1. **Passage coverage**: The retrieval system's job is to surface the
   *evidence* an expert annotator identified. A chunk that covers less
   than half of the gold passage may miss critical context (e.g., the
   qualifying clause of a compliance statement). The 50% threshold
   balances strictness (requiring substantial coverage) with tolerance
   for chunking boundary effects.

2. **Two-tier matching**: The primary path locates both gold passage and
   chunk as character spans inside the normalised full-report text. This
   is exact and position-aware. When exact substring matching fails
   (e.g., due to formatting differences between the manually curated
   golden standard and the preprocessed report), a token-level overlap
   fallback is used. The fallback uses multiplicity-aware bag-of-words
   overlap to avoid inflated matches from repeated common words.

References:
    - Adlakha et al. (2023) "Evaluating Correctness and Faithfulness
      of Instruction-Following Models for Question Answering" (token F1
      for passage-level matching)
    - TREC passage retrieval evaluation methodology (NIST)
    - Karpukhin et al. (2020) DPR evaluation: passage-level exact match
      with relaxed string matching
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from statistics import mean, stdev

from rag.data.text_utils import normalize_text
from rag.types import ChunkRecord, ConfigResult, EvalSample, GoldEvidence, RetrievalHit

_logger = logging.getLogger(__name__)

# Word-boundary tokenizer shared across matching functions.
_TOKEN_RE = re.compile(r"\w+")

# Default relevance threshold: fraction of gold passage that must be
# covered by a chunk to count as relevant.
DEFAULT_RELEVANCE_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Low-level text matching utilities
# ---------------------------------------------------------------------------

def _find_span(text: str, needle: str) -> tuple[int, int] | None:
    """Find the first occurrence of *needle* in *text* and return (start, end).

    Returns None if *needle* is empty or not found. This is the core of
    the span-based matching: both gold evidence and chunk are located as
    character ranges inside the normalised full-report text so that their
    overlap can be computed positionally.
    """
    if not needle:
        return None
    idx = text.find(needle)
    if idx >= 0:
        return (idx, idx + len(needle))
    return None


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Compute character overlap between two half-open intervals [start, end)."""
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    return max(0, hi - lo)


def _tokenize(text: str) -> list[str]:
    """Simple word tokenization (alphanumeric sequences, lowercased).

    Used for the token-overlap fallback. Deliberately simple so it is
    reproducible and does not depend on external NLP libraries.
    """
    return _TOKEN_RE.findall(text.lower())


def _token_overlap_ratio(gold_text: str, chunk_text: str) -> float:
    """Compute fraction of gold tokens covered by the chunk.

    Uses a multiplicity-aware bag-of-words overlap: each gold token is
    matched at most as many times as it appears in the chunk. This
    prevents a chunk that mentions a common word once from matching a
    gold passage that repeats it many times.

    Returns a value in [0.0, 1.0].
    """
    gold_tokens = _tokenize(gold_text)
    if not gold_tokens:
        return 0.0
    chunk_bag = Counter(_tokenize(chunk_text))
    gold_bag = Counter(gold_tokens)
    matched = 0
    for token, count in gold_bag.items():
        matched += min(count, chunk_bag.get(token, 0))
    return matched / len(gold_tokens)


# ---------------------------------------------------------------------------
# Diagnostics tracking
# ---------------------------------------------------------------------------

@dataclass
class MatchDiagnostics:
    """Accumulated diagnostics for relevance matching paths.

    Tracks how many relevance checks took the span-based vs. token-fallback
    path and how many resulted in hits. This is reported alongside evaluation
    results to verify that the matching pipeline is functioning correctly.
    """
    span_checks: int = 0
    span_hits: int = 0
    token_fallback_checks: int = 0
    token_fallback_hits: int = 0
    gold_span_misses: int = 0
    chunk_span_misses: int = 0
    short_evidence_skips: int = 0


# ---------------------------------------------------------------------------
# Core relevance judgement
# ---------------------------------------------------------------------------

def _is_relevant(
    chunk: ChunkRecord,
    evidence: GoldEvidence,
    report_text_by_company: dict[str, str],
    chunk_span_cache: dict[str, tuple[int, int] | None],
    gold_span_cache: dict[str, tuple[int, int] | None],
    diagnostics: MatchDiagnostics | None = None,
    threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
) -> bool:
    """Determine whether *chunk* is relevant to *evidence*.

    The relevance rule is a two-tier approach:

    **Primary (span-based)**: Both the gold passage and chunk are located as
    character spans inside the normalised full-report text. A chunk is
    relevant when the overlap covers >= *threshold* of the gold passage
    length (in characters). This is position-aware and handles partial
    coverage from chunk boundary effects.

    **Fallback (token-level)**: When the gold passage cannot be found as an
    exact substring (e.g., formatting differences between the golden standard
    and the preprocessed report), a token-level overlap is used. A chunk is
    relevant when >= *threshold* of the gold tokens are present in the chunk.
    Gold passages shorter than 20 characters are skipped to avoid
    false positives from very short or empty annotations.

    Parameters
    ----------
    threshold : float
        Minimum fraction of gold passage that must be covered. Default 0.5.
    """
    if not evidence.normalized_text.strip():
        return False

    report_text = report_text_by_company.get(evidence.company, "")
    if not report_text:
        return False

    # --- Primary: span-based matching in normalised report text ---
    if evidence.passage_id not in gold_span_cache:
        gold_span_cache[evidence.passage_id] = _find_span(
            report_text, evidence.normalized_text
        )
    gold_span = gold_span_cache[evidence.passage_id]

    if chunk.chunk_id not in chunk_span_cache:
        chunk_span_cache[chunk.chunk_id] = _find_span(
            report_text, chunk.normalized_text
        )
    chunk_span = chunk_span_cache[chunk.chunk_id]

    if gold_span is not None and chunk_span is not None:
        if diagnostics is not None:
            diagnostics.span_checks += 1
        ov = _overlap(chunk_span, gold_span)
        gold_len = max(1, gold_span[1] - gold_span[0])
        hit = ov >= int(threshold * gold_len)
        if hit and diagnostics is not None:
            diagnostics.span_hits += 1
        return hit

    # Track why we fell through to the fallback.
    if diagnostics is not None:
        if gold_span is None:
            diagnostics.gold_span_misses += 1
        if chunk_span is None:
            diagnostics.chunk_span_misses += 1

    # --- Fallback: token-overlap matching ---
    # Skip very short evidence passages (< 20 chars) to avoid false positives
    # from trivially short strings that may match many chunks spuriously.
    if len(evidence.normalized_text) < 20:
        if diagnostics is not None:
            diagnostics.short_evidence_skips += 1
        return False

    if diagnostics is not None:
        diagnostics.token_fallback_checks += 1
    hit = _token_overlap_ratio(evidence.normalized_text, chunk.normalized_text) >= threshold
    if hit and diagnostics is not None:
        diagnostics.token_fallback_hits += 1
    return hit


# ---------------------------------------------------------------------------
# Ranking metrics (standard IR formulations)
# ---------------------------------------------------------------------------

def _dcg_at_k(rels: list[int], k: int) -> float:
    """Discounted Cumulative Gain at k (binary relevance)."""
    out = 0.0
    for i, rel in enumerate(rels[:k], start=1):
        if rel:
            out += 1.0 / math.log2(i + 1)
    return out


def _ndcg_at_k(rels: list[int], k: int) -> float:
    """Normalised DCG at k. Returns 0 when no relevant items exist."""
    dcg = _dcg_at_k(rels, k)
    ideal = sorted(rels, reverse=True)
    idcg = _dcg_at_k(ideal, k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def _mrr_at_k(rels: list[int], k: int) -> float:
    """Mean Reciprocal Rank at k. Returns 0 if no relevant item in top-k."""
    for rank, rel in enumerate(rels[:k], start=1):
        if rel:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Budgeted recall (normalises for chunk size differences)
# ---------------------------------------------------------------------------

def _budgeted_recall(
    hits: list[RetrievalHit],
    evidences: list[GoldEvidence],
    report_text_by_company: dict[str, str],
    chunk_span_cache: dict[str, tuple[int, int] | None],
    gold_span_cache: dict[str, tuple[int, int] | None],
    token_budget: int = 4096,
    diagnostics: MatchDiagnostics | None = None,
    threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
) -> float:
    """Recall within a fixed token budget.

    This metric equalises the comparison across chunk sizes: a system
    using 1024-token chunks gets fewer chunks in the budget than one
    using 256-token chunks, making results directly comparable.
    """
    seen_gold: set[str] = set()
    consumed = 0
    for hit in hits:
        consumed += hit.chunk.token_count
        if consumed > token_budget:
            break
        for ev in evidences:
            if ev.passage_id in seen_gold:
                continue
            if _is_relevant(
                hit.chunk,
                ev,
                report_text_by_company,
                chunk_span_cache=chunk_span_cache,
                gold_span_cache=gold_span_cache,
                diagnostics=diagnostics,
                threshold=threshold,
            ):
                seen_gold.add(ev.passage_id)
    if not evidences:
        return 0.0
    return len(seen_gold) / len(evidences)


# ---------------------------------------------------------------------------
# Per-sample metric computation
# ---------------------------------------------------------------------------

def _compute_for_sample(
    hits: list[RetrievalHit],
    sample: EvalSample,
    report_text_by_company: dict[str, str],
    chunk_span_cache: dict[str, tuple[int, int] | None],
    gold_span_cache: dict[str, tuple[int, int] | None],
    diagnostics: MatchDiagnostics | None = None,
    threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
) -> dict[str, float]:
    """Compute all RQ1 metrics for one question-company sample.

    Metrics computed:
    - Recall@k (k=5, 10, 15): fraction of gold evidence passages found
    - Precision@k (k=5, 10, 15): fraction of top-k chunks that are relevant
    - F1@k (k=5, 10, 15): harmonic mean of Precision@k and Recall@k
    - nDCG@10: ranking quality accounting for position of relevant chunks
    - MRR@10: reciprocal rank of the first relevant chunk
    - Budgeted Recall@4096: recall within a 4096-token budget
    """
    unique_gold_ids = {e.passage_id for e in sample.evidences}
    gold_n = len(unique_gold_ids)

    rel_by_hit: list[int] = []
    matched_gold_at_k: dict[int, set[str]] = {5: set(), 10: set(), 15: set()}
    relevant_chunks_at_k: dict[int, int] = {5: 0, 10: 0, 15: 0}

    for i, hit in enumerate(hits[:15], start=1):
        matched_ids: set[str] = set()
        for ev in sample.evidences:
            if _is_relevant(
                hit.chunk,
                ev,
                report_text_by_company,
                chunk_span_cache=chunk_span_cache,
                gold_span_cache=gold_span_cache,
                diagnostics=diagnostics,
                threshold=threshold,
            ):
                matched_ids.add(ev.passage_id)
        rel = 1 if matched_ids else 0
        rel_by_hit.append(rel)
        for k in (5, 10, 15):
            if i <= k:
                if rel:
                    relevant_chunks_at_k[k] += 1
                matched_gold_at_k[k].update(matched_ids)

    def recall_at(k: int) -> float:
        if gold_n == 0:
            return 0.0
        return len(matched_gold_at_k[k]) / gold_n

    def precision_at(k: int) -> float:
        return relevant_chunks_at_k[k] / k

    def f1_at(k: int) -> float:
        p = precision_at(k)
        r = recall_at(k)
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    return {
        "recall@5": recall_at(5),
        "recall@10": recall_at(10),
        "recall@15": recall_at(15),
        "precision@5": precision_at(5),
        "precision@10": precision_at(10),
        "precision@15": precision_at(15),
        "f1@5": f1_at(5),
        "f1@10": f1_at(10),
        "f1@15": f1_at(15),
        "ndcg@10": _ndcg_at_k(rel_by_hit, 10),
        "mrr@10": _mrr_at_k(rel_by_hit, 10),
        "budgeted_recall@4096": _budgeted_recall(
            hits,
            sample.evidences,
            report_text_by_company,
            chunk_span_cache=chunk_span_cache,
            gold_span_cache=gold_span_cache,
            token_budget=4096,
            diagnostics=diagnostics,
            threshold=threshold,
        ),
    }


# ---------------------------------------------------------------------------
# Configuration-level evaluation
# ---------------------------------------------------------------------------

def evaluate_configuration(
    *,
    config_id: str,
    chunk_strategy: str,
    chunk_size: int,
    retrieval_method: str,
    samples: list[EvalSample],
    retrieve_fn,
    report_text_by_company: dict[str, str],
    threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
) -> ConfigResult:
    """Evaluate one RQ1 configuration over all question-company pairs.

    Parameters
    ----------
    config_id : str
        Unique identifier for this configuration (e.g. "fixed_1024_sparse").
    chunk_strategy, chunk_size, retrieval_method : str/int
        Configuration factors for result metadata.
    samples : list[EvalSample]
        Golden standard evaluation samples (one per question-company pair).
    retrieve_fn : callable
        ``retrieve_fn(query, company, top_k) -> list[RetrievalHit]``
    report_text_by_company : dict[str, str]
        Normalised full-report text keyed by company name. Used for
        span-based relevance matching.
    threshold : float
        Relevance overlap threshold (default 0.5).

    Returns
    -------
    ConfigResult
        Aggregated and per-standard metrics with evaluation diagnostics.
    """
    metric_rows: list[dict[str, float]] = []
    per_sample_rows: list[dict] = []  # for per-sample detail export
    per_standard_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    per_standard_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"evaluated": 0, "skipped": 0}
    )
    chunk_span_cache: dict[str, tuple[int, int] | None] = {}
    gold_span_cache: dict[str, tuple[int, int] | None] = {}
    diag = MatchDiagnostics()
    skipped = 0

    for sample in samples:
        std = sample.esrs_standard
        usable_evidences = [e for e in sample.evidences if e.normalized_text.strip()]
        if not usable_evidences:
            skipped += 1
            per_standard_counts[std]["skipped"] += 1
            continue

        per_standard_counts[std]["evaluated"] += 1

        # Build a clean sample with only usable evidences.
        sample_for_eval = EvalSample(
            question_id=sample.question_id,
            question=sample.question,
            esrs_standard=sample.esrs_standard,
            company=sample.company,
            evidences=usable_evidences,
        )
        hits = retrieve_fn(sample_for_eval.question, sample_for_eval.company, 15)
        row = _compute_for_sample(
            hits,
            sample_for_eval,
            report_text_by_company,
            chunk_span_cache=chunk_span_cache,
            gold_span_cache=gold_span_cache,
            diagnostics=diag,
            threshold=threshold,
        )
        metric_rows.append(row)
        per_standard_rows[sample_for_eval.esrs_standard].append(row)

        # Store per-sample detail for downstream analysis.
        per_sample_rows.append({
            "question_id": sample.question_id,
            "company": sample.company,
            "esrs_standard": std,
            "n_gold_evidences": len(usable_evidences),
            **row,
        })

    if skipped:
        _logger.info(
            "Config %s: skipped %d/%d samples (no usable evidence text)",
            config_id, skipped, len(samples),
        )

    # --- Aggregate metrics (macro-average over samples) ---
    metric_names = [
        "recall@5", "recall@10", "recall@15",
        "precision@5", "precision@10", "precision@15",
        "f1@5", "f1@10", "f1@15",
        "ndcg@10", "mrr@10",
        "budgeted_recall@4096",
    ]
    aggregate = {m: mean([r[m] for r in metric_rows]) for m in metric_names}

    # Standard deviation for key metrics (useful for thesis tables).
    for m in ("recall@15", "precision@15", "f1@15", "ndcg@10", "mrr@10"):
        values = [r[m] for r in metric_rows]
        aggregate[f"{m}_std"] = stdev(values) if len(values) > 1 else 0.0

    per_standard = {
        std: {m: mean([r[m] for r in rows]) for m in metric_names}
        for std, rows in per_standard_rows.items()
    }

    # --- Match diagnostics ---
    total_relevance_checks = diag.span_checks + diag.token_fallback_checks
    match_diagnostics = {
        "span_checks": diag.span_checks,
        "span_hits": diag.span_hits,
        "token_fallback_checks": diag.token_fallback_checks,
        "token_fallback_hits": diag.token_fallback_hits,
        "gold_span_misses": diag.gold_span_misses,
        "chunk_span_misses": diag.chunk_span_misses,
        "short_evidence_skips": diag.short_evidence_skips,
        "span_path_ratio": (
            diag.span_checks / total_relevance_checks
            if total_relevance_checks > 0 else 0.0
        ),
    }

    return ConfigResult(
        config_id=config_id,
        chunk_strategy=chunk_strategy,
        chunk_size=chunk_size,
        retrieval_method=retrieval_method,
        metrics=aggregate,
        per_standard_metrics=per_standard,
        details={
            "n_total": len(samples),
            "n_evaluated": len(metric_rows),
            "n_skipped": skipped,
            "relevance_threshold": threshold,
            "per_standard_counts": {k: dict(v) for k, v in per_standard_counts.items()},
            "match_diagnostics": match_diagnostics,
            "per_sample_metrics": per_sample_rows,
        },
    )
