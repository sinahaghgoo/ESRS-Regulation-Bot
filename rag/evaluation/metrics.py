"""RQ1 metric computation with explicit relevance alignment rules."""

from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import mean

from rag.data.text_utils import normalize_text
from rag.types import ChunkRecord, ConfigResult, EvalSample, GoldEvidence, RetrievalHit

_logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\w+")


def _find_span(text: str, needle: str) -> tuple[int, int] | None:
    if not needle:
        return None
    idx = text.find(needle)
    if idx >= 0:
        return (idx, idx + len(needle))
    return None


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> int:
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    return max(0, hi - lo)


def _tokenize(text: str) -> list[str]:
    """Simple word tokenization for token-overlap matching."""
    return _TOKEN_RE.findall(text.lower())


def _token_overlap_ratio(gold_text: str, chunk_text: str) -> float:
    """Compute fraction of gold tokens covered by the chunk.

    Uses a bag-of-words overlap that counts each token at most as many
    times as it appears in the gold passage (i.e. respects multiplicity).
    This prevents a chunk that contains one mention of a common word from
    matching a gold passage that repeats it many times.
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


def _is_relevant(
    chunk: ChunkRecord,
    evidence: GoldEvidence,
    report_text_by_company: dict[str, str],
    chunk_span_cache: dict[str, tuple[int, int] | None],
    gold_span_cache: dict[str, tuple[int, int] | None],
) -> bool:
    """50% gold passage overlap rule — span-based with token-overlap fallback.

    Primary path: locate both gold passage and chunk as spans inside the
    normalized full-report text and check >=50% character overlap.

    Fallback (when the gold passage cannot be found as an exact substring,
    e.g. due to markdown formatting differences between the manually curated
    golden standard and the preprocessed report text): compute token-level
    overlap between the gold passage and the chunk.  A chunk is relevant
    when >=50% of the gold tokens are present in the chunk text.
    """
    # Skip empty evidence passages — they cannot be meaningfully matched.
    if not evidence.normalized_text.strip():
        return False

    report_text = report_text_by_company.get(evidence.company, "")
    if not report_text:
        return False

    # --- Primary: span-based matching in report text ---
    if evidence.passage_id not in gold_span_cache:
        gold_span_cache[evidence.passage_id] = _find_span(report_text, evidence.normalized_text)
    gold_span = gold_span_cache[evidence.passage_id]
    if chunk.chunk_id not in chunk_span_cache:
        chunk_span_cache[chunk.chunk_id] = _find_span(report_text, chunk.normalized_text)
    chunk_span = chunk_span_cache[chunk.chunk_id]

    if gold_span is not None and chunk_span is not None:
        ov = _overlap(chunk_span, gold_span)
        return ov >= int(0.5 * max(1, gold_span[1] - gold_span[0]))

    # --- Fallback: token-overlap matching ---
    # This handles cases where the golden standard text was cleaned or
    # reformulated relative to the markdown source (e.g. markdown bullets,
    # escaped characters, footnote markers, or table reformulations removed).
    if len(evidence.normalized_text) < 20:
        return False
    return _token_overlap_ratio(evidence.normalized_text, chunk.normalized_text) >= 0.5


def _dcg_at_k(rels: list[int], k: int) -> float:
    vals = rels[:k]
    out = 0.0
    for i, rel in enumerate(vals, start=1):
        if rel:
            out += 1.0 / math.log2(i + 1)
    return out


def _ndcg_at_k(rels: list[int], k: int) -> float:
    dcg = _dcg_at_k(rels, k)
    ideal = sorted(rels, reverse=True)
    idcg = _dcg_at_k(ideal, k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def _mrr_at_k(rels: list[int], k: int) -> float:
    for rank, rel in enumerate(rels[:k], start=1):
        if rel:
            return 1.0 / rank
    return 0.0


def _budgeted_recall(
    hits: list[RetrievalHit],
    evidences: list[GoldEvidence],
    report_text_by_company: dict[str, str],
    chunk_span_cache: dict[str, tuple[int, int] | None],
    gold_span_cache: dict[str, tuple[int, int] | None],
    token_budget: int = 4096,
) -> float:
    seen_gold: set[str] = set()
    consumed = 0
    for hit in hits:
        consumed += hit.chunk.token_count
        for ev in evidences:
            if ev.passage_id in seen_gold:
                continue
            if _is_relevant(
                hit.chunk,
                ev,
                report_text_by_company,
                chunk_span_cache=chunk_span_cache,
                gold_span_cache=gold_span_cache,
            ):
                seen_gold.add(ev.passage_id)
        if consumed > token_budget:
            break
    if not evidences:
        return 0.0
    return len(seen_gold) / len(evidences)


def _compute_for_sample(
    hits: list[RetrievalHit],
    sample: EvalSample,
    report_text_by_company: dict[str, str],
    chunk_span_cache: dict[str, tuple[int, int] | None],
    gold_span_cache: dict[str, tuple[int, int] | None],
) -> dict[str, float]:
    """Compute all RQ1 metrics for one question-company sample."""
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

    return {
        "recall@5": recall_at(5),
        "recall@10": recall_at(10),
        "recall@15": recall_at(15),
        "precision@5": precision_at(5),
        "precision@10": precision_at(10),
        "precision@15": precision_at(15),
        "ndcg@10": _ndcg_at_k(rel_by_hit, 10),
        "mrr@10": _mrr_at_k(rel_by_hit, 10),
        "budgeted_recall@4096": _budgeted_recall(
            hits,
            sample.evidences,
            report_text_by_company,
            chunk_span_cache=chunk_span_cache,
            gold_span_cache=gold_span_cache,
            token_budget=4096,
        ),
    }


def evaluate_configuration(
    *,
    config_id: str,
    chunk_strategy: str,
    chunk_size: int,
    retrieval_method: str,
    samples: list[EvalSample],
    retrieve_fn,
    report_text_by_company: dict[str, str],
) -> ConfigResult:
    """
    Evaluate one RQ1 configuration over all question-company pairs.

    `retrieve_fn(query, company, top_k) -> list[RetrievalHit]`
    """
    metric_rows: list[dict[str, float]] = []
    per_standard_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    chunk_span_cache: dict[str, tuple[int, int] | None] = {}
    gold_span_cache: dict[str, tuple[int, int] | None] = {}
    skipped = 0
    for sample in samples:
        # Skip samples where ALL evidence passages have empty text —
        # they cannot be meaningfully evaluated for retrieval quality.
        usable_evidences = [e for e in sample.evidences if e.normalized_text.strip()]
        if not usable_evidences:
            skipped += 1
            continue
        # Replace sample evidences with only usable ones for scoring.
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
        )
        metric_rows.append(row)
        per_standard_rows[sample_for_eval.esrs_standard].append(row)
    if skipped:
        _logger.info(
            "Config %s: skipped %d/%d samples with empty evidence",
            config_id, skipped, len(samples),
        )

    metric_names = [
        "recall@5",
        "recall@10",
        "recall@15",
        "precision@5",
        "precision@10",
        "precision@15",
        "ndcg@10",
        "mrr@10",
        "budgeted_recall@4096",
    ]
    aggregate = {m: mean([r[m] for r in metric_rows]) for m in metric_names}
    per_standard = {
        std: {m: mean([r[m] for r in rows]) for m in metric_names}
        for std, rows in per_standard_rows.items()
    }
    return ConfigResult(
        config_id=config_id,
        chunk_strategy=chunk_strategy,
        chunk_size=chunk_size,
        retrieval_method=retrieval_method,
        metrics=aggregate,
        per_standard_metrics=per_standard,
        details={"n_samples": len(metric_rows)},
    )

