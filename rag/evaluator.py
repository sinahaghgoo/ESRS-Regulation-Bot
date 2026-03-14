"""Evaluation harness for benchmarking ESRS RAG configurations against Golden Standard."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from rag.config import ARTIFACTS_DIR, BENCHMARK_QUESTION_IDS, LLM_MODEL

# Lazy imports for modules that require the openai package.
# This allows metric helper functions to be used without the full LLM stack.
def _import_generate_answer():
    from rag.generator import generate_answer
    return generate_answer

def _import_chat_json():
    from rag.llm_client import chat_json
    return chat_json

EVAL_OUTPUT_DIR = ARTIFACTS_DIR / "evaluation_results"
EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Golden Standard normalization
# ---------------------------------------------------------------------------

def normalize_page_ref(page_ref: Any) -> Set[int]:
    """Normalize page references from Golden Standard/evaluation outputs."""
    if isinstance(page_ref, int):
        return {page_ref} if page_ref > 0 else set()
    if page_ref is None:
        return set()
    text = str(page_ref).strip()
    if not text:
        return set()
    if text.lower() in {"not explicitly marked", "not available", "n/a", "na", "unknown"}:
        return set()

    pages: Set[int] = set()
    # Accept separators and ranges like "43-44", "43–44", "43, 44"
    for part in re.split(r"[,;|]", text):
        item = part.strip()
        if not item:
            continue
        match = re.match(r"^(\d+)\s*[-–]\s*(\d+)$", item)
        if match:
            start, end = int(match.group(1)), int(match.group(2))
            lo, hi = sorted((start, end))
            pages.update(range(lo, hi + 1))
            continue
        if item.isdigit():
            value = int(item)
            if value > 0:
                pages.add(value)
    return pages


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


# Stop words excluded from relevance overlap to avoid inflated matching
# when evidence passages contain many common English words.
_STOP_WORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "shall", "should", "may", "might", "must", "can", "could",
    "not", "no", "nor", "if", "then", "than", "that", "this", "these",
    "those", "it", "its", "we", "our", "their", "they", "he", "she",
    "which", "who", "whom", "what", "where", "when", "how", "all", "each",
    "any", "both", "such", "so", "also", "very", "more", "most", "other",
    "only", "same", "into", "about", "up", "out", "over", "after", "before",
}


def _content_tokens(text: str) -> Set[str]:
    """Tokenize text and remove stop words for content-based matching."""
    return {t for t in _tokenize(text) if t not in _STOP_WORDS and len(t) > 1}


def _safe_mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(mean(vals)) if vals else 0.0


def _chunk_pages(chunk: Dict[str, Any]) -> Set[int]:
    pages = set()
    for p in chunk.get("page_span", []) or []:
        if isinstance(p, int) and p > 0:
            pages.add(p)
    page_number = chunk.get("page_number")
    if isinstance(page_number, int) and page_number > 0:
        pages.add(page_number)
    return pages


def _chunk_header_text(chunk: Dict[str, Any]) -> str:
    return " > ".join(chunk.get("header_hierarchy", []) or []).strip().lower()


def _gold_pages(gold_evidence: List[Dict[str, Any]]) -> Set[int]:
    pages: Set[int] = set()
    for ev in gold_evidence:
        pages.update(normalize_page_ref(ev.get("page_number", "")))
    return pages


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def _evidence_text(ev: Dict[str, Any]) -> str:
    """Return evidence text, handling both 'text' and 'exact_text' field names."""
    return str(ev.get("text", "") or ev.get("exact_text", "")).strip()


def _chunk_relevant(chunk: Dict[str, Any], gold_evidence: List[Dict[str, Any]]) -> bool:
    """Mark a chunk relevant via page overlap or moderate lexical overlap to evidence text.

    Lexical overlap uses content tokens (stop words removed) to avoid inflated
    matching from common English words.
    """
    for ev in gold_evidence:
        if _chunk_matches_evidence(chunk, ev):
            return True
    return False


def _chunk_matches_evidence(chunk: Dict[str, Any], evidence: Dict[str, Any]) -> bool:
    """Check whether a chunk matches one evidence entry."""
    c_pages = _chunk_pages(chunk)
    c_text_tokens = _content_tokens(chunk.get("text", ""))
    c_header = _chunk_header_text(chunk)

    ev_pages = normalize_page_ref(evidence.get("page_number", ""))
    ev_tokens = _content_tokens(_evidence_text(evidence))

    # If golden evidence specifies pages, require page overlap when chunk has page metadata.
    # This keeps retrieval relevance aligned with citation-grounded evaluation.
    if ev_pages:
        if c_pages:
            return bool(c_pages & ev_pages)
        # Fallback when chunk lacks page metadata: require strong lexical overlap.
        if ev_tokens and c_text_tokens:
            overlap = len(ev_tokens & c_text_tokens) / max(1, len(ev_tokens))
            return overlap >= 0.60
        return False

    ev_header = str(evidence.get("source_section_header", "")).strip().lower()
    if ev_header and c_header and ev_header in c_header:
        return True

    if ev_tokens and c_text_tokens:
        overlap = len(ev_tokens & c_text_tokens) / max(1, len(ev_tokens))
        if overlap >= 0.40:
            return True

    return False


def compute_retrieval_metrics(
    retrieved_chunks: List[Dict[str, Any]],
    gold_evidence: List[Dict[str, Any]],
    k_values: Sequence[int] = (5, 10, 15),
) -> Dict[str, float]:
    """Compute retrieval metrics required by README."""
    metrics: Dict[str, float] = {}

    gold_pages = _gold_pages(gold_evidence)
    relevant_flags = [_chunk_relevant(chunk, gold_evidence) for chunk in retrieved_chunks]

    for k in k_values:
        top_chunks = retrieved_chunks[:k]
        top_flags = relevant_flags[:k]

        # Recall@k:
        # - page-based when page references exist in gold
        # - evidence-hit fallback when gold page refs are unavailable
        if gold_pages:
            top_pages: Set[int] = set()
            for ch in top_chunks:
                top_pages.update(_chunk_pages(ch))
            recall = len(gold_pages & top_pages) / len(gold_pages)
        elif gold_evidence:
            matched_evidence = sum(
                1 for ev in gold_evidence if any(_chunk_matches_evidence(ch, ev) for ch in top_chunks)
            )
            recall = matched_evidence / len(gold_evidence)
        else:
            recall = 0.0

        # Chunk-level precision at k using relevance predicate.
        precision = sum(1 for flag in top_flags if flag) / max(1, len(top_flags))

        # MRR at k on chunk relevance.
        mrr = 0.0
        for rank, is_rel in enumerate(top_flags, start=1):
            if is_rel:
                mrr = 1.0 / rank
                break

        metrics[f"recall@{k}"] = recall
        metrics[f"precision@{k}"] = precision
        metrics[f"mrr@{k}"] = mrr

    # Coverage ratio: fraction of gold evidence entries hit by any retrieved chunk.
    covered = sum(
        1 for ev in gold_evidence if any(_chunk_matches_evidence(ch, ev) for ch in retrieved_chunks)
    )

    metrics["coverage_ratio"] = covered / max(1, len(gold_evidence))
    return metrics


# ---------------------------------------------------------------------------
# Answer quality metrics
# ---------------------------------------------------------------------------

def compute_token_f1(prediction: str, reference: str) -> Dict[str, float]:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return {"token_precision": 0.0, "token_recall": 0.0, "token_f1": 0.0}

    from collections import Counter

    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)
    common = sum((pred_counter & ref_counter).values())

    precision = common / max(1, len(pred_tokens))
    recall = common / max(1, len(ref_tokens))
    f1 = 2 * precision * recall / max(1e-9, (precision + recall))

    return {
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": f1,
    }


def compute_bertscore(prediction: str, reference: str) -> float:
    """Compute BERTScore F1; fallback to embedding cosine if unavailable."""
    try:
        from bert_score import score as bert_score

        _, _, f1 = bert_score([prediction], [reference], lang="en", verbose=False)
        return float(f1[0].item())
    except Exception:
        # Fallback to embedding cosine for deterministic availability.
        try:
            from rag.embedder import embed_texts

            vecs = embed_texts([prediction[:3000], reference[:3000]])
            v1 = vecs[0] / (np.linalg.norm(vecs[0]) + 1e-10)
            v2 = vecs[1] / (np.linalg.norm(vecs[1]) + 1e-10)
            return float(np.dot(v1, v2))
        except Exception:
            return compute_token_f1(prediction, reference)["token_f1"]


def llm_judge_score(query: str, prediction: str, reference: str) -> Dict[str, Any]:
    """Use LLM-as-a-judge with persistent caching."""
    cache_dir = EVAL_OUTPUT_DIR / "llm_judge_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    key_raw = f"{LLM_MODEL}::{query}::{prediction[:1500]}::{reference[:1500]}"
    key = hashlib.sha256(key_raw.encode("utf-8")).hexdigest()
    cache_file = cache_dir / f"{key}.json"
    if cache_file.exists():
        with cache_file.open("r", encoding="utf-8") as f:
            return json.load(f)

    system = (
        "You are evaluating generated ESRS answers."
        "Return strict JSON with keys: score (1-5), rationale."
        "Score based on correctness, completeness, and grounding against reference."
    )
    user = (
        f"Question: {query}\n\n"
        f"Reference answer:\n{reference[:4000]}\n\n"
        f"Generated answer:\n{prediction[:4000]}"
    )
    parsed, _usage = _import_chat_json()(
        system_prompt=system,
        user_prompt=user,
        model=LLM_MODEL,
        temperature=0.0,
        max_completion_tokens=300,
        max_retries=2,
    )

    if not isinstance(parsed, dict):
        parsed = {"score": 0, "rationale": "Judge call failed"}

    score = parsed.get("score", 0)
    try:
        score = int(score)
    except (TypeError, ValueError):
        score = 0
    parsed["score"] = max(0, min(5, score))

    with cache_file.open("w", encoding="utf-8") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)
    return parsed


def compute_citation_correctness(page_references: List[Dict[str, Any]], gold_evidence: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compare cited pages against gold evidence pages."""
    gold_pages = _gold_pages(gold_evidence)
    cited_pages: Set[int] = set()

    for ref in page_references:
        for p in ref.get("pages", []) or []:
            if isinstance(p, int) and p > 0:
                cited_pages.add(p)

    if not cited_pages and not gold_pages:
        return {
            "citation_precision": 1.0,
            "citation_recall": 1.0,
            "citation_f1": 1.0,
        }

    tp = len(cited_pages & gold_pages)
    precision = tp / max(1, len(cited_pages))
    recall = tp / max(1, len(gold_pages))
    f1 = 2 * precision * recall / max(1e-9, (precision + recall))

    return {
        "citation_precision": precision,
        "citation_recall": recall,
        "citation_f1": f1,
    }


# ---------------------------------------------------------------------------
# Hallucination metrics
# ---------------------------------------------------------------------------

def _sentence_split(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    # Split by punctuation while preserving concise claims.
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def compute_faithfulness(generated_answer: str, retrieved_passages: List[str]) -> Dict[str, float]:
    """Compute claim support ratio and hallucination rate against retrieved evidence."""
    claims = [s for s in _sentence_split(generated_answer) if len(_tokenize(s)) >= 5]
    passages = [_content_tokens(p) for p in retrieved_passages if p]

    if not claims:
        return {
            "faithfulness_score": 0.0,
            "supported_claim_ratio": 0.0,
            "hallucination_rate": 1.0,
        }

    claim_scores: List[float] = []
    supported = 0

    for claim in claims:
        claim_tokens = _content_tokens(claim)
        if not claim_tokens:
            claim_scores.append(0.0)
            continue

        best_overlap = 0.0
        for p_tokens in passages:
            if not p_tokens:
                continue
            overlap = len(claim_tokens & p_tokens) / len(claim_tokens)
            best_overlap = max(best_overlap, overlap)

        claim_scores.append(best_overlap)
        if best_overlap >= 0.35:
            supported += 1

    supported_ratio = supported / len(claims)
    faithfulness = _safe_mean(claim_scores)

    return {
        "faithfulness_score": faithfulness,
        "supported_claim_ratio": supported_ratio,
        "hallucination_rate": 1.0 - supported_ratio,
    }


# ---------------------------------------------------------------------------
# Efficiency metrics
# ---------------------------------------------------------------------------

def compute_efficiency_metrics(latency_seconds: float, token_usage: Dict[str, int], n_chunks_retrieved: int) -> Dict[str, float]:
    """Compute latency and rough token-cost metrics."""
    prompt_tokens = int(token_usage.get("prompt_tokens", 0))
    completion_tokens = int(token_usage.get("completion_tokens", 0))
    total_tokens = int(token_usage.get("total_tokens", prompt_tokens + completion_tokens))

    # Conservative estimate in USD (placeholder rates configurable later).
    # Input: $2.5 / 1M tokens, Output: $10 / 1M tokens.
    estimated_cost = (prompt_tokens * 2.5 + completion_tokens * 10.0) / 1_000_000

    return {
        "latency_seconds": float(latency_seconds),
        "prompt_tokens": float(prompt_tokens),
        "completion_tokens": float(completion_tokens),
        "total_tokens": float(total_tokens),
        "estimated_cost_usd": float(estimated_cost),
        "n_chunks_retrieved": float(n_chunks_retrieved),
    }


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def _question_lookup(golden_standard: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {q.get("id", ""): q for q in golden_standard.get("questions", [])}


def _rq_group(config_name: str) -> str:
    if config_name.startswith("RQ1"):
        return "RQ1"
    if config_name.startswith("RQ2"):
        return "RQ2"
    if config_name.startswith("RQ3"):
        return "RQ3"
    return "Other"


def evaluate_pipeline_config(
    config_name: str,
    retriever: Any,
    config: Dict[str, Any],
    golden_standard: Dict[str, Any],
    question_ids: Optional[List[str]] = None,
    companies: Optional[List[str]] = None,
    skip_llm_judge: bool = False,
    skip_bertscore: bool = False,
) -> Dict[str, Any]:
    """Evaluate one pipeline configuration end-to-end."""
    from rag.esrs_context import get_esrs_requirements_text
    from rag.pipeline import _normalize_company_names
    from rag.retriever import Retriever as _Retriever

    q_lookup = _question_lookup(golden_standard)
    selected_questions = question_ids or BENCHMARK_QUESTION_IDS
    companies = _normalize_company_names(companies)

    active_retriever = retriever
    if active_retriever is None:
        from rag.pipeline import build_index

        vec_index, bm25_index, _chunks = build_index(
            config=config,
            force_rebuild=False,
            companies=companies,
        )
        active_retriever = _Retriever(vec_index, bm25_index, config)

    per_sample: List[Dict[str, Any]] = []
    per_question: Dict[str, Dict[str, Any]] = {}
    per_question_results: List[Dict[str, Any]] = []

    for qid in selected_questions:
        question = q_lookup.get(qid)
        if not question:
            continue

        query = question.get("question", "")
        esrs_standard = question.get("esrs_standard", "")
        esrs_context = get_esrs_requirements_text(query=query, esrs_standard=esrs_standard)

        per_question[qid] = {}

        for company_entry in question.get("per_company", []):
            company = company_entry.get("company", "")
            if companies and company not in companies:
                continue

            gold_answer = str(company_entry.get("ground_truth_answer", "")).strip()
            if not gold_answer:
                continue

            gold_evidence = company_entry.get("evidence", []) or []
            gold_fulfillment = str(company_entry.get("fulfillment_level", "")).strip()

            t0 = time.time()
            retrieved = active_retriever.retrieve(query, company=company)
            retrieval_latency = time.time() - t0

            ret_metrics = compute_retrieval_metrics(retrieved, gold_evidence)

            t1 = time.time()
            answer = _import_generate_answer()(
                query=query,
                retrieved_chunks=retrieved,
                esrs_context=esrs_context,
                company=company,
                config=config,
            )
            generation_latency = time.time() - t1

            direct_answer = str(answer.get("direct_answer", ""))
            token_metrics = compute_token_f1(direct_answer, gold_answer)
            bertscore = 0.0 if skip_bertscore else compute_bertscore(direct_answer, gold_answer)
            judge = {"score": 0, "rationale": "skipped"}
            if not skip_llm_judge:
                judge = llm_judge_score(query, direct_answer, gold_answer)
            judge_score_raw = float(judge.get("score", 0))
            judge_score_normalized = max(
                0.0,
                min((judge_score_raw / 5.0) if judge_score_raw > 1.0 else judge_score_raw, 1.0),
            )

            citation_metrics = compute_citation_correctness(answer.get("page_references", []), gold_evidence)
            hallucination_metrics = compute_faithfulness(direct_answer, [c.get("text", "") for c in retrieved])
            efficiency = compute_efficiency_metrics(
                latency_seconds=retrieval_latency + generation_latency,
                token_usage=answer.get("_token_usage", {}),
                n_chunks_retrieved=len(retrieved),
            )

            predicted_fulfillment = str(answer.get("compliance_level", "")).strip()
            compliance_match = bool(predicted_fulfillment and gold_fulfillment and predicted_fulfillment.lower() == gold_fulfillment.lower())

            sample = {
                "config_name": config_name,
                "rq_group": _rq_group(config_name),
                "question_id": qid,
                "question": query,
                "company": company,
                "esrs_standard": esrs_standard,
                "gold_fulfillment": gold_fulfillment,
                "predicted_fulfillment": predicted_fulfillment,
                "compliance_match": compliance_match,
                "retrieval": ret_metrics,
                "answer_quality": {
                    **token_metrics,
                    "bertscore": bertscore,
                    "llm_judge_score": judge_score_raw,
                    "llm_judge_score_normalized": judge_score_normalized,
                    "llm_judge_rationale": str(judge.get("rationale", "")),
                },
                "citation": citation_metrics,
                "hallucination": hallucination_metrics,
                "efficiency": efficiency,
                "retrieved_chunk_count": len(retrieved),
                "sources_used": answer.get("sources_used", []),
                "page_references": answer.get("page_references", []),
                "generated_answer_preview": direct_answer[:700],
            }
            per_sample.append(sample)
            per_question[qid][company] = sample
            per_question_results.append(
                {
                    "question_id": qid,
                    "company": company,
                    "query": query,
                    "generated_answer": direct_answer,
                    "reference_answer": gold_answer,
                    "predicted_fulfillment": predicted_fulfillment,
                    "gold_fulfillment": gold_fulfillment,
                }
            )

            print(
                f"    {qid}/{company}: R@15={ret_metrics.get('recall@15', 0):.3f} "
                f"F1={token_metrics.get('token_f1', 0):.3f} "
                f"CitF1={citation_metrics.get('citation_f1', 0):.3f}"
            )

    # Aggregate metrics for this config.
    def collect(path: Tuple[str, ...]) -> List[float]:
        vals: List[float] = []
        for row in per_sample:
            cur: Any = row
            for key in path:
                if not isinstance(cur, dict):
                    cur = None
                    break
                cur = cur.get(key)
            if isinstance(cur, (int, float)):
                vals.append(float(cur))
        return vals

    aggregate = {
        "n_evaluated": len(per_sample),
        "retrieval": {
            "avg_recall@5": _safe_mean(collect(("retrieval", "recall@5"))),
            "avg_recall@10": _safe_mean(collect(("retrieval", "recall@10"))),
            "avg_recall@15": _safe_mean(collect(("retrieval", "recall@15"))),
            "avg_precision@5": _safe_mean(collect(("retrieval", "precision@5"))),
            "avg_precision@10": _safe_mean(collect(("retrieval", "precision@10"))),
            "avg_precision@15": _safe_mean(collect(("retrieval", "precision@15"))),
            "avg_mrr@5": _safe_mean(collect(("retrieval", "mrr@5"))),
            "avg_mrr@10": _safe_mean(collect(("retrieval", "mrr@10"))),
            "avg_mrr@15": _safe_mean(collect(("retrieval", "mrr@15"))),
            "avg_coverage_ratio": _safe_mean(collect(("retrieval", "coverage_ratio"))),
        },
        "answer_quality": {
            "avg_token_f1": _safe_mean(collect(("answer_quality", "token_f1"))),
            "avg_token_precision": _safe_mean(collect(("answer_quality", "token_precision"))),
            "avg_token_recall": _safe_mean(collect(("answer_quality", "token_recall"))),
            "avg_bertscore": _safe_mean(collect(("answer_quality", "bertscore"))),
            "avg_llm_judge": _safe_mean(collect(("answer_quality", "llm_judge_score"))),
            "avg_llm_judge_normalized": _safe_mean(collect(("answer_quality", "llm_judge_score_normalized"))),
            "compliance_accuracy": _safe_mean([1.0 if row["compliance_match"] else 0.0 for row in per_sample]),
        },
        "citation": {
            "avg_citation_precision": _safe_mean(collect(("citation", "citation_precision"))),
            "avg_citation_recall": _safe_mean(collect(("citation", "citation_recall"))),
            "avg_citation_f1": _safe_mean(collect(("citation", "citation_f1"))),
        },
        "hallucination": {
            "avg_faithfulness": _safe_mean(collect(("hallucination", "faithfulness_score"))),
            "avg_supported_claim_ratio": _safe_mean(collect(("hallucination", "supported_claim_ratio"))),
            "avg_hallucination_rate": _safe_mean(collect(("hallucination", "hallucination_rate"))),
        },
        "efficiency": {
            "avg_latency": _safe_mean(collect(("efficiency", "latency_seconds"))),
            "avg_prompt_tokens": _safe_mean(collect(("efficiency", "prompt_tokens"))),
            "avg_total_tokens": _safe_mean(collect(("efficiency", "total_tokens"))),
            "avg_cost_usd": _safe_mean(collect(("efficiency", "estimated_cost_usd"))),
            "avg_chunks_retrieved": _safe_mean(collect(("efficiency", "n_chunks_retrieved"))),
        },
    }
    # Backward-compatible flattened aliases used by existing notebooks.
    aggregate["answer_token_f1"] = aggregate["answer_quality"]["avg_token_f1"]
    aggregate["answer_citation_f1"] = aggregate["citation"]["avg_citation_f1"]
    aggregate["answer_faithfulness_score"] = aggregate["hallucination"]["avg_faithfulness"]

    return {
        "config_name": config_name,
        "rq_group": _rq_group(config_name),
        "config": config,
        "per_question": per_question,
        "per_question_results": per_question_results,
        "per_sample": per_sample,
        "aggregate": aggregate,
    }


# ---------------------------------------------------------------------------
# Approach B: LLM-as-a-judge with explicit rubrics
# ---------------------------------------------------------------------------

LLM_JUDGE_RUBRIC_SYSTEM = """You are a rigorous evaluator of ESRS sustainability report answers.

Score the generated answer on EACH of the following five dimensions using a 1-5 scale.

1. FACTUAL_CORRECTNESS (1-5): Are the stated facts accurate compared to the reference?
   1=Mostly wrong, 2=Several errors, 3=Mostly correct with minor errors,
   4=Correct with negligible issues, 5=Fully correct

2. COMPLETENESS (1-5): Does the answer cover all key points from the reference?
   1=Major gaps, 2=Covers <50%, 3=Covers ~50-70%, 4=Covers ~70-90%, 5=Comprehensive

3. GROUNDING_FAITHFULNESS (1-5): Is the answer grounded in retrieved passages only?
   1=Mostly hallucinated, 2=Significant ungrounded claims, 3=Some unverifiable claims,
   4=Almost fully grounded, 5=Fully grounded in evidence

4. CITATION_QUALITY (1-5): Are sources/pages cited correctly and helpfully?
   1=No citations, 2=Sparse/wrong citations, 3=Some correct citations,
   4=Good citations with minor gaps, 5=Precise and complete citations

5. COMPLIANCE_ASSESSMENT (1-5): Is the ESRS compliance evaluation appropriate?
   1=Completely wrong level, 2=Off by one level with poor rationale,
   3=Reasonable but imprecise, 4=Correct level with good rationale,
   5=Correct level with thorough, well-evidenced rationale

Return strict JSON:
{
  "factual_correctness": {"score": int, "rationale": str},
  "completeness": {"score": int, "rationale": str},
  "grounding_faithfulness": {"score": int, "rationale": str},
  "citation_quality": {"score": int, "rationale": str},
  "compliance_assessment": {"score": int, "rationale": str},
  "overall_score": float,
  "overall_rationale": str
}

overall_score = mean of the five dimension scores.
"""


def llm_judge_detailed(
    query: str,
    prediction: str,
    reference: str,
    retrieved_passages: Optional[List[str]] = None,
    predicted_compliance: str = "",
    gold_compliance: str = "",
) -> Dict[str, Any]:
    """Approach B: Multi-dimensional LLM-as-a-judge with explicit rubrics and caching."""
    cache_dir = EVAL_OUTPUT_DIR / "llm_judge_detailed_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    passages_sig = "||".join((p or "")[:250] for p in (retrieved_passages or [])[:10])
    key_raw = (
        f"{LLM_MODEL}::detailed::{query}::{prediction[:1500]}::{reference[:1500]}"
        f"::{predicted_compliance}::{gold_compliance}::{passages_sig}"
    )
    key = hashlib.sha256(key_raw.encode("utf-8")).hexdigest()
    cache_file = cache_dir / f"{key}.json"
    if cache_file.exists():
        with cache_file.open("r", encoding="utf-8") as f:
            return json.load(f)

    passages_text = ""
    if retrieved_passages:
        passages_text = "\n---\n".join(p[:500] for p in retrieved_passages[:10])

    user_prompt = (
        f"Question: {query}\n\n"
        f"Reference answer:\n{reference[:4000]}\n\n"
        f"Generated answer:\n{prediction[:4000]}\n\n"
    )
    if passages_text:
        user_prompt += f"Retrieved passages (for grounding check):\n{passages_text[:3000]}\n\n"
    if predicted_compliance or gold_compliance:
        user_prompt += (
            f"Predicted compliance level: {predicted_compliance}\n"
            f"Gold compliance level: {gold_compliance}\n"
        )

    parsed, _usage = _import_chat_json()(
        system_prompt=LLM_JUDGE_RUBRIC_SYSTEM,
        user_prompt=user_prompt,
        model=LLM_MODEL,
        temperature=0.0,
        max_completion_tokens=800,
        max_retries=2,
    )

    if not isinstance(parsed, dict):
        parsed = {
            "factual_correctness": {"score": 0, "rationale": "Judge call failed"},
            "completeness": {"score": 0, "rationale": "Judge call failed"},
            "grounding_faithfulness": {"score": 0, "rationale": "Judge call failed"},
            "citation_quality": {"score": 0, "rationale": "Judge call failed"},
            "compliance_assessment": {"score": 0, "rationale": "Judge call failed"},
            "overall_score": 0.0,
            "overall_rationale": "Judge call failed",
        }

    # Ensure scores are ints in [0,5]
    for dim in ["factual_correctness", "completeness", "grounding_faithfulness",
                 "citation_quality", "compliance_assessment"]:
        entry = parsed.get(dim, {})
        if isinstance(entry, dict):
            try:
                entry["score"] = max(0, min(5, int(entry.get("score", 0))))
            except (TypeError, ValueError):
                entry["score"] = 0
            parsed[dim] = entry
        else:
            parsed[dim] = {"score": 0, "rationale": "Missing dimension"}

    # Compute overall
    dim_scores = [parsed[d]["score"] for d in [
        "factual_correctness", "completeness", "grounding_faithfulness",
        "citation_quality", "compliance_assessment"
    ]]
    parsed["overall_score"] = sum(dim_scores) / max(1, len(dim_scores))

    with cache_file.open("w", encoding="utf-8") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)
    return parsed


def select_best_pipeline(all_results: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Select the best pipeline configuration based on a weighted composite score.

    Weighting (thesis-aligned):
      - Retrieval recall@15: 0.20
      - Token F1: 0.15
      - BERTScore: 0.10
      - Citation F1: 0.15
      - Faithfulness: 0.15
      - LLM judge (normalized to 0-1): 0.15
      - Cost efficiency (inverted): 0.10

    Returns:
        Tuple of (best_config_name, result_dict)
    """
    weights = {
        "retrieval_avg_recall@15": 0.20,
        "answer_quality_avg_token_f1": 0.15,
        "answer_quality_avg_bertscore": 0.10,
        "citation_avg_citation_f1": 0.15,
        "hallucination_avg_faithfulness": 0.15,
        "answer_quality_avg_llm_judge_normalized": 0.15,
        "efficiency_cost_inverted": 0.10,
    }

    scored: List[Tuple[str, float, Dict[str, float]]] = []

    for name, result in all_results.items():
        agg = result.get("aggregate", {})
        if agg.get("n_evaluated", 0) == 0:
            continue

        metrics = {}
        for category in ["retrieval", "answer_quality", "citation", "hallucination", "efficiency"]:
            for k, v in agg.get(category, {}).items():
                metrics[f"{category}_{k}"] = float(v)

        judge_raw = metrics.get("answer_quality_avg_llm_judge", 0.0)
        judge_normalized = (judge_raw / 5.0) if judge_raw > 1.0 else judge_raw
        metrics["answer_quality_avg_llm_judge_normalized"] = max(0.0, min(judge_normalized, 1.0))

        # Invert cost for scoring (lower cost = higher score; normalize to ~[0,1])
        avg_cost = metrics.get("efficiency_avg_cost_usd", 0.05)
        metrics["efficiency_cost_inverted"] = max(0.0, 1.0 - min(avg_cost * 10, 1.0))

        composite = 0.0
        for metric_key, weight in weights.items():
            val = metrics.get(metric_key, 0.0)
            composite += weight * val

        metrics["composite_score"] = composite
        scored.append((name, composite, metrics))

    if not scored:
        # Fallback
        first_name = next(iter(all_results))
        return first_name, all_results[first_name]

    scored.sort(key=lambda x: x[1], reverse=True)
    best_name = scored[0][0]
    return best_name, all_results[best_name]


# ---------------------------------------------------------------------------
# Reporting and output files
# ---------------------------------------------------------------------------

def _flatten_aggregate_row(config_name: str, result: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "config": config_name,
        "rq_group": result.get("rq_group", "Other"),
        "n_evaluated": result.get("aggregate", {}).get("n_evaluated", 0),
    }
    aggregate = result.get("aggregate", {})
    for category in ["retrieval", "answer_quality", "citation", "hallucination", "efficiency"]:
        for key, value in aggregate.get(category, {}).items():
            row[f"{category}_{key}"] = value
    return row


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = sorted({k for row in rows for k in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_evaluation_results(all_results: Dict[str, Any], output_dir: Path = EVAL_OUTPUT_DIR) -> None:
    """Persist evaluation results in JSON, CSV, and Markdown formats."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "evaluation_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    aggregate_rows = [_flatten_aggregate_row(name, result) for name, result in all_results.items()]
    _write_csv(output_dir / "evaluation_summary.csv", aggregate_rows)

    sample_rows: List[Dict[str, Any]] = []
    for name, result in all_results.items():
        for sample in result.get("per_sample", []):
            row = {
                "config": name,
                "rq_group": result.get("rq_group", "Other"),
                "question_id": sample.get("question_id", ""),
                "company": sample.get("company", ""),
                "gold_fulfillment": sample.get("gold_fulfillment", ""),
                "predicted_fulfillment": sample.get("predicted_fulfillment", ""),
                "compliance_match": sample.get("compliance_match", False),
                "retrieval_recall@15": sample.get("retrieval", {}).get("recall@15", 0),
                "retrieval_precision@15": sample.get("retrieval", {}).get("precision@15", 0),
                "retrieval_mrr@15": sample.get("retrieval", {}).get("mrr@15", 0),
                "retrieval_coverage_ratio": sample.get("retrieval", {}).get("coverage_ratio", 0),
                "answer_token_f1": sample.get("answer_quality", {}).get("token_f1", 0),
                "answer_bertscore": sample.get("answer_quality", {}).get("bertscore", 0),
                "answer_llm_judge": sample.get("answer_quality", {}).get("llm_judge_score", 0),
                "citation_f1": sample.get("citation", {}).get("citation_f1", 0),
                "hallucination_rate": sample.get("hallucination", {}).get("hallucination_rate", 0),
                "faithfulness_score": sample.get("hallucination", {}).get("faithfulness_score", 0),
                "latency_seconds": sample.get("efficiency", {}).get("latency_seconds", 0),
                "total_tokens": sample.get("efficiency", {}).get("total_tokens", 0),
                "estimated_cost_usd": sample.get("efficiency", {}).get("estimated_cost_usd", 0),
                "retrieved_chunk_count": sample.get("retrieved_chunk_count", 0),
            }
            sample_rows.append(row)
    _write_csv(output_dir / "evaluation_per_sample.csv", sample_rows)

    md_path = output_dir / "evaluation_summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# ESRS RAG Evaluation Summary\n\n")
        f.write(f"Configurations evaluated: {len(all_results)}\n\n")
        f.write("## Aggregate Results by Configuration\n\n")

        for name, result in all_results.items():
            agg = result.get("aggregate", {})
            f.write(f"### {name} ({result.get('rq_group', 'Other')})\n\n")
            f.write(f"Evaluated samples: {agg.get('n_evaluated', 0)}\n\n")

            for category in ["retrieval", "answer_quality", "citation", "hallucination", "efficiency"]:
                f.write(f"**{category.replace('_', ' ').title()}**\n\n")
                f.write("| Metric | Value |\n")
                f.write("|---|---:|\n")
                for key, value in agg.get(category, {}).items():
                    f.write(f"| {key} | {value:.4f} |\n")
                f.write("\n")

        # RQ-level compact section
        f.write("## Compact RQ Comparison\n\n")
        rq_groups: Dict[str, List[Dict[str, Any]]] = {}
        for row in aggregate_rows:
            rq_groups.setdefault(str(row.get("rq_group", "Other")), []).append(row)

        for rq, rows in sorted(rq_groups.items()):
            f.write(f"### {rq}\n\n")
            f.write("| Config | Recall@15 | Token F1 | Citation F1 | Hallucination Rate | Cost USD |\n")
            f.write("|---|---:|---:|---:|---:|---:|\n")
            for row in rows:
                f.write(
                    "| "
                    f"{row.get('config', '')} | "
                    f"{float(row.get('retrieval_avg_recall@15', 0)):.4f} | "
                    f"{float(row.get('answer_quality_avg_token_f1', 0)):.4f} | "
                    f"{float(row.get('citation_avg_citation_f1', 0)):.4f} | "
                    f"{float(row.get('hallucination_avg_hallucination_rate', 0)):.4f} | "
                    f"{float(row.get('efficiency_avg_cost_usd', 0)):.6f} |\n"
                )
            f.write("\n")

    print(f"\nSaved evaluation JSON: {json_path}")
    print(f"Saved evaluation CSV:  {output_dir / 'evaluation_summary.csv'}")
    print(f"Saved sample CSV:      {output_dir / 'evaluation_per_sample.csv'}")
    print(f"Saved Markdown summary:{md_path}")
