"""LLM-as-a-judge module for RQ3: multi-dimensional generation quality evaluation.

This module uses Claude Sonnet 4.5 as a G-Eval-style, reference-based,
pointwise, rubric-based judge to evaluate generated ESRS compliance answers
across five quality dimensions in a SINGLE API call per sample.

Design rationale
----------------
RQ3 is a single-condition characterisation study of the best pipeline.
Unlike RQ2 (which evaluates rationale alignment on one dimension), RQ3
evaluates five orthogonal quality dimensions simultaneously:

  1. Factual correctness   — accuracy of stated facts vs gold answer
  2. Completeness          — coverage of key points from the gold answer
  3. Grounding faithfulness — whether claims are grounded in retrieved evidence
  4. Citation usefulness   — relevance and sufficiency of cited pages
  5. Compliance assessment quality — correctness of the Low/Medium/High judgment

All five scores are returned in a single JSON response to minimise API cost
(~204 calls for the full dataset instead of ~1020).

Anti-anchoring: the generated compliance level is intentionally WITHHELD
from the judge prompt (same principle as RQ2 v2 judge). The judge evaluates
answer quality independently of whether the level prediction matched.

References:
  - Liu et al. (2023). "G-Eval: NLG Evaluation using GPT-4 with Better
    Human Alignment."  EMNLP 2023.
  - Zheng et al. (2024). "Judging LLM-as-a-Judge with MT-Bench and
    Chatbot Arena."  NeurIPS 2024.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests

from rag.config.settings import ARTIFACTS_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JUDGE_MODEL = "claude-sonnet-4-5-20250929"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_VERSION = "2023-06-01"

JUDGE_PROMPT_VERSION = "rq3_v1"

VALID_WEAKNESSES = frozenset({
    "none",
    "unsupported_claim",
    "incomplete_answer",
    "wrong_compliance_level",
    "weak_citation_support",
    "factual_error",
})

DIMENSION_NAMES = (
    "factual_correctness",
    "completeness",
    "grounding_faithfulness",
    "citation_usefulness",
    "compliance_assessment_quality",
)

JUDGE_CACHE_DIR = ARTIFACTS_DIR / "evaluation_results" / "rq3" / "judge_cache"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RQ3JudgeInput:
    """All context the judge needs for one RQ3 sample."""

    question: str
    esrs_standard: str
    esrs_disclosure_id: str
    company: str
    # Gold reference
    gold_fulfillment_level: str
    gold_fulfillment_rationale: str
    gold_ground_truth_answer: str
    # Generated answer (compliance_level intentionally excluded)
    generated_direct_answer: str
    generated_compliance_rationale: str
    generated_key_findings: list[str]
    generated_disclosure_gaps: list[str]
    generated_cited_pages: list[str]


@dataclass
class RQ3JudgeOutput:
    """Structured multi-dimensional judge scores for one sample."""

    factual_correctness: int          # 1-5
    completeness: int                 # 1-5
    grounding_faithfulness: int       # 1-5
    citation_usefulness: int          # 1-5
    compliance_assessment_quality: int  # 1-5
    mean_score: float                 # arithmetic mean of all 5
    reasoning: str = ""
    primary_weakness: str = "none"
    judge_explanation: str = ""


# ---------------------------------------------------------------------------
# Prompt template — G-Eval style, all 5 dimensions in one call
# ---------------------------------------------------------------------------

_JUDGE_PROMPT_TEMPLATE = """\
You are an expert judge evaluating the quality of a generated answer for an \
ESRS sustainability disclosure question. Evaluate the answer across five \
dimensions.

## Input

**Question:** {question}
**ESRS Standard:** {esrs_standard}
**Disclosure ID(s):** {esrs_disclosure_id}
**Company:** {company}

**Gold Fulfillment Level:** {gold_fulfillment_level}

**Gold Fulfillment Rationale:**
{gold_fulfillment_rationale}

**Gold Ground-Truth Answer:**
{gold_ground_truth_answer}

**Generated Direct Answer:**
{generated_direct_answer}

**Generated Compliance Rationale:**
{generated_compliance_rationale}

**Generated Key Findings:**
{generated_key_findings}

**Generated Disclosure Gaps:**
{generated_disclosure_gaps}

**Generated Cited Pages:**
{generated_cited_pages}

NOTE: The generated compliance level is intentionally WITHHELD to prevent \
anchoring bias. Evaluate answer quality independently.

## Evaluation Steps (follow each step in order)

1. Read the gold answer and identify all key factual claims and evidence points.
2. For each dimension below, evaluate the generated answer against the rubric.
3. Assign a score (1-5) for each dimension.
4. Identify the single most significant weakness category.

## Dimensions and Rubrics (each scored 1-5)

### 1. Factual Correctness
[5] All stated facts match the gold answer; no fabrications
[4] Nearly all facts correct; at most one minor inaccuracy
[3] Most facts correct but one notable error or imprecision
[2] Multiple factual errors or one major contradiction
[1] Predominantly incorrect or fabricated claims

### 2. Completeness
[5] All key points from the gold answer are covered
[4] Most key points covered; one minor omission
[3] Main points present but notable gaps in coverage
[2] Significant omissions of key information
[1] Most critical information missing

### 3. Grounding Faithfulness
[5] Every claim is traceable to retrieved evidence; no unsupported assertions
[4] Nearly all claims grounded; at most one minor unsupported claim
[3] Most claims grounded but some unsupported assertions present
[2] Multiple claims lack grounding in retrieved evidence
[1] Answer introduces substantial information not in the evidence

### 4. Citation Usefulness
[5] Cited pages are specific, relevant, and sufficient for verification
[4] Citations mostly useful; one less relevant or missing citation
[3] Some citations helpful but gaps in coverage or relevance
[2] Citations are largely unhelpful or incomplete
[1] No useful citations or predominantly wrong page references

### 5. Compliance Assessment Quality
[5] Compliance judgment matches gold level with comprehensive, well-evidenced reasoning
[4] Judgment is correct or reasonably justified; minor reasoning gaps
[3] Judgment is plausible but reasoning has notable weaknesses
[2] Judgment appears incorrect or poorly supported
[1] Judgment clearly wrong with contradictory or absent reasoning

## Output

Return ONLY valid JSON (no markdown fences, no text outside the JSON object):

{{"reasoning": "<your step-by-step evaluation, 3-5 sentences>", \
"factual_correctness": <1-5>, \
"completeness": <1-5>, \
"grounding_faithfulness": <1-5>, \
"citation_usefulness": <1-5>, \
"compliance_assessment_quality": <1-5>, \
"primary_weakness": "<one of: none, unsupported_claim, incomplete_answer, wrong_compliance_level, weak_citation_support, factual_error>", \
"judge_explanation": "<max 50 words: concise summary>"}}"""


def _prompt_hash() -> str:
    """Content hash of the prompt template for cache invalidation."""
    return hashlib.sha256(_JUDGE_PROMPT_TEMPLATE.encode()).hexdigest()[:10]


def get_judge_version() -> str:
    """Combined version string: prompt version + prompt content hash."""
    return f"{JUDGE_PROMPT_VERSION}_{_prompt_hash()}"


# ---------------------------------------------------------------------------
# Cache helpers (same architecture as rq2_judge.py)
# ---------------------------------------------------------------------------

def _cache_key(inp: RQ3JudgeInput, version: str) -> str:
    """Content-hashed cache key combining all judge inputs + prompt version."""
    blob = (
        f"{inp.question}|{inp.company}|"
        f"{inp.gold_fulfillment_level}|{inp.gold_fulfillment_rationale}|"
        f"{inp.gold_ground_truth_answer}|"
        f"{inp.generated_direct_answer}|{inp.generated_compliance_rationale}|"
        f"{'|'.join(inp.generated_key_findings)}|"
        f"{'|'.join(inp.generated_disclosure_gaps)}|"
        f"{'|'.join(inp.generated_cited_pages)}|"
        f"{version}"
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def _load_cached(key: str) -> RQ3JudgeOutput | None:
    path = JUDGE_CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text("utf-8"))
        o = data["judge_output"]
        return RQ3JudgeOutput(
            factual_correctness=int(o["factual_correctness"]),
            completeness=int(o["completeness"]),
            grounding_faithfulness=int(o["grounding_faithfulness"]),
            citation_usefulness=int(o["citation_usefulness"]),
            compliance_assessment_quality=int(o["compliance_assessment_quality"]),
            mean_score=float(o["mean_score"]),
            reasoning=o.get("reasoning", ""),
            primary_weakness=o.get("primary_weakness", "none"),
            judge_explanation=o.get("judge_explanation", ""),
        )
    except Exception as exc:
        logger.warning("Corrupt RQ3 cache %s: %s", key, exc)
        return None


def _save_cache(key: str, out: RQ3JudgeOutput, inp: RQ3JudgeInput) -> None:
    JUDGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_key": key,
        "judge_version": get_judge_version(),
        "judge_input": asdict(inp),
        "judge_output": asdict(out),
    }
    try:
        (JUDGE_CACHE_DIR / f"{key}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), "utf-8",
        )
    except Exception as exc:
        logger.error("RQ3 cache write failed for %s: %s", key, exc)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(raw: dict[str, Any]) -> tuple[bool, str]:
    """Validate raw judge JSON against the expected RQ3 schema."""
    for dim in DIMENSION_NAMES:
        if dim not in raw:
            return False, f"missing dimension: {dim}"
        try:
            s = int(raw[dim])
        except (TypeError, ValueError):
            return False, f"bad score type for {dim}: {raw[dim]}"
        if not 1 <= s <= 5:
            return False, f"{dim} score out of range: {s}"

    pw = raw.get("primary_weakness", "none")
    if pw not in VALID_WEAKNESSES:
        return False, f"invalid primary_weakness: {pw}"

    if "reasoning" in raw and not isinstance(raw["reasoning"], str):
        return False, "reasoning not a string"
    if "judge_explanation" in raw and not isinstance(raw["judge_explanation"], str):
        return False, "judge_explanation not a string"
    return True, ""


# ---------------------------------------------------------------------------
# API call (same pattern as rq2_judge.py)
# ---------------------------------------------------------------------------

def _call_anthropic(prompt: str, api_key: str, max_tokens: int = 1024) -> str:
    """Call Anthropic Messages API via requests and return assistant text."""
    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_API_VERSION,
        "content-type": "application/json",
    }
    body = {
        "model": JUDGE_MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    resp = requests.post(
        ANTHROPIC_API_URL,
        headers=headers,
        json=body,
        timeout=90,
        verify=False,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["content"][0]["text"]


def _parse_json(text: str) -> dict[str, Any]:
    """Extract JSON object from possibly-wrapped text."""
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return {}


# ---------------------------------------------------------------------------
# Core judge function
# ---------------------------------------------------------------------------

def _build_output(raw: dict[str, Any]) -> RQ3JudgeOutput:
    """Convert validated raw judge dict into an RQ3JudgeOutput."""
    scores = {dim: int(raw[dim]) for dim in DIMENSION_NAMES}
    mean_score = sum(scores.values()) / len(scores)
    return RQ3JudgeOutput(
        factual_correctness=scores["factual_correctness"],
        completeness=scores["completeness"],
        grounding_faithfulness=scores["grounding_faithfulness"],
        citation_usefulness=scores["citation_usefulness"],
        compliance_assessment_quality=scores["compliance_assessment_quality"],
        mean_score=round(mean_score, 3),
        reasoning=str(raw.get("reasoning", ""))[:500],
        primary_weakness=raw.get("primary_weakness", "none"),
        judge_explanation=str(raw.get("judge_explanation", ""))[:200],
    )


def _fallback(msg: str) -> RQ3JudgeOutput:
    return RQ3JudgeOutput(
        factual_correctness=1,
        completeness=1,
        grounding_faithfulness=1,
        citation_usefulness=1,
        compliance_assessment_quality=1,
        mean_score=1.0,
        reasoning="",
        primary_weakness="none",
        judge_explanation=f"Judge error: {msg[:150]}",
    )


def judge_sample(
    inp: RQ3JudgeInput,
    *,
    api_key: str | None = None,
    use_cache: bool = True,
    _retry: bool = True,
) -> tuple[RQ3JudgeOutput, bool]:
    """Judge one RQ3 sample across all 5 dimensions.

    Returns ``(output, was_cached)``.
    """
    if api_key is None:
        api_key = os.getenv("CLAUDE_API_KEY", "")
    if not api_key:
        raise RuntimeError("CLAUDE_API_KEY not set")

    ver = get_judge_version()
    key = _cache_key(inp, ver)

    if use_cache:
        cached = _load_cached(key)
        if cached is not None:
            return cached, True

    prompt = _JUDGE_PROMPT_TEMPLATE.format(
        question=inp.question,
        esrs_standard=inp.esrs_standard,
        esrs_disclosure_id=inp.esrs_disclosure_id,
        company=inp.company,
        gold_fulfillment_level=inp.gold_fulfillment_level,
        gold_fulfillment_rationale=inp.gold_fulfillment_rationale,
        gold_ground_truth_answer=inp.gold_ground_truth_answer,
        generated_direct_answer=inp.generated_direct_answer,
        generated_compliance_rationale=inp.generated_compliance_rationale,
        generated_key_findings="\n".join(
            f"- {f}" for f in inp.generated_key_findings
        ) if inp.generated_key_findings else "(none)",
        generated_disclosure_gaps="\n".join(
            f"- {g}" for g in inp.generated_disclosure_gaps
        ) if inp.generated_disclosure_gaps else "(none)",
        generated_cited_pages=", ".join(inp.generated_cited_pages)
        if inp.generated_cited_pages else "(none)",
    )

    try:
        raw_text = _call_anthropic(prompt, api_key)
        raw = _parse_json(raw_text)
        ok, err = _validate(raw)
        if not ok:
            if _retry:
                logger.warning("RQ3 validation failed (%s), retrying", err)
                time.sleep(1)
                return judge_sample(inp, api_key=api_key,
                                    use_cache=False, _retry=False)
            out = _fallback(err)
            _save_cache(key, out, inp)
            return out, False

        out = _build_output(raw)
        _save_cache(key, out, inp)
        return out, False

    except Exception as exc:
        logger.error("RQ3 judge call failed: %s", exc)
        out = _fallback(str(exc))
        return out, False


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def run_judge_batch(
    inputs: list[RQ3JudgeInput],
    *,
    api_key: str | None = None,
    dry_run: bool = False,
) -> list[tuple[RQ3JudgeOutput, bool]]:
    """Run the RQ3 judge on a list of inputs.

    Returns a list of ``(output, was_cached)`` tuples, one per input.
    """
    if api_key is None:
        api_key = os.getenv("CLAUDE_API_KEY", "")

    results: list[tuple[RQ3JudgeOutput, bool]] = []
    api_calls = 0

    for i, inp in enumerate(inputs):
        ver = get_judge_version()
        key = _cache_key(inp, ver)
        cached = _load_cached(key)

        if cached is not None:
            results.append((cached, True))
            continue

        if dry_run:
            results.append((_fallback("dry_run: not cached"), False))
            continue

        out, was_cached = judge_sample(inp, api_key=api_key, use_cache=False)
        api_calls += 1
        results.append((out, was_cached))

        if api_calls % 20 == 0:
            logger.info("RQ3 judge progress: %d/%d (API calls: %d)",
                        i + 1, len(inputs), api_calls)
        # Rate-limit: 0.3 s between API calls
        time.sleep(0.3)

    logger.info("RQ3 judge batch done: %d samples, %d API calls, %d cached",
                len(inputs), api_calls, len(inputs) - api_calls)
    return results
