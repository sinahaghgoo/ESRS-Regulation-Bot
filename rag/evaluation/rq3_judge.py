"""LLM-as-a-judge module for RQ3: multi-dimensional generation quality evaluation.

This module uses Claude Sonnet 4.5 as a G-Eval-style, reference-based,
pointwise, rubric-based judge to evaluate generated ESRS compliance answers
across five quality dimensions in a SINGLE API call per sample.

Design rationale
----------------
RQ3 is a single-condition characterisation study of the best pipeline.
Unlike RQ2 (which evaluates rationale alignment on one dimension), RQ3
evaluates five orthogonal quality dimensions simultaneously:

  1. Factual correctness    — accuracy of stated facts vs gold answer
  2. Completeness           — coverage of key points from the gold answer
  3. Grounding faithfulness — whether claims are grounded in retrieved evidence
  4. Citation usefulness    — relevance and sufficiency of cited pages
  5. Coherence              — logical structure, flow, and internal consistency

All five scores are returned in a single JSON response to minimise API cost
(~204 calls for the full dataset instead of ~1020).

Anti-anchoring: the generated compliance level is intentionally WITHHELD
from the judge prompt (same principle as RQ2 v2 judge). The judge evaluates
answer quality independently of whether the level prediction matched.

Dimension selection — scientific backing
-----------------------------------------
Each dimension is grounded in established NLG / RAG evaluation literature:

  - **Factual correctness**: Core accuracy metric used across all reference-
    based evaluation frameworks.  See G-Eval (Liu et al., 2023, EMNLP) and
    FActScore (Min et al., 2023, EMNLP).
  - **Completeness**: Measures recall of key information from the reference.
    Analogous to "coverage" in SummEval (Fabbri et al., 2021, TACL) and
    "answer completeness" in ARES (Saad-Falcon et al., 2024, NAACL).
  - **Grounding faithfulness**: Central to RAG evaluation — whether claims
    are supported by retrieved evidence.  Core metric in RAGAS (Es et al.,
    2024, EACL), ARES, and TruLens.
  - **Citation usefulness**: Evaluates verifiability of RAG outputs via
    source attribution quality.  Based on ALCE (Gao et al., 2023, EMNLP)
    and HAGRID (Kamalloo et al., 2023, EMNLP Findings).
  - **Coherence**: Measures logical structure, flow, and consistency of the
    generated text.  Established dimension in G-Eval (Liu et al., 2023),
    UniEval (Zhong et al., 2022, EMNLP), and SummEval (Fabbri et al., 2021).
    Replaces "compliance assessment quality" from v2, which overlapped with
    RQ2's rationale alignment judge.

Ceiling-effect mitigation (v3)
-------------------------------
The v2 prompt suffered from score inflation (~89% of scores were 4-5).
The v3 prompt addresses this via three evidence-based strategies:

  1. **Count-based anchoring** — the judge must enumerate key points and
     count hits/misses before scoring, preventing gestalt-based leniency.
     (Ye et al., 2024, "Improving LLM-as-a-Judge", ACL Findings)
  2. **Explicit distribution guidance** — the judge is told that 3 is the
     expected midpoint for competent answers, and 5 should be rare (<10%).
     (Zheng et al., 2024, "Judging LLM-as-a-Judge")
  3. **Deficit-first evaluation** — the judge must list ALL deficits before
     assigning scores, counteracting positivity bias.
     (Liu et al., 2023, G-Eval; Kim et al., 2024, "Prometheus 2", ICML)

References:
  - Liu et al. (2023). "G-Eval: NLG Evaluation using GPT-4 with Better
    Human Alignment."  EMNLP 2023.
  - Zheng et al. (2024). "Judging LLM-as-a-Judge with MT-Bench and
    Chatbot Arena."  NeurIPS 2024.
  - Zhong et al. (2022). "Towards a Unified Multi-Dimensional Evaluator
    for Text Generation."  EMNLP 2022.
  - Fabbri et al. (2021). "SummEval: Reevaluating Summarization
    Evaluation."  TACL 2021.
  - Es et al. (2024). "RAGAS: Automated Evaluation of Retrieval Augmented
    Generation."  EACL 2024.
  - Saad-Falcon et al. (2024). "ARES: An Automated Evaluation Framework
    for Retrieval-Augmented Generation Systems."  NAACL 2024.
  - Gao et al. (2023). "Enabling Large Language Models to Generate Text
    with Citations."  EMNLP 2023.
  - Min et al. (2023). "FActScore: Fine-grained Atomic Evaluation of
    Factual Precision."  EMNLP 2023.
  - Kim et al. (2024). "Prometheus 2: An Open Source Language Model
    Specialized in Evaluating Other Language Models."  ICML 2024.
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

JUDGE_PROMPT_VERSION = "rq3_v3"

VALID_WEAKNESSES = frozenset({
    "none",
    "unsupported_claim",
    "incomplete_answer",
    "structural_incoherence",
    "weak_citation_support",
    "factual_error",
})

DIMENSION_NAMES = (
    "factual_correctness",
    "completeness",
    "grounding_faithfulness",
    "citation_usefulness",
    "coherence",
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
    coherence: int                    # 1-5
    mean_score: float                 # arithmetic mean of all 5
    reasoning: str = ""
    primary_weakness: str = "none"
    judge_explanation: str = ""


# ---------------------------------------------------------------------------
# Prompt template — G-Eval style, all 5 dimensions in one call
# ---------------------------------------------------------------------------

_JUDGE_PROMPT_TEMPLATE = """\
You are a strict expert judge evaluating the quality of a generated answer \
for an ESRS sustainability disclosure question. Evaluate the answer across \
five dimensions using calibrated, deficit-focused scoring.

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

## Scoring Philosophy — READ CAREFULLY

You must be a STRICT evaluator. Most generated answers have real deficits — \
your job is to find and quantify them, not to be generous.

- Score 3 ("adequate") is the expected baseline for a competent answer that \
gets the main points right but has clear, identifiable shortcomings. This \
should be the most common score.
- Score 4 ("good") requires near-complete quality with only trivial flaws. \
At most ~25% of answers should earn a 4 on any dimension.
- Score 5 ("excellent") is exceptional and should be RARE — reserve it for \
answers that match the gold standard with no meaningful deficit at all. \
Fewer than 10% of answers should earn a 5 on any dimension.
- Score 2 ("poor") indicates multiple substantial deficits.
- Score 1 ("very poor") indicates a fundamentally flawed answer.

If you catch yourself giving mostly 4s and 5s, you are being too lenient. \
A well-calibrated judge uses the full 1-5 range and centers around 3.

## Evaluation Steps (follow each step IN ORDER before scoring)

Step 1 — ENUMERATE gold key points:
Read the gold answer carefully. Make a numbered list of EVERY distinct \
factual claim, evidence point, specific metric, regulatory requirement, \
and nuanced conclusion. Count them: N_gold = total number of key points.

Step 2 — CHECK coverage point-by-point:
For EACH gold key point, determine: (a) HIT — present and accurate in \
the generated answer, (b) PARTIAL — present but vague, imprecise, or \
lacking detail, (c) MISS — absent from the generated answer, \
(d) DISTORTED — present but factually wrong or misleading. Count each: \
N_hit, N_partial, N_miss, N_distorted.

Step 3 — IDENTIFY unsupported claims:
List any claims in the generated answer that are NOT in the gold answer \
and NOT clearly grounded in the retrieved evidence. Count them: N_unsupported.

Step 4 — ASSESS structural quality:
Check whether the answer is logically organized, flows naturally between \
sections (direct answer, key findings, gaps, rationale), avoids redundancy \
across sections, and maintains internal consistency.

Step 5 — SCORE using the rubrics below. Use your counts from Steps 1-3 \
as hard anchors — do NOT override the count-based thresholds with \
subjective impressions.

Step 6 — IDENTIFY the single most significant weakness category.

## Dimensions and Rubrics (each scored 1-5)

### 1. Factual Correctness
Use your distortion count (N_distorted) and partial count as anchors:
[5] N_distorted = 0 AND N_partial = 0 — every fact, metric, and entity \
matches the gold answer precisely with full specificity preserved
[4] N_distorted = 0 AND N_partial <= 1 — no factual errors, at most one \
minor loss of precision that does not change the substantive meaning
[3] N_distorted <= 1 AND N_partial <= 3 — core facts mostly correct but \
one factual error or several imprecise restatements weaken overall accuracy
[2] N_distorted = 2-3 OR one major contradiction with the gold answer
[1] N_distorted >= 4 OR predominantly incorrect / fabricated claims

### 2. Completeness
Use your hit rate as an anchor: coverage = N_hit / N_gold (partials count \
as 0.5 hits):
[5] Coverage >= 95% — covers ALL key points with comparable depth and \
specificity, no substantive omissions whatsoever
[4] Coverage 80-94% — covers most key points; only minor details missing \
that do not materially affect the answer's utility
[3] Coverage 60-79% — main themes present but omits at least one \
significant point or provides notably less depth on important topics
[2] Coverage 40-59% — significant omissions, several key points absent
[1] Coverage < 40% — most critical information from the gold answer missing

### 3. Grounding Faithfulness
Use your unsupported-claim count (N_unsupported) as an anchor:
[5] N_unsupported = 0 — every factual claim can be directly traced to \
specific retrieved evidence with zero extrapolation or generalisation
[4] N_unsupported = 1 AND the unsupported claim is minor and does not \
contradict the evidence
[3] N_unsupported = 2-3 OR one claim that substantially extrapolates \
beyond the evidence (even if plausible)
[2] N_unsupported = 4-5 OR one claim that directly contradicts the evidence
[1] N_unsupported >= 6 OR answer introduces substantial information not \
present in any evidence

### 4. Citation Usefulness
Evaluate both over-citing (irrelevant pages) and under-citing (uncited claims):
[5] Every cited page is directly relevant to a specific claim, no \
redundant or padding references, AND all major claims have supporting \
citations — zero excess, zero gaps
[4] At most 1-2 slightly tangential pages OR one important claim lacks \
a citation, but overall citation quality supports verifiability well
[3] Noticeable over-citing (3+ irrelevant pages) OR 2-3 important claims \
left uncited OR citations are too broad (e.g. citing entire sections \
rather than specific pages)
[2] Many irrelevant citations OR most important claims lack citations
[1] No useful citations or predominantly wrong page references

### 5. Coherence
Evaluate the logical structure, flow, and internal consistency of the \
generated answer across all its components (direct answer, key findings, \
disclosure gaps, compliance rationale):
[5] Perfectly organized: each section serves a distinct purpose with no \
redundancy, arguments flow logically, transitions are smooth, and all \
components are internally consistent with each other — reads as a unified, \
professional assessment
[4] Well-organized overall but with 1-2 minor structural issues: slight \
redundancy between sections, one awkward transition, or one point that \
would fit better in a different section
[3] Adequate structure but noticeable issues: information repeated across \
sections, some logical jumps, findings and gaps not clearly delineated, \
or sections that feel disconnected from each other
[2] Poor organization: substantial redundancy, contradictions between \
sections, key information buried or misplaced, difficult to follow the \
overall argument
[1] Incoherent: contradictory statements across sections, no logical \
flow, or answer structure actively impedes understanding

## Output

Return ONLY valid JSON (no markdown fences, no text outside the JSON object):

{{"reasoning": "<your deficit-focused analysis: state N_gold, N_hit, N_partial, N_miss, N_distorted, N_unsupported counts, then justify each score, 4-8 sentences>", \
"factual_correctness": <1-5>, \
"completeness": <1-5>, \
"grounding_faithfulness": <1-5>, \
"citation_usefulness": <1-5>, \
"coherence": <1-5>, \
"primary_weakness": "<one of: none, unsupported_claim, incomplete_answer, structural_incoherence, weak_citation_support, factual_error>", \
"judge_explanation": "<max 50 words: concise summary of key deficits found>"}}"""


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
            coherence=int(o["coherence"]),
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
        coherence=scores["coherence"],
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
        coherence=1,
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
