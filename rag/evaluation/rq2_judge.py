"""LLM-as-a-judge module for RQ2: compliance rationale alignment evaluation.

This module uses Claude Sonnet 4.5 as a G-Eval-style, reference-based,
pointwise, rubric-based judge to assess whether the generated compliance
rationale substantively agrees with the gold fulfillment rationale.

Design rationale — G-Eval (Liu et al., 2023, EMNLP)
-----------------------------------------------------
G-Eval uses LLM-based evaluation against a rubric with explicit
chain-of-thought evaluation steps.  We adapt this for compliance-
assessment evaluation: the judge receives the gold rationale and the
generated rationale, follows pre-defined evaluation steps, and assigns
a 1–5 Likert score.

Key design decisions:
  1. The judge does NOT see the *generated* compliance level.  Showing
     both gold and generated levels creates anchoring bias: when levels
     match, the judge is disposed toward 'aligned', collapsing the
     rationale evaluation to a proxy for level correctness.  Removing
     the generated level forces the judge to evaluate rationale substance
     independently.  (Zheng et al., 2024, "Judging LLM-as-a-Judge")
  2. The 1–5 Likert scale follows the original G-Eval paper rather than
     a coarser 3-tier system, providing finer-grained discrimination.
  3. Verdicts (aligned/partially_aligned/misaligned) are derived
     deterministically from the score, not judged independently.

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
from dataclasses import asdict, dataclass, field
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

JUDGE_PROMPT_VERSION = "v2"

VALID_ERROR_TAGS = frozenset({
    "contradiction",
    "omission",
    "different_level_support",
    "none",
})

VALID_VERDICTS = frozenset({"aligned", "partially_aligned", "misaligned"})

JUDGE_CACHE_DIR = ARTIFACTS_DIR / "evaluation_results" / "rq2" / "judge_cache"

# ---------------------------------------------------------------------------
# Score ↔ verdict mapping
# ---------------------------------------------------------------------------

def score_to_verdict(score: int) -> str:
    """Map a 1-5 Likert score to a categorical verdict.

    4-5 → aligned, 3 → partially_aligned, 1-2 → misaligned.
    """
    if score >= 4:
        return "aligned"
    if score == 3:
        return "partially_aligned"
    return "misaligned"


def score_to_normalized(score: int) -> float:
    """Normalize a 1-5 Likert score to [0, 1]: (score - 1) / 4."""
    return (score - 1) / 4.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class JudgeInput:
    """All context the judge needs for one sample."""
    question: str
    esrs_standard: str
    esrs_disclosure_id: str
    gold_fulfillment_level: str
    gold_fulfillment_rationale: str
    generated_compliance_level: str   # kept for downstream joint metrics, NOT sent to judge
    generated_compliance_rationale: str
    company: str = ""


@dataclass
class JudgeOutput:
    """Structured judge verdict for one sample."""
    rationale_score: int               # 1-5 Likert scale
    rationale_score_normalized: float  # (score - 1) / 4 → [0, 1]
    rationale_verdict: str             # aligned | partially_aligned | misaligned
    level_consistent_with_gold: bool
    joint_strict_match: bool           # level match AND verdict == aligned
    joint_lenient_match: bool          # level match AND verdict in {aligned, partially_aligned}
    error_tags: list[str] = field(default_factory=list)
    judge_reasoning: str = ""          # chain-of-thought evaluation reasoning
    judge_explanation: str = ""        # short auditable summary


# ---------------------------------------------------------------------------
# Prompt template — G-Eval style with chain-of-thought
# ---------------------------------------------------------------------------

_JUDGE_PROMPT_TEMPLATE = """\
You are an expert judge evaluating whether a generated compliance rationale \
substantively aligns with a gold-standard rationale for an ESRS sustainability \
disclosure question.

## Task

Assess whether the GENERATED RATIONALE captures the same key evidence points, \
reasoning, and conclusions as the GOLD RATIONALE regarding the company's \
fulfillment of the ESRS disclosure requirement.

## Input

**Question:** {question}
**ESRS Standard:** {esrs_standard}
**Disclosure ID(s):** {esrs_disclosure_id}
**Company:** {company}

**Gold Fulfillment Level:** {gold_fulfillment_level}

**Gold Fulfillment Rationale:**
{gold_fulfillment_rationale}

**Generated Compliance Rationale:**
{generated_compliance_rationale}

## Evaluation Steps (follow each step in order)

1. IDENTIFY the key evidence claims and reasoning points in the GOLD rationale.
2. For each gold key point, CHECK whether the GENERATED rationale covers it \
substantively (paraphrasing is acceptable).
3. CHECK for any direct contradictions between the rationales.
4. ASSESS whether the generated rationale's overall fulfillment judgment aligns \
with the gold level direction (even if it uses different words).
5. ASSIGN a score from the rubric below based on your evaluation.

## Scoring Rubric (1–5 Likert Scale)

5 — **Full alignment**: All critical gold points are substantively covered. \
No contradictions. Both rationales clearly support the same fulfillment judgment. \
Differences in wording, ordering, or additional non-conflicting detail are acceptable.

4 — **High alignment**: Most critical gold points are covered. At most one \
minor gap or slight inconsistency, but no material contradictions. The overall \
fulfillment direction is the same.

3 — **Moderate alignment**: The main thrust of the gold reasoning is captured, \
but there are notable gaps in coverage of key points. No direct contradictions \
of material claims. The rationales generally point in the same direction.

2 — **Weak alignment**: Significant gaps in gold point coverage OR one material \
inconsistency. The rationales may point in roughly the same direction but with \
major omissions of critical reasoning.

1 — **Misaligned**: The generated rationale contradicts the gold on material \
points, its reasoning better supports a materially different fulfillment level, \
or it omits the majority of critical gold reasoning.

## Error Tags (assign ALL that apply; use "none" if score >= 4)

- contradiction — directly contradicts a gold claim
- omission — omits essential justifications present in the gold rationale
- different_level_support — the generated reasoning better supports a different \
fulfillment level than the gold
- none — no specific errors

## Output

Respond with ONLY a valid JSON object (no markdown fences, no text outside):

{{"reasoning": "<2-4 sentences: your step-by-step evaluation>", \
"score": <integer 1-5>, \
"error_tags": ["<tag>"], \
"judge_explanation": "<max 50 words: concise summary>"}}"""


def _prompt_hash() -> str:
    """Content hash of the prompt template for cache invalidation."""
    return hashlib.sha256(_JUDGE_PROMPT_TEMPLATE.encode()).hexdigest()[:10]


def get_judge_version() -> str:
    """Combined version string: prompt version + prompt content hash."""
    return f"{JUDGE_PROMPT_VERSION}_{_prompt_hash()}"


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_key(inp: JudgeInput, condition: str, version: str) -> str:
    blob = (
        f"{inp.question}|{inp.company}|{condition}|"
        f"{inp.gold_fulfillment_level}|{inp.gold_fulfillment_rationale}|"
        f"{inp.generated_compliance_level}|{inp.generated_compliance_rationale}|"
        f"{version}"
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def _load_cached(key: str) -> JudgeOutput | None:
    path = JUDGE_CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text("utf-8"))
        o = data["judge_output"]
        return JudgeOutput(
            rationale_score=int(o["rationale_score"]),
            rationale_score_normalized=float(o["rationale_score_normalized"]),
            rationale_verdict=o["rationale_verdict"],
            level_consistent_with_gold=o["level_consistent_with_gold"],
            joint_strict_match=o["joint_strict_match"],
            joint_lenient_match=o["joint_lenient_match"],
            error_tags=o.get("error_tags", []),
            judge_reasoning=o.get("judge_reasoning", ""),
            judge_explanation=o.get("judge_explanation", ""),
        )
    except Exception as exc:
        logger.warning("Corrupt cache %s: %s", key, exc)
        return None


def _save_cache(key: str, out: JudgeOutput, inp: JudgeInput, condition: str) -> None:
    JUDGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_key": key,
        "judge_version": get_judge_version(),
        "condition": condition,
        "judge_input": asdict(inp),
        "judge_output": asdict(out),
    }
    try:
        (JUDGE_CACHE_DIR / f"{key}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), "utf-8",
        )
    except Exception as exc:
        logger.error("Cache write failed for %s: %s", key, exc)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(raw: dict[str, Any]) -> tuple[bool, str]:
    """Validate raw judge JSON against the expected schema."""
    for f in ("score", "error_tags"):
        if f not in raw:
            return False, f"missing field: {f}"
    try:
        s = int(raw["score"])
    except (TypeError, ValueError):
        return False, f"bad score type: {raw['score']}"
    if not 1 <= s <= 5:
        return False, f"score out of range: {s}"
    if not isinstance(raw["error_tags"], list):
        return False, "error_tags not a list"
    bad = set(raw["error_tags"]) - VALID_ERROR_TAGS
    if bad:
        return False, f"invalid tags: {bad}"
    # reasoning and judge_explanation are optional but should be strings
    if "reasoning" in raw and not isinstance(raw["reasoning"], str):
        return False, "reasoning not a string"
    if "judge_explanation" in raw and not isinstance(raw["judge_explanation"], str):
        return False, "judge_explanation not a string"
    return True, ""


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def _call_anthropic(prompt: str, api_key: str, max_tokens: int = 512) -> str:
    """Call Anthropic Messages API via ``requests`` and return assistant text."""
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
        verify=False,  # proxy environment uses self-signed cert
    )
    resp.raise_for_status()
    data = resp.json()
    return data["content"][0]["text"]


def _parse_json(text: str) -> dict[str, Any]:
    """Extract JSON object from possibly-wrapped text."""
    text = text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find embedded JSON
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

def _build_output(inp: JudgeInput, raw: dict[str, Any]) -> JudgeOutput:
    """Convert validated raw judge dict into a JudgeOutput."""
    score = int(raw["score"])
    tags = raw.get("error_tags", [])
    reasoning = str(raw.get("reasoning", ""))[:500]
    explanation = str(raw.get("judge_explanation", ""))[:200]

    verdict = score_to_verdict(score)
    normalized = score_to_normalized(score)

    level_match = (
        inp.generated_compliance_level.strip().lower()
        == inp.gold_fulfillment_level.strip().lower()
    )
    return JudgeOutput(
        rationale_score=score,
        rationale_score_normalized=normalized,
        rationale_verdict=verdict,
        level_consistent_with_gold=level_match,
        joint_strict_match=level_match and verdict == "aligned",
        joint_lenient_match=level_match and verdict in ("aligned", "partially_aligned"),
        error_tags=list(tags),
        judge_reasoning=reasoning,
        judge_explanation=explanation,
    )


def _fallback(msg: str) -> JudgeOutput:
    return JudgeOutput(
        rationale_score=1,
        rationale_score_normalized=0.0,
        rationale_verdict="misaligned",
        level_consistent_with_gold=False,
        joint_strict_match=False,
        joint_lenient_match=False,
        error_tags=["none"],
        judge_reasoning="",
        judge_explanation=f"Judge error: {msg[:150]}",
    )


def judge_sample(
    inp: JudgeInput,
    condition: str,
    *,
    api_key: str | None = None,
    use_cache: bool = True,
    _retry: bool = True,
) -> tuple[JudgeOutput, bool]:
    """Judge one sample.  Returns ``(output, was_cached)``."""
    if api_key is None:
        api_key = os.getenv("CLAUDE_API_KEY", "")
    if not api_key:
        raise RuntimeError("CLAUDE_API_KEY not set")

    ver = get_judge_version()
    key = _cache_key(inp, condition, ver)

    if use_cache:
        cached = _load_cached(key)
        if cached is not None:
            return cached, True

    # NOTE: generated_compliance_level is intentionally excluded from the
    # prompt to prevent anchoring bias.  The judge evaluates rationale
    # substance independently of level prediction correctness.
    prompt = _JUDGE_PROMPT_TEMPLATE.format(
        question=inp.question,
        esrs_standard=inp.esrs_standard,
        esrs_disclosure_id=inp.esrs_disclosure_id,
        gold_fulfillment_level=inp.gold_fulfillment_level,
        gold_fulfillment_rationale=inp.gold_fulfillment_rationale,
        generated_compliance_rationale=inp.generated_compliance_rationale,
        company=inp.company,
    )

    try:
        raw_text = _call_anthropic(prompt, api_key)
        raw = _parse_json(raw_text)
        ok, err = _validate(raw)
        if not ok:
            if _retry:
                logger.warning("Validation failed (%s), retrying", err)
                time.sleep(1)
                return judge_sample(inp, condition, api_key=api_key,
                                    use_cache=False, _retry=False)
            out = _fallback(err)
            _save_cache(key, out, inp, condition)
            return out, False

        out = _build_output(inp, raw)
        _save_cache(key, out, inp, condition)
        return out, False

    except Exception as exc:
        logger.error("Judge call failed: %s", exc)
        out = _fallback(str(exc))
        return out, False


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def run_judge_batch(
    samples: list[dict[str, str]],
    condition: str,
    *,
    api_key: str | None = None,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Run the judge on a list of sample dicts.

    Parameters
    ----------
    samples
        Each dict must have keys: question, esrs_standard, esrs_disclosure_id,
        gold_fulfillment_level, gold_fulfillment_rationale,
        generated_compliance_level, generated_compliance_rationale, company.
    condition
        ``"full_system"`` or ``"no_esrs_context"``.
    api_key
        Anthropic key.  Falls back to ``CLAUDE_API_KEY`` env var.
    dry_run
        If *True*, only return cached results; skip API calls for uncached.

    Returns
    -------
    list[dict]
        One dict per sample with all JudgeOutput fields plus metadata.
    """
    if api_key is None:
        api_key = os.getenv("CLAUDE_API_KEY", "")

    ver = get_judge_version()
    results: list[dict[str, Any]] = []
    api_calls = 0

    for i, sd in enumerate(samples):
        inp = JudgeInput(
            question=sd.get("question", ""),
            esrs_standard=sd.get("esrs_standard", ""),
            esrs_disclosure_id=sd.get("esrs_disclosure_id", ""),
            gold_fulfillment_level=sd.get("gold_fulfillment_level", ""),
            gold_fulfillment_rationale=sd.get("gold_fulfillment_rationale", ""),
            generated_compliance_level=sd.get("generated_compliance_level", ""),
            generated_compliance_rationale=sd.get("generated_compliance_rationale", ""),
            company=sd.get("company", ""),
        )
        key = _cache_key(inp, condition, ver)
        cached = _load_cached(key)

        if cached is not None:
            row = asdict(cached)
            row.update(was_cached=True, sample_index=i, condition=condition)
            results.append(row)
            continue

        if dry_run:
            row = asdict(_fallback("dry_run: not cached"))
            row.update(was_cached=False, sample_index=i, condition=condition,
                       skipped_dry_run=True)
            results.append(row)
            continue

        # Actual API call
        try:
            out, _ = judge_sample(inp, condition, api_key=api_key, use_cache=False)
            api_calls += 1
            row = asdict(out)
            row.update(was_cached=False, sample_index=i, condition=condition)
            results.append(row)
            if api_calls % 20 == 0:
                logger.info("Judge progress: %d/%d (API calls: %d)",
                            i + 1, len(samples), api_calls)
            # Light rate-limit: 0.3 s between calls
            time.sleep(0.3)
        except Exception as exc:
            logger.error("Sample %d failed: %s", i, exc)
            row = asdict(_fallback(str(exc)))
            row.update(was_cached=False, sample_index=i, condition=condition)
            results.append(row)

    logger.info("Judge batch done: %d samples, %d API calls, %d cached",
                len(samples), api_calls, len(samples) - api_calls)
    return results


# ---------------------------------------------------------------------------
# Utility: load all cached results for a condition
# ---------------------------------------------------------------------------

def load_all_cached(condition: str | None = None) -> list[dict[str, Any]]:
    """Load every cached judge artifact, optionally filtered by condition."""
    if not JUDGE_CACHE_DIR.exists():
        return []
    out: list[dict[str, Any]] = []
    for p in JUDGE_CACHE_DIR.glob("*.json"):
        try:
            d = json.loads(p.read_text("utf-8"))
            if condition is None or d.get("condition") == condition:
                out.append(d)
        except Exception:
            continue
    return out
