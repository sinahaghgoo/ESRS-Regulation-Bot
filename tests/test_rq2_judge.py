"""Tests for RQ2 judge module (v2 — G-Eval 1-5 Likert, no anchoring).

Tests cover validation, JSON parsing, dataclass construction, score mapping,
and utility functions. No network calls or API dependencies required.
"""

import json
import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path for imports.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.evaluation.rq2_judge import (
    VALID_ERROR_TAGS,
    VALID_VERDICTS,
    JudgeInput,
    JudgeOutput,
    _build_output,
    _cache_key,
    _parse_json,
    _validate,
    get_judge_version,
    score_to_verdict,
    score_to_normalized,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_judge_input() -> JudgeInput:
    return JudgeInput(
        question="Does the company have a transition plan?",
        esrs_standard="ESRS E1",
        esrs_disclosure_id="E1-1",
        gold_fulfillment_level="High",
        gold_fulfillment_rationale="The company published a detailed transition plan aligned with 1.5°C.",
        generated_compliance_level="High",
        generated_compliance_rationale="The company has a published transition plan.",
        company="TestCo",
    )


@pytest.fixture
def valid_judge_output_dict() -> dict:
    """Valid judge output dict — v2 format (score as integer 1-5)."""
    return {
        "score": 4,
        "error_tags": ["none"],
        "reasoning": "Gold points are covered.",
        "judge_explanation": "The generated rationale covers the main judgment.",
    }


@pytest.fixture
def valid_judge_output_with_errors() -> dict:
    return {
        "score": 2,
        "error_tags": ["omission", "different_level_support"],
        "reasoning": "Major gaps in coverage.",
        "judge_explanation": "Missing details on implementation timeline.",
    }


# ---------------------------------------------------------------------------
# Tests for score mapping
# ---------------------------------------------------------------------------

class TestScoreMapping:
    def test_score_to_verdict_aligned(self) -> None:
        assert score_to_verdict(5) == "aligned"
        assert score_to_verdict(4) == "aligned"

    def test_score_to_verdict_partial(self) -> None:
        assert score_to_verdict(3) == "partially_aligned"

    def test_score_to_verdict_misaligned(self) -> None:
        assert score_to_verdict(2) == "misaligned"
        assert score_to_verdict(1) == "misaligned"

    def test_score_to_normalized(self) -> None:
        assert score_to_normalized(1) == 0.0
        assert score_to_normalized(3) == 0.5
        assert score_to_normalized(5) == 1.0
        assert score_to_normalized(4) == 0.75
        assert score_to_normalized(2) == 0.25


# ---------------------------------------------------------------------------
# Tests for JudgeInput
# ---------------------------------------------------------------------------

class TestJudgeInput:
    def test_construction_with_all_fields(self) -> None:
        inp = JudgeInput(
            question="Q1", esrs_standard="ESRS E1", esrs_disclosure_id="E1-1",
            gold_fulfillment_level="High", gold_fulfillment_rationale="Gold",
            generated_compliance_level="High", generated_compliance_rationale="Gen",
            company="MyCompany",
        )
        assert inp.question == "Q1"
        assert inp.company == "MyCompany"

    def test_company_defaults_to_empty(self) -> None:
        inp = JudgeInput(
            question="Q1", esrs_standard="ESRS E1", esrs_disclosure_id="E1-1",
            gold_fulfillment_level="High", gold_fulfillment_rationale="Gold",
            generated_compliance_level="High", generated_compliance_rationale="Gen",
        )
        assert inp.company == ""


# ---------------------------------------------------------------------------
# Tests for JudgeOutput
# ---------------------------------------------------------------------------

class TestJudgeOutput:
    def test_construction_with_all_fields(self) -> None:
        out = JudgeOutput(
            rationale_score=4, rationale_score_normalized=0.75,
            rationale_verdict="aligned", level_consistent_with_gold=True,
            joint_strict_match=True, joint_lenient_match=True,
            error_tags=["none"], judge_reasoning="Good.", judge_explanation="Aligned.",
        )
        assert out.rationale_score == 4
        assert out.rationale_score_normalized == 0.75
        assert out.rationale_verdict == "aligned"

    def test_error_tags_default_to_empty_list(self) -> None:
        out = JudgeOutput(
            rationale_score=4, rationale_score_normalized=0.75,
            rationale_verdict="aligned", level_consistent_with_gold=True,
            joint_strict_match=True, joint_lenient_match=True,
        )
        assert out.error_tags == []

    def test_reasoning_defaults_to_empty(self) -> None:
        out = JudgeOutput(
            rationale_score=1, rationale_score_normalized=0.0,
            rationale_verdict="misaligned", level_consistent_with_gold=False,
            joint_strict_match=False, joint_lenient_match=False,
        )
        assert out.judge_reasoning == ""
        assert out.judge_explanation == ""


# ---------------------------------------------------------------------------
# Tests for _validate (v2: expects 'score' integer 1-5)
# ---------------------------------------------------------------------------

class TestValidate:
    def test_valid_output(self, valid_judge_output_dict: dict) -> None:
        ok, err = _validate(valid_judge_output_dict)
        assert ok is True
        assert err == ""

    def test_valid_output_with_errors(self, valid_judge_output_with_errors: dict) -> None:
        ok, err = _validate(valid_judge_output_with_errors)
        assert ok is True

    def test_missing_score(self, valid_judge_output_dict: dict) -> None:
        d = valid_judge_output_dict.copy()
        del d["score"]
        ok, err = _validate(d)
        assert ok is False
        assert "missing" in err.lower()

    def test_missing_error_tags(self, valid_judge_output_dict: dict) -> None:
        d = valid_judge_output_dict.copy()
        del d["error_tags"]
        ok, err = _validate(d)
        assert ok is False

    def test_score_below_range(self, valid_judge_output_dict: dict) -> None:
        d = valid_judge_output_dict.copy()
        d["score"] = 0
        ok, err = _validate(d)
        assert ok is False
        assert "range" in err.lower()

    def test_score_above_range(self, valid_judge_output_dict: dict) -> None:
        d = valid_judge_output_dict.copy()
        d["score"] = 6
        ok, err = _validate(d)
        assert ok is False
        assert "range" in err.lower()

    def test_score_at_boundaries(self, valid_judge_output_dict: dict) -> None:
        d = valid_judge_output_dict.copy()
        d["score"] = 1
        ok, _ = _validate(d)
        assert ok is True
        d["score"] = 5
        ok, _ = _validate(d)
        assert ok is True

    def test_score_not_integer(self, valid_judge_output_dict: dict) -> None:
        d = valid_judge_output_dict.copy()
        d["score"] = "not_a_number"
        ok, err = _validate(d)
        assert ok is False

    def test_error_tags_not_list(self, valid_judge_output_dict: dict) -> None:
        d = valid_judge_output_dict.copy()
        d["error_tags"] = "not_a_list"
        ok, err = _validate(d)
        assert ok is False

    def test_invalid_error_tags(self, valid_judge_output_dict: dict) -> None:
        d = valid_judge_output_dict.copy()
        d["error_tags"] = ["none", "invalid_tag"]
        ok, err = _validate(d)
        assert ok is False

    def test_allows_all_valid_error_tags(self, valid_judge_output_dict: dict) -> None:
        for tag in VALID_ERROR_TAGS:
            d = valid_judge_output_dict.copy()
            d["error_tags"] = [tag]
            ok, err = _validate(d)
            assert ok is True, f"Failed for tag: {tag}"

    def test_reasoning_not_string(self, valid_judge_output_dict: dict) -> None:
        d = valid_judge_output_dict.copy()
        d["reasoning"] = 123
        ok, err = _validate(d)
        assert ok is False

    def test_optional_fields_missing_is_ok(self) -> None:
        """score and error_tags are required; reasoning/explanation are optional."""
        d = {"score": 3, "error_tags": ["none"]}
        ok, err = _validate(d)
        assert ok is True


# ---------------------------------------------------------------------------
# Tests for _parse_json
# ---------------------------------------------------------------------------

class TestParseJson:
    def test_clean_json(self) -> None:
        result = _parse_json('{"key": "value", "number": 42}')
        assert result == {"key": "value", "number": 42}

    def test_markdown_fences(self) -> None:
        result = _parse_json('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_embedded_json(self) -> None:
        result = _parse_json('Here is output: {"key": "value"} and more')
        assert result == {"key": "value"}

    def test_invalid_json_returns_empty_dict(self) -> None:
        assert _parse_json('Not valid JSON') == {}

    def test_judge_v2_output_format(self) -> None:
        text = '{"reasoning": "Points covered.", "score": 4, "error_tags": ["none"], "judge_explanation": "Good."}'
        result = _parse_json(text)
        assert result["score"] == 4
        assert result["error_tags"] == ["none"]


# ---------------------------------------------------------------------------
# Tests for _build_output (v2: score integer → verdict derived)
# ---------------------------------------------------------------------------

class TestBuildOutput:
    def test_matching_levels_aligned(self, sample_judge_input: JudgeInput) -> None:
        raw = {"score": 5, "error_tags": ["none"], "reasoning": "All covered."}
        out = _build_output(sample_judge_input, raw)
        assert out.rationale_score == 5
        assert out.rationale_score_normalized == 1.0
        assert out.rationale_verdict == "aligned"
        assert out.level_consistent_with_gold is True
        assert out.joint_strict_match is True
        assert out.joint_lenient_match is True

    def test_mismatched_levels(self, sample_judge_input: JudgeInput) -> None:
        sample_judge_input.generated_compliance_level = "Low"
        raw = {"score": 4, "error_tags": ["none"]}
        out = _build_output(sample_judge_input, raw)
        assert out.level_consistent_with_gold is False
        assert out.joint_strict_match is False
        assert out.joint_lenient_match is False

    def test_partial_verdict(self, sample_judge_input: JudgeInput) -> None:
        raw = {"score": 3, "error_tags": ["omission"]}
        out = _build_output(sample_judge_input, raw)
        assert out.rationale_verdict == "partially_aligned"
        assert out.rationale_score_normalized == 0.5
        assert out.joint_strict_match is False
        assert out.joint_lenient_match is True  # level matches + partial

    def test_misaligned_verdict(self, sample_judge_input: JudgeInput) -> None:
        raw = {"score": 1, "error_tags": ["contradiction", "different_level_support"]}
        out = _build_output(sample_judge_input, raw)
        assert out.rationale_verdict == "misaligned"
        assert out.rationale_score_normalized == 0.0

    def test_case_insensitive_level_matching(self, sample_judge_input: JudgeInput) -> None:
        sample_judge_input.generated_compliance_level = "high"
        sample_judge_input.gold_fulfillment_level = "HIGH"
        raw = {"score": 4, "error_tags": ["none"]}
        out = _build_output(sample_judge_input, raw)
        assert out.level_consistent_with_gold is True

    def test_explanation_truncated(self, sample_judge_input: JudgeInput) -> None:
        raw = {"score": 4, "error_tags": ["none"], "judge_explanation": "X" * 300}
        out = _build_output(sample_judge_input, raw)
        assert len(out.judge_explanation) == 200

    def test_reasoning_preserved(self, sample_judge_input: JudgeInput) -> None:
        raw = {"score": 4, "error_tags": ["none"], "reasoning": "Detailed analysis."}
        out = _build_output(sample_judge_input, raw)
        assert out.judge_reasoning == "Detailed analysis."


# ---------------------------------------------------------------------------
# Tests for _cache_key
# ---------------------------------------------------------------------------

class TestCacheKey:
    def test_deterministic(self, sample_judge_input: JudgeInput) -> None:
        key1 = _cache_key(sample_judge_input, "full_system", "v2_abc123")
        key2 = _cache_key(sample_judge_input, "full_system", "v2_abc123")
        assert key1 == key2

    def test_changes_with_condition(self, sample_judge_input: JudgeInput) -> None:
        key1 = _cache_key(sample_judge_input, "full_system", "v2_abc123")
        key2 = _cache_key(sample_judge_input, "no_esrs_context", "v2_abc123")
        assert key1 != key2

    def test_changes_with_version(self, sample_judge_input: JudgeInput) -> None:
        key1 = _cache_key(sample_judge_input, "full_system", "v1_abc")
        key2 = _cache_key(sample_judge_input, "full_system", "v2_xyz")
        assert key1 != key2

    def test_is_hex_string(self, sample_judge_input: JudgeInput) -> None:
        key = _cache_key(sample_judge_input, "full_system", "v2_abc123")
        int(key, 16)  # Should not raise


# ---------------------------------------------------------------------------
# Tests for get_judge_version
# ---------------------------------------------------------------------------

class TestGetJudgeVersion:
    def test_returns_string(self) -> None:
        assert isinstance(get_judge_version(), str)

    def test_contains_v2(self) -> None:
        assert get_judge_version().startswith("v2_")

    def test_deterministic(self) -> None:
        assert get_judge_version() == get_judge_version()


# ---------------------------------------------------------------------------
# Tests for constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_valid_verdicts(self) -> None:
        assert VALID_VERDICTS == {"aligned", "partially_aligned", "misaligned"}

    def test_valid_error_tags(self) -> None:
        assert "contradiction" in VALID_ERROR_TAGS
        assert "omission" in VALID_ERROR_TAGS
        assert "different_level_support" in VALID_ERROR_TAGS
        assert "none" in VALID_ERROR_TAGS
