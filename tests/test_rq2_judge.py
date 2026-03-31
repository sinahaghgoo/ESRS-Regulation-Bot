"""Tests for RQ2 judge module.

Tests cover validation, JSON parsing, dataclass construction, and utility functions.
No network calls or API dependencies required.
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
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_judge_input() -> JudgeInput:
    """Create a sample JudgeInput for testing."""
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
    """Create a valid judge output dictionary."""
    return {
        "rationale_verdict": "aligned",
        "rationale_score": 0.85,
        "error_tags": ["none"],
        "judge_explanation": "The generated rationale covers the main judgment."
    }


@pytest.fixture
def valid_judge_output_with_errors() -> dict:
    """Create a valid judge output with error tags."""
    return {
        "rationale_verdict": "partially_aligned",
        "rationale_score": 0.55,
        "error_tags": ["omission_of_critical_reasons", "different_interpretation"],
        "judge_explanation": "Missing details on implementation timeline."
    }


# ---------------------------------------------------------------------------
# Tests for JudgeInput dataclass
# ---------------------------------------------------------------------------

class TestJudgeInput:
    """Tests for JudgeInput construction and fields."""

    def test_judge_input_construction_with_all_fields(self) -> None:
        """JudgeInput can be constructed with all fields."""
        inp = JudgeInput(
            question="Q1",
            esrs_standard="ESRS E1",
            esrs_disclosure_id="E1-1",
            gold_fulfillment_level="High",
            gold_fulfillment_rationale="Gold rationale",
            generated_compliance_level="High",
            generated_compliance_rationale="Generated rationale",
            company="MyCompany",
        )
        assert inp.question == "Q1"
        assert inp.esrs_standard == "ESRS E1"
        assert inp.esrs_disclosure_id == "E1-1"
        assert inp.gold_fulfillment_level == "High"
        assert inp.gold_fulfillment_rationale == "Gold rationale"
        assert inp.generated_compliance_level == "High"
        assert inp.generated_compliance_rationale == "Generated rationale"
        assert inp.company == "MyCompany"

    def test_judge_input_company_defaults_to_empty(self) -> None:
        """JudgeInput.company defaults to empty string."""
        inp = JudgeInput(
            question="Q1",
            esrs_standard="ESRS E1",
            esrs_disclosure_id="E1-1",
            gold_fulfillment_level="High",
            gold_fulfillment_rationale="Gold",
            generated_compliance_level="High",
            generated_compliance_rationale="Generated",
        )
        assert inp.company == ""

    def test_judge_input_fields_are_strings(self, sample_judge_input: JudgeInput) -> None:
        """All JudgeInput fields are strings."""
        assert isinstance(sample_judge_input.question, str)
        assert isinstance(sample_judge_input.esrs_standard, str)
        assert isinstance(sample_judge_input.esrs_disclosure_id, str)
        assert isinstance(sample_judge_input.gold_fulfillment_level, str)
        assert isinstance(sample_judge_input.gold_fulfillment_rationale, str)
        assert isinstance(sample_judge_input.generated_compliance_level, str)
        assert isinstance(sample_judge_input.generated_compliance_rationale, str)
        assert isinstance(sample_judge_input.company, str)


# ---------------------------------------------------------------------------
# Tests for JudgeOutput dataclass
# ---------------------------------------------------------------------------

class TestJudgeOutput:
    """Tests for JudgeOutput construction and fields."""

    def test_judge_output_construction_with_all_fields(self) -> None:
        """JudgeOutput can be constructed with all fields."""
        out = JudgeOutput(
            rationale_verdict="aligned",
            rationale_score=0.9,
            level_consistent_with_gold=True,
            joint_strict_match=True,
            joint_lenient_match=True,
            error_tags=["none"],
            judge_explanation="Fully aligned.",
        )
        assert out.rationale_verdict == "aligned"
        assert out.rationale_score == 0.9
        assert out.level_consistent_with_gold is True
        assert out.joint_strict_match is True
        assert out.joint_lenient_match is True
        assert out.error_tags == ["none"]
        assert out.judge_explanation == "Fully aligned."

    def test_judge_output_error_tags_default_to_empty_list(self) -> None:
        """JudgeOutput.error_tags defaults to empty list."""
        out = JudgeOutput(
            rationale_verdict="aligned",
            rationale_score=0.8,
            level_consistent_with_gold=True,
            joint_strict_match=True,
            joint_lenient_match=True,
        )
        assert out.error_tags == []

    def test_judge_output_explanation_defaults_to_empty(self) -> None:
        """JudgeOutput.judge_explanation defaults to empty string."""
        out = JudgeOutput(
            rationale_verdict="misaligned",
            rationale_score=0.2,
            level_consistent_with_gold=False,
            joint_strict_match=False,
            joint_lenient_match=False,
        )
        assert out.judge_explanation == ""

    def test_judge_output_score_is_float(self) -> None:
        """JudgeOutput.rationale_score is a float."""
        out = JudgeOutput(
            rationale_verdict="aligned",
            rationale_score=0.5,
            level_consistent_with_gold=False,
            joint_strict_match=False,
            joint_lenient_match=False,
        )
        assert isinstance(out.rationale_score, float)


# ---------------------------------------------------------------------------
# Tests for _validate
# ---------------------------------------------------------------------------

class TestValidate:
    """Tests for _validate(raw) function."""

    def test_validate_with_valid_output(self, valid_judge_output_dict: dict) -> None:
        """_validate returns (True, '') for valid judge output."""
        ok, err = _validate(valid_judge_output_dict)
        assert ok is True
        assert err == ""

    def test_validate_with_valid_output_with_errors(
        self, valid_judge_output_with_errors: dict
    ) -> None:
        """_validate accepts valid error tags."""
        ok, err = _validate(valid_judge_output_with_errors)
        assert ok is True
        assert err == ""

    def test_validate_missing_rationale_verdict(self, valid_judge_output_dict: dict) -> None:
        """_validate rejects output missing rationale_verdict."""
        d = valid_judge_output_dict.copy()
        del d["rationale_verdict"]
        ok, err = _validate(d)
        assert ok is False
        assert "missing field" in err.lower()

    def test_validate_missing_rationale_score(self, valid_judge_output_dict: dict) -> None:
        """_validate rejects output missing rationale_score."""
        d = valid_judge_output_dict.copy()
        del d["rationale_score"]
        ok, err = _validate(d)
        assert ok is False
        assert "missing field" in err.lower()

    def test_validate_missing_error_tags(self, valid_judge_output_dict: dict) -> None:
        """_validate rejects output missing error_tags."""
        d = valid_judge_output_dict.copy()
        del d["error_tags"]
        ok, err = _validate(d)
        assert ok is False
        assert "missing field" in err.lower()

    def test_validate_missing_judge_explanation(self, valid_judge_output_dict: dict) -> None:
        """_validate rejects output missing judge_explanation."""
        d = valid_judge_output_dict.copy()
        del d["judge_explanation"]
        ok, err = _validate(d)
        assert ok is False
        assert "missing field" in err.lower()

    def test_validate_bad_verdict(self, valid_judge_output_dict: dict) -> None:
        """_validate rejects invalid verdict values."""
        d = valid_judge_output_dict.copy()
        d["rationale_verdict"] = "not_a_valid_verdict"
        ok, err = _validate(d)
        assert ok is False
        assert "bad verdict" in err.lower()

    def test_validate_score_not_float(self, valid_judge_output_dict: dict) -> None:
        """_validate rejects non-numeric scores."""
        d = valid_judge_output_dict.copy()
        d["rationale_score"] = "not_a_number"
        ok, err = _validate(d)
        assert ok is False
        assert "bad score" in err.lower() or "cannot" in err.lower()

    def test_validate_score_below_zero(self, valid_judge_output_dict: dict) -> None:
        """_validate rejects scores below 0.0."""
        d = valid_judge_output_dict.copy()
        d["rationale_score"] = -0.1
        ok, err = _validate(d)
        assert ok is False
        assert "range" in err.lower()

    def test_validate_score_above_one(self, valid_judge_output_dict: dict) -> None:
        """_validate rejects scores above 1.0."""
        d = valid_judge_output_dict.copy()
        d["rationale_score"] = 1.1
        ok, err = _validate(d)
        assert ok is False
        assert "range" in err.lower()

    def test_validate_score_at_boundaries(self, valid_judge_output_dict: dict) -> None:
        """_validate accepts scores at 0.0 and 1.0 boundaries."""
        d = valid_judge_output_dict.copy()
        d["rationale_score"] = 0.0
        ok, err = _validate(d)
        assert ok is True

        d["rationale_score"] = 1.0
        ok, err = _validate(d)
        assert ok is True

    def test_validate_error_tags_not_list(self, valid_judge_output_dict: dict) -> None:
        """_validate rejects error_tags if not a list."""
        d = valid_judge_output_dict.copy()
        d["error_tags"] = "not_a_list"
        ok, err = _validate(d)
        assert ok is False
        assert "not a list" in err.lower()

    def test_validate_invalid_error_tags(self, valid_judge_output_dict: dict) -> None:
        """_validate rejects invalid error tag values."""
        d = valid_judge_output_dict.copy()
        d["error_tags"] = ["none", "invalid_tag"]
        ok, err = _validate(d)
        assert ok is False
        assert "invalid" in err.lower() or "tag" in err.lower()

    def test_validate_judge_explanation_not_string(self, valid_judge_output_dict: dict) -> None:
        """_validate rejects judge_explanation if not a string."""
        d = valid_judge_output_dict.copy()
        d["judge_explanation"] = 123
        ok, err = _validate(d)
        assert ok is False
        assert "not a string" in err.lower()

    def test_validate_allows_all_valid_verdicts(self, valid_judge_output_dict: dict) -> None:
        """_validate accepts all verdicts in VALID_VERDICTS."""
        for verdict in VALID_VERDICTS:
            d = valid_judge_output_dict.copy()
            d["rationale_verdict"] = verdict
            ok, err = _validate(d)
            assert ok is True, f"Failed for verdict: {verdict}"

    def test_validate_allows_all_valid_error_tags(self, valid_judge_output_dict: dict) -> None:
        """_validate accepts all tags in VALID_ERROR_TAGS."""
        for tag in VALID_ERROR_TAGS:
            d = valid_judge_output_dict.copy()
            d["error_tags"] = [tag]
            ok, err = _validate(d)
            assert ok is True, f"Failed for tag: {tag}"


# ---------------------------------------------------------------------------
# Tests for _parse_json
# ---------------------------------------------------------------------------

class TestParseJson:
    """Tests for _parse_json(text) function."""

    def test_parse_json_with_clean_json(self) -> None:
        """_parse_json extracts valid JSON from clean input."""
        text = '{"key": "value", "number": 42}'
        result = _parse_json(text)
        assert result == {"key": "value", "number": 42}

    def test_parse_json_with_markdown_fences(self) -> None:
        """_parse_json removes markdown code fences."""
        text = '```json\n{"key": "value"}\n```'
        result = _parse_json(text)
        assert result == {"key": "value"}

    def test_parse_json_with_markdown_fences_no_language(self) -> None:
        """_parse_json removes markdown code fences without language tag."""
        text = '```\n{"key": "value"}\n```'
        result = _parse_json(text)
        assert result == {"key": "value"}

    def test_parse_json_with_embedded_json(self) -> None:
        """_parse_json extracts JSON embedded in text."""
        text = 'Here is some output: {"key": "value"} and more text'
        result = _parse_json(text)
        assert result == {"key": "value"}

    def test_parse_json_with_leading_trailing_text(self) -> None:
        """_parse_json ignores text before and after JSON."""
        text = 'Before {"a": 1} after'
        result = _parse_json(text)
        assert result == {"a": 1}

    def test_parse_json_with_nested_json(self) -> None:
        """_parse_json handles nested structures."""
        text = '{"outer": {"inner": "value"}, "list": [1, 2, 3]}'
        result = _parse_json(text)
        assert result == {"outer": {"inner": "value"}, "list": [1, 2, 3]}

    def test_parse_json_with_whitespace(self) -> None:
        """_parse_json handles leading/trailing whitespace."""
        text = '   \n{"key": "value"}\n   '
        result = _parse_json(text)
        assert result == {"key": "value"}

    def test_parse_json_invalid_json_returns_empty_dict(self) -> None:
        """_parse_json returns empty dict for invalid JSON."""
        text = 'Not valid JSON at all'
        result = _parse_json(text)
        assert result == {}

    def test_parse_json_with_multiline_json(self) -> None:
        """_parse_json handles multiline JSON."""
        text = """{
            "key1": "value1",
            "key2": "value2"
        }"""
        result = _parse_json(text)
        assert result == {"key1": "value1", "key2": "value2"}

    def test_parse_json_with_escaped_quotes(self) -> None:
        """_parse_json handles escaped quotes in strings."""
        text = '{"key": "value with \\"quotes\\""}'
        result = _parse_json(text)
        assert result["key"] == 'value with "quotes"'

    def test_parse_json_judge_output_format(self) -> None:
        """_parse_json handles typical judge output format."""
        text = """{
            "rationale_verdict": "aligned",
            "rationale_score": 0.85,
            "error_tags": ["none"],
            "judge_explanation": "Rationales align."
        }"""
        result = _parse_json(text)
        assert result["rationale_verdict"] == "aligned"
        assert result["rationale_score"] == 0.85
        assert result["error_tags"] == ["none"]


# ---------------------------------------------------------------------------
# Tests for _build_output
# ---------------------------------------------------------------------------

class TestBuildOutput:
    """Tests for _build_output(inp, raw) function."""

    def test_build_output_with_matching_levels_and_aligned_verdict(
        self, sample_judge_input: JudgeInput, valid_judge_output_dict: dict
    ) -> None:
        """_build_output produces correct output for matched levels and aligned verdict."""
        sample_judge_input.generated_compliance_level = "High"
        sample_judge_input.gold_fulfillment_level = "High"

        out = _build_output(sample_judge_input, valid_judge_output_dict)

        assert out.rationale_verdict == "aligned"
        assert out.rationale_score == 0.85
        assert out.level_consistent_with_gold is True
        assert out.joint_strict_match is True
        assert out.joint_lenient_match is True

    def test_build_output_with_mismatched_levels(
        self, sample_judge_input: JudgeInput, valid_judge_output_dict: dict
    ) -> None:
        """_build_output sets level_consistent_with_gold to False when levels don't match."""
        sample_judge_input.generated_compliance_level = "High"
        sample_judge_input.gold_fulfillment_level = "Low"

        out = _build_output(sample_judge_input, valid_judge_output_dict)

        assert out.level_consistent_with_gold is False
        assert out.joint_strict_match is False
        assert out.joint_lenient_match is False

    def test_build_output_with_partially_aligned_verdict_and_matched_levels(
        self, sample_judge_input: JudgeInput, valid_judge_output_with_errors: dict
    ) -> None:
        """_build_output handles partially_aligned verdict correctly."""
        sample_judge_input.generated_compliance_level = "Medium"
        sample_judge_input.gold_fulfillment_level = "Medium"

        out = _build_output(sample_judge_input, valid_judge_output_with_errors)

        assert out.rationale_verdict == "partially_aligned"
        assert out.level_consistent_with_gold is True
        assert out.joint_strict_match is False
        assert out.joint_lenient_match is True

    def test_build_output_with_misaligned_verdict(
        self, sample_judge_input: JudgeInput, valid_judge_output_dict: dict
    ) -> None:
        """_build_output handles misaligned verdict correctly."""
        valid_judge_output_dict["rationale_verdict"] = "misaligned"
        valid_judge_output_dict["rationale_score"] = 0.2

        out = _build_output(sample_judge_input, valid_judge_output_dict)

        assert out.rationale_verdict == "misaligned"
        assert out.joint_strict_match is False
        assert out.joint_lenient_match is False

    def test_build_output_case_insensitive_level_matching(
        self, sample_judge_input: JudgeInput, valid_judge_output_dict: dict
    ) -> None:
        """_build_output performs case-insensitive level matching."""
        sample_judge_input.generated_compliance_level = "high"
        sample_judge_input.gold_fulfillment_level = "HIGH"

        out = _build_output(sample_judge_input, valid_judge_output_dict)

        assert out.level_consistent_with_gold is True

    def test_build_output_whitespace_normalized_in_level_matching(
        self, sample_judge_input: JudgeInput, valid_judge_output_dict: dict
    ) -> None:
        """_build_output strips whitespace from levels before matching."""
        sample_judge_input.generated_compliance_level = "  High  "
        sample_judge_input.gold_fulfillment_level = "High"

        out = _build_output(sample_judge_input, valid_judge_output_dict)

        assert out.level_consistent_with_gold is True

    def test_build_output_error_tags_preserved(
        self, sample_judge_input: JudgeInput, valid_judge_output_with_errors: dict
    ) -> None:
        """_build_output preserves error tags from raw dict."""
        out = _build_output(sample_judge_input, valid_judge_output_with_errors)

        assert out.error_tags == ["omission_of_critical_reasons", "different_interpretation"]

    def test_build_output_explanation_truncated_to_200_chars(
        self, sample_judge_input: JudgeInput
    ) -> None:
        """_build_output truncates judge_explanation to 200 characters."""
        raw = {
            "rationale_verdict": "aligned",
            "rationale_score": 0.8,
            "error_tags": ["none"],
            "judge_explanation": "X" * 300,  # 300 characters
        }

        out = _build_output(sample_judge_input, raw)

        assert len(out.judge_explanation) == 200
        assert out.judge_explanation == "X" * 200

    def test_build_output_missing_explanation_defaults_to_empty(
        self, sample_judge_input: JudgeInput
    ) -> None:
        """_build_output handles missing judge_explanation gracefully."""
        raw = {
            "rationale_verdict": "aligned",
            "rationale_score": 0.8,
            "error_tags": ["none"],
        }

        out = _build_output(sample_judge_input, raw)

        assert out.judge_explanation == ""

    def test_build_output_missing_error_tags_defaults_to_empty_list(
        self, sample_judge_input: JudgeInput
    ) -> None:
        """_build_output handles missing error_tags gracefully."""
        raw = {
            "rationale_verdict": "aligned",
            "rationale_score": 0.8,
            "judge_explanation": "Good.",
        }

        out = _build_output(sample_judge_input, raw)

        assert out.error_tags == []


# ---------------------------------------------------------------------------
# Tests for _cache_key
# ---------------------------------------------------------------------------

class TestCacheKey:
    """Tests for _cache_key(inp, condition, version) function."""

    def test_cache_key_is_string(self, sample_judge_input: JudgeInput) -> None:
        """_cache_key returns a string."""
        key = _cache_key(sample_judge_input, "full_system", "v1_abc123")
        assert isinstance(key, str)

    def test_cache_key_is_deterministic(self, sample_judge_input: JudgeInput) -> None:
        """_cache_key returns same hash for identical inputs."""
        key1 = _cache_key(sample_judge_input, "full_system", "v1_abc123")
        key2 = _cache_key(sample_judge_input, "full_system", "v1_abc123")
        assert key1 == key2

    def test_cache_key_changes_with_different_condition(
        self, sample_judge_input: JudgeInput
    ) -> None:
        """_cache_key differs when condition changes."""
        key1 = _cache_key(sample_judge_input, "full_system", "v1_abc123")
        key2 = _cache_key(sample_judge_input, "no_esrs_context", "v1_abc123")
        assert key1 != key2

    def test_cache_key_changes_with_different_version(
        self, sample_judge_input: JudgeInput
    ) -> None:
        """_cache_key differs when version changes."""
        key1 = _cache_key(sample_judge_input, "full_system", "v1_abc123")
        key2 = _cache_key(sample_judge_input, "full_system", "v1_xyz789")
        assert key1 != key2

    def test_cache_key_changes_with_different_question(
        self, sample_judge_input: JudgeInput
    ) -> None:
        """_cache_key differs when question changes."""
        key1 = _cache_key(sample_judge_input, "full_system", "v1_abc123")

        sample_judge_input.question = "Different question"
        key2 = _cache_key(sample_judge_input, "full_system", "v1_abc123")

        assert key1 != key2

    def test_cache_key_changes_with_different_company(
        self, sample_judge_input: JudgeInput
    ) -> None:
        """_cache_key differs when company changes."""
        key1 = _cache_key(sample_judge_input, "full_system", "v1_abc123")

        sample_judge_input.company = "DifferentCo"
        key2 = _cache_key(sample_judge_input, "full_system", "v1_abc123")

        assert key1 != key2

    def test_cache_key_changes_with_different_gold_level(
        self, sample_judge_input: JudgeInput
    ) -> None:
        """_cache_key differs when gold fulfillment level changes."""
        key1 = _cache_key(sample_judge_input, "full_system", "v1_abc123")

        sample_judge_input.gold_fulfillment_level = "Low"
        key2 = _cache_key(sample_judge_input, "full_system", "v1_abc123")

        assert key1 != key2

    def test_cache_key_changes_with_different_gold_rationale(
        self, sample_judge_input: JudgeInput
    ) -> None:
        """_cache_key differs when gold rationale changes."""
        key1 = _cache_key(sample_judge_input, "full_system", "v1_abc123")

        sample_judge_input.gold_fulfillment_rationale = "Different rationale"
        key2 = _cache_key(sample_judge_input, "full_system", "v1_abc123")

        assert key1 != key2

    def test_cache_key_changes_with_different_generated_level(
        self, sample_judge_input: JudgeInput
    ) -> None:
        """_cache_key differs when generated compliance level changes."""
        key1 = _cache_key(sample_judge_input, "full_system", "v1_abc123")

        sample_judge_input.generated_compliance_level = "Low"
        key2 = _cache_key(sample_judge_input, "full_system", "v1_abc123")

        assert key1 != key2

    def test_cache_key_changes_with_different_generated_rationale(
        self, sample_judge_input: JudgeInput
    ) -> None:
        """_cache_key differs when generated rationale changes."""
        key1 = _cache_key(sample_judge_input, "full_system", "v1_abc123")

        sample_judge_input.generated_compliance_rationale = "Different generated rationale"
        key2 = _cache_key(sample_judge_input, "full_system", "v1_abc123")

        assert key1 != key2

    def test_cache_key_is_hexadecimal(self, sample_judge_input: JudgeInput) -> None:
        """_cache_key returns a hexadecimal string."""
        key = _cache_key(sample_judge_input, "full_system", "v1_abc123")
        try:
            int(key, 16)
            is_hex = True
        except ValueError:
            is_hex = False
        assert is_hex, f"Cache key is not hexadecimal: {key}"

    def test_cache_key_length_is_consistent(self, sample_judge_input: JudgeInput) -> None:
        """_cache_key always returns same length string."""
        key1 = _cache_key(sample_judge_input, "condition1", "version1")
        key2 = _cache_key(sample_judge_input, "condition2", "version2")
        assert len(key1) == len(key2)


# ---------------------------------------------------------------------------
# Tests for get_judge_version
# ---------------------------------------------------------------------------

class TestGetJudgeVersion:
    """Tests for get_judge_version() function."""

    def test_get_judge_version_returns_string(self) -> None:
        """get_judge_version returns a string."""
        version = get_judge_version()
        assert isinstance(version, str)

    def test_get_judge_version_not_empty(self) -> None:
        """get_judge_version returns non-empty string."""
        version = get_judge_version()
        assert len(version) > 0

    def test_get_judge_version_contains_version_marker(self) -> None:
        """get_judge_version includes version prefix."""
        version = get_judge_version()
        assert "v" in version.lower()

    def test_get_judge_version_deterministic(self) -> None:
        """get_judge_version returns same value across calls."""
        version1 = get_judge_version()
        version2 = get_judge_version()
        assert version1 == version2

    def test_get_judge_version_format(self) -> None:
        """get_judge_version returns format like 'v1_<hash>'."""
        version = get_judge_version()
        assert "_" in version, "Version should contain underscore separator"
        parts = version.split("_")
        assert len(parts) == 2, "Version should have two parts separated by underscore"
        assert "v" in parts[0].lower(), "First part should be version"


# ---------------------------------------------------------------------------
# Tests for Constants
# ---------------------------------------------------------------------------

class TestConstants:
    """Tests for module constants."""

    def test_valid_verdicts_is_frozenset(self) -> None:
        """VALID_VERDICTS is a frozenset."""
        assert isinstance(VALID_VERDICTS, frozenset)

    def test_valid_verdicts_contains_expected_values(self) -> None:
        """VALID_VERDICTS contains expected verdict strings."""
        assert "aligned" in VALID_VERDICTS
        assert "partially_aligned" in VALID_VERDICTS
        assert "misaligned" in VALID_VERDICTS

    def test_valid_error_tags_is_frozenset(self) -> None:
        """VALID_ERROR_TAGS is a frozenset."""
        assert isinstance(VALID_ERROR_TAGS, frozenset)

    def test_valid_error_tags_contains_expected_values(self) -> None:
        """VALID_ERROR_TAGS contains expected tag strings."""
        assert "contradiction" in VALID_ERROR_TAGS
        assert "different_interpretation" in VALID_ERROR_TAGS
        assert "omission_of_critical_reasons" in VALID_ERROR_TAGS
        assert "supports_different_level" in VALID_ERROR_TAGS
        assert "none" in VALID_ERROR_TAGS

    def test_valid_error_tags_not_empty(self) -> None:
        """VALID_ERROR_TAGS is not empty."""
        assert len(VALID_ERROR_TAGS) > 0

    def test_valid_verdicts_not_empty(self) -> None:
        """VALID_VERDICTS is not empty."""
        assert len(VALID_VERDICTS) > 0
