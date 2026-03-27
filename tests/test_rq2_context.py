"""Tests for full-standard ESRS context injection in RQ2."""

import json
import sys
from pathlib import Path

# Ensure project root is on sys.path for imports.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.config.settings import ESRS_JSON_DIR, GOLDEN_STANDARD_PATH
from rag.generation.esrs_context import ESRSContextExtractor, build_full_standard_context
from rag.types import GenerationSample


def _make_generation_sample() -> GenerationSample:
    return GenerationSample(
        question_id="Q3",
        question=(
            "Does the undertaking have a transition plan for climate change mitigation? "
            "If yes, is the transition plan aligned with the objective of limiting global "
            "warming to 1.5°C?"
        ),
        esrs_standard="ESRS E1",
        esrs_disclosure_id="E1-1",
        company="TestCo",
        ground_truth_answer="",
        fulfillment_level="High",
        fulfillment_rationale="",
        evidences=[],
    )


def test_build_full_standard_context_includes_multiple_disclosures() -> None:
    payload = json.loads((ESRS_JSON_DIR / "ESRS_E1.json").read_text(encoding="utf-8"))
    context_text, selected = build_full_standard_context(
        payload,
        esrs_standard="ESRS E1",
        esrs_disclosure_id="E1-1",
        question="Transition plan question",
    )

    assert "Full ESRS standard context:" in context_text
    assert "Disclosure Requirement E1-1" in context_text
    assert "Disclosure Requirement E1-4" in context_text
    assert len(selected) > 20


def test_context_extractor_serializes_full_standard_without_openai(tmp_path: Path) -> None:
    extractor = ESRSContextExtractor(cache_path=tmp_path / "context_cache.json", require_openai=False)
    result = extractor.get_or_extract(_make_generation_sample())

    assert result.raw_response == ""
    assert "Disclosure Requirement E1-1" in result.context_text
    assert "Disclosure Requirement E1-4" in result.context_text
    assert result.model == "deterministic_full_standard"


def test_question_to_standard_mapping_matches_expected() -> None:
    payload = json.loads(GOLDEN_STANDARD_PATH.read_text(encoding="utf-8"))
    mapping = {question["id"]: question["esrs_standard"] for question in payload["questions"]}

    assert mapping == {
        "Q1": "ESRS 2",
        "Q2": "ESRS 2",
        "Q3": "ESRS E1",
        "Q4": "ESRS E2",
        "Q5": "ESRS E3",
        "Q6": "ESRS E4",
        "Q7": "ESRS E5",
        "Q8": "ESRS S1",
        "Q9": "ESRS S2",
        "Q10": "ESRS S3",
        "Q11": "ESRS S4",
        "Q12": "ESRS G1",
    }
