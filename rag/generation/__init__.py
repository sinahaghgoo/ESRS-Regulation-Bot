"""Generation helpers for RQ2 and later end-to-end experiments."""

from .esrs_context import (
    ESRSContextExtractor,
    build_disclosure_context,
    build_full_standard_context,
    build_question_extraction_input,
    extract_esrs_context,
    flatten_esrs_standard,
    load_esrs_standards,
)
from .generator import AnswerGenerator, generate_answer

__all__ = [
    "AnswerGenerator",
    "ESRSContextExtractor",
    "build_disclosure_context",
    "build_full_standard_context",
    "build_question_extraction_input",
    "extract_esrs_context",
    "flatten_esrs_standard",
    "generate_answer",
    "load_esrs_standards",
]
