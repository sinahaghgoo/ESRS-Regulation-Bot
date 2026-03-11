"""Generation layer exports."""

from rag.esrs_context import extract_relevant_requirements, get_esrs_requirements_text
from rag.generator import generate_answer, generate_comparison

__all__ = [
    "extract_relevant_requirements",
    "get_esrs_requirements_text",
    "generate_answer",
    "generate_comparison",
]
