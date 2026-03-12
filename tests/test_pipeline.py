"""Tests for pipeline compatibility helpers."""

from rag.pipeline import _normalize_company_names


def test_normalize_company_names_accepts_stems_and_display_names():
    companies = ["AkerBP_2024", "BASF", "aker bp", "Aker BP"]
    normalized = _normalize_company_names(companies)
    assert normalized == ["Aker BP", "BASF"]
