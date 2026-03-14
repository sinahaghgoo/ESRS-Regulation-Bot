"""Tests for pipeline compatibility helpers."""

from rag.pipeline import _collection_name, _chunking_cache_tag, _normalize_company_names


def test_normalize_company_names_accepts_stems_and_display_names():
    companies = ["AkerBP_2024", "BASF", "aker bp", "Aker BP"]
    normalized = _normalize_company_names(companies)
    assert normalized == ["Aker BP", "BASF"]


def test_chunking_cache_tag_includes_size_and_overlap():
    config = {
        "chunking_strategy": "structure_aware",
        "chunk_size_tokens": 512,
        "chunk_overlap_tokens": 64,
    }
    assert _chunking_cache_tag(config) == "structure_aware_s512_o64"


def test_collection_name_changes_with_chunk_parameters():
    base = {
        "qdrant_collection_prefix": "esrs_reports",
        "chunking_strategy": "structure_aware",
        "chunk_overlap_tokens": 64,
    }
    cfg_512 = {**base, "chunk_size_tokens": 512}
    cfg_256 = {**base, "chunk_size_tokens": 256}

    name_512 = _collection_name(cfg_512)
    name_256 = _collection_name(cfg_256)

    assert name_512 == "esrs_reports_structure_aware_s512_o64"
    assert name_256 == "esrs_reports_structure_aware_s256_o64"
    assert name_512 != name_256
