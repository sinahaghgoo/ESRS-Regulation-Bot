"""Tests for evaluator metric helpers."""

from rag.evaluator import (
    compute_citation_correctness,
    compute_faithfulness,
    compute_retrieval_metrics,
    normalize_page_ref,
)


def test_normalize_page_ref_variants():
    assert normalize_page_ref(43) == {43}
    assert normalize_page_ref("43") == {43}
    assert normalize_page_ref("43-45") == {43, 44, 45}
    assert normalize_page_ref("43–44") == {43, 44}
    assert normalize_page_ref("43, 50") == {43, 50}
    assert normalize_page_ref("Not explicitly marked") == set()
    assert normalize_page_ref("") == set()


def test_compute_citation_correctness():
    page_references = [
        {"pages": [42]},
        {"pages": [43, 44]},
    ]
    gold_evidence = [
        {"page_number": "42"},
        {"page_number": "44-45"},
    ]
    metrics = compute_citation_correctness(page_references, gold_evidence)
    # cited={42,43,44}, gold={42,44,45}, overlap={42,44}
    assert 0 < metrics["citation_precision"] < 1
    assert 0 < metrics["citation_recall"] < 1
    assert 0 < metrics["citation_f1"] < 1


def test_compute_faithfulness_supported_claims():
    answer = (
        "The company reports scope 1 and 2 emissions reduction. "
        "It also details methane intensity improvements."
    )
    passages = [
        "The report describes scope 1 and 2 emissions reduction targets and progress.",
        "Methane intensity has improved year over year according to monitoring data.",
    ]
    metrics = compute_faithfulness(answer, passages)
    assert metrics["supported_claim_ratio"] >= 0.5
    assert metrics["hallucination_rate"] <= 0.5


def test_compute_retrieval_metrics_page_overlap():
    retrieved = [
        {"page_span": [42, 43], "page_number": 42, "text": "materiality process", "header_hierarchy": ["DMA"]},
        {"page_span": [60], "page_number": 60, "text": "other section", "header_hierarchy": ["Other"]},
    ]
    gold = [
        {"page_number": "42", "text": "materiality process", "source_section_header": "DMA"},
        {"page_number": "44", "text": "threshold scoring", "source_section_header": "Methodology"},
    ]
    metrics = compute_retrieval_metrics(retrieved, gold, k_values=[1, 2])
    assert metrics["recall@1"] > 0
    assert metrics["precision@1"] > 0
    assert "coverage_ratio" in metrics
