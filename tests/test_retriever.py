"""
Tests for retrieval components.
Validates BM25, RRF, page reference handling, and evaluation metrics.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag.retriever import BM25Index, reciprocal_rank_fusion
from rag.evaluator import normalize_page_ref, compute_token_f1, compute_retrieval_metrics
from rag.chunker import Chunk


def test_bm25_basic():
    """Test BM25 index build and search."""
    chunks = [
        Chunk(chunk_id="c1", text="Climate change mitigation greenhouse gas emissions scope 1",
              company="TestCo", reporting_year=2024, source_file="test.md",
              header_hierarchy=["Environment", "Climate"], page_number=10,
              page_span=[10], chunk_index=0, chunking_strategy="test"),
        Chunk(chunk_id="c2", text="Water consumption and marine resources pollution",
              company="TestCo", reporting_year=2024, source_file="test.md",
              header_hierarchy=["Environment", "Water"], page_number=20,
              page_span=[20], chunk_index=1, chunking_strategy="test"),
        Chunk(chunk_id="c3", text="Employee diversity workforce gender pay gap",
              company="TestCo", reporting_year=2024, source_file="test.md",
              header_hierarchy=["Social", "Workforce"], page_number=30,
              page_span=[30], chunk_index=2, chunking_strategy="test"),
    ]
    
    bm25 = BM25Index()
    bm25.build(chunks)
    
    results = bm25.search("greenhouse gas emissions", top_k=3)
    assert len(results) > 0, "Should return results"
    assert results[0]["chunk_id"] == "c1", "Climate chunk should rank first"
    
    results2 = bm25.search("workforce diversity", top_k=3)
    assert results2[0]["chunk_id"] == "c3", "Social chunk should rank first"
    
    print("  [PASS] test_bm25_basic")


def test_bm25_company_filter():
    """Test BM25 company filtering."""
    chunks = [
        Chunk(chunk_id="c1", text="emissions targets", company="CompanyA",
              reporting_year=2024, source_file="a.md", header_hierarchy=["Env"],
              page_number=1, page_span=[1], chunk_index=0, chunking_strategy="t"),
        Chunk(chunk_id="c2", text="emissions reduction", company="CompanyB",
              reporting_year=2024, source_file="b.md", header_hierarchy=["Env"],
              page_number=2, page_span=[2], chunk_index=1, chunking_strategy="t"),
    ]
    
    bm25 = BM25Index()
    bm25.build(chunks)
    
    results = bm25.search("emissions", top_k=5, company_filter="CompanyA")
    assert all(r["company"] == "CompanyA" for r in results)
    
    print("  [PASS] test_bm25_company_filter")


def test_rrf():
    """Test Reciprocal Rank Fusion merging."""
    list1 = [
        {"chunk_id": "a", "text": "doc a", "score": 0.9},
        {"chunk_id": "b", "text": "doc b", "score": 0.7},
        {"chunk_id": "c", "text": "doc c", "score": 0.5},
    ]
    list2 = [
        {"chunk_id": "b", "text": "doc b", "score": 0.95},
        {"chunk_id": "d", "text": "doc d", "score": 0.8},
        {"chunk_id": "a", "text": "doc a", "score": 0.6},
    ]
    
    merged = reciprocal_rank_fusion([list1, list2], k=60)
    
    assert len(merged) == 4, f"Expected 4 unique docs, got {len(merged)}"
    # "b" appears in both lists at high ranks, should be top
    ids = [m["chunk_id"] for m in merged]
    # "a" and "b" should be in top 2 (they appear in both lists)
    assert "b" in ids[:2] and "a" in ids[:2]
    
    print("  [PASS] test_rrf")


def test_normalize_page_ref():
    """Test page reference normalization."""
    assert normalize_page_ref(42) == {42}
    assert normalize_page_ref("42") == {42}
    assert normalize_page_ref("43-44") == {43, 44}
    assert normalize_page_ref("43-45") == {43, 44, 45}
    assert normalize_page_ref("") == set()
    assert normalize_page_ref("Not explicitly marked") == set()
    assert normalize_page_ref("10, 12") == {10, 12}
    
    print("  [PASS] test_normalize_page_ref")


def test_token_f1():
    """Test token-level F1 computation."""
    pred = "The company has reduced its greenhouse gas emissions by 20 percent"
    ref = "The company reduced greenhouse gas emissions by 20 percent through clean energy"
    
    result = compute_token_f1(pred, ref)
    assert 0 < result["token_f1"] < 1
    assert 0 < result["token_precision"] <= 1
    assert 0 < result["token_recall"] <= 1
    
    # Perfect match
    result2 = compute_token_f1("hello world", "hello world")
    assert result2["token_f1"] == 1.0
    
    # No overlap
    result3 = compute_token_f1("alpha beta", "gamma delta")
    assert result3["token_f1"] == 0.0
    
    print("  [PASS] test_token_f1")


def test_retrieval_metrics():
    """Test retrieval metrics computation."""
    retrieved = [
        {"page_span": [42, 43], "page_number": 42},
        {"page_span": [50], "page_number": 50},
        {"page_span": [60], "page_number": 60},
    ]
    
    gold_evidence = [
        {"page_number": "42", "source_section_header": "Climate"},
        {"page_number": "43-44", "source_section_header": "Emissions"},
        {"page_number": "55", "source_section_header": "Targets"},
    ]
    
    metrics = compute_retrieval_metrics(retrieved, gold_evidence, k_values=[3])
    
    # Gold pages: {42, 43, 44, 55}
    # Retrieved pages: {42, 43, 50, 60}
    # Overlap: {42, 43}
    assert metrics["recall@3"] > 0, "Should have some recall"
    assert "mrr@3" in metrics
    assert "coverage_ratio" in metrics
    
    print(f"  [PASS] test_retrieval_metrics (recall@3={metrics['recall@3']:.2f})")


if __name__ == "__main__":
    print("Running retriever tests...")
    test_bm25_basic()
    test_bm25_company_filter()
    test_rrf()
    test_normalize_page_ref()
    test_token_f1()
    test_retrieval_metrics()
    print("\nAll retriever tests passed!")
