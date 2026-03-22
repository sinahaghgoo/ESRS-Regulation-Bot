"""
Unit tests for the evaluation metrics module.

Tests cover:
- Span-based relevance matching
- Token-overlap fallback matching
- Metric computation (Recall, Precision, F1, nDCG, MRR)
- Edge cases (empty inputs, short evidence, no relevant chunks)
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path for imports.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.evaluation.metrics import (
    DEFAULT_RELEVANCE_THRESHOLD,
    MatchDiagnostics,
    _compute_for_sample,
    _dcg_at_k,
    _find_span,
    _is_relevant,
    _mrr_at_k,
    _ndcg_at_k,
    _overlap,
    _token_overlap_ratio,
    _tokenize,
)
from rag.types import ChunkRecord, EvalSample, GoldEvidence, RetrievalHit


def _make_chunk(chunk_id: str, text: str, company: str = "TestCo") -> ChunkRecord:
    """Helper: create a minimal ChunkRecord for testing."""
    from rag.data.text_utils import normalize_text
    return ChunkRecord(
        chunk_id=chunk_id,
        company=company,
        year="2024",
        strategy="fixed",
        chunk_size=1024,
        text=text,
        normalized_text=normalize_text(text),
        token_count=len(text.split()),
        start_char=0,
        end_char=len(text),
        page_start=1,
        page_end=1,
    )


def _make_evidence(passage_id: str, text: str, company: str = "TestCo") -> GoldEvidence:
    """Helper: create a minimal GoldEvidence for testing."""
    from rag.data.text_utils import normalize_text
    return GoldEvidence(
        passage_id=passage_id,
        company=company,
        question_id="Q1",
        question="Test question",
        esrs_standard="ESRS E1",
        text=text,
        normalized_text=normalize_text(text),
        page_number="1",
        source_section_header="Test",
    )


def _make_hit(chunk: ChunkRecord, rank: int, score: float = 1.0) -> RetrievalHit:
    """Helper: create a minimal RetrievalHit."""
    return RetrievalHit(chunk=chunk, score=score, rank=rank, method="test")


# ---------------------------------------------------------------------------
# Test low-level utilities
# ---------------------------------------------------------------------------

class TestFindSpan:
    def test_found(self):
        assert _find_span("hello world", "world") == (6, 11)

    def test_not_found(self):
        assert _find_span("hello world", "xyz") is None

    def test_empty_needle(self):
        assert _find_span("hello", "") is None

    def test_beginning(self):
        assert _find_span("hello world", "hello") == (0, 5)


class TestOverlap:
    def test_full_overlap(self):
        assert _overlap((0, 10), (0, 10)) == 10

    def test_partial_overlap(self):
        assert _overlap((0, 10), (5, 15)) == 5

    def test_no_overlap(self):
        assert _overlap((0, 5), (10, 15)) == 0

    def test_contained(self):
        assert _overlap((0, 20), (5, 10)) == 5


class TestTokenize:
    def test_basic(self):
        assert _tokenize("Hello World!") == ["hello", "world"]

    def test_numbers(self):
        assert _tokenize("CO2 emissions 2024") == ["co2", "emissions", "2024"]

    def test_empty(self):
        assert _tokenize("") == []


class TestTokenOverlapRatio:
    def test_identical(self):
        assert _token_overlap_ratio("hello world", "hello world") == 1.0

    def test_no_overlap(self):
        assert _token_overlap_ratio("hello world", "foo bar") == 0.0

    def test_partial(self):
        ratio = _token_overlap_ratio("hello world test", "hello world foo")
        assert abs(ratio - 2/3) < 0.01

    def test_empty_gold(self):
        assert _token_overlap_ratio("", "hello") == 0.0

    def test_multiplicity(self):
        # Gold has "hello" twice, chunk has it once -> only 1 match.
        ratio = _token_overlap_ratio("hello hello world", "hello world")
        assert abs(ratio - 2/3) < 0.01


# ---------------------------------------------------------------------------
# Test relevance matching
# ---------------------------------------------------------------------------

class TestIsRelevant:
    def test_span_match_above_threshold(self):
        """Chunk covers >50% of gold passage -> relevant."""
        from rag.data.text_utils import normalize_text
        # Simulate: report text contains both gold and chunk text.
        gold_text = "the company reported significant emissions reductions"
        chunk_text = "the company reported significant emissions reductions in 2024 with detailed data"
        report_text = normalize_text(chunk_text)  # chunk is superset

        chunk = _make_chunk("c1", chunk_text)
        evidence = _make_evidence("e1", gold_text)
        report_map = {"TestCo": report_text}

        result = _is_relevant(
            chunk, evidence, report_map,
            chunk_span_cache={}, gold_span_cache={},
        )
        assert result is True

    def test_span_match_below_threshold(self):
        """Chunk covers <50% of gold passage -> not relevant."""
        from rag.data.text_utils import normalize_text
        gold_text = "the company reported significant emissions reductions throughout all operations in europe and asia"
        # Chunk only contains a small part.
        chunk_text = "unrelated text about financial performance the company reported"
        report_text = normalize_text(
            "unrelated text about financial performance the company reported "
            "significant emissions reductions throughout all operations in europe and asia"
        )

        chunk = _make_chunk("c1", chunk_text)
        evidence = _make_evidence("e1", gold_text)
        report_map = {"TestCo": report_text}

        result = _is_relevant(
            chunk, evidence, report_map,
            chunk_span_cache={}, gold_span_cache={},
        )
        assert result is False

    def test_empty_evidence_returns_false(self):
        chunk = _make_chunk("c1", "some text")
        evidence = _make_evidence("e1", "")
        result = _is_relevant(
            chunk, evidence, {"TestCo": "some text"},
            chunk_span_cache={}, gold_span_cache={},
        )
        assert result is False

    def test_short_evidence_skipped_in_fallback(self):
        """Evidence <20 chars should be skipped in token fallback."""
        chunk = _make_chunk("c1", "short text here")
        evidence = _make_evidence("e1", "short")  # 5 chars
        diag = MatchDiagnostics()
        result = _is_relevant(
            chunk, evidence, {"TestCo": "different text"},
            chunk_span_cache={}, gold_span_cache={},
            diagnostics=diag,
        )
        assert result is False
        assert diag.short_evidence_skips > 0

    def test_custom_threshold(self):
        """Custom threshold should change relevance decision."""
        from rag.data.text_utils import normalize_text
        gold_text = "emissions reductions in all operations"
        chunk_text = "emissions reductions in all operations with detailed monitoring data and quarterly reports"
        report_text = normalize_text(chunk_text)

        chunk = _make_chunk("c1", chunk_text)
        evidence = _make_evidence("e1", gold_text)
        report_map = {"TestCo": report_text}

        # Should pass at 0.3 threshold.
        assert _is_relevant(
            chunk, evidence, report_map,
            chunk_span_cache={}, gold_span_cache={},
            threshold=0.3,
        ) is True


# ---------------------------------------------------------------------------
# Test ranking metrics
# ---------------------------------------------------------------------------

class TestDCG:
    def test_perfect_ranking(self):
        rels = [1, 1, 1, 0, 0]
        assert _dcg_at_k(rels, 3) > 0

    def test_no_relevant(self):
        assert _dcg_at_k([0, 0, 0], 3) == 0.0

    def test_single_relevant(self):
        # DCG with single hit at rank 1: 1/log2(2) = 1.0
        assert abs(_dcg_at_k([1, 0, 0], 3) - 1.0) < 0.01


class TestNDCG:
    def test_perfect_ranking(self):
        assert _ndcg_at_k([1, 1, 0, 0], 4) == 1.0

    def test_no_relevant(self):
        assert _ndcg_at_k([0, 0, 0], 3) == 0.0

    def test_imperfect_ranking(self):
        # Relevant at rank 2 instead of 1.
        ndcg = _ndcg_at_k([0, 1, 0], 3)
        assert 0 < ndcg < 1.0


class TestMRR:
    def test_first_position(self):
        assert _mrr_at_k([1, 0, 0], 3) == 1.0

    def test_second_position(self):
        assert _mrr_at_k([0, 1, 0], 3) == 0.5

    def test_no_relevant(self):
        assert _mrr_at_k([0, 0, 0], 3) == 0.0


# ---------------------------------------------------------------------------
# Test per-sample computation
# ---------------------------------------------------------------------------

class TestComputeForSample:
    def test_perfect_retrieval(self):
        """All gold evidence found in top-5 -> perfect scores."""
        from rag.data.text_utils import normalize_text

        gold_text = "emissions were reduced by thirty percent in 2024"
        chunk_text = "emissions were reduced by thirty percent in 2024 according to the annual report"
        report_text = normalize_text(chunk_text)

        chunk = _make_chunk("c1", chunk_text)
        evidence = _make_evidence("e1", gold_text)
        hit = _make_hit(chunk, rank=1)

        sample = EvalSample(
            question_id="Q1", question="Test?",
            esrs_standard="ESRS E1", company="TestCo",
            evidences=[evidence],
        )

        metrics = _compute_for_sample(
            [hit], sample, {"TestCo": report_text},
            chunk_span_cache={}, gold_span_cache={},
        )

        assert metrics["recall@5"] == 1.0
        assert metrics["recall@15"] == 1.0
        assert metrics["mrr@10"] == 1.0
        assert metrics["f1@5"] > 0

    def test_no_relevant_chunks(self):
        """No overlap -> all metrics should be 0."""
        from rag.data.text_utils import normalize_text

        chunk = _make_chunk("c1", "completely unrelated text about financial markets")
        evidence = _make_evidence("e1", "emissions were reduced by thirty percent in 2024")
        hit = _make_hit(chunk, rank=1)

        report_text = normalize_text(
            "completely unrelated text about financial markets. "
            "emissions were reduced by thirty percent in 2024"
        )

        sample = EvalSample(
            question_id="Q1", question="Test?",
            esrs_standard="ESRS E1", company="TestCo",
            evidences=[evidence],
        )

        metrics = _compute_for_sample(
            [hit], sample, {"TestCo": report_text},
            chunk_span_cache={}, gold_span_cache={},
        )

        assert metrics["recall@15"] == 0.0
        assert metrics["mrr@10"] == 0.0

    def test_f1_computation(self):
        """F1 should be harmonic mean of precision and recall."""
        from rag.data.text_utils import normalize_text

        gold_text = "climate change mitigation strategies"
        chunk_text = "climate change mitigation strategies were implemented across all business units"
        report_text = normalize_text(chunk_text)

        chunk = _make_chunk("c1", chunk_text)
        evidence = _make_evidence("e1", gold_text)
        hit = _make_hit(chunk, rank=1)

        sample = EvalSample(
            question_id="Q1", question="Test?",
            esrs_standard="ESRS E1", company="TestCo",
            evidences=[evidence],
        )

        metrics = _compute_for_sample(
            [hit], sample, {"TestCo": report_text},
            chunk_span_cache={}, gold_span_cache={},
        )

        p = metrics["precision@5"]
        r = metrics["recall@5"]
        expected_f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        assert abs(metrics["f1@5"] - expected_f1) < 0.001


# ---------------------------------------------------------------------------
# Test statistical tests module
# ---------------------------------------------------------------------------

class TestStatisticalTests:
    def test_bootstrap_ci(self):
        from rag.evaluation.statistical_tests import bootstrap_ci
        values = [0.8, 0.85, 0.9, 0.75, 0.88, 0.92, 0.87, 0.83]
        result = bootstrap_ci(values, metric_name="recall@15")
        assert result.ci_lower < result.mean < result.ci_upper
        assert result.n_samples == len(values)
        assert result.confidence_level == 0.95

    def test_bootstrap_empty(self):
        from rag.evaluation.statistical_tests import bootstrap_ci
        result = bootstrap_ci([], metric_name="recall@15")
        assert result.mean == 0.0

    def test_permutation_test_identical(self):
        from rag.evaluation.statistical_tests import paired_permutation_test
        values = [0.8, 0.85, 0.9, 0.75]
        result = paired_permutation_test(values, values, "A", "B")
        assert result.p_value > 0.05  # Not significant (identical).
        assert result.significant is False

    def test_permutation_test_different(self):
        from rag.evaluation.statistical_tests import paired_permutation_test
        a = [0.9, 0.95, 0.92, 0.88, 0.93, 0.91, 0.94, 0.90] * 25
        b = [0.5, 0.55, 0.52, 0.48, 0.53, 0.51, 0.54, 0.50] * 25
        result = paired_permutation_test(a, b, "A", "B")
        assert result.p_value < 0.05  # Should be significant.
        assert result.significant is True
        assert result.observed_diff > 0


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

def run_all_tests():
    """Run all test classes and report results."""
    import traceback

    test_classes = [
        TestFindSpan, TestOverlap, TestTokenize, TestTokenOverlapRatio,
        TestIsRelevant, TestDCG, TestNDCG, TestMRR,
        TestComputeForSample, TestStatisticalTests,
    ]

    total = 0
    passed = 0
    failed = 0
    errors = []

    for cls in test_classes:
        instance = cls()
        for method_name in dir(instance):
            if not method_name.startswith("test_"):
                continue
            total += 1
            method = getattr(instance, method_name)
            try:
                method()
                passed += 1
                print(f"  PASS: {cls.__name__}.{method_name}")
            except Exception as e:
                failed += 1
                tb = traceback.format_exc()
                errors.append((f"{cls.__name__}.{method_name}", str(e), tb))
                print(f"  FAIL: {cls.__name__}.{method_name}: {e}")

    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if errors:
        print(f"\nFailed tests:")
        for name, err, tb in errors:
            print(f"  {name}: {err}")
    print(f"{'='*60}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
