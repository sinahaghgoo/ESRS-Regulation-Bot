"""
Unit tests for chunking strategies.

Tests cover:
- Fixed-size chunking: correct token counts, overlap, boundary handling
- Structure-aware chunking: heading hierarchy, section boundaries
- Semantic chunking: merging behaviour, size constraints
- Edge cases: empty input, single sentence, oversized sections
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.chunking.strategies import (
    _iter_structured_sections,
    _merge_small_groups,
    _split_by_tokens,
    _encoding,
    _token_count,
    build_chunks_for_config,
)
from rag.config.settings import PipelineConfig
from rag.types import ReportDocument
from rag.data.text_utils import normalize_text, extract_page_markers


def _make_report(text: str, company: str = "TestCo") -> dict[str, ReportDocument]:
    return {
        company: ReportDocument(
            company=company,
            year="2024",
            source_path="test.md",
            raw_text=text,
            normalized_text=normalize_text(text),
            page_markers=extract_page_markers(text),
        )
    }


# ---------------------------------------------------------------------------
# Structure parsing tests
# ---------------------------------------------------------------------------

class TestStructuredSections:
    def test_single_heading(self):
        text = "# Introduction\nThis is the introduction paragraph."
        sections = _iter_structured_sections(text)
        assert len(sections) >= 1
        # The section text should contain the paragraph.
        assert "introduction paragraph" in sections[0][0].lower()

    def test_multiple_headings(self):
        text = "# Section A\nContent A.\n# Section B\nContent B."
        sections = _iter_structured_sections(text)
        assert len(sections) == 2

    def test_heading_hierarchy(self):
        text = "# H1\n## H2\nContent under H2."
        sections = _iter_structured_sections(text)
        # The deepest section should have hierarchy ["H1", "H2"].
        last = sections[-1]
        assert "H1" in last[3]
        assert "H2" in last[3]

    def test_page_markers_excluded(self):
        text = "# Test\n<!-- Report Page 1 -->\nActual content here."
        sections = _iter_structured_sections(text)
        for sec_text, _, _, _ in sections:
            assert "Report Page" not in sec_text


# ---------------------------------------------------------------------------
# Token window splitting tests
# ---------------------------------------------------------------------------

class TestSplitByTokens:
    def test_short_text_single_chunk(self):
        enc = _encoding()
        text = "This is a short text."
        chunks = _split_by_tokens(text, 0, enc, max_tokens=100, overlap_tokens=10, hierarchy=[])
        assert len(chunks) == 1
        assert chunks[0][0].strip() == text

    def test_overlap_produces_more_chunks(self):
        enc = _encoding()
        # Create text that's about 20 tokens.
        text = " ".join(["word"] * 20)
        no_overlap = _split_by_tokens(text, 0, enc, max_tokens=10, overlap_tokens=0, hierarchy=[])
        with_overlap = _split_by_tokens(text, 0, enc, max_tokens=10, overlap_tokens=3, hierarchy=[])
        # Overlap should produce at least as many chunks.
        assert len(with_overlap) >= len(no_overlap)

    def test_respects_max_tokens(self):
        enc = _encoding()
        text = " ".join(["word"] * 100)
        chunks = _split_by_tokens(text, 0, enc, max_tokens=20, overlap_tokens=2, hierarchy=[])
        for chunk_text, _, _, _ in chunks:
            assert _token_count(chunk_text, enc) <= 21  # Allow 1 token tolerance.


# ---------------------------------------------------------------------------
# Semantic group merging tests
# ---------------------------------------------------------------------------

class TestMergeSmallGroups:
    def test_merge_below_ceiling(self):
        enc = _encoding()
        groups = [["Hello."], ["World."], ["Test."]]
        merged = _merge_small_groups(groups, max_tokens=100, enc=enc)
        # All short groups should merge into one.
        assert len(merged) == 1

    def test_no_merge_above_ceiling(self):
        enc = _encoding()
        # Each group is ~10 tokens. With max_tokens=12, can merge at most 1 pair.
        groups = [
            ["word " * 10],
            ["word " * 10],
            ["word " * 10],
        ]
        merged = _merge_small_groups(groups, max_tokens=12, enc=enc)
        # Should not merge all into one (would exceed ceiling).
        assert len(merged) >= 2

    def test_empty_input(self):
        enc = _encoding()
        assert _merge_small_groups([], max_tokens=100, enc=enc) == []

    def test_single_group(self):
        enc = _encoding()
        groups = [["Single sentence."]]
        merged = _merge_small_groups(groups, max_tokens=100, enc=enc)
        assert len(merged) == 1


# ---------------------------------------------------------------------------
# End-to-end chunking tests
# ---------------------------------------------------------------------------

class TestBuildChunksForConfig:
    def _sample_report_text(self) -> str:
        """Generate a realistic-ish report text for testing."""
        paragraphs = []
        for i in range(5):
            paragraphs.append(f"# Section {i+1}")
            paragraphs.append(
                f"This is section {i+1} of the sustainability report. "
                f"It contains information about environmental performance, "
                f"climate targets, emission reductions, and regulatory compliance. "
                f"The company has made significant progress in reducing its "
                f"carbon footprint through investment in renewable energy sources. " * 3
            )
        return "\n".join(paragraphs)

    def test_fixed_chunking_produces_chunks(self):
        text = self._sample_report_text()
        reports = _make_report(text)
        config = PipelineConfig(chunk_strategy="fixed", chunk_size=256, retrieval_method="dense")
        chunks, stats = build_chunks_for_config(reports, config)
        assert len(chunks) > 0
        assert stats.total_chunks == len(chunks)
        assert stats.mean_tokens > 0

    def test_structure_aware_preserves_hierarchy(self):
        text = "# Main Section\n## Subsection\nContent about emissions."
        reports = _make_report(text)
        config = PipelineConfig(
            chunk_strategy="structure_aware", chunk_size=1024, retrieval_method="dense"
        )
        chunks, stats = build_chunks_for_config(reports, config)
        assert len(chunks) > 0
        # At least one chunk should have heading hierarchy.
        has_hierarchy = any(len(c.header_hierarchy) > 0 for c in chunks)
        assert has_hierarchy

    def test_semantic_chunking_reasonable_sizes(self):
        """After the fix, semantic chunks should be larger than single sentences."""
        text = self._sample_report_text()
        reports = _make_report(text)
        config = PipelineConfig(
            chunk_strategy="semantic", chunk_size=512, retrieval_method="dense"
        )
        chunks, stats = build_chunks_for_config(reports, config)
        assert len(chunks) > 0
        # Mean tokens should be substantially above single-sentence size (~15-20 tokens).
        # With merging, we expect at least 50+ tokens on average.
        assert stats.mean_tokens > 30, (
            f"Semantic chunks still too small: mean={stats.mean_tokens:.1f} tokens"
        )

    def test_chunk_ids_are_unique(self):
        text = self._sample_report_text()
        reports = _make_report(text)
        config = PipelineConfig(chunk_strategy="fixed", chunk_size=256, retrieval_method="dense")
        chunks, _ = build_chunks_for_config(reports, config)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs are not unique"

    def test_all_chunks_have_normalised_text(self):
        text = self._sample_report_text()
        reports = _make_report(text)
        config = PipelineConfig(chunk_strategy="fixed", chunk_size=256, retrieval_method="dense")
        chunks, _ = build_chunks_for_config(reports, config)
        for chunk in chunks:
            assert chunk.normalized_text, f"Chunk {chunk.chunk_id} has empty normalised text"
            assert chunk.normalized_text == chunk.normalized_text.lower()

    def test_chunk_stats_quartiles(self):
        """ChunkStats should include median and quartiles."""
        text = self._sample_report_text()
        reports = _make_report(text)
        config = PipelineConfig(chunk_strategy="fixed", chunk_size=256, retrieval_method="dense")
        _, stats = build_chunks_for_config(reports, config)
        assert stats.median_tokens > 0
        assert stats.p25_tokens <= stats.median_tokens <= stats.p75_tokens


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

def run_all_tests():
    import traceback

    test_classes = [
        TestStructuredSections, TestSplitByTokens,
        TestMergeSmallGroups, TestBuildChunksForConfig,
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
            print(f"    {tb}")
    print(f"{'='*60}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
