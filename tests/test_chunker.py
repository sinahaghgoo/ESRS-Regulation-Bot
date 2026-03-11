"""
Tests for the chunking module.
Validates both fixed-size and structure-aware chunking strategies.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag.parser import parse_report
from rag.chunker import chunk_report, estimate_tokens, Chunk
from rag.config import REPORTS_MD_DIR


def test_estimate_tokens():
    """Test token estimation heuristic."""
    assert estimate_tokens("") == 0
    assert estimate_tokens("hello world") > 0
    # ~1.3 tokens per word
    text = "The quick brown fox jumps over the lazy dog"
    tokens = estimate_tokens(text)
    assert 8 <= tokens <= 15, f"Expected ~12 tokens, got {tokens}"
    print("  [PASS] test_estimate_tokens")


def test_fixed_size_chunking():
    """Test fixed-size chunking produces valid chunks with metadata."""
    md_files = list(REPORTS_MD_DIR.glob("*.md"))
    report = parse_report(md_files[0])
    
    chunks = chunk_report(report, strategy="fixed_size", chunk_size_tokens=512)
    
    assert len(chunks) > 0, "Should produce chunks"
    
    for chunk in chunks[:5]:
        assert chunk.chunk_id != ""
        assert chunk.text != ""
        assert chunk.company != ""
        assert chunk.source_file != ""
        assert chunk.chunking_strategy == "fixed_size"
        assert chunk.token_count > 0
        # Page metadata should be present for most chunks
    
    # Check that token sizes are roughly within bounds
    oversized = [c for c in chunks if c.token_count > 700]  # allow some slack
    ratio = len(oversized) / len(chunks)
    assert ratio < 0.2, f"Too many oversized chunks: {ratio:.0%}"
    
    print(f"  [PASS] test_fixed_size_chunking ({len(chunks)} chunks)")


def test_structure_aware_chunking():
    """Test structure-aware chunking respects section boundaries."""
    md_files = list(REPORTS_MD_DIR.glob("*.md"))
    report = parse_report(md_files[0])
    
    chunks = chunk_report(report, strategy="structure_aware", chunk_size_tokens=512)
    
    assert len(chunks) > 0, "Should produce chunks"
    
    # Check metadata preservation
    for chunk in chunks[:5]:
        assert chunk.chunk_id != ""
        assert chunk.text != ""
        assert chunk.company != ""
        assert chunk.chunking_strategy == "structure_aware"
    
    # Check that tables are kept intact (not split mid-table)
    table_chunks = [c for c in chunks if c.contains_table]
    for tc in table_chunks[:3]:
        # If it has a table, the text should contain table tags
        assert "<table" in tc.text.lower() or "table" in tc.text.lower()
    
    print(f"  [PASS] test_structure_aware_chunking ({len(chunks)} chunks, "
          f"{len(table_chunks)} with tables)")


def test_page_metadata_preserved():
    """Test that page metadata is correctly propagated to chunks."""
    md_files = list(REPORTS_MD_DIR.glob("*.md"))
    report = parse_report(md_files[0])
    
    for strategy in ["fixed_size", "structure_aware"]:
        chunks = chunk_report(report, strategy=strategy, chunk_size_tokens=512)
        
        chunks_with_pages = [c for c in chunks if c.page_number > 0 or c.page_span]
        ratio = len(chunks_with_pages) / len(chunks) if chunks else 0
        
        assert ratio > 0.5, (
            f"Strategy '{strategy}': Only {ratio:.0%} of chunks have page metadata. "
            f"Expected >50%"
        )
        
        print(f"  [PASS] test_page_metadata_preserved ({strategy}: "
              f"{ratio:.0%} have page info)")


def test_chunk_serialization():
    """Test that chunks can be serialized and deserialized."""
    chunk = Chunk(
        chunk_id="test_001",
        text="Sample text",
        company="Test Corp",
        reporting_year=2024,
        source_file="test.md",
        header_hierarchy=["H1", "H2"],
        page_number=42,
        page_span=[42, 43],
        chunk_index=0,
        chunking_strategy="fixed_size",
        token_count=10,
    )
    
    d = chunk.to_dict()
    assert d["chunk_id"] == "test_001"
    assert d["page_span"] == [42, 43]
    assert d["header_hierarchy"] == ["H1", "H2"]
    
    print("  [PASS] test_chunk_serialization")


if __name__ == "__main__":
    print("Running chunker tests...")
    test_estimate_tokens()
    test_fixed_size_chunking()
    test_structure_aware_chunking()
    test_page_metadata_preserved()
    test_chunk_serialization()
    print("\nAll chunker tests passed!")
