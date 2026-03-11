"""
Tests for the report parser module.
Validates page marker extraction, heading hierarchy, and table handling.
"""

import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag.parser import parse_report, PAGE_MARKER_RE
from rag.config import REPORTS_MD_DIR


def test_page_marker_regex():
    """Test that page marker regex matches correctly."""
    assert PAGE_MARKER_RE.search("<!-- Report Page 35 -->")
    assert PAGE_MARKER_RE.search("<!-- Report Page 100 -->")
    assert PAGE_MARKER_RE.search("<!--  Report Page 42  -->")
    assert not PAGE_MARKER_RE.search("<!-- Some other comment -->")
    print("  [PASS] test_page_marker_regex")


def test_parse_single_report():
    """Test parsing a single report file."""
    md_files = list(REPORTS_MD_DIR.glob("*.md"))
    assert len(md_files) > 0, "No markdown files found"
    
    report = parse_report(md_files[0])
    
    assert report.company != "", "Company name should not be empty"
    assert report.reporting_year == 2024
    assert len(report.elements) > 100, f"Expected >100 elements, got {len(report.elements)}"
    assert report.total_pages > 0, "Should have detected page markers"
    
    # Check that page numbers are propagated
    elements_with_pages = [e for e in report.elements if e.page_number > 0]
    assert len(elements_with_pages) > 0, "Some elements should have page numbers"
    
    # Check heading hierarchy
    headings = [e for e in report.elements if e.element_type == "heading"]
    assert len(headings) > 0, "Should have parsed headings"
    
    # Check that tables are preserved
    tables = [e for e in report.elements if e.element_type == "table"]
    assert len(tables) > 0, "Should have parsed tables"
    
    print(f"  [PASS] test_parse_single_report ({report.company}: "
          f"{len(report.elements)} elements, {report.total_pages} pages)")


def test_heading_hierarchy_tracking():
    """Test that heading hierarchy is correctly maintained."""
    md_files = list(REPORTS_MD_DIR.glob("*.md"))
    report = parse_report(md_files[0])
    
    # Find an element with a non-empty header hierarchy
    elements_with_hierarchy = [
        e for e in report.elements 
        if len(e.header_hierarchy) > 1
    ]
    assert len(elements_with_hierarchy) > 0, "Should have elements with multi-level headers"
    
    # Verify hierarchy is a list of strings
    for e in elements_with_hierarchy[:5]:
        assert all(isinstance(h, str) for h in e.header_hierarchy)
        assert all(len(h) > 0 for h in e.header_hierarchy)
    
    print(f"  [PASS] test_heading_hierarchy_tracking")


def test_page_span_on_tables():
    """Test that tables spanning multiple pages track page spans."""
    md_files = list(REPORTS_MD_DIR.glob("*.md"))
    report = parse_report(md_files[0])
    
    tables = [e for e in report.elements if e.element_type == "table"]
    tables_with_span = [t for t in tables if len(t.page_span) > 1]
    
    # It's OK if no tables span multiple pages, but if they do, verify format
    if tables_with_span:
        for t in tables_with_span[:3]:
            assert all(isinstance(p, int) for p in t.page_span)
            assert t.page_span == sorted(t.page_span)
        print(f"  [PASS] test_page_span_on_tables ({len(tables_with_span)} multi-page tables)")
    else:
        print(f"  [PASS] test_page_span_on_tables (no multi-page tables found, OK)")


if __name__ == "__main__":
    print("Running parser tests...")
    test_page_marker_regex()
    test_parse_single_report()
    test_heading_hierarchy_tracking()
    test_page_span_on_tables()
    print("\nAll parser tests passed!")
