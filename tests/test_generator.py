"""Tests for generator citation/page injection helpers."""

from rag.generator import _inject_page_references_in_text


def test_inject_page_references_replaces_source_tags():
    text = "The company has a transition plan [Source 1] and milestones [Source 2]."
    refs = [
        {"source_index": 1, "page_inline": "Page 5-6"},
        {"source_index": 2, "page_inline": "Page 7"},
    ]
    out = _inject_page_references_in_text(text, refs)
    assert "[Page 5-6]" in out
    assert "[Page 7]" in out
    assert "Source 1" not in out


def test_inject_page_references_appends_when_missing_tags():
    text = "The company reports a transition plan."
    refs = [{"source_index": 1, "page_inline": "Page 5-6"}]
    out = _inject_page_references_in_text(text, refs)
    assert "References:" in out
    assert "[Page 5-6]" in out
