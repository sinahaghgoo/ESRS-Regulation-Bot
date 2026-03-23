"""Unit tests for lightweight retrieval helpers."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.retrieval.methods import _build_rerank_text, _rerank_blend_weights
from rag.types import ChunkRecord


def _make_chunk(text: str, header: list[str] | None = None) -> ChunkRecord:
    from rag.data.text_utils import normalize_text

    return ChunkRecord(
        chunk_id="c1",
        company="TestCo",
        year="2024",
        strategy="fixed",
        chunk_size=1024,
        text=text,
        normalized_text=normalize_text(text),
        token_count=len(text.split()),
        start_char=0,
        end_char=len(text),
        page_start=1,
        page_end=2,
        header_hierarchy=header or [],
    )


def test_rerank_snippet_prefers_late_relevant_passage():
    prefix = " ".join(["introductory"] * 180)
    relevant = (
        "The undertaking generated waste and directed most of it to recycling, "
        "while the remainder went to incineration and landfill."
    )
    suffix = " ".join(["closing"] * 120)
    chunk = _make_chunk(f"{prefix} {relevant} {suffix}")

    snippet = _build_rerank_text(
        "How much total waste was directed to recycling, incineration, and landfill?",
        chunk,
        max_words=60,
    ).lower()

    assert "recycling" in snippet
    assert "incineration" in snippet
    assert "landfill" in snippet
    assert snippet.count("introductory") < 20


def test_rerank_snippet_keeps_section_context():
    chunk = _make_chunk(
        "This section describes community engagement and grievance mechanisms in detail.",
        header=["Sustainability Statement", "Affected Communities"],
    )

    snippet = _build_rerank_text(
        "What grievance mechanisms are available for affected communities?",
        chunk,
        max_words=40,
    )

    assert snippet.startswith("[Section: Sustainability Statement > Affected Communities]")


def test_long_chunk_rerank_blend_prefers_base_signal():
    weights = _rerank_blend_weights("hybrid_rerank", chunk_size=1024)

    assert weights["base"] > weights["cross"]
    assert abs(sum(weights.values()) - 1.0) < 1e-9
