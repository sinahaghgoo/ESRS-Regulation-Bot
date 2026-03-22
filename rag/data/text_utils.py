"""
Text helpers for normalisation, sentence splitting, and offset mapping.

These utilities are shared across the chunking and evaluation modules.
All normalisation is deterministic and does not depend on external NLP
libraries, ensuring reproducibility across environments.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# Whitespace normalisation regex: collapses all whitespace (including
# non-breaking spaces) into single spaces.
_WS_RE = re.compile(r"\s+")

# Page marker regex: matches comments injected during PDF preprocessing.
_PAGE_RE = re.compile(r"<!--\s*Report Page\s+(\d+)\s*-->")


def normalize_text(text: str) -> str:
    """Normalise text for stable matching between chunks and gold evidences.

    Operations:
    1. Replace non-breaking spaces (\\u00a0) with regular spaces.
    2. Collapse all whitespace sequences into single spaces.
    3. Strip leading/trailing whitespace.
    4. Lowercase.

    This normalisation ensures that minor formatting differences between
    the golden standard annotations and the preprocessed report text do
    not cause false negatives in relevance matching.
    """
    text = text.replace("\u00a0", " ")
    text = _WS_RE.sub(" ", text).strip().lower()
    return text


def sentence_split(text: str) -> list[str]:
    """Lightweight sentence splitting suitable for semantic chunking.

    Splits on sentence-ending punctuation (.!?) followed by whitespace
    and a capital letter or digit. This is a simple heuristic that works
    well for the formal prose style of sustainability reports.

    Returns an empty list for blank input.
    """
    if not text.strip():
        return []
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text.strip())
    return [p.strip() for p in pieces if p.strip()]


def extract_page_markers(markdown_text: str) -> list[tuple[int, int]]:
    """Extract (char_offset, page_number) tuples from report markdown.

    Page markers are HTML comments like ``<!-- Report Page 42 -->``
    injected during PDF-to-Markdown preprocessing. The returned list
    is sorted by character offset and used for page number resolution.
    """
    markers: list[tuple[int, int]] = []
    for match in _PAGE_RE.finditer(markdown_text):
        markers.append((match.start(), int(match.group(1))))
    return markers


def page_for_offset(markers: list[tuple[int, int]], char_offset: int) -> int | None:
    """Resolve the page number for a character offset.

    Finds the last page marker that precedes *char_offset*. Returns None
    if no markers are available (e.g., when the report has no page markers).
    """
    if not markers:
        return None
    active = None
    for pos, page in markers:
        if pos <= char_offset:
            active = page
        else:
            break
    return active


def extract_marker_spans(raw_text: str) -> list[tuple[int, int]]:
    """Return (start, end) character spans of every page marker in *raw_text*.

    These spans are used by ``stripped_to_raw_offset`` to map positions
    in the page-marker-stripped text back to the original raw text.
    """
    return [(m.start(), m.end()) for m in _PAGE_RE.finditer(raw_text)]


def stripped_to_raw_offset(
    marker_spans: list[tuple[int, int]], stripped_offset: int
) -> int:
    """Convert an offset in page-marker-stripped text to raw text offset.

    After ``re.sub(PAGE_RE, "", raw)``, every character shifts left by
    the total bytes of markers that preceded it. This function reverses
    that mapping so ``page_for_offset`` can be called with raw-text
    page markers.
    """
    cum_removed = 0
    for ms, me in marker_spans:
        if ms - cum_removed <= stripped_offset:
            cum_removed += me - ms
        else:
            break
    return stripped_offset + cum_removed


def iter_ngrams(tokens: list[str], n: int) -> Iterable[tuple[str, ...]]:
    """Yield token n-grams of length *n*."""
    if n <= 0:
        return
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i : i + n])
