"""Text helpers for normalization, sentence splitting, and offsets."""

from __future__ import annotations

import re
from collections.abc import Iterable

_WS_RE = re.compile(r"\s+")
_PAGE_RE = re.compile(r"<!--\s*Report Page\s+(\d+)\s*-->")


def normalize_text(text: str) -> str:
    """Normalize text for stable matching between chunks and gold evidences."""
    text = text.replace("\u00a0", " ")
    text = _WS_RE.sub(" ", text).strip().lower()
    return text


def sentence_split(text: str) -> list[str]:
    """Lightweight sentence splitting suitable for semantic chunking fallback."""
    if not text.strip():
        return []
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text.strip())
    return [p.strip() for p in pieces if p.strip()]


def extract_page_markers(markdown_text: str) -> list[tuple[int, int]]:
    """Return tuples of (char_offset, page_number) from report markdown."""
    markers: list[tuple[int, int]] = []
    for match in _PAGE_RE.finditer(markdown_text):
        markers.append((match.start(), int(match.group(1))))
    return markers


def page_for_offset(markers: list[tuple[int, int]], char_offset: int) -> int | None:
    """Resolve page number for a character offset using preceding marker."""
    if not markers:
        return None
    active = None
    for pos, page in markers:
        if pos <= char_offset:
            active = page
        else:
            break
    return active


def iter_ngrams(tokens: list[str], n: int) -> Iterable[tuple[str, ...]]:
    """Yield token n-grams."""
    if n <= 0:
        return
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i : i + n])

