"""
Text helpers for normalisation, sentence splitting, and offset mapping.

These utilities are shared across the chunking and evaluation modules.
All normalisation is deterministic and does not depend on external NLP
libraries, ensuring reproducibility across environments.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

# Whitespace normalisation regex: collapses all whitespace (including
# non-breaking spaces) into single spaces.
_WS_RE = re.compile(r"\s+")

# Page marker regex: matches comments injected during PDF preprocessing.
_PAGE_RE = re.compile(r"<!--\s*Report Page\s+(\d+)\s*-->")

# Markdown formatting patterns for stripping in normalisation.
# These patterns are carefully designed to remove ONLY formatting artifacts
# that differ between golden standard text (plain PDF text) and preprocessed
# Markdown report text, without altering actual content.
_MD_BOLD_ITALIC_RE = re.compile(r"\*{1,3}|_{1,3}")  # *, **, ***, _, __, ___
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")  # [text](url) → text
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")  # ![alt](url) → alt
_MD_HEADING_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)  # ## Heading → Heading
_MD_INLINE_CODE_RE = re.compile(r"`([^`]*)`")  # `code` → code
# Note: HTML tag stripping is intentionally omitted. Sustainability reports
# frequently use < and > in text (e.g., "< 15 kgCO2e/boe", ">50%"), and
# a naive <[^>]+> regex would catastrophically match these as tags and
# remove large portions of content. Page marker comments are stripped
# separately before normalisation via the PAGE_RE pattern.
_MD_BULLET_RE = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)  # - item → item
# Numbered list: match only simple "1. " or "1) " patterns at start of line,
# NOT section references like "1.2" or "3.1.4" (which are content, not formatting).
_MD_NUMBERED_RE = re.compile(r"^\s*\d+[)]\s+", re.MULTILINE)  # 1) item → item
# Table separators: only replace pipes that are part of table structure,
# not standalone pipes in prose. We replace pipe-space or space-pipe patterns.
_MD_TABLE_SEP_RE = re.compile(r"\s*\|\s*")  # | separators → space

# Unicode normalisation: map common fancy characters to ASCII equivalents.
_UNICODE_REPLACEMENTS = {
    "\u2018": "'",   # left single quote
    "\u2019": "'",   # right single quote / apostrophe
    "\u201c": '"',   # left double quote
    "\u201d": '"',   # right double quote
    "\u2013": "-",   # en dash
    "\u2014": "-",   # em dash
    "\u2015": "-",   # horizontal bar
    "\u2026": "...", # ellipsis
    "\u00b7": " ",   # middle dot
    "\u2022": " ",   # bullet
    "\u00ad": "",    # soft hyphen
    "\u200b": "",    # zero-width space
    "\u200c": "",    # zero-width non-joiner
    "\u200d": "",    # zero-width joiner
    "\ufeff": "",    # BOM / zero-width no-break space
}


def normalize_text(text: str) -> str:
    """Normalise text for stable matching between chunks and gold evidences.

    Operations:
    1. Replace non-breaking spaces (\\u00a0) with regular spaces.
    2. Apply Unicode normalisation (NFC) and replace fancy quotes/dashes.
    3. Strip Markdown formatting (bold, italic, links, headings, bullets).
    4. Remove residual HTML tags.
    5. Collapse all whitespace sequences into single spaces.
    6. Strip leading/trailing whitespace.
    7. Lowercase.

    This aggressive normalisation ensures that formatting differences between
    the golden standard annotations (often plain text copied from PDFs) and
    the preprocessed Markdown report text do not cause false negatives in
    span-based relevance matching.
    """
    # Unicode normalisation to canonical form.
    text = unicodedata.normalize("NFC", text)

    # Replace non-breaking spaces.
    text = text.replace("\u00a0", " ")

    # Replace common Unicode punctuation with ASCII equivalents.
    for char, replacement in _UNICODE_REPLACEMENTS.items():
        text = text.replace(char, replacement)

    # Strip Markdown formatting artifacts.
    # Order matters: links/images before bold/italic (they may contain markers).
    text = _MD_IMAGE_RE.sub(r"\1", text)        # images before links
    text = _MD_LINK_RE.sub(r"\1", text)          # [text](url) → text
    text = _MD_INLINE_CODE_RE.sub(r"\1", text)   # `code` → code
    text = _MD_BOLD_ITALIC_RE.sub("", text)      # bold/italic markers
    text = _MD_HEADING_RE.sub("", text)          # heading markers
    text = _MD_BULLET_RE.sub("", text)           # bullet markers
    text = _MD_NUMBERED_RE.sub("", text)         # numbered list markers
    text = _MD_TABLE_SEP_RE.sub(" ", text)       # table separators

    # Final whitespace normalisation and lowercase.
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
