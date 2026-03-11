"""Chunk sustainability report elements while preserving citation-critical metadata."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from rag.parser import ParsedElement, ParsedReport


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Approximate token count from whitespace words."""
    if not text:
        return 0
    return max(1, int(len(text.split()) * 1.3))


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """Serializable chunk structure used by indexing/retrieval/evaluation."""

    chunk_id: str
    text: str
    company: str
    reporting_year: int
    source_file: str
    header_hierarchy: List[str]
    page_number: int = -1
    page_span: List[int] = field(default_factory=list)
    chunk_index: int = 0
    chunking_strategy: str = ""
    token_count: int = 0
    contains_table: bool = False
    contains_figure: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class _WordUnit:
    word: str
    pages: Set[int]
    headers: Tuple[str, ...]
    contains_table: bool
    contains_figure: bool


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _clean_company_for_id(company: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", company)


def _generate_chunk_id(company: str, strategy: str, index: int) -> str:
    raw = f"{company}::{strategy}::{index}"
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]
    return f"{_clean_company_for_id(company)}_{strategy[:6]}_{index:05d}_{digest}"


def _element_pages(elem: ParsedElement) -> Set[int]:
    pages: Set[int] = set()
    for p in elem.page_span:
        if isinstance(p, int) and p > 0:
            pages.add(p)
    if not pages and isinstance(elem.page_number, int) and elem.page_number > 0:
        pages.add(elem.page_number)
    return pages


def _element_text(elem: ParsedElement) -> str:
    text = (elem.content or "").strip()
    if not text:
        return ""
    if elem.element_type == "heading":
        return f"{'#' * max(1, elem.heading_level)} {text}"
    return text


def _chunk_from_word_units(
    word_units: Sequence[_WordUnit],
    report: ParsedReport,
    chunk_index: int,
    strategy: str,
) -> Chunk:
    """Build a Chunk object from a list of token-level metadata units."""
    text = " ".join(w.word for w in word_units).strip()

    pages: Set[int] = set()
    contains_table = False
    contains_figure = False
    header_counts: Dict[Tuple[str, ...], int] = {}

    for unit in word_units:
        pages.update(unit.pages)
        contains_table = contains_table or unit.contains_table
        contains_figure = contains_figure or unit.contains_figure
        header_counts[unit.headers] = header_counts.get(unit.headers, 0) + 1

    header_hierarchy = list(max(header_counts.items(), key=lambda x: x[1])[0]) if header_counts else []
    page_span = sorted(pages)
    page_number = page_span[0] if page_span else -1

    return Chunk(
        chunk_id=_generate_chunk_id(report.company, strategy, chunk_index),
        text=text,
        company=report.company,
        reporting_year=report.reporting_year,
        source_file=report.source_file,
        header_hierarchy=header_hierarchy,
        page_number=page_number,
        page_span=page_span,
        chunk_index=chunk_index,
        chunking_strategy=strategy,
        token_count=estimate_tokens(text),
        contains_table=contains_table,
        contains_figure=contains_figure,
    )


# ---------------------------------------------------------------------------
# Fixed-size chunking
# ---------------------------------------------------------------------------

def _to_word_units(report: ParsedReport) -> List[_WordUnit]:
    """Expand parsed elements into word-level units with metadata."""
    units: List[_WordUnit] = []
    for elem in report.elements:
        text = _element_text(elem)
        if not text:
            continue
        pages = _element_pages(elem)
        headers = tuple(elem.header_hierarchy)
        words = text.split()
        for word in words:
            units.append(
                _WordUnit(
                    word=word,
                    pages=set(pages),
                    headers=headers,
                    contains_table=(elem.element_type == "table"),
                    contains_figure=(elem.element_type == "figure"),
                )
            )
    return units


def chunk_fixed_size(report: ParsedReport, chunk_size_tokens: int = 512, chunk_overlap_tokens: int = 64) -> List[Chunk]:
    """Create fixed-size chunks with overlap, preserving page/header metadata."""
    word_units = _to_word_units(report)
    if not word_units:
        return []

    words_per_chunk = max(40, int(chunk_size_tokens / 1.3))
    overlap_words = max(0, min(words_per_chunk - 1, int(chunk_overlap_tokens / 1.3)))
    step = max(1, words_per_chunk - overlap_words)

    chunks: List[Chunk] = []
    chunk_index = 0

    for start in range(0, len(word_units), step):
        end = min(start + words_per_chunk, len(word_units))
        window = word_units[start:end]
        if not window:
            continue
        chunk = _chunk_from_word_units(window, report=report, chunk_index=chunk_index, strategy="fixed_size")
        chunks.append(chunk)
        chunk_index += 1
        if end >= len(word_units):
            break

    return chunks


# ---------------------------------------------------------------------------
# Structure-aware chunking
# ---------------------------------------------------------------------------

def _group_sections(elements: List[ParsedElement]) -> List[List[ParsedElement]]:
    """Group elements into sections split by H1-H3 headings."""
    sections: List[List[ParsedElement]] = []
    current: List[ParsedElement] = []

    for elem in elements:
        is_major_heading = elem.element_type == "heading" and elem.heading_level <= 3
        if is_major_heading and current:
            sections.append(current)
            current = [elem]
            continue
        current.append(elem)

    if current:
        sections.append(current)
    return sections


def _element_to_units(elem: ParsedElement) -> List[_WordUnit]:
    text = _element_text(elem)
    if not text:
        return []
    pages = _element_pages(elem)
    headers = tuple(elem.header_hierarchy)
    words = text.split()
    return [
        _WordUnit(
            word=w,
            pages=set(pages),
            headers=headers,
            contains_table=(elem.element_type == "table"),
            contains_figure=(elem.element_type == "figure"),
        )
        for w in words
    ]


def chunk_structure_aware(report: ParsedReport, chunk_size_tokens: int = 512, chunk_overlap_tokens: int = 64) -> List[Chunk]:
    """Chunk by document structure while respecting page and section boundaries."""
    words_per_chunk = max(40, int(chunk_size_tokens / 1.3))
    overlap_words = max(0, min(words_per_chunk - 1, int(chunk_overlap_tokens / 1.3)))

    chunks: List[Chunk] = []
    chunk_index = 0

    for section in _group_sections(report.elements):
        if not section:
            continue

        # Convert section into unit blocks per element to keep table/figure integrity.
        unit_blocks: List[List[_WordUnit]] = []
        for elem in section:
            units = _element_to_units(elem)
            if units:
                unit_blocks.append(units)

        buffer: List[_WordUnit] = []

        def flush() -> None:
            nonlocal buffer, chunk_index
            if not buffer:
                return
            chunks.append(
                _chunk_from_word_units(
                    buffer,
                    report=report,
                    chunk_index=chunk_index,
                    strategy="structure_aware",
                )
            )
            chunk_index += 1
            if overlap_words > 0:
                buffer = buffer[-overlap_words:]
            else:
                buffer = []

        for block in unit_blocks:
            block_len = len(block)

            # If a single element block is huge, split it internally to stay within limits.
            if block_len > words_per_chunk:
                flush()
                start = 0
                while start < block_len:
                    end = min(start + words_per_chunk, block_len)
                    piece = block[start:end]
                    chunks.append(
                        _chunk_from_word_units(
                            piece,
                            report=report,
                            chunk_index=chunk_index,
                            strategy="structure_aware",
                        )
                    )
                    chunk_index += 1
                    if end >= block_len:
                        break
                    start = max(0, end - overlap_words)
                buffer = []
                continue

            # Flush when adding this block would exceed window size.
            if len(buffer) + block_len > words_per_chunk and buffer:
                flush()

            # Preserve table/figure as atomic block.
            buffer.extend(block)

        flush()

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_report(
    report: ParsedReport,
    strategy: str = "structure_aware",
    chunk_size_tokens: int = 512,
    chunk_overlap_tokens: int = 64,
) -> List[Chunk]:
    """Chunk a parsed report using the selected strategy."""
    if strategy == "fixed_size":
        return chunk_fixed_size(
            report=report,
            chunk_size_tokens=chunk_size_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
        )
    if strategy == "structure_aware":
        return chunk_structure_aware(
            report=report,
            chunk_size_tokens=chunk_size_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
        )
    raise ValueError(f"Unknown chunking strategy: {strategy}")
