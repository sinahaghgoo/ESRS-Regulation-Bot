"""Shared dataclasses and type aliases for RQ1 modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReportDocument:
    """Normalized report document loaded from markdown source."""

    company: str
    year: str
    source_path: str
    raw_text: str
    normalized_text: str
    page_markers: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class ChunkRecord:
    """Chunk unit used for indexing, retrieval, and evaluation."""

    chunk_id: str
    company: str
    year: str
    strategy: str
    chunk_size: int
    text: str
    normalized_text: str
    token_count: int
    start_char: int
    end_char: int
    page_start: int | None
    page_end: int | None
    header_hierarchy: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalHit:
    """Retrieved chunk and scoring details."""

    chunk: ChunkRecord
    score: float
    rank: int
    method: str
    aux: dict[str, Any] = field(default_factory=dict)


@dataclass
class GoldEvidence:
    """A single evidence passage from the golden standard."""

    passage_id: str
    company: str
    question_id: str
    question: str
    esrs_standard: str
    text: str
    normalized_text: str
    page_number: str
    source_section_header: str


@dataclass
class EvalSample:
    """Evaluation sample for one question/company pair."""

    question_id: str
    question: str
    esrs_standard: str
    company: str
    evidences: list[GoldEvidence]


@dataclass
class ConfigResult:
    """Metrics and metadata for one RQ1 configuration."""

    config_id: str
    chunk_strategy: str
    chunk_size: int
    retrieval_method: str
    metrics: dict[str, float]
    per_standard_metrics: dict[str, dict[str, float]]
    details: dict[str, Any] = field(default_factory=dict)

