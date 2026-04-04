"""
Shared dataclasses and type aliases for the RQ1 evaluation pipeline.

These types form the data model that connects all pipeline stages:
    ReportDocument -> ChunkRecord -> RetrievalHit -> EvalSample -> ConfigResult

Design decisions:
- All types are plain dataclasses (no ORM, no Pydantic) for simplicity
  and zero external dependencies.
- Normalised text is stored alongside raw text to avoid repeated
  normalisation during evaluation (which processes thousands of comparisons).
- ChunkRecord carries positional metadata (start_char, end_char, page numbers)
  to enable span-based relevance matching and citation generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReportDocument:
    """Normalised report document loaded from markdown source.

    Represents one company's sustainability report after PDF-to-Markdown
    preprocessing. The raw_text preserves page markers for offset tracking;
    normalized_text is used for evaluation matching.
    """

    company: str
    year: str
    source_path: str
    raw_text: str
    normalized_text: str
    page_markers: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class ChunkRecord:
    """Chunk unit used for indexing, retrieval, and evaluation.

    Each chunk carries both its raw text (for display and embedding) and
    normalised text (for evaluation matching). The positional fields
    (start_char, end_char) refer to the page-marker-stripped report text.
    """

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

    @property
    def retrieval_text(self) -> str:
        """Text representation used for indexing and ranking.

        Structure-aware and semantic chunks often rely on their section
        headings for semantic context. Prefixing the chunk body with the
        heading hierarchy makes that context available to dense retrieval,
        BM25, and reranking without changing the chunk text used for
        evaluation matching.
        """
        header = " > ".join(part.strip() for part in self.header_hierarchy if part.strip())
        if header:
            return f"[Section: {header}] {self.text}"
        return self.text


@dataclass
class RetrievalHit:
    """Retrieved chunk with scoring details.

    The aux dict carries method-specific metadata such as pre-rerank scores,
    RRF provenance (dense/sparse ranks), and cache status.
    """

    chunk: ChunkRecord
    score: float
    rank: int
    method: str
    aux: dict[str, Any] = field(default_factory=dict)


@dataclass
class GoldEvidence:
    """A single evidence passage from the golden standard.

    Expert-annotated passage that should be retrieved for a given
    question-company pair. The normalised_text is used for matching
    against retrieved chunks.
    """

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
    """Evaluation sample for one question-company pair.

    Groups all gold evidence passages for a single (question, company)
    combination. The evaluation computes how many of these evidence
    passages are covered by the top-k retrieved chunks.
    """

    question_id: str
    question: str
    esrs_standard: str
    company: str
    evidences: list[GoldEvidence]


@dataclass
class ConfigResult:
    """Metrics and metadata for one RQ1 configuration.

    Stores both aggregate metrics (macro-averaged over all samples) and
    per-ESRS-standard breakdowns. The details dict contains evaluation
    coverage statistics, match diagnostics, and per-sample metrics for
    detailed analysis.
    """

    config_id: str
    chunk_strategy: str
    chunk_size: int
    retrieval_method: str
    metrics: dict[str, float]
    per_standard_metrics: dict[str, dict[str, float]]
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationSample:
    """Full gold sample for RQ2/RQ3 generation evaluation."""

    question_id: str
    question: str
    esrs_standard: str
    esrs_disclosure_id: str
    company: str
    ground_truth_answer: str
    fulfillment_level: str
    fulfillment_rationale: str
    evidences: list[GoldEvidence]


@dataclass
class RetrievedEvidenceChunk:
    """Serializable retrieved chunk payload used by RQ2 generation."""

    chunk_id: str
    text: str
    score: float
    rank: int
    page_start: int | None
    page_end: int | None
    header_hierarchy: list[str] = field(default_factory=list)


@dataclass
class ESRSContextResult:
    """Cached ESRS context payload for one thesis question."""

    question_id: str
    esrs_standard: str
    esrs_disclosure_id: str
    context_text: str
    selected_requirements: list[dict[str, str]] = field(default_factory=list)
    raw_response: str = ""
    model: str = ""


@dataclass
class GeneratedAnswer:
    """Structured answer produced by the generation model."""

    direct_answer: str
    compliance_level: str
    compliance_rationale: str
    key_findings: list[str] = field(default_factory=list)
    disclosure_gaps: list[str] = field(default_factory=list)
    cited_pages: list[str] = field(default_factory=list)
    raw_response: str = ""
    parse_error: str | None = None


@dataclass
class RQ2SampleResult:
    """Per-sample evaluation record for one RQ2 condition."""

    condition: str
    question_id: str
    question: str
    esrs_standard: str
    esrs_disclosure_id: str
    company: str
    gold_fulfillment_level: str
    predicted_fulfillment_level: str
    direct_answer_token_f1: float
    rationale_token_f1: float
    exact_match: bool = False
    ordinal_error: int = 0
    success: bool = True
    error: str | None = None


@dataclass
class RQ2Summary:
    """Aggregate RQ2 metrics and metadata."""

    results_version: str
    model: str
    best_pipeline_name: str
    n_samples: int
    n_conditions: int
    aggregate_metrics: dict[str, dict[str, Any]]
    per_standard_metrics: dict[str, dict[str, dict[str, Any]]]
    majority_baseline: dict[str, Any] = field(default_factory=dict)
