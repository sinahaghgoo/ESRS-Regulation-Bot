"""
RQ1 chunking implementations: fixed, structure-aware, and semantic.

Three chunking strategies are compared in the RQ1 factorial experiment:

1. **Fixed-size chunking** — Token-based sliding window with configurable
   overlap. Ignores document structure; serves as the baseline. The overlap
   ratio (default 12.5%) is chosen to provide context continuity at chunk
   boundaries without excessive duplication.

2. **Structure-aware chunking** — Respects H1-H3 section boundaries
   extracted during preprocessing. Sections smaller than the target size
   are kept intact; oversized sections are split with the same token-window
   approach as the fixed strategy, preserving the heading hierarchy as
   metadata.

3. **Semantic chunking** — Groups adjacent sentences by embedding
   similarity. Uses sentence-transformers (all-MiniLM-L6-v2) to detect
   topic boundaries, then merges semantically coherent sentence groups
   up to the target token ceiling. This ensures chunks are topically
   cohesive while respecting the size constraint.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import tiktoken

from rag.config.settings import PipelineConfig
from rag.data.text_utils import (
    extract_marker_spans,
    normalize_text,
    page_for_offset,
    sentence_split,
    stripped_to_raw_offset,
)
from rag.types import ChunkRecord, ReportDocument

# Heading regex: matches Markdown H1-H3 lines (e.g. "## Section Title").
_H_RE = re.compile(r"^(#{1,3})\s+(.*)$")

# Page marker regex: matches "<!-- Report Page N -->" comments injected
# during PDF-to-Markdown preprocessing.
_PAGE_MARKER_RE = re.compile(r"<!--\s*Report Page\s+\d+\s*-->")

# Lazily initialised sentence-transformer model for semantic chunking.
_SEMANTIC_MODEL = None


def _get_semantic_model():
    """Lazily initialise sentence-transformer once per process.

    Uses all-MiniLM-L6-v2 (22 M params, ~80 MB) — a lightweight model
    suitable for sentence-level similarity that runs efficiently on CPU.
    """
    global _SEMANTIC_MODEL
    if _SEMANTIC_MODEL is None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        _SEMANTIC_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _SEMANTIC_MODEL


def _encoding():
    """Get the tiktoken encoder for the embedding model.

    Uses the same tokenizer as the embedding model (text-embedding-3-large)
    to ensure token counts are consistent between chunking and embedding.
    """
    try:
        return tiktoken.encoding_for_model("text-embedding-3-large")
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def _token_count(text: str, enc) -> int:
    """Count tokens in *text* using the given tiktoken encoder."""
    return len(enc.encode(text))


def _chunk_id(company: str, strategy: str, size: int, start: int, text: str) -> str:
    """Generate a deterministic chunk ID from its defining properties.

    Uses SHA1 of (company, strategy, size, start_offset, text_prefix) to
    produce a short, collision-resistant identifier.
    """
    digest = hashlib.sha1(
        f"{company}:{strategy}:{size}:{start}:{text[:80]}".encode("utf-8")
    ).hexdigest()
    return digest[:20]


# ---------------------------------------------------------------------------
# Structure parsing
# ---------------------------------------------------------------------------

def _iter_structured_sections(raw_text: str) -> list[tuple[str, int, int, list[str]]]:
    """Split report text by H1-H3 heading boundaries.

    Returns a list of (section_text, start_offset, end_offset, heading_hierarchy)
    tuples. The heading hierarchy is a list of strings representing the active
    H1 > H2 > H3 nesting at that point.

    Page marker lines are excluded from section text but their offsets are
    still tracked so character positions remain correct.
    """
    lines = raw_text.splitlines()
    sections: list[tuple[str, int, int, list[str]]] = []
    hierarchy: list[str] = []
    buf: list[str] = []
    start_offset = 0
    current_offset = 0

    for line in lines:
        line_with_nl = f"{line}\n"
        match = _H_RE.match(line.strip())
        if match:
            # Flush the current section buffer.
            if buf:
                text = "".join(buf).strip()
                if text:
                    sections.append((text, start_offset, current_offset, hierarchy.copy()))
                buf = []
            level = len(match.group(1))
            heading = match.group(2).strip()
            # Update hierarchy: keep headings up to this level, replace current.
            hierarchy = hierarchy[: level - 1] + [heading]
            start_offset = current_offset
        # Exclude page marker lines from section text.
        if not _PAGE_MARKER_RE.match(line.strip()):
            buf.append(line_with_nl)
        current_offset += len(line_with_nl)

    # Flush remaining buffer.
    if buf:
        text = "".join(buf).strip()
        if text:
            sections.append((text, start_offset, current_offset, hierarchy.copy()))

    return sections


# ---------------------------------------------------------------------------
# Fixed-size token-window splitting
# ---------------------------------------------------------------------------

def _split_by_tokens(
    text: str,
    start_offset: int,
    enc,
    max_tokens: int,
    overlap_tokens: int,
    hierarchy: list[str],
) -> list[tuple[str, int, int, list[str]]]:
    """Split text into token-sized windows with overlap.

    Each window contains at most *max_tokens* tokens. Consecutive windows
    overlap by *overlap_tokens* tokens to preserve context at boundaries.

    Returns a list of (chunk_text, start_char, end_char, hierarchy) tuples,
    where start_char/end_char are relative to the original text's coordinate
    system (offset by *start_offset*).
    """
    ids = enc.encode(text)
    if not ids:
        return []

    chunks: list[tuple[str, int, int, list[str]]] = []
    # Track search position to avoid repeatedly matching early occurrences
    # when windows overlap.
    search_from = 0
    i = 0
    while i < len(ids):
        j = min(len(ids), i + max_tokens)
        piece = enc.decode(ids[i:j]).strip()
        if piece:
            rel_start = text.find(piece, search_from)
            if rel_start < 0:
                # Tokenizer round-trip may alter whitespace; fall back to
                # a broader search from the start.
                rel_start = text.find(piece)
            rel_start = rel_start if rel_start >= 0 else search_from
            s = start_offset + rel_start
            e = s + len(piece)
            chunks.append((piece, s, e, hierarchy.copy()))
            search_from = rel_start + 1
        if j == len(ids):
            break
        i = max(i + 1, j - overlap_tokens)

    return chunks


# ---------------------------------------------------------------------------
# Semantic boundary detection
# ---------------------------------------------------------------------------

def _semantic_boundaries(sentences: list[str], threshold: float = 0.5) -> list[int]:
    """Detect semantic topic boundaries between adjacent sentences.

    Computes cosine similarity between consecutive sentence embeddings.
    Where similarity drops below *threshold*, a boundary is placed.
    Returns a list of boundary indices (exclusive end of each group).

    Falls back to a lexical Jaccard similarity proxy if the sentence-
    transformer model cannot be loaded.
    """
    if len(sentences) < 2:
        return [len(sentences)]

    try:
        from sentence_transformers.util import cos_sim  # type: ignore

        model = _get_semantic_model()
        embeddings = model.encode(sentences, normalize_embeddings=True)
        bounds: list[int] = []
        for idx in range(len(sentences) - 1):
            sim = float(cos_sim(embeddings[idx], embeddings[idx + 1]))
            if sim < threshold:
                bounds.append(idx + 1)
        bounds.append(len(sentences))
        return bounds
    except Exception:
        # Fallback: lexical Jaccard similarity.
        bounds = []
        for idx in range(len(sentences) - 1):
            left = set(re.findall(r"\w+", sentences[idx].lower()))
            right = set(re.findall(r"\w+", sentences[idx + 1].lower()))
            denom = max(len(left | right), 1)
            sim = len(left & right) / denom
            if sim < 0.1:
                bounds.append(idx + 1)
        bounds.append(len(sentences))
        return bounds


def _merge_small_groups(
    groups: list[list[str]],
    max_tokens: int,
    enc,
) -> list[list[str]]:
    """Merge consecutive sentence groups that are below the token ceiling.

    This is the key fix for semantic chunking: raw semantic boundaries
    often produce very small groups (single sentences). This function
    greedily merges adjacent groups as long as the combined token count
    stays within *max_tokens*, producing chunks that are closer to the
    target size while preserving semantic coherence.
    """
    if not groups:
        return []

    merged: list[list[str]] = []
    current_group: list[str] = list(groups[0])
    current_tokens = _token_count(" ".join(current_group), enc)

    for group in groups[1:]:
        group_text = " ".join(group)
        group_tokens = _token_count(group_text, enc)

        # If merging would stay within budget, merge.
        if current_tokens + group_tokens <= max_tokens:
            current_group.extend(group)
            current_tokens += group_tokens
        else:
            # Flush current group and start a new one.
            merged.append(current_group)
            current_group = list(group)
            current_tokens = group_tokens

    # Flush remaining.
    if current_group:
        merged.append(current_group)

    return merged


# ---------------------------------------------------------------------------
# Chunk materialisation
# ---------------------------------------------------------------------------

def _materialize_chunk(
    report: ReportDocument,
    strategy: str,
    size: int,
    text: str,
    start_char: int,
    end_char: int,
    hierarchy: list[str],
    enc,
    marker_spans: list[tuple[int, int]] | None = None,
) -> ChunkRecord:
    """Create a ChunkRecord from raw chunk data.

    Converts character offsets from the page-marker-stripped text back to
    the original raw text coordinates for accurate page number resolution.
    """
    if marker_spans:
        raw_start = stripped_to_raw_offset(marker_spans, start_char)
        raw_end = stripped_to_raw_offset(marker_spans, end_char)
    else:
        raw_start = start_char
        raw_end = end_char

    p_start = page_for_offset(report.page_markers, raw_start)
    p_end = page_for_offset(report.page_markers, raw_end)
    norm = normalize_text(text)

    return ChunkRecord(
        chunk_id=_chunk_id(report.company, strategy, size, start_char, text),
        company=report.company,
        year=report.year,
        strategy=strategy,
        chunk_size=size,
        text=text.strip(),
        normalized_text=norm,
        token_count=_token_count(text, enc),
        start_char=start_char,
        end_char=end_char,
        page_start=p_start,
        page_end=p_end,
        header_hierarchy=hierarchy,
        metadata={"source_path": report.source_path},
    )


# ---------------------------------------------------------------------------
# Chunk statistics
# ---------------------------------------------------------------------------

@dataclass
class ChunkStats:
    """Descriptive statistics required by RQ1 for reporting chunk distributions."""

    total_chunks: int
    mean_tokens: float
    std_tokens: float
    min_tokens: int
    max_tokens: int
    median_tokens: float = 0.0
    p25_tokens: float = 0.0
    p75_tokens: float = 0.0


# ---------------------------------------------------------------------------
# Main entry point: build chunks for one configuration
# ---------------------------------------------------------------------------

def build_chunks_for_config(
    reports: dict[str, ReportDocument],
    config: PipelineConfig,
) -> tuple[list[ChunkRecord], ChunkStats]:
    """Build chunks for all reports under a single strategy/size configuration.

    This function dispatches to the appropriate chunking strategy and
    materialises ChunkRecord objects with normalised text, token counts,
    page numbers, and heading hierarchy metadata.

    Parameters
    ----------
    reports : dict[str, ReportDocument]
        Reports keyed by company name.
    config : PipelineConfig
        Configuration specifying chunk_strategy, chunk_size, and
        chunk_overlap_ratio.

    Returns
    -------
    (chunks, stats) : tuple[list[ChunkRecord], ChunkStats]
        All chunks and their descriptive statistics.
    """
    enc = _encoding()
    overlap = max(1, int(config.chunk_size * config.chunk_overlap_ratio))
    all_chunks: list[ChunkRecord] = []

    for report in reports.values():
        # Remove page markers before chunking so they don't affect token
        # counts or text matching.
        text = _PAGE_MARKER_RE.sub("", report.raw_text)
        marker_spans = extract_marker_spans(report.raw_text)

        if config.chunk_strategy == "fixed":
            # --- Fixed-size chunking: token-based sliding window ---
            windows = _split_by_tokens(text, 0, enc, config.chunk_size, overlap, [])

        elif config.chunk_strategy == "structure_aware":
            # --- Structure-aware chunking: respect heading boundaries ---
            windows = []
            for section_text, s_off, _, hierarchy in _iter_structured_sections(text):
                if _token_count(section_text, enc) <= config.chunk_size:
                    # Section fits within the target size — keep intact.
                    windows.append(
                        (section_text, s_off, s_off + len(section_text), hierarchy)
                    )
                else:
                    # Section exceeds target — split with token windows.
                    windows.extend(
                        _split_by_tokens(
                            section_text, s_off, enc,
                            config.chunk_size, overlap, hierarchy,
                        )
                    )

        elif config.chunk_strategy == "semantic":
            # --- Semantic chunking: group sentences by embedding similarity ---
            windows = []
            sections = _iter_structured_sections(text) or [(text, 0, len(text), [])]

            for sec_text, sec_start, _, hierarchy in sections:
                sents = sentence_split(sec_text)
                if not sents:
                    continue

                # Detect semantic boundaries between adjacent sentences.
                bounds = _semantic_boundaries(sents, threshold=0.5)

                # Build raw sentence groups from boundary indices.
                raw_groups: list[list[str]] = []
                prev = 0
                for b in bounds:
                    group = sents[prev:b]
                    if group:
                        raw_groups.append(group)
                    prev = b

                # Merge small groups to approach the target chunk size.
                # This prevents the excessive fragmentation (avg ~75 tokens)
                # that occurred without merging.
                merged_groups = _merge_small_groups(raw_groups, config.chunk_size, enc)

                for group in merged_groups:
                    block = " ".join(group).strip()
                    if not block:
                        continue

                    block_tokens = _token_count(block, enc)
                    if block_tokens <= config.chunk_size:
                        # Block fits — find its position in the section text.
                        rel = sec_text.find(block)
                        rel = rel if rel >= 0 else 0
                        windows.append(
                            (block, sec_start + rel, sec_start + rel + len(block), hierarchy)
                        )
                    else:
                        # Merged group still exceeds ceiling — split further.
                        block_start = sec_text.find(block)
                        block_start = block_start if block_start >= 0 else 0
                        windows.extend(
                            _split_by_tokens(
                                block,
                                sec_start + block_start,
                                enc,
                                config.chunk_size,
                                overlap,
                                hierarchy,
                            )
                        )
        else:
            raise ValueError(f"Unknown chunk strategy: {config.chunk_strategy}")

        # Materialise ChunkRecord objects from raw windows.
        for c_text, c_start, c_end, hierarchy in windows:
            chunk = _materialize_chunk(
                report=report,
                strategy=config.chunk_strategy,
                size=config.chunk_size,
                text=c_text,
                start_char=c_start,
                end_char=c_end,
                hierarchy=hierarchy,
                enc=enc,
                marker_spans=marker_spans,
            )
            all_chunks.append(chunk)

    # --- Compute chunk statistics ---
    token_counts = [c.token_count for c in all_chunks]
    if token_counts:
        tc_mean = sum(token_counts) / len(token_counts)
        tc_var = sum((x - tc_mean) ** 2 for x in token_counts) / len(token_counts)
        tc_std = tc_var ** 0.5
        sorted_tc = sorted(token_counts)
        n = len(sorted_tc)
        stats = ChunkStats(
            total_chunks=n,
            mean_tokens=tc_mean,
            std_tokens=tc_std,
            min_tokens=min(token_counts),
            max_tokens=max(token_counts),
            median_tokens=sorted_tc[n // 2],
            p25_tokens=sorted_tc[n // 4],
            p75_tokens=sorted_tc[3 * n // 4],
        )
    else:
        stats = ChunkStats(0, 0.0, 0.0, 0, 0)

    return all_chunks, stats
