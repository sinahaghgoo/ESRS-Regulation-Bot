"""RQ1 chunking implementations: fixed, structure-aware, and semantic."""

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

_H_RE = re.compile(r"^(#{1,3})\s+(.*)$")
_PAGE_MARKER_RE = re.compile(r"<!--\s*Report Page\s+\d+\s*-->")
_SEMANTIC_MODEL = None


def _get_semantic_model():
    """Lazily initialize sentence-transformer once per process."""
    global _SEMANTIC_MODEL
    if _SEMANTIC_MODEL is None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        _SEMANTIC_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _SEMANTIC_MODEL


def _encoding():
    try:
        return tiktoken.encoding_for_model("text-embedding-3-large")
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def _token_count(text: str, enc) -> int:
    return len(enc.encode(text))


def _chunk_id(company: str, strategy: str, size: int, start: int, text: str) -> str:
    digest = hashlib.sha1(f"{company}:{strategy}:{size}:{start}:{text[:80]}".encode("utf-8")).hexdigest()
    return digest[:20]


def _iter_structured_sections(raw_text: str) -> list[tuple[str, int, int, list[str]]]:
    """Split by H1-H3 boundaries and preserve heading hierarchy."""
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
            if buf:
                text = "".join(buf).strip()
                if text:
                    sections.append((text, start_offset, current_offset, hierarchy.copy()))
                buf = []
            level = len(match.group(1))
            heading = match.group(2).strip()
            hierarchy = hierarchy[: level - 1] + [heading]
            start_offset = current_offset
        if not _PAGE_MARKER_RE.match(line.strip()):
            buf.append(line_with_nl)
        current_offset += len(line_with_nl)
    if buf:
        text = "".join(buf).strip()
        if text:
            sections.append((text, start_offset, current_offset, hierarchy.copy()))
    return sections


def _split_by_tokens(
    text: str,
    start_offset: int,
    enc,
    max_tokens: int,
    overlap_tokens: int,
    hierarchy: list[str],
) -> list[tuple[str, int, int, list[str]]]:
    """Split text into token-sized windows with overlap."""
    ids = enc.encode(text)
    if not ids:
        return []
    chunks: list[tuple[str, int, int, list[str]]] = []
    # Track the search-start position to avoid repeatedly matching
    # the first occurrence for overlapping windows.
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


def _semantic_boundaries(sentences: list[str], threshold: float = 0.5) -> list[int]:
    """Compute split boundaries on sentence similarity drops."""
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
        # Fallback: lexical similarity proxy based on token overlap.
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


@dataclass
class ChunkStats:
    """Descriptive statistics required by RQ1."""

    total_chunks: int
    mean_tokens: float
    std_tokens: float
    min_tokens: int
    max_tokens: int


def build_chunks_for_config(
    reports: dict[str, ReportDocument],
    config: PipelineConfig,
) -> tuple[list[ChunkRecord], ChunkStats]:
    """Build chunks for all reports under a single strategy/size config."""
    enc = _encoding()
    overlap = max(1, int(config.chunk_size * config.chunk_overlap_ratio))
    all_chunks: list[ChunkRecord] = []

    for report in reports.values():
        text = _PAGE_MARKER_RE.sub("", report.raw_text)
        marker_spans = extract_marker_spans(report.raw_text)
        if config.chunk_strategy == "fixed":
            windows = _split_by_tokens(text, 0, enc, config.chunk_size, overlap, [])
        elif config.chunk_strategy == "structure_aware":
            windows = []
            for section_text, s_off, _, hierarchy in _iter_structured_sections(text):
                if _token_count(section_text, enc) <= config.chunk_size:
                    windows.append((section_text, s_off, s_off + len(section_text), hierarchy))
                else:
                    windows.extend(
                        _split_by_tokens(
                            section_text,
                            s_off,
                            enc,
                            config.chunk_size,
                            overlap,
                            hierarchy,
                        )
                    )
        elif config.chunk_strategy == "semantic":
            windows = []
            sections = _iter_structured_sections(text) or [(text, 0, len(text), [])]
            for sec_text, sec_start, _, hierarchy in sections:
                sents = sentence_split(sec_text)
                if not sents:
                    continue
                bounds = _semantic_boundaries(sents, threshold=0.5)
                prev = 0
                for b in bounds:
                    block = " ".join(sents[prev:b]).strip()
                    prev = b
                    if not block:
                        continue
                    if _token_count(block, enc) <= config.chunk_size:
                        rel = sec_text.find(block)
                        rel = rel if rel >= 0 else 0
                        windows.append((block, sec_start + rel, sec_start + rel + len(block), hierarchy))
                    else:
                        windows.extend(
                            _split_by_tokens(
                                block,
                                sec_start + max(sec_text.find(block), 0),
                                enc,
                                config.chunk_size,
                                overlap,
                                hierarchy,
                            )
                        )
        else:
            raise ValueError(f"Unknown chunk strategy: {config.chunk_strategy}")

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

    token_counts = [c.token_count for c in all_chunks]
    if token_counts:
        mean = sum(token_counts) / len(token_counts)
        var = sum((x - mean) ** 2 for x in token_counts) / len(token_counts)
        std = var**0.5
        stats = ChunkStats(
            total_chunks=len(token_counts),
            mean_tokens=mean,
            std_tokens=std,
            min_tokens=min(token_counts),
            max_tokens=max(token_counts),
        )
    else:
        stats = ChunkStats(0, 0.0, 0.0, 0, 0)

    return all_chunks, stats

