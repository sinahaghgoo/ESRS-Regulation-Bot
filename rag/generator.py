"""LLM answer generation with strict citation grounding and page-resolution helpers."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from rag.config import ARTIFACTS_DIR, LLM_MODEL
from rag.llm_client import chat_json

GENERATION_CACHE_DIR = ARTIFACTS_DIR / "generation_cache"
GENERATION_CACHE_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = """You are an expert ESRS sustainability analyst.
Only use retrieved passages as evidence.
Never invent facts.
If evidence is insufficient, say so explicitly.

Return strict JSON with keys:
- direct_answer: concise but complete answer with inline [Source N] markers
- compliance_level: one of Low, Medium, High
- compliance_rationale: rationale grounded in evidence
- disclosure_gaps: list of missing disclosures
- key_findings: list of key findings
- sources_used: list of integer source indices you used
"""

COMPARISON_SYSTEM_PROMPT = """You are an ESRS analyst comparing companies.
Use only provided evidence passages.
Return strict JSON:
- comparative_summary
- per_company (object of company -> {compliance_level,key_strengths,key_gaps})
- ranking_rationale
- common_gaps
- best_practices
"""


def _format_page_label(pages: List[int]) -> str:
    if not pages:
        return "Page not available"
    if len(pages) == 1:
        return f"p. {pages[0]}"
    return f"pp. {pages[0]}-{pages[-1]}"


def _format_page_inline(pages: List[int]) -> str:
    """Format pages for inline answer citations, e.g. 'Page 5-6'."""
    if not pages:
        return "Page not available"

    uniq = sorted(set(pages))
    ranges: List[tuple[int, int]] = []
    start = uniq[0]
    end = uniq[0]

    for page in uniq[1:]:
        if page == end + 1:
            end = page
            continue
        ranges.append((start, end))
        start = page
        end = page
    ranges.append((start, end))

    parts = [f"{a}" if a == b else f"{a}-{b}" for a, b in ranges]
    if len(parts) == 1:
        return f"Page {parts[0]}"
    return f"Pages {', '.join(parts)}"


def _normalize_pages(chunk: Dict[str, Any]) -> List[int]:
    pages = [p for p in chunk.get("page_span", []) if isinstance(p, int) and p > 0]
    if pages:
        return sorted(set(pages))
    page_number = chunk.get("page_number")
    if isinstance(page_number, int) and page_number > 0:
        return [page_number]
    return []


def _format_passages_for_prompt(results: List[Dict[str, Any]], company: str = "") -> str:
    """Format retrieved chunks into source-indexed context for prompting."""
    parts: List[str] = []
    for idx, r in enumerate(results, 1):
        pages = _normalize_pages(r)
        header = " > ".join(r.get("header_hierarchy", []) or ["Unknown section"])
        label = f"[Source {idx}]" if not company else f"[{company} - Source {idx}]"
        page_label = _format_page_label(pages)
        chunk_id = r.get("chunk_id", "")
        text = r.get("text", "")

        parts.append(label)
        parts.append(f"Chunk ID: {chunk_id}")
        parts.append(f"Section: {header}")
        parts.append(f"Location: {page_label}")
        parts.append(f"Content: {text}")
        parts.append("")
    return "\n".join(parts)


def _format_esrs_requirements(esrs_context: str) -> str:
    return f"--- APPLICABLE ESRS REQUIREMENTS ---\n{esrs_context}\n---"


def _extract_referenced_sources(text: str) -> List[int]:
    refs = re.findall(r"\[\s*Source\s+(\d+)\s*\]", text or "", flags=re.IGNORECASE)
    unique_sorted = sorted({int(r) for r in refs})
    return unique_sorted


def _coerce_sources_used(raw: Any, max_source: int, answer_text: str = "") -> List[int]:
    sources: List[int] = []
    if isinstance(raw, list):
        for x in raw:
            try:
                value = int(x)
            except (TypeError, ValueError):
                continue
            if 1 <= value <= max_source:
                sources.append(value)
    if not sources:
        sources = _extract_referenced_sources(answer_text)
    sources = sorted(set(s for s in sources if 1 <= s <= max_source))
    return sources


def _resolve_page_references(chunks: List[Dict[str, Any]], source_indices: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """Resolve citations to concrete pages from chunk metadata only."""
    citations: List[Dict[str, Any]] = []
    selected = set(source_indices or [])
    for idx, chunk in enumerate(chunks, 1):
        if selected and idx not in selected:
            continue
        pages = _normalize_pages(chunk)
        citations.append(
            {
                "source_index": idx,
                "pages": pages,
                "page_display": _format_page_label(pages),
                "page_inline": _format_page_inline(pages),
                "section": " > ".join(chunk.get("header_hierarchy", []) or ["Unknown section"]),
                "chunk_id": chunk.get("chunk_id", ""),
                "text_preview": (chunk.get("text", "")[:250]).strip(),
            }
        )
    return citations


def _inject_page_references_in_text(text: str, page_refs: List[Dict[str, Any]]) -> str:
    """
    Replace `Source N` markers in answer text with concrete page markers.
    Example: `[Source 1]` -> `[Page 5-6]`.
    """
    if not text:
        return text

    page_map = {
        int(ref.get("source_index")): str(ref.get("page_inline", "Page not available"))
        for ref in page_refs
        if isinstance(ref.get("source_index"), int)
    }
    if not page_map:
        return text

    pattern = re.compile(r"Source\s+(\d+)", flags=re.IGNORECASE)

    def _replace(match: re.Match[str]) -> str:
        idx = int(match.group(1))
        return page_map.get(idx, "Page not available")

    transformed = pattern.sub(_replace, text)

    # If model omitted markers, append explicit page references.
    if transformed == text:
        refs = " ".join(f"[{page_map[i]}]" for i in sorted(page_map))
        transformed = f"{text}\n\nReferences: {refs}"

    return transformed


def _generation_cache_key(
    query: str,
    company: str,
    esrs_context: str,
    chunks: List[Dict[str, Any]],
    model: str,
) -> str:
    chunk_sig = "||".join(f"{c.get('chunk_id','')}::{c.get('score',0):.6f}" for c in chunks)
    raw = f"{model}::{company}::{query}::{esrs_context[:2000]}::{chunk_sig}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _call_llm(system_prompt: str, user_prompt: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Call LLM JSON endpoint with robust defaults and token usage tracking."""
    parsed, usage = chat_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=config.get("llm_model", LLM_MODEL),
        temperature=float(config.get("temperature", 0.1)),
        max_completion_tokens=int(config.get("max_generation_tokens", 2048)),
        max_retries=3,
    )
    if not isinstance(parsed, dict):
        parsed = {"direct_answer": "Error: invalid model output", "sources_used": []}
    parsed["_token_usage"] = usage
    return parsed


def generate_answer(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    esrs_context: str,
    company: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate a structured answer for a single company with grounded citations."""
    passages_text = _format_passages_for_prompt(retrieved_chunks)
    user_prompt = (
        f"Company: {company}\n"
        f"Question: {query}\n\n"
        f"{_format_esrs_requirements(esrs_context)}\n\n"
        "--- RETRIEVED REPORT PASSAGES ---\n"
        f"{passages_text}\n"
        "---\n"
        "Use only these passages."
    )

    model = config.get("llm_model", LLM_MODEL)
    cache_key = _generation_cache_key(query, company, esrs_context, retrieved_chunks, model)
    cache_file = GENERATION_CACHE_DIR / f"{cache_key}.json"

    if cache_file.exists():
        with cache_file.open("r", encoding="utf-8") as f:
            result = json.load(f)
    else:
        result = _call_llm(SYSTEM_PROMPT, user_prompt, config)
        with cache_file.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    direct_answer = str(result.get("direct_answer", ""))
    source_indices = _coerce_sources_used(result.get("sources_used", []), max_source=len(retrieved_chunks), answer_text=direct_answer)
    page_refs = _resolve_page_references(retrieved_chunks, source_indices)

    # Keep raw source-tagged answer for traceability, but present page-tagged answer.
    direct_answer_with_pages = _inject_page_references_in_text(direct_answer, page_refs)
    compliance_rationale = _inject_page_references_in_text(
        str(result.get("compliance_rationale", "")),
        page_refs,
    )
    key_findings_raw = result.get("key_findings", [])
    key_findings = (
        [_inject_page_references_in_text(str(item), page_refs) for item in key_findings_raw]
        if isinstance(key_findings_raw, list)
        else key_findings_raw
    )

    return {
        **result,
        "direct_answer_raw": direct_answer,
        "direct_answer": direct_answer_with_pages,
        "compliance_rationale": compliance_rationale,
        "key_findings": key_findings,
        "sources_used": source_indices,
        "page_references": page_refs,
        "company": company,
        "query": query,
    }


def generate_comparison(
    query: str,
    per_company_results: Dict[str, List[Dict[str, Any]]],
    esrs_context: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate comparative analysis for multiple companies."""
    company_passages: List[str] = []
    for company, chunks in per_company_results.items():
        passages_text = _format_passages_for_prompt(chunks, company=company)
        company_passages.append(f"=== {company} ===\n{passages_text}")

    user_prompt = (
        f"Question: {query}\n\n"
        f"{_format_esrs_requirements(esrs_context)}\n\n"
        "--- RETRIEVED REPORT PASSAGES BY COMPANY ---\n"
        + "\n\n".join(company_passages)
        + "\n---"
    )

    result = _call_llm(COMPARISON_SYSTEM_PROMPT, user_prompt, config)
    result["per_company_pages"] = {
        company: _resolve_page_references(chunks)
        for company, chunks in per_company_results.items()
    }
    return result
