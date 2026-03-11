"""Extract query-relevant ESRS disclosure requirements from ESRS JSON standards."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from rag.config import ARTIFACTS_DIR, ESRS_JSON_DIR, ESRS_JSON_FILES, LLM_MODEL
from rag.llm_client import chat_json

ESRS_CACHE_DIR = ARTIFACTS_DIR / "esrs_requirements_cache"
ESRS_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_esrs_standard(standard_name: str) -> Dict[str, Any]:
    """Load a structured ESRS standard JSON file by standard name."""
    filename = ESRS_JSON_FILES.get(standard_name)
    if not filename:
        raise ValueError(f"Unknown ESRS standard: {standard_name}")
    path = ESRS_JSON_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing ESRS file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_text_nodes(obj: Any, path: str = "") -> List[str]:
    """Recursively flatten nested JSON into readable text lines."""
    out: List[str] = []
    if isinstance(obj, dict):
        for key, val in obj.items():
            child_path = f"{path}.{key}" if path else str(key)
            if isinstance(val, (dict, list)):
                out.extend(_iter_text_nodes(val, child_path))
            elif isinstance(val, str):
                text = val.strip()
                if text:
                    out.append(f"[{child_path}] {text}")
            elif isinstance(val, (int, float)):
                out.append(f"[{child_path}] {val}")
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            child_path = f"{path}[{idx}]"
            out.extend(_iter_text_nodes(item, child_path))
    return out


def flatten_esrs_text(standard_data: Dict[str, Any], max_chars: int = 40000) -> str:
    """Flatten ESRS JSON into compact text for prompt context."""
    lines: List[str] = []
    standard = standard_data.get("standard", "")
    title = standard_data.get("title", "")
    if standard:
        lines.append(f"Standard: {standard}")
    if title:
        lines.append(f"Title: {title}")
    lines.extend(_iter_text_nodes(standard_data.get("objective", {}), "objective"))
    lines.extend(_iter_text_nodes(standard_data.get("content", {}), "content"))
    lines.extend(_iter_text_nodes(standard_data.get("appendix", {}), "appendix"))

    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[... truncated ...]"
    return text


def _cache_key(query: str, esrs_standard: str) -> str:
    return hashlib.sha256(f"{LLM_MODEL}::{esrs_standard}::{query}".encode("utf-8")).hexdigest()


def _extract_keyword_requirements(query: str, flattened_text: str, max_items: int = 8) -> List[Dict[str, Any]]:
    """Fallback lexical extraction when LLM call fails."""
    query_tokens = {t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 3}
    candidates: List[tuple[int, str]] = []
    for line in flattened_text.splitlines():
        low = line.lower()
        overlap = sum(1 for t in query_tokens if t in low)
        if overlap > 0:
            candidates.append((overlap, line.strip()))
    candidates.sort(key=lambda x: x[0], reverse=True)

    items: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for _, line in candidates:
        if line in seen:
            continue
        seen.add(line)
        paragraph_refs = re.findall(r"\b\d+\b", line)[:2]
        items.append(
            {
                "requirement_id": "heuristic",
                "description": line[:300],
                "paragraph_refs": [f"§{p}" for p in paragraph_refs],
            }
        )
        if len(items) >= max_items:
            break
    return items


def extract_relevant_requirements(query: str, esrs_standard: str, max_retries: int = 3) -> Dict[str, Any]:
    """Identify applicable ESRS requirements for a user question."""
    key = _cache_key(query, esrs_standard)
    cache_file = ESRS_CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        with cache_file.open("r", encoding="utf-8") as f:
            return json.load(f)

    standard_data = load_esrs_standard(esrs_standard)
    esrs_text = flatten_esrs_text(standard_data)

    system_prompt = (
        "You are an ESRS domain expert. Extract all ESRS disclosure requirements relevant "
        "to the user question. Return strict JSON with fields: relevant_requirements, summary."
        "Each relevant requirement must include requirement_id, description, paragraph_refs."
    )
    user_prompt = (
        f"Question: {query}\n\n"
        f"ESRS Standard: {esrs_standard}\n"
        f"{esrs_text}\n\n"
        "Return only JSON."
    )

    parsed, _usage = chat_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_completion_tokens=1500,
        temperature=0.0,
        max_retries=max_retries,
    )

    requirements = parsed.get("relevant_requirements") if isinstance(parsed, dict) else None
    summary = parsed.get("summary", "") if isinstance(parsed, dict) else ""

    if not isinstance(requirements, list):
        requirements = _extract_keyword_requirements(query, esrs_text)
        if not summary:
            summary = "Fallback lexical extraction was used due LLM parsing/API issue."

    result = {
        "requirements": requirements,
        "summary": summary,
        "standard_name": esrs_standard,
        "full_context": esrs_text[:6000],
    }

    with cache_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def get_esrs_requirements_text(query: str, esrs_standard: str) -> str:
    """Render extracted requirements as compact prompt text."""
    result = extract_relevant_requirements(query=query, esrs_standard=esrs_standard)
    parts = [f"ESRS Standard: {esrs_standard}"]

    reqs = result.get("requirements", [])
    if reqs:
        parts.append("Relevant Disclosure Requirements:")
        for idx, req in enumerate(reqs, 1):
            rid = str(req.get("requirement_id", f"R{idx}"))
            desc = str(req.get("description", "")).strip()
            refs = req.get("paragraph_refs", [])
            refs_text = ", ".join(str(r) for r in refs if r)
            line = f"{idx}. [{rid}] {desc}"
            if refs_text:
                line += f" (refs: {refs_text})"
            parts.append(line)
    else:
        parts.append("No specific requirement extraction available; use the standard context cautiously.")

    summary = str(result.get("summary", "")).strip()
    if summary:
        parts.append(f"Summary: {summary}")
    return "\n".join(parts)
