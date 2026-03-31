"""ESRS standard loading, flattening, and full-standard context serialization."""

from __future__ import annotations

import json
import re
from pathlib import Path

from rag.config.settings import ESRS_JSON_DIR, RQ2_RESULTS_DIR, RQ2_RESULTS_VERSION
from rag.types import ESRSContextResult, GenerationSample


def _standard_filename(esrs_standard: str) -> str:
    suffix = esrs_standard.replace("ESRS ", "").replace(" ", "_")
    return f"ESRS_{suffix}.json"


def _flatten_node(node: object, headings: list[str], out: list[dict[str, str]]) -> None:
    if isinstance(node, dict):
        title = str(node.get("title", "")).strip() if "title" in node else ""
        next_headings = headings + ([title] if title else [])

        if "objective" in node and isinstance(node["objective"], str):
            out.append(
                {
                    "reference": "Objective",
                    "title": " > ".join(next_headings or ["Objective"]),
                    "text": str(node["objective"]).strip(),
                }
            )

        paragraphs = node.get("paragraphs")
        if isinstance(paragraphs, list):
            for paragraph in paragraphs:
                if isinstance(paragraph, dict):
                    for reference, text in paragraph.items():
                        out.append(
                            {
                                "reference": str(reference).strip(),
                                "title": " > ".join(next_headings),
                                "text": str(text).strip(),
                            }
                        )
                elif isinstance(paragraph, str) and paragraph.strip():
                    out.append(
                        {
                            "reference": "",
                            "title": " > ".join(next_headings),
                            "text": paragraph.strip(),
                        }
                    )

        for key, value in node.items():
            if key in {"title", "paragraphs", "objective"}:
                continue
            _flatten_node(value, next_headings, out)
        return

    if isinstance(node, list):
        for item in node:
            _flatten_node(item, headings, out)
        return

    if isinstance(node, str) and node.strip():
        out.append(
            {
                "reference": "",
                "title": " > ".join(headings),
                "text": node.strip(),
            }
        )


def flatten_esrs_standard(payload: dict) -> list[dict[str, str]]:
    """Flatten the nested ESRS JSON into reference-title-text rows."""
    out: list[dict[str, str]] = []
    _flatten_node(payload, [str(payload.get("standard", "")).strip(), str(payload.get("title", "")).strip()], out)
    cleaned: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in out:
        key = (row["reference"], row["title"], row["text"])
        if key in seen or not row["text"]:
            continue
        seen.add(key)
        cleaned.append(row)
    return cleaned


def _format_context_label(row: dict[str, str], *, fallback_title: str) -> str:
    title = row["title"] or fallback_title
    parts = []
    if row["reference"]:
        parts.append(f"Paragraph {row['reference']}")
    if title:
        parts.append(title)
    return " / ".join(parts) if parts else fallback_title


def build_full_standard_context(
    payload: dict,
    *,
    esrs_standard: str,
    esrs_disclosure_id: str,
    question: str | None = None,
) -> tuple[str, list[dict[str, str]]]:
    """Serialize the full ESRS standard into a deterministic prompt block."""
    rows = flatten_esrs_standard(payload)
    lines = []
    if question:
        lines.append(f"Question: {question}")
    lines.extend(
        [
            f"ESRS standard: {esrs_standard}",
            f"Mapped disclosure ID: {esrs_disclosure_id or 'n/a'}",
            "",
            "Full ESRS standard context:",
        ]
    )

    selected: list[dict[str, str]] = []
    for row in rows:
        lines.append(f"[{_format_context_label(row, fallback_title=esrs_standard)}] {row['text']}".strip())
        selected.append({"reference": row["reference"], "text": row["text"]})

    return "\n".join(lines).strip(), selected


def build_question_extraction_input(
    payload: dict,
    *,
    esrs_standard: str,
    esrs_disclosure_id: str,
    question: str,
) -> str:
    """Backward-compatible alias for full-standard ESRS context building."""
    context_text, _ = build_full_standard_context(
        payload,
        esrs_standard=esrs_standard,
        esrs_disclosure_id=esrs_disclosure_id,
        question=question,
    )
    return context_text


# ---------------------------------------------------------------------------
# Disclosure-level ESRS context filtering
# ---------------------------------------------------------------------------

_DISCLOSURE_ID_RE = re.compile(r"[A-Za-z]+-?\d+|[A-Z]{2,3}-\d+", re.IGNORECASE)


def _parse_disclosure_ids(raw: str) -> list[str]:
    """Split a comma-separated disclosure-id string into normalised IDs."""
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return parts


def _node_matches_disclosure(title: str, disclosure_ids: list[str]) -> bool:
    """Check whether a JSON node title references any of the target disclosure IDs."""
    title_upper = title.upper().replace("–", "-").replace("—", "-")
    for did in disclosure_ids:
        did_norm = did.upper().strip().replace("–", "-").replace("—", "-")
        if did_norm in title_upper:
            return True
    return False


def _collect_matching_subtrees(
    node: object,
    disclosure_ids: list[str],
    headings: list[str],
    out: list[dict[str, str]],
    *,
    force_include: bool = False,
) -> None:
    """Recursively collect paragraphs from subtrees that match disclosure IDs."""
    if isinstance(node, dict):
        title = str(node.get("title", "")).strip() if "title" in node else ""
        next_headings = headings + ([title] if title else [])

        include_this = force_include or (
            title and _node_matches_disclosure(title, disclosure_ids)
        )

        if include_this:
            if "objective" in node and isinstance(node["objective"], str):
                out.append({
                    "reference": "Objective",
                    "title": " > ".join(next_headings or ["Objective"]),
                    "text": str(node["objective"]).strip(),
                })

            paragraphs = node.get("paragraphs")
            if isinstance(paragraphs, list):
                for paragraph in paragraphs:
                    if isinstance(paragraph, dict):
                        for reference, text in paragraph.items():
                            out.append({
                                "reference": str(reference).strip(),
                                "title": " > ".join(next_headings),
                                "text": str(text).strip(),
                            })
                    elif isinstance(paragraph, str) and paragraph.strip():
                        out.append({
                            "reference": "",
                            "title": " > ".join(next_headings),
                            "text": paragraph.strip(),
                        })

        for key, value in node.items():
            if key in {"title", "paragraphs", "objective"}:
                continue
            _collect_matching_subtrees(
                value, disclosure_ids, next_headings, out,
                force_include=include_this,
            )
        return

    if isinstance(node, list):
        for item in node:
            _collect_matching_subtrees(
                item, disclosure_ids, headings, out,
                force_include=force_include,
            )
        return


def build_disclosure_context(
    payload: dict,
    *,
    esrs_standard: str,
    esrs_disclosure_id: str,
    question: str | None = None,
) -> tuple[str, list[dict[str, str]]]:
    """Serialize only the disclosure-relevant sections of an ESRS standard.

    Filters the ESRS JSON tree to include only subsections whose titles
    contain one of the target disclosure IDs (e.g. "E1-1", "S1-6").
    Falls back to the full standard if no matching sections are found.
    """
    disclosure_ids = _parse_disclosure_ids(esrs_disclosure_id)
    if not disclosure_ids:
        return build_full_standard_context(
            payload, esrs_standard=esrs_standard,
            esrs_disclosure_id=esrs_disclosure_id, question=question,
        )

    root_headings = [
        str(payload.get("standard", "")).strip(),
        str(payload.get("title", "")).strip(),
    ]
    rows: list[dict[str, str]] = []
    _collect_matching_subtrees(payload, disclosure_ids, root_headings, rows)

    seen: set[tuple[str, str, str]] = set()
    cleaned: list[dict[str, str]] = []
    for row in rows:
        key = (row["reference"], row["title"], row["text"])
        if key in seen or not row["text"]:
            continue
        seen.add(key)
        cleaned.append(row)

    if not cleaned:
        return build_full_standard_context(
            payload, esrs_standard=esrs_standard,
            esrs_disclosure_id=esrs_disclosure_id, question=question,
        )

    lines: list[str] = []
    if question:
        lines.append(f"Question: {question}")
    lines.extend([
        f"ESRS standard: {esrs_standard}",
        f"Mapped disclosure ID: {esrs_disclosure_id or 'n/a'}",
        "",
        "Relevant ESRS disclosure requirements:",
    ])

    selected: list[dict[str, str]] = []
    for row in cleaned:
        lines.append(
            f"[{_format_context_label(row, fallback_title=esrs_standard)}] "
            f"{row['text']}".strip()
        )
        selected.append({"reference": row["reference"], "text": row["text"]})

    return "\n".join(lines).strip(), selected


# ---------------------------------------------------------------------------
# Convenience functions for notebook use
# ---------------------------------------------------------------------------

def load_esrs_standards() -> dict[str, dict]:
    """Load all ESRS standard JSON files into a dict keyed by standard name."""
    standards: dict[str, dict] = {}
    for path in sorted(ESRS_JSON_DIR.glob("ESRS_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        std_name = str(payload.get("standard", path.stem.replace("_", " ")))
        standards[std_name] = payload
    return standards


def extract_esrs_context(
    esrs_standard: str,
    standards: dict[str, dict],
    question: str | None = None,
    esrs_disclosure_id: str = "",
) -> str:
    """Extract disclosure-level ESRS context for a given standard.

    Uses disclosure-level filtering when *esrs_disclosure_id* is provided,
    otherwise falls back to the full standard.
    """
    payload = standards.get(esrs_standard)
    if payload is None:
        return ""
    if esrs_disclosure_id:
        context_text, _ = build_disclosure_context(
            payload, esrs_standard=esrs_standard,
            esrs_disclosure_id=esrs_disclosure_id, question=question,
        )
    else:
        context_text, _ = build_full_standard_context(
            payload, esrs_standard=esrs_standard,
            esrs_disclosure_id=esrs_disclosure_id, question=question,
        )
    return context_text


class ESRSContextExtractor:
    """Cache the full mapped ESRS standard for each evaluation question."""

    def __init__(
        self,
        *,
        model: str | None = None,
        cache_path: Path | None = None,
        client: object | None = None,
        require_openai: bool = True,
    ) -> None:
        del client
        del require_openai
        self.model = model or "deterministic_full_standard"
        self.cache_path = cache_path or (RQ2_RESULTS_DIR / "context_cache.json")
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: dict[str, dict] = {}
        if self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                self.cache = {}

    def _cache_key(self, sample: GenerationSample) -> str:
        return f"{RQ2_RESULTS_VERSION}:{self.model}:{sample.question_id}:{sample.esrs_standard}"

    def _persist(self) -> None:
        self.cache_path.write_text(json.dumps(self.cache, indent=2), encoding="utf-8")

    def _load_standard_payload(self, esrs_standard: str) -> dict:
        path = ESRS_JSON_DIR / _standard_filename(esrs_standard)
        return json.loads(path.read_text(encoding="utf-8"))

    def get_or_extract(self, sample: GenerationSample) -> ESRSContextResult:
        """Load cached ESRS context for a question or serialize it once."""
        key = self._cache_key(sample)
        cached = self.cache.get(key)
        if cached and cached.get("results_version") == RQ2_RESULTS_VERSION:
            return ESRSContextResult(
                question_id=sample.question_id,
                esrs_standard=sample.esrs_standard,
                esrs_disclosure_id=sample.esrs_disclosure_id,
                context_text=str(cached.get("context_text", "")),
                selected_requirements=list(cached.get("selected_requirements", [])),
                raw_response=str(cached.get("raw_response", "")),
                model=str(cached.get("model", self.model)),
            )

        payload = self._load_standard_payload(sample.esrs_standard)
        context_text, selected = build_disclosure_context(
            payload,
            esrs_standard=sample.esrs_standard,
            esrs_disclosure_id=sample.esrs_disclosure_id,
            question=sample.question,
        )

        out = ESRSContextResult(
            question_id=sample.question_id,
            esrs_standard=sample.esrs_standard,
            esrs_disclosure_id=sample.esrs_disclosure_id,
            context_text=context_text,
            selected_requirements=selected,
            raw_response="",
            model=self.model,
        )
        self.cache[key] = {
            "results_version": RQ2_RESULTS_VERSION,
            "question_id": out.question_id,
            "esrs_standard": out.esrs_standard,
            "esrs_disclosure_id": out.esrs_disclosure_id,
            "context_text": out.context_text,
            "selected_requirements": out.selected_requirements,
            "raw_response": out.raw_response,
            "model": out.model,
        }
        self._persist()
        return out
