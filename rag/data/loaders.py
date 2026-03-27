"""
Load report corpus and RQ1 golden-standard evaluation samples.

This module handles two data sources:

1. **Report corpus** — Preprocessed sustainability reports stored as
   Markdown files in ``data/Reports_md_clean/``. Each file is loaded
   as a ``ReportDocument`` with normalised text and page markers.

2. **Golden standard** — Expert-annotated evaluation dataset stored as
   JSON in ``golden_standard/golden_standard.json``. Contains 12
   questions (one per ESRS standard) × 17 companies = 204 evaluation
   pairs. Each pair includes evidence passages with page references.

Company name mapping handles filename variations (e.g., "AkerBP" ->
"Aker BP", "SBMOffshore" -> "SBM Offshore") to ensure consistent
matching between report files and golden standard entries.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from rag.config.settings import GOLDEN_STANDARD_PATH, REPORTS_MD_DIR
from rag.data.text_utils import extract_page_markers, normalize_text
from rag.types import EvalSample, GenerationSample, GoldEvidence, ReportDocument

# Regex for parsing report filenames: expects "CompanyName" or "CompanyName_2024".
_REPORT_RE = re.compile(r"^(?P<slug>[A-Za-z]+)_?(?P<year>\d{4})?$")

# Map from filename slugs to canonical company names used in the golden standard.
_COMPANY_MAP = {
    "AkerBP": "Aker BP",
    "SBMOffshore": "SBM Offshore",
    "Mol": "MOL",
    "TotalEnergies": "TotalEnergies",
}


def _company_from_filename(path: Path) -> tuple[str, str]:
    """Extract (company_name, year) from a report filename.

    Applies the company name mapping to convert filename slugs to
    the canonical names used in the golden standard.
    """
    stem = path.stem
    match = _REPORT_RE.match(stem)
    if not match:
        return stem, "2024"
    slug = match.group("slug")
    year = match.group("year") or "2024"
    return _COMPANY_MAP.get(slug, slug), year


def load_reports(reports_dir: Path | None = None) -> dict[str, ReportDocument]:
    """Load all markdown reports keyed by company name.

    Parameters
    ----------
    reports_dir : Path | None
        Directory containing .md report files. Defaults to
        ``data/Reports_md_clean/``.

    Returns
    -------
    dict[str, ReportDocument]
        Reports keyed by canonical company name.

    Raises
    ------
    FileNotFoundError
        If no markdown files are found in the directory.
    """
    root = reports_dir or REPORTS_MD_DIR
    reports: dict[str, ReportDocument] = {}

    for path in sorted(root.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        company, year = _company_from_filename(path)
        reports[company] = ReportDocument(
            company=company,
            year=year,
            source_path=str(path),
            raw_text=raw,
            normalized_text=normalize_text(raw),
            page_markers=extract_page_markers(raw),
        )

    if not reports:
        raise FileNotFoundError(f"No markdown reports found in {root}")
    return reports


def load_golden_samples(path: Path | None = None) -> list[EvalSample]:
    """Load question-company evaluation samples from the golden standard.

    Each sample represents one (question, company) pair with its associated
    expert-annotated evidence passages.

    Parameters
    ----------
    path : Path | None
        Path to the golden standard JSON. Defaults to
        ``golden_standard/golden_standard.json``.

    Returns
    -------
    list[EvalSample]
        Evaluation samples (expected: 204 for the full dataset).

    Raises
    ------
    RuntimeError
        If no samples are found (indicates a parsing or file error).
    """
    golden_path = path or GOLDEN_STANDARD_PATH
    payload = json.loads(golden_path.read_text(encoding="utf-8"))
    samples: list[EvalSample] = []

    for question_item in payload.get("questions", []):
        qid = question_item["id"]
        question = question_item["question"]
        esrs = question_item.get("esrs_standard", "")

        for company_item in question_item.get("per_company", []):
            company = company_item["company"]
            evidences: list[GoldEvidence] = []

            for ev in company_item.get("evidence", []):
                text = ev.get("text", "") or ev.get("exact_text", "")
                evidences.append(
                    GoldEvidence(
                        passage_id=ev["passage_id"],
                        company=company,
                        question_id=qid,
                        question=question,
                        esrs_standard=esrs,
                        text=text,
                        normalized_text=normalize_text(text),
                        page_number=str(ev.get("page_number", "")),
                        source_section_header=str(ev.get("source_section_header", "")),
                    )
                )

            samples.append(
                EvalSample(
                    question_id=qid,
                    question=question,
                    esrs_standard=esrs,
                    company=company,
                    evidences=evidences,
                )
            )

    if not samples:
        raise RuntimeError(f"No evaluation samples found in {golden_path}")
    return samples


def load_generation_samples(path: Path | None = None) -> list[GenerationSample]:
    """Load full question-company samples for RQ2/RQ3 generation evaluation."""
    golden_path = path or GOLDEN_STANDARD_PATH
    payload = json.loads(golden_path.read_text(encoding="utf-8"))
    samples: list[GenerationSample] = []

    for question_item in payload.get("questions", []):
        qid = question_item["id"]
        question = question_item["question"]
        esrs = question_item.get("esrs_standard", "")
        disclosure_id = question_item.get("esrs_disclosure_id", "")

        for company_item in question_item.get("per_company", []):
            evidences: list[GoldEvidence] = []
            for ev in company_item.get("evidence", []):
                text = ev.get("text", "") or ev.get("exact_text", "")
                evidences.append(
                    GoldEvidence(
                        passage_id=ev["passage_id"],
                        company=company_item["company"],
                        question_id=qid,
                        question=question,
                        esrs_standard=esrs,
                        text=text,
                        normalized_text=normalize_text(text),
                        page_number=str(ev.get("page_number", "")),
                        source_section_header=str(ev.get("source_section_header", "")),
                    )
                )

            samples.append(
                GenerationSample(
                    question_id=qid,
                    question=question,
                    esrs_standard=esrs,
                    esrs_disclosure_id=disclosure_id,
                    company=company_item["company"],
                    ground_truth_answer=str(company_item.get("ground_truth_answer", "")),
                    fulfillment_level=str(company_item.get("fulfillment_level", "")),
                    fulfillment_rationale=str(company_item.get("fulfillment_rationale", "")),
                    evidences=evidences,
                )
            )

    if not samples:
        raise RuntimeError(f"No generation samples found in {golden_path}")
    return samples
