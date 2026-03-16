"""Load report corpus and RQ1 golden-standard samples."""

from __future__ import annotations

import json
import re
from pathlib import Path

from rag.config.settings import GOLDEN_STANDARD_PATH, REPORTS_MD_DIR
from rag.data.text_utils import extract_page_markers, normalize_text
from rag.types import EvalSample, GoldEvidence, ReportDocument

_REPORT_RE = re.compile(r"^(?P<slug>[A-Za-z]+)_?(?P<year>\d{4})?$")


def _company_from_filename(path: Path) -> tuple[str, str]:
    stem = path.stem
    match = _REPORT_RE.match(stem)
    if not match:
        return stem, "2024"
    slug = match.group("slug")
    year = match.group("year") or "2024"
    company_map = {
        "AkerBP": "Aker BP",
        "SBMOffshore": "SBM Offshore",
        "Mol": "MOL",
        "TotalEnergies": "TotalEnergies",
    }
    return company_map.get(slug, slug), year


def load_reports(reports_dir: Path | None = None) -> dict[str, ReportDocument]:
    """Load all markdown reports keyed by company name."""
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
    """Load question/company evaluation samples from the golden standard."""
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

