"""
Report Parser Module
====================
Parses Markdown sustainability reports preserving:
  - Page markers (<!-- Report Page N -->)
  - Heading hierarchy (h1-h6)
  - HTML tables (kept intact within chunks)
  - Figure descriptions
  - Section structure for downstream chunking

Each parsed element carries metadata: company, year, source file,
current page number(s), and the header breadcrumb trail.
"""

import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from rag.config import COMPANY_MAP, REPORTS_MD_DIR


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ParsedElement:
    """A single structural element from the parsed report."""
    element_type: str          # "heading", "paragraph", "table", "figure", "list"
    content: str               # raw text/html content
    heading_level: int = 0     # 1-6 for headings, 0 for others
    page_number: int = -1      # current page at this element
    page_span: List[int] = field(default_factory=list)  # pages this element spans
    header_hierarchy: List[str] = field(default_factory=list)  # breadcrumb of parent headings
    source_file: str = ""
    company: str = ""
    reporting_year: int = 2024


@dataclass
class ParsedReport:
    """Container for a fully parsed report."""
    company: str
    reporting_year: int
    source_file: str
    elements: List[ParsedElement]
    total_pages: int = 0


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Matches <!-- Report Page 35 --> (with flexible whitespace)
PAGE_MARKER_RE = re.compile(r'<!--\s*Report\s+Page\s+(\d+)\s*-->')

# Matches markdown headings: # through ######
HEADING_RE = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

# Matches HTML table blocks (may span multiple lines)
TABLE_START_RE = re.compile(r'<table\b', re.IGNORECASE)
TABLE_END_RE = re.compile(r'</table>', re.IGNORECASE)

# Matches figure blocks
FIGURE_START_RE = re.compile(r'<figure\b', re.IGNORECASE)
FIGURE_END_RE = re.compile(r'</figure>', re.IGNORECASE)


def _extract_company_year(filename: str) -> Tuple[str, int]:
    """Extract company display name and year from filename stem."""
    stem = Path(filename).stem
    if stem in COMPANY_MAP:
        return COMPANY_MAP[stem], 2024
    # Fallback: try to parse "CompanyName_YYYY"
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0], int(parts[1])
    return stem, 2024


def parse_report(filepath: Path) -> ParsedReport:
    """
    Parse a Markdown report file into structured elements.
    
    Preserves page markers, heading hierarchy, tables, and figures.
    Each element tracks which page(s) it belongs to and its position
    in the heading hierarchy.
    
    Args:
        filepath: Path to the .md report file
    
    Returns:
        ParsedReport with all structural elements
    """
    text = filepath.read_text(encoding="utf-8")
    filename = filepath.name
    company, year = _extract_company_year(filename)

    elements: List[ParsedElement] = []
    current_page = -1
    max_page = -1
    # Track heading hierarchy: index = level-1, value = heading text
    heading_stack: List[Optional[str]] = [None] * 6

    lines = text.split('\n')
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        # --- Check for page marker ---
        page_match = PAGE_MARKER_RE.search(line)
        if page_match:
            current_page = int(page_match.group(1))
            max_page = max(max_page, current_page)
            # If the line ONLY contains the page marker, skip to next line
            stripped = PAGE_MARKER_RE.sub('', line).strip()
            if not stripped:
                i += 1
                continue
            # Otherwise, the line has content after the marker
            line = stripped

        # --- Check for HTML table block ---
        if TABLE_START_RE.search(line):
            table_lines = [line]
            table_start_page = current_page
            table_pages = {current_page} if current_page > 0 else set()
            # Accumulate until </table>
            while not TABLE_END_RE.search(line) and i + 1 < n:
                i += 1
                line = lines[i]
                # Check for page markers inside tables
                pm = PAGE_MARKER_RE.search(line)
                if pm:
                    current_page = int(pm.group(1))
                    max_page = max(max_page, current_page)
                    table_pages.add(current_page)
                    line = PAGE_MARKER_RE.sub('', line)
                table_lines.append(line)
            
            table_content = '\n'.join(table_lines)
            page_list = sorted(table_pages) if table_pages else ([current_page] if current_page > 0 else [])
            hierarchy = [h for h in heading_stack if h is not None]
            
            elements.append(ParsedElement(
                element_type="table",
                content=table_content,
                page_number=table_start_page,
                page_span=page_list,
                header_hierarchy=list(hierarchy),
                source_file=filename,
                company=company,
                reporting_year=year,
            ))
            i += 1
            continue

        # --- Check for figure block ---
        if FIGURE_START_RE.search(line):
            fig_lines = [line]
            fig_start_page = current_page
            fig_pages = {current_page} if current_page > 0 else set()
            while not FIGURE_END_RE.search(line) and i + 1 < n:
                i += 1
                line = lines[i]
                pm = PAGE_MARKER_RE.search(line)
                if pm:
                    current_page = int(pm.group(1))
                    max_page = max(max_page, current_page)
                    fig_pages.add(current_page)
                    line = PAGE_MARKER_RE.sub('', line)
                fig_lines.append(line)

            fig_content = '\n'.join(fig_lines)
            page_list = sorted(fig_pages) if fig_pages else ([current_page] if current_page > 0 else [])
            hierarchy = [h for h in heading_stack if h is not None]
            
            elements.append(ParsedElement(
                element_type="figure",
                content=fig_content,
                page_number=fig_start_page,
                page_span=page_list,
                header_hierarchy=list(hierarchy),
                source_file=filename,
                company=company,
                reporting_year=year,
            ))
            i += 1
            continue

        # --- Check for heading ---
        heading_match = HEADING_RE.match(line)
        if heading_match:
            level = len(heading_match.group(1))  # 1-6
            heading_text = heading_match.group(2).strip()
            
            # Update heading stack: set this level, clear all deeper levels
            heading_stack[level - 1] = heading_text
            for j in range(level, 6):
                heading_stack[j] = None
            
            hierarchy = [h for h in heading_stack if h is not None]
            
            elements.append(ParsedElement(
                element_type="heading",
                content=heading_text,
                heading_level=level,
                page_number=current_page,
                page_span=[current_page] if current_page > 0 else [],
                header_hierarchy=list(hierarchy),
                source_file=filename,
                company=company,
                reporting_year=year,
            ))
            i += 1
            continue

        # --- Regular text (paragraph/list) ---
        stripped = line.strip()
        if stripped:
            # Detect list items
            is_list = stripped.startswith(('-', '*', '+')) or re.match(r'^\d+\.', stripped)
            etype = "list" if is_list else "paragraph"
            hierarchy = [h for h in heading_stack if h is not None]
            
            elements.append(ParsedElement(
                element_type=etype,
                content=stripped,
                page_number=current_page,
                page_span=[current_page] if current_page > 0 else [],
                header_hierarchy=list(hierarchy),
                source_file=filename,
                company=company,
                reporting_year=year,
            ))

        i += 1

    return ParsedReport(
        company=company,
        reporting_year=year,
        source_file=filename,
        elements=elements,
        total_pages=max_page,
    )


def parse_all_reports() -> Dict[str, ParsedReport]:
    """
    Parse all Markdown reports in the Reports_md directory.
    
    Returns:
        Dictionary mapping filename stem to ParsedReport
    """
    reports = {}
    md_files = sorted(REPORTS_MD_DIR.glob("*.md"))
    
    for md_file in md_files:
        stem = md_file.stem
        print(f"  Parsing {md_file.name}...")
        reports[stem] = parse_report(md_file)
        print(f"    -> {len(reports[stem].elements)} elements, "
              f"{reports[stem].total_pages} pages")
    
    return reports
