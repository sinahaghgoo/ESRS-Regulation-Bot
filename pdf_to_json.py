#!/usr/bin/env python3
"""
ESRS Regulation PDF to JSON Converter

Converts ESRS regulation PDFs from data/Regulation_Split_PDF/ into structured
JSON files in data/final_regulation_split_json/, matching the format of the
existing reference JSONs in data/Regulation_Split_JSON/.
"""

import json
import os
import re
from pathlib import Path

import pdfplumber


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "data" / "Regulation_Split_PDF"
OUTPUT_DIR = BASE_DIR / "data" / "final_regulation_split_json"
REFERENCE_DIR = BASE_DIR / "data" / "Regulation_Split_JSON"

# Page header noise to strip  
PAGE_HEADER_PATTERNS = [
    re.compile(r"^EN$"),
    re.compile(r"^OJ L,?\s*22\.12\.2023$"),
    re.compile(r"^ELI:\s*http.*$"),
    re.compile(r"^\d+/284\s*ELI.*$"),
    re.compile(r"^ELI.*\d+/284$"),
]


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_lines(pdf_path: str) -> list[str]:
    """Extract text lines from PDF, removing page headers."""
    all_lines: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            for line in text.split("\n"):
                stripped = line.strip()
                if not stripped:
                    continue
                if any(pat.match(stripped) for pat in PAGE_HEADER_PATTERNS):
                    continue
                all_lines.append(stripped)
    return all_lines


# ---------------------------------------------------------------------------
# Line classification
# ---------------------------------------------------------------------------

def is_paragraph_start(line: str):
    """Check if line starts a numbered paragraph. 
    Returns (number, text) or (None, None).
    AR paragraphs: 'AR 1. text...'
    Normal paragraphs: '1. text...' (but NOT section headers like '1. Categories of...')
    """
    # AR paragraph
    m = re.match(r"^(AR\s*\d+)\.\s+(.*)", line, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    # Normal paragraph — must start with a number followed by '. ' and NOT be a section header
    m = re.match(r"^(\d+)\.\s+(.*)", line, re.DOTALL)
    if m:
        num = m.group(1)
        rest = m.group(2)
        # Section headers: "1. Categories of ESRS Standards, reporting areas..."
        # These start with a capital letter and describe a section topic
        # Paragraphs: "1. The objective of this Standard is..."
        # Heuristic: if num <= ~15 and text looks like a section title (short, title-like),
        # it could be a section. But we need context. For now, return the match
        # and let the caller decide based on context.
        return num, rest.strip()
    return None, None


def is_toc_line(line: str) -> bool:
    """Check if a line looks like a TOC entry."""
    return line.startswith("–") or line.startswith("—") or line.startswith("- ")


def is_continuation(line: str) -> bool:
    """Check if a line is a continuation of a previous paragraph."""
    # Starts with lowercase, or with special chars like (a), (b), i., ii., etc.
    if not line:
        return False
    if line[0].islower():
        return True
    if re.match(r"^\([a-z]\)", line):
        return True
    if re.match(r"^\(i+\)", line):
        return True
    if re.match(r"^[ivxl]+\.", line):
        return True
    return False


# ---------------------------------------------------------------------------
# Section heading detection
# ---------------------------------------------------------------------------

CATEGORY_HEADERS = {
    "Governance", "Strategy", "Impact, risk and opportunity management",
    "Impacts, risks and opportunities management",
    "Metrics and targets", "ESRS 2 General disclosures",
    "Interaction with other ESRS", "Interactions with other ESRS",
    "Disclosure Requirements", "Basis for preparation",
}

NUMBERED_SECTION_RE = re.compile(r"^(\d+)\.\s+([A-Z].*)")
NUMBERED_SUBSECTION_RE = re.compile(r"^(\d+\.\d+)\s+(.*)")


def classify_line(line: str) -> str:
    """Classify a line as: 'category', 'section', 'subsection', 'disclosure', 
    'appendix', 'objective', 'paragraph', 'ar_paragraph', 'toc', 'continuation', 'other'."""
    stripped = line.strip()
    
    if stripped == "Objective":
        return "objective_header"
    if stripped == "TABLE OF CONTENTS":
        return "toc_header"
    # Check for standalone "Disclosure Requirements" / "Disclosure requirements" as category
    if stripped.lower() == "disclosure requirements":
        return "category"
    if stripped in CATEGORY_HEADERS:
        return "category"
    if re.match(r"^Appendix\s+[A-Z]", stripped):
        return "appendix"
    if stripped == "Application Requirements":
        return "appendix_title"
    if stripped.lower().startswith("disclosure requirement"):
        return "disclosure"
    if NUMBERED_SUBSECTION_RE.match(stripped):
        return "subsection"
    if is_toc_line(stripped):
        return "toc"
    
    para_num, para_text = is_paragraph_start(stripped)
    if para_num:
        if para_num.startswith("AR"):
            return "ar_paragraph"
        return "paragraph"
    
    return "other"


# ---------------------------------------------------------------------------
# Core Parser (complete rewrite with cleaner state machine)
# ---------------------------------------------------------------------------

class ESRSParser:
    def __init__(self, lines: list[str], pdf_name: str):
        self.lines = lines
        self.pdf_name = pdf_name
        self.pos = 0
        
    def parse(self) -> dict:
        """Main parse entry point."""
        result = {}
        
        # Step 1: Extract standard and title from the very first lines
        standard, title = self._extract_standard_and_title()
        result["standard"] = standard
        result["title"] = title
        
        # Step 2: Skip past the TABLE OF CONTENTS
        self._skip_toc()
        
        # Step 3: Parse Objective
        result["objective"] = self._parse_objective()
        
        # Step 4: Parse content and appendix
        content, appendix = self._parse_body()
        result["content"] = content
        if appendix:
            result["appendix"] = appendix
            
        return result
    
    def _extract_standard_and_title(self) -> tuple[str, str]:
        """Extract 'ESRS X1' and 'TITLE' from the lines before TOC."""
        standard = ""
        title = ""
        
        # Find lines before TABLE OF CONTENTS
        toc_idx = len(self.lines)
        for i, line in enumerate(self.lines[:50]):
            if line.strip() == "TABLE OF CONTENTS":
                toc_idx = i
                break
        
        # Search backwards from TOC for standard and title
        # Pattern: "ESRS G1" then "BUSINESS CONDUCT" then "TABLE OF CONTENTS"
        for i in range(toc_idx - 1, -1, -1):
            stripped = self.lines[i].strip()
            # Skip known noise
            if stripped in ("ANNEX I", "") or stripped.startswith("European sustainability"):
                continue
            # Skip ESRS listing lines like "ESRS E1 Climate change"
            if re.match(r"^ESRS\s+\S+\s+.+", stripped):
                continue
            # Match standalone standard identifier
            m = re.match(r"^(ESRS\s+\S+)$", stripped)
            if m:
                standard = m.group(1)
                break
            # All-caps title before standard
            if stripped == stripped.upper() and len(stripped) > 2:
                title = stripped
        
        return standard, title
    
    def _skip_toc(self):
        """Skip past the TABLE OF CONTENTS to find the actual 'Objective' start."""
        # Find TABLE OF CONTENTS
        toc_start = -1
        for i, line in enumerate(self.lines):
            if line.strip() == "TABLE OF CONTENTS":
                toc_start = i
                break
        
        if toc_start == -1:
            self.pos = 0
            return
        
        # Find the second 'Objective' (first is in TOC, second is actual content)
        obj_count = 0
        for i in range(toc_start, len(self.lines)):
            if self.lines[i].strip() == "Objective":
                obj_count += 1
                if obj_count >= 2:
                    self.pos = i
                    return
        
        # Fallback: find first numbered paragraph after TOC
        for i in range(toc_start + 1, len(self.lines)):
            line = self.lines[i].strip()
            if not is_toc_line(line) and not line.startswith("Appendix"):
                para_num, _ = is_paragraph_start(line)
                if para_num == "1":
                    self.pos = i
                    return
        
        self.pos = toc_start + 1
    
    def _parse_objective(self) -> dict:
        """Parse the Objective section."""
        objective = {}
        
        # Skip 'Objective' header if present
        if self.pos < len(self.lines) and self.lines[self.pos].strip() == "Objective":
            self.pos += 1
        
        # Objective ends when we hit a structural header (not another paragraph).
        # Key distinction: objective paragraphs like "1. The objective of..." should
        # NOT be confused with section headers like "1. Categories of ESRS".
        # We stop only at known structural boundaries.
        objective_stop_markers = (
            "category", "disclosure", "subsection", "appendix",
            "appendix_title", "objective_header", "toc_header"
        )
        
        while self.pos < len(self.lines):
            line = self.lines[self.pos].strip()
            cls = classify_line(line)
            
            if cls in objective_stop_markers:
                break
            
            para_num, para_text = is_paragraph_start(line)
            if para_num and not para_num.startswith("AR"):
                # Collect the full paragraph text (may span multiple lines)
                full_text = para_text
                self.pos += 1
                while self.pos < len(self.lines):
                    next_line = self.lines[self.pos].strip()
                    next_cls = classify_line(next_line)
                    next_para_num, _ = is_paragraph_start(next_line)
                    # Stop at new paragraph or structural element
                    if next_para_num or next_cls in objective_stop_markers:
                        break
                    full_text += " " + next_line
                    self.pos += 1
                objective[para_num] = full_text.strip()
            else:
                self.pos += 1
        
        return objective
    
    def _parse_body(self) -> tuple[list[dict], list[dict]]:
        """Parse everything after Objective into content and appendix."""
        content = []
        appendix = []
        in_appendix = False
        
        while self.pos < len(self.lines):
            line = self.lines[self.pos].strip()
            cls = classify_line(line)
            
            # Detect appendix transition
            if cls == "appendix":
                in_appendix = True
                app_section = self._parse_appendix_section()
                if app_section:
                    appendix.append(app_section)
                continue
            
            if cls == "appendix_title":
                in_appendix = True
                self.pos += 1
                continue
            
            if in_appendix:
                # Within appendix, parse subsections
                if cls == "category":
                    cat_section = self._parse_category(is_appendix=True)
                    if cat_section and appendix:
                        if "subsections" not in appendix[-1]:
                            appendix[-1]["subsections"] = []
                        appendix[-1]["subsections"].append(cat_section)
                    continue
                elif cls == "disclosure":
                    dr = self._parse_disclosure_requirement(is_appendix=True)
                    if dr and appendix:
                        if "subsections" not in appendix[-1]:
                            appendix[-1]["subsections"] = []
                        appendix[-1]["subsections"].append(dr)
                    continue
                elif cls in ("ar_paragraph", "paragraph"):
                    paras = self._collect_paragraphs()
                    if paras and appendix:
                        last = appendix[-1]
                        if "subsections" in last and last["subsections"]:
                            last_sub = last["subsections"][-1]
                            if "paragraphs" not in last_sub:
                                last_sub["paragraphs"] = []
                            last_sub["paragraphs"].extend(paras)
                        else:
                            if "paragraphs" not in last:
                                last["paragraphs"] = []
                            last["paragraphs"].extend(paras)
                    continue
                else:
                    if line and cls == "other":
                        titled_sub = self._parse_titled_subsection(is_appendix=True)
                        if titled_sub and appendix:
                            if "subsections" not in appendix[-1]:
                                appendix[-1]["subsections"] = []
                            appendix[-1]["subsections"].append(titled_sub)
                        continue
                    self.pos += 1
                    continue
            
            # Content parsing
            if cls == "category":
                section = self._parse_category(is_appendix=False)
                if section:
                    content.append(section)
            elif cls == "subsection":
                section = self._parse_numbered_subsection()
                if section:
                    content.append(section)
            elif cls == "disclosure":
                dr = self._parse_disclosure_requirement(is_appendix=False)
                if dr:
                    content.append(dr)
            elif cls == "paragraph":
                # Check if this is truly a numbered section header (ESRS_1 style)
                # or a regular paragraph that should be attached to the previous section
                if self._is_section_header(line):
                    section = self._parse_numbered_section()
                    if section:
                        content.append(section)
                else:
                    # Regular paragraph — attach to the last content section
                    paras = self._collect_paragraphs()
                    if paras and content:
                        if "paragraphs" not in content[-1]:
                            content[-1]["paragraphs"] = []
                        content[-1]["paragraphs"].extend(paras)
                    elif paras:
                        content.append({"title": "", "paragraphs": paras})
            else:
                self.pos += 1
        
        return content, appendix
    
    def _is_section_header(self, line: str) -> bool:
        """Check if a numbered line is truly a section header (not a paragraph).
        
        Section headers: '1. Categories of ESRS Standards, reporting areas...'
        Paragraphs: '3. The content of this Standard on general disclosures...'
        
        Heuristic: ESRS_1 section headers are high-level topics. They are 
        typically in a known set or are relatively short noun phrases.
        """
        m = NUMBERED_SECTION_RE.match(line)
        if not m:
            return False
        num = int(m.group(1))
        text = m.group(2)
        # Only ESRS_1 has numbered sections (numbers 1-10)
        # Real section headers don't start with "The", "This", "In", "An"
        sentence_starters = ("The ", "This ", "In ", "An ", "A ", "As ", "For ", "When ", "Where ")
        if text.startswith(sentence_starters):
            return False
        # Section headers are relatively short and typically < 120 chars
        if len(line) > 120:
            return False
        return num <= 15
    
    def _parse_category(self, is_appendix: bool) -> dict | None:
        """Parse a category section (Governance, Strategy, etc.).
        
        For 'Disclosure Requirements', acts as a container that nests
        child categories and DRs as subsections.
        """
        title = self.lines[self.pos].strip()
        is_container = title.lower() == "disclosure requirements"
        self.pos += 1
        section: dict = {"title": title}
        subsections = []
        
        while self.pos < len(self.lines):
            line = self.lines[self.pos].strip()
            cls = classify_line(line)
            
            # Stop conditions
            if cls in ("appendix", "appendix_title"):
                break
            # Another category at same level — stop (unless we are a container)
            if cls == "category" and line.strip() != title:
                if is_container:
                    # Container consumes child categories
                    sub_cat = self._parse_category(is_appendix=is_appendix)
                    if sub_cat:
                        subsections.append(sub_cat)
                    continue
                else:
                    break
            
            if cls == "disclosure":
                dr = self._parse_disclosure_requirement(is_appendix=is_appendix)
                if dr:
                    subsections.append(dr)
            elif cls in ("paragraph", "ar_paragraph"):
                paras = self._collect_paragraphs()
                if paras:
                    if subsections:
                        if "paragraphs" not in subsections[-1]:
                            subsections[-1]["paragraphs"] = []
                        subsections[-1]["paragraphs"].extend(paras)
                    else:
                        if "paragraphs" not in section:
                            section["paragraphs"] = []
                        section["paragraphs"].extend(paras)
            elif cls == "subsection":
                sub = self._parse_numbered_subsection()
                if sub:
                    subsections.append(sub)
            elif cls == "other":
                if is_appendix:
                    titled = self._parse_titled_subsection(is_appendix=True)
                    if titled:
                        subsections.append(titled)
                else:
                    self.pos += 1
            else:
                self.pos += 1
        
        if subsections:
            section["subsections"] = subsections
        
        return section
    
    def _parse_disclosure_requirement(self, is_appendix: bool) -> dict | None:
        """Parse a Disclosure Requirement section.
        
        Distinguishes between:
        - Container "Disclosure Requirements" → consumes categories/DRs as subsections
        - Individual DRs (e.g., "Disclosure Requirement G1-1 – ...") → only collect paragraphs
        """
        title = self.lines[self.pos].strip()
        is_container = (title == "Disclosure Requirements")
        self.pos += 1
        
        # DR titles may span multiple lines
        if not is_container:
            while self.pos < len(self.lines):
                next_line = self.lines[self.pos].strip()
                next_cls = classify_line(next_line)
                # If next line is clearly a new element, stop
                if next_cls in ("paragraph", "ar_paragraph", "category", "disclosure",
                                "subsection", "appendix", "appendix_title"):
                    break
                # If next line looks like title continuation
                if (next_line and not next_line[0].isdigit() and 
                    next_cls == "other"):
                    title += " " + next_line
                    self.pos += 1
                else:
                    break
        
        dr: dict = {"title": title}
        
        if is_container:
            # "Disclosure Requirements" is a container — collect categories and DRs as subsections
            subsections = []
            while self.pos < len(self.lines):
                line = self.lines[self.pos].strip()
                cls = classify_line(line)
                if cls in ("appendix", "appendix_title"):
                    break
                if NUMBERED_SECTION_RE.match(line):
                    break
                if cls == "category":
                    sub_cat = self._parse_category(is_appendix=is_appendix)
                    if sub_cat:
                        subsections.append(sub_cat)
                elif cls == "disclosure":
                    sub_dr = self._parse_disclosure_requirement(is_appendix=is_appendix)
                    if sub_dr:
                        subsections.append(sub_dr)
                elif cls in ("paragraph", "ar_paragraph"):
                    paras = self._collect_paragraphs()
                    if paras:
                        if subsections:
                            if "paragraphs" not in subsections[-1]:
                                subsections[-1]["paragraphs"] = []
                            subsections[-1]["paragraphs"].extend(paras)
                        else:
                            if "paragraphs" not in dr:
                                dr["paragraphs"] = []
                            dr["paragraphs"].extend(paras)
                else:
                    self.pos += 1
            if subsections:
                dr["subsections"] = subsections
        else:
            # Individual DR — just collect its own paragraphs, don't consume siblings
            paragraphs = self._collect_paragraphs()
            if paragraphs:
                dr["paragraphs"] = paragraphs
        
        return dr
    
    def _parse_numbered_section(self) -> dict | None:
        """Parse a top-level numbered section like '1. Categories of ESRS'."""
        line = self.lines[self.pos].strip()
        self.pos += 1
        section: dict = {"title": line}
        subsections = []
        
        while self.pos < len(self.lines):
            next_line = self.lines[self.pos].strip()
            cls = classify_line(next_line)
            
            if cls in ("appendix", "appendix_title"):
                break
            if cls == "category":
                break
            # Another numbered section at same level
            if NUMBERED_SECTION_RE.match(next_line):
                # Check if it's a deeper subsection
                m_current = NUMBERED_SECTION_RE.match(line)
                m_next = NUMBERED_SECTION_RE.match(next_line)
                if m_current and m_next:
                    # Same level — stop
                    break
            
            if cls == "subsection":
                sub = self._parse_numbered_subsection()
                if sub:
                    subsections.append(sub)
            elif cls == "disclosure":
                dr = self._parse_disclosure_requirement(is_appendix=False)
                if dr:
                    subsections.append(dr)
            elif cls in ("paragraph",):
                if NUMBERED_SECTION_RE.match(next_line):
                    break
                paras = self._collect_paragraphs()
                if paras:
                    if subsections:
                        if "paragraphs" not in subsections[-1]:
                            subsections[-1]["paragraphs"] = []
                        subsections[-1]["paragraphs"].extend(paras)
                    else:
                        if "paragraphs" not in section:
                            section["paragraphs"] = []
                        section["paragraphs"].extend(paras)
            else:
                self.pos += 1
        
        if subsections:
            section["subsections"] = subsections
        
        return section
    
    def _parse_numbered_subsection(self) -> dict | None:
        """Parse a numbered subsection like '1.1 Categories of ESRS standards'."""
        line = self.lines[self.pos].strip()
        self.pos += 1
        section: dict = {"title": line}
        
        paragraphs = self._collect_paragraphs()
        if paragraphs:
            section["paragraphs"] = paragraphs
        
        return section
    
    def _parse_titled_subsection(self, is_appendix: bool) -> dict | None:
        """Parse a free-form titled subsection (e.g. 'Methodologies' in appendix)."""
        title = self.lines[self.pos].strip()
        self.pos += 1
        
        sub: dict = {"title": title}
        
        # Check what follows
        if self.pos < len(self.lines):
            cls = classify_line(self.lines[self.pos].strip())
            if cls in ("ar_paragraph", "paragraph"):
                paragraphs = self._collect_paragraphs()
                if paragraphs:
                    sub["paragraphs"] = paragraphs
                    return sub
            elif cls == "disclosure":
                subsections = []
                while self.pos < len(self.lines):
                    line = self.lines[self.pos].strip()
                    cls2 = classify_line(line)
                    if cls2 == "disclosure":
                        dr = self._parse_disclosure_requirement(is_appendix=is_appendix)
                        if dr:
                            subsections.append(dr)
                    elif cls2 == "category":
                        break
                    elif cls2 in ("appendix", "appendix_title"):
                        break
                    elif cls2 == "other":
                        # Might be nested titled subsection
                        inner = self._parse_titled_subsection(is_appendix=is_appendix)
                        if inner:
                            subsections.append(inner)
                    else:
                        break
                if subsections:
                    sub["subsections"] = subsections
                    return sub
        
        # If nothing useful follows, this might be noise
        return sub if ("paragraphs" in sub or "subsections" in sub) else None
    
    def _parse_appendix_section(self) -> dict:
        """Parse an appendix header and its description."""
        line = self.lines[self.pos].strip()
        self.pos += 1
        
        # Parse appendix title
        m = re.match(r"^Appendix\s+([A-Z])(?:[:\s]+)?(.*)", line)
        if m:
            letter = m.group(1)
            rest = m.group(2).strip()
            if rest:
                title = f"Appendix {letter}: {rest}"
            else:
                # Check next line
                if self.pos < len(self.lines):
                    next_line = self.lines[self.pos].strip()
                    cls = classify_line(next_line)
                    if cls == "other" or next_line == "Application Requirements":
                        title = f"Appendix {letter}: {next_line}"
                        self.pos += 1
                    else:
                        title = f"Appendix {letter}"
                else:
                    title = f"Appendix {letter}"
        else:
            title = line
        
        app: dict = {"title": title}
        
        # Check for description (starts with "This appendix..." or "This Appendix...")
        if self.pos < len(self.lines):
            desc_line = self.lines[self.pos].strip()
            if desc_line.lower().startswith("this appendix") or desc_line.lower().startswith("this annex"):
                desc = desc_line
                self.pos += 1
                # Continue collecting description lines
                while self.pos < len(self.lines):
                    next_line = self.lines[self.pos].strip()
                    cls = classify_line(next_line)
                    if cls in ("category", "disclosure", "paragraph", "ar_paragraph",
                               "appendix", "subsection"):
                        break
                    desc += " " + next_line
                    self.pos += 1
                app["description"] = desc.strip()
        
        # Parse appendix body (subsections)
        subsections = []
        while self.pos < len(self.lines):
            line2 = self.lines[self.pos].strip()
            cls = classify_line(line2)
            
            # Stop at next appendix
            if cls == "appendix":
                break
            
            if cls == "category":
                cat = self._parse_category(is_appendix=True)
                if cat:
                    subsections.append(cat)
            elif cls == "disclosure":
                dr = self._parse_disclosure_requirement(is_appendix=True)
                if dr:
                    subsections.append(dr)
            elif cls in ("ar_paragraph", "paragraph"):
                paras = self._collect_paragraphs()
                if paras:
                    if subsections:
                        if "paragraphs" not in subsections[-1]:
                            subsections[-1]["paragraphs"] = []
                        subsections[-1]["paragraphs"].extend(paras)
                    else:
                        if "paragraphs" not in app:
                            app["paragraphs"] = []
                        app["paragraphs"].extend(paras)
            elif cls == "other":
                titled = self._parse_titled_subsection(is_appendix=True)
                if titled:
                    subsections.append(titled)
            else:
                self.pos += 1
        
        if subsections:
            app["subsections"] = subsections
        
        return app
    
    def _collect_paragraphs(self) -> list[dict]:
        """Collect consecutive paragraphs (numbered or AR-numbered)."""
        paragraphs = []
        
        while self.pos < len(self.lines):
            line = self.lines[self.pos].strip()
            
            para_num, para_text = is_paragraph_start(line)
            if not para_num:
                break
            
            # Collect full paragraph text
            full_text = para_text
            self.pos += 1
            
            while self.pos < len(self.lines):
                next_line = self.lines[self.pos].strip()
                next_cls = classify_line(next_line)
                
                # Stop at any structural element
                if next_cls in ("paragraph", "ar_paragraph", "category", "disclosure",
                                "subsection", "appendix", "appendix_title",
                                "objective_header"):
                    break
                # Also check for numbered section
                if NUMBERED_SECTION_RE.match(next_line):
                    break
                
                # This is a continuation line
                full_text += " " + next_line
                self.pos += 1
            
            paragraphs.append({para_num: full_text.strip()})
        
        return paragraphs


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Clean extracted text."""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_structure(obj):
    """Recursively clean all text in the structure."""
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if k == "paragraphs" and isinstance(v, list):
                result[k] = [{pk: clean_text(pv) for pk, pv in p.items()} for p in v]
            elif isinstance(v, str):
                result[k] = clean_text(v)
            else:
                result[k] = clean_structure(v)
        return result
    elif isinstance(obj, list):
        return [clean_structure(item) for item in obj]
    return obj


def remove_empty(obj):
    """Remove empty subsections/paragraphs lists."""
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            v2 = remove_empty(v)
            if k == "subsections" and isinstance(v2, list) and len(v2) == 0:
                continue
            if k == "paragraphs" and isinstance(v2, list) and len(v2) == 0:
                continue
            result[k] = v2
        return result
    elif isinstance(obj, list):
        return [remove_empty(item) for item in obj]
    return obj


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert_pdf_to_json(pdf_path: Path) -> dict:
    """Convert a single PDF to the JSON structure."""
    lines = extract_lines(str(pdf_path))
    parser = ESRSParser(lines, pdf_path.stem)
    result = parser.parse()
    result = clean_structure(result)
    result = remove_empty(result)
    return result


def main():
    """Main entry point: convert all PDFs in the input directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(INPUT_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {INPUT_DIR}")
        return

    print(f"Found {len(pdf_files)} PDF files to convert")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 60)

    for pdf_path in pdf_files:
        json_name = pdf_path.stem + ".json"
        output_path = OUTPUT_DIR / json_name

        print(f"Converting: {pdf_path.name} -> {json_name}")
        try:
            result = convert_pdf_to_json(pdf_path)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"  ✓ Written to {output_path}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()

    print("-" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
