"""
Extract company report text from PDF files using Azure Document Intelligence
and GPT-5.2 for figure semantic descriptions.

Output: structured markdown matching the format in data/company reports final/
"""

import argparse
import base64
import json
import os
import pickle
import re
import sys
import time

import fitz  # PyMuPDF
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

AZURE_ENDPOINT = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_DOCUMENT_INTELLIGENCE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5.2")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(SCRIPT_DIR, "data", "company reports")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "data", "company reports final")
CACHE_DIR = os.path.join(SCRIPT_DIR, "data", ".cache")

# ---------------------------------------------------------------------------
# Azure Document Intelligence — analyse PDF (with caching)
# ---------------------------------------------------------------------------

def analyse_pdf(pdf_path: str, report_name: str):
    """Send PDF to Azure Document Intelligence and return the result.
    
    Caches the result to avoid repeated API calls during development.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{report_name}_azure.pkl")
    
    # Check cache
    if os.path.exists(cache_path):
        print(f"[1/4] Loading cached Azure DI result for {report_name}")
        with open(cache_path, "rb") as f:
            result = pickle.load(f)
        print(f"   Pages: {len(result.pages)}")
        print(f"   Tables: {len(result.tables) if result.tables else 0}")
        print(f"   Figures: {len(result.figures) if result.figures else 0}")
        print(f"   Paragraphs: {len(result.paragraphs) if result.paragraphs else 0}")
        return result
    
    print(f"[1/4] Analysing PDF with Azure Document Intelligence...")
    client = DocumentIntelligenceClient(
        endpoint=AZURE_ENDPOINT,
        credential=AzureKeyCredential(AZURE_KEY),
    )
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    
    pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")
    
    poller = client.begin_analyze_document(
        "prebuilt-layout",
        AnalyzeDocumentRequest(bytes_source=pdf_base64),
        output_content_format="markdown",
    )
    result = poller.result()
    
    # Cache the result
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    
    print(f"   Pages: {len(result.pages)}")
    print(f"   Tables: {len(result.tables) if result.tables else 0}")
    print(f"   Figures: {len(result.figures) if result.figures else 0}")
    print(f"   Paragraphs: {len(result.paragraphs) if result.paragraphs else 0}")
    return result


# ---------------------------------------------------------------------------
# Figure image extraction from PDF
# ---------------------------------------------------------------------------

def extract_figure_image(pdf_path: str, figure, dpi: int = 200) -> str | None:
    """Extract a figure region from the PDF as a base64-encoded PNG.
    
    Returns None if the region is too small (likely decorative).
    """
    if not figure.bounding_regions:
        return None

    region = figure.bounding_regions[0]
    page_num = region.page_number  # 1-indexed

    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]

    polygon = region.polygon
    if not polygon or len(polygon) < 4:
        doc.close()
        return None

    # Convert polygon points to page coordinates (Azure uses inches -> 72 pts/inch)
    xs = [polygon[i] for i in range(0, len(polygon), 2)]
    ys = [polygon[i] for i in range(1, len(polygon), 2)]

    x0 = min(xs) * 72
    y0 = min(ys) * 72
    x1 = max(xs) * 72
    y1 = max(ys) * 72

    # Clamp to page bounds
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(page.rect.width, x1)
    y1 = min(page.rect.height, y1)

    width = x1 - x0
    height = y1 - y0

    # Skip very small figures (likely icons or decorative elements)
    if width < 50 or height < 50:
        doc.close()
        return None

    clip = fitz.Rect(x0, y0, x1, y1)
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, clip=clip)

    img_bytes = pix.tobytes("png")
    doc.close()

    return base64.b64encode(img_bytes).decode("utf-8")


def classify_and_describe_figure(base64_image: str, caption_hint: str = "") -> str | None:
    """Use GPT-5.2 to classify a figure and generate a semantic description.
    
    Returns the semantic description or None if the figure is decorative.
    """
    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = """You are analysing a figure from a company sustainability/annual report.

Your task:
1. Determine if this figure contains MEANINGFUL information (charts, diagrams, process flows, infographics, data visualizations, organizational charts, framework illustrations, matrices, timelines, maps with data) or if it is DECORATIVE (title images, logos, photographs of people/buildings/nature, header banners, background images, cover pages).

2. If DECORATIVE, respond with exactly: DECORATIVE

3. If MEANINGFUL, write a detailed semantic description of the figure that captures ALL the data, relationships, and insights shown. The description should be comprehensive enough that someone reading only the text description would understand the figure's full content without seeing it. Include specific numbers, categories, relationships, and trends shown in the figure.

Important: Write the description as a continuous paragraph, not as bullet points. Do NOT include any prefix like "Description:" - just write the description directly."""

    if caption_hint:
        prompt += f"\n\nThe figure has the following caption/title in the document: {caption_hint}"

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                    "detail": "high",
                                },
                            },
                            {
                                "type": "text",
                                "text": "Analyse this figure and provide the semantic description as instructed.",
                            },
                        ],
                    },
                ],
                max_completion_tokens=2000,
                temperature=0.1,
            )
            result = response.choices[0].message.content.strip()

            if result.upper().startswith("DECORATIVE"):
                return None

            # Clean up any accidental prefix
            for prefix in ("Description:", "MEANINGFUL:", "Meaningful:"):
                if result.startswith(prefix):
                    result = result[len(prefix):].strip()

            return result

        except Exception as e:
            print(f"   GPT-5.2 error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                print(f"   WARNING: Skipping figure after {max_retries} failed attempts")
                return None


# ---------------------------------------------------------------------------
# Table conversion to HTML
# ---------------------------------------------------------------------------

def table_to_html(table) -> str:
    """Convert an Azure Document Intelligence table to an HTML <table> string."""
    if not table.cells:
        return ""

    max_row = max(c.row_index for c in table.cells) + 1
    max_col = max(c.column_index for c in table.cells) + 1

    # Build grid to track which cells are covered by spans
    grid = [[None for _ in range(max_col)] for _ in range(max_row)]

    # Place cells into grid
    for cell in table.cells:
        r, c = cell.row_index, cell.column_index
        if grid[r][c] is None:
            grid[r][c] = cell

    lines = []
    
    # Check for caption
    caption_text = ""
    if hasattr(table, 'caption') and table.caption and hasattr(table.caption, 'content'):
        caption_text = table.caption.content.strip()

    lines.append("<table>")
    if caption_text:
        lines.append(f"<caption>{caption_text}</caption>")

    for r in range(max_row):
        lines.append("<tr>")
        for c_idx in range(max_col):
            cell = grid[r][c_idx]
            if cell is None or cell == "SPAN":
                continue  # Empty or covered by a span

            # Determine tag
            kind = getattr(cell, 'kind', '')
            tag = "th" if kind in ("columnHeader", "rowHeader") else "td"

            # Span attributes
            attrs = ""
            rs = getattr(cell, 'row_span', 1) or 1
            cs = getattr(cell, 'column_span', 1) or 1
            if rs > 1:
                attrs += f' rowspan="{rs}"'
            if cs > 1:
                attrs += f' colspan="{cs}"'

            # Mark spanned cells as covered
            for dr in range(rs):
                for dc in range(cs):
                    if dr == 0 and dc == 0:
                        continue
                    if r + dr < max_row and c_idx + dc < max_col:
                        grid[r + dr][c_idx + dc] = "SPAN"

            content = getattr(cell, 'content', '') or ''
            content = content.strip()
            lines.append(f"<{tag}{attrs}>{content}</{tag}>")

        lines.append("</tr>")

    lines.append("</table>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Build the markdown output
# ---------------------------------------------------------------------------

def get_element_page(element) -> int:
    """Get the page number of an element."""
    if hasattr(element, 'bounding_regions') and element.bounding_regions:
        return element.bounding_regions[0].page_number
    return 0


def build_markdown(result, pdf_path: str, empty_figcaptions: bool = False) -> str:
    """Assemble all elements into a single markdown string."""
    
    content = result.content
    
    # -------------------------------------------------------------------
    # Step 1: Collect all replacements (tables + figures) with their spans
    # -------------------------------------------------------------------
    replacements = []  # (offset, length, replacement_text, type, index)
    
    # Tables -> HTML
    print(f"[2/4] Processing tables...")
    if result.tables:
        for i, table in enumerate(result.tables):
            if table.spans:
                offset = table.spans[0].offset
                length = table.spans[0].length
                html = table_to_html(table)
                replacements.append((offset, length, "\n" + html + "\n", "table", i))
    
    # Figures -> process with GPT-5.2 or insert empty figcaptions
    figure_count = len(result.figures) if result.figures else 0
    if empty_figcaptions:
        print(f"[3/4] Processing {figure_count} figures with empty figcaptions...")
    else:
        print(f"[3/4] Processing {figure_count} figures with GPT-5.2...")
    if result.figures:
        if empty_figcaptions:
            for i, figure in enumerate(result.figures):
                if not figure.spans:
                    continue
                offset = figure.spans[0].offset
                length = figure.spans[0].length
                figure_html = "\n<figure>\n<figcaption></figcaption>\n</figure>\n"
                replacements.append((offset, length, figure_html, "figure_empty", i))
        else:
        # Cache for figure descriptions to avoid re-processing
            figure_cache_path = os.path.join(CACHE_DIR, f"{os.path.basename(pdf_path).replace('.pdf', '')}_figures.json")
            figure_cache = {}
            if os.path.exists(figure_cache_path):
                with open(figure_cache_path, "r") as f:
                    figure_cache = json.load(f)
            
            for i, figure in enumerate(result.figures):
                if not figure.spans:
                    continue
                
                offset = figure.spans[0].offset
                length = figure.spans[0].length
                page = get_element_page(figure)
                
                # Check figure description cache
                cache_key = f"{i}_{page}_{offset}"
                if cache_key in figure_cache:
                    desc = figure_cache[cache_key]
                    print(f"   Figure {i + 1}/{len(result.figures)} (page {page}) — cached")
                else:
                    print(f"   Figure {i + 1}/{len(result.figures)} (page {page})...")
                    
                    # Extract caption hint
                    caption_hint = ""
                    if hasattr(figure, 'caption') and figure.caption:
                        caption_hint = getattr(figure.caption, 'content', str(figure.caption))
                    
                    # Extract image from PDF
                    base64_img = extract_figure_image(pdf_path, figure)
                    
                    if base64_img is None:
                        desc = "__SKIP__"
                        print(f"      → Skipped (too small)")
                    else:
                        desc = classify_and_describe_figure(base64_img, caption_hint)
                        if desc is None:
                            desc = "__DECORATIVE__"
                            print(f"      → Decorative (skipped)")
                        else:
                            print(f"      → Meaningful ({len(desc)} chars)")
                    
                    figure_cache[cache_key] = desc
                    # Save cache after each figure
                    with open(figure_cache_path, "w") as f:
                        json.dump(figure_cache, f)
                
                if desc in ("__SKIP__", "__DECORATIVE__"):
                    replacements.append((offset, length, "", "figure_skip", i))
                else:
                    figure_html = f"\n<figure>\n<figcaption>{desc}</figcaption>\n</figure>\n"
                    replacements.append((offset, length, figure_html, "figure", i))
    
    # -------------------------------------------------------------------
    # Step 2: Apply all replacements (sort by offset descending)
    # -------------------------------------------------------------------
    replacements.sort(key=lambda x: x[0], reverse=True)
    
    for offset, length, replacement, rtype, idx in replacements:
        content = content[:offset] + replacement + content[offset + length:]
    
    # -------------------------------------------------------------------
    # Step 3: Post-processing cleanup
    # -------------------------------------------------------------------
    print(f"[4/4] Post-processing...")
    content = postprocess_markdown(content)
    
    return content


def postprocess_markdown(content: str) -> str:
    """Clean up the markdown output."""
    
    # Fix heading formatting: ensure space after # 
    content = re.sub(r'^(#{1,6})([^ #\n])', r'\1 \2', content, flags=re.MULTILINE)
    
    # Remove :selected: and :unselected: markers from checkboxes
    content = content.replace(":selected:", "").replace(":unselected:", "")
    
    # Remove Azure-specific comments
    content = re.sub(r'<!-- PageBreak -->', '', content)
    content = re.sub(r'<!-- PageHeader="[^"]*" -->', '', content)
    content = re.sub(r'<!-- PageFooter="[^"]*" -->', '', content)
    content = re.sub(r'<!--.*?-->', '', content)
    
    # Remove image references like ![](image_url) that Azure may include
    content = re.sub(r'!\[.*?\]\(.*?\)', '', content)
    
    # Clean up excessive blank lines (max 2 consecutive)
    content = re.sub(r'\n{4,}', '\n\n\n', content)
    
    # Clean up spaces around table/figure tags
    content = re.sub(r'\n{3,}(<table>)', r'\n\n\1', content)
    content = re.sub(r'(</table>)\n{3,}', r'\1\n\n', content)
    content = re.sub(r'\n{3,}(<figure>)', r'\n\n\1', content)
    content = re.sub(r'(</figure>)\n{3,}', r'\1\n\n', content)
    
    # Trim trailing whitespace on each line
    lines = content.split('\n')
    lines = [line.rstrip() for line in lines]
    content = '\n'.join(lines)
    
    # Ensure file ends with single newline
    content = content.strip() + '\n'
    
    return content


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def extract_report(report_name: str, output_dir: str = None, empty_figcaptions: bool = False):
    """Extract a single report from PDF to markdown."""
    
    pdf_path = os.path.join(PDF_DIR, f"{report_name}.pdf")
    if not os.path.exists(pdf_path):
        print(f"ERROR: PDF not found: {pdf_path}")
        sys.exit(1)
    
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"{report_name}.md")
    
    print(f"=" * 60)
    print(f"Extracting: {report_name}")
    print(f"  PDF: {pdf_path}")
    print(f"  Output: {output_path}")
    print(f"=" * 60)
    
    # Step 1: Analyse PDF (cached)
    result = analyse_pdf(pdf_path, report_name)
    
    # Step 2-4: Build markdown
    markdown = build_markdown(result, pdf_path, empty_figcaptions=empty_figcaptions)
    
    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown)
    
    print(f"\nDone! Output written to: {output_path}")
    print(f"Output size: {len(markdown):,} characters, {markdown.count(chr(10)):,} lines")
    
    # Quick stats
    table_count = markdown.count("<table>")
    figure_count = markdown.count("<figure>")
    heading_count = len(re.findall(r'^#+\s', markdown, re.MULTILINE))
    print(f"Stats: {heading_count} headings, {table_count} tables, {figure_count} figures")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Extract company report text from PDF")
    parser.add_argument(
        "--report",
        type=str,
        help="Report name without extension, e.g. 'AkerBP_2024'",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory (default: data/company reports final/)",
    )
    parser.add_argument(
        "--all-three",
        action="store_true",
        help="Process all three test reports: AkerBP, OMV, Repsol",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear the Azure DI cache before processing",
    )
    parser.add_argument(
        "--empty-figcaptions",
        action="store_true",
        help="Insert empty <figcaption></figcaption> for all detected figures and skip figure descriptions",
    )
    args = parser.parse_args()
    
    if args.clear_cache and os.path.exists(CACHE_DIR):
        import shutil
        shutil.rmtree(CACHE_DIR)
        print("Cache cleared.")
    
    if args.all_three:
        for name in ["AkerBP_2024", "OMV_2024", "Repsol_2024"]:
            extract_report(name, args.output_dir, empty_figcaptions=args.empty_figcaptions)
            print()
    elif args.report:
        extract_report(args.report, args.output_dir, empty_figcaptions=args.empty_figcaptions)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
