"""
ESRS RAG Chatbot — Streamlit Application
==========================================
Interactive frontend for analyzing sustainability reports using the
best-performing RAG pipeline configuration from RQ1, with ESRS context
injection controlled by the RQ2/RQ3 finding.

Configuration is loaded dynamically:
  - Pipeline config (chunk strategy, size, retrieval method) from
    ``config/best_pipeline.yaml`` (written by RQ1 evaluation).
  - ESRS context decision from ``rq2_aggregate_results.json``
    (written by RQ2 evaluation) / ``rq3_summary.json``.
"""

import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

# ---------------------------------------------------------------------------
# Suppress noisy warnings from torch / torchvision / transformers that appear
# when Streamlit's file watcher scans installed packages.
#
# 1. Python warnings filters — catch UserWarning / FutureWarning from torch.
# 2. Logging level on Streamlit's local_sources_watcher — the watcher tries
#    to resolve __path__ on every imported module; when it hits the
#    transformers lazy-import stubs that depend on torchvision (not
#    installed), it logs a traceback at INFO level for each model.
#    Setting the logger to ERROR silences all of them.
# ---------------------------------------------------------------------------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", message=".*torch\\.classes.*")
warnings.filterwarnings("ignore", message=".*_register_pytree_node.*")
warnings.filterwarnings("ignore", message=".*Examining the path.*")

logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from rag.config import (
    COMPANY_MAP,
    ESRS_STANDARDS,
    load_best_pipeline_config,
    load_esrs_context_decision,
    BEST_PIPELINE_PATH,
)
from rag.pipeline import load_pipeline, run_query

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ESRS Sustainability Report Analyzer",
    page_icon="\U0001f4ca",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — Premium dark theme with glassmorphism
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ============ GLOBAL THEME ============ */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ============ SIDEBAR ============ */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1628 0%, #132743 60%, #0f3460 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] * {
        color: #e0e6ed !important;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stMultiSelect label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stTextInput label {
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.3px;
        color: #7eb8da !important;
    }

    /* ============ HEADER HERO ============ */
    .hero-container {
        background: linear-gradient(135deg, #0a1628 0%, #0f3460 50%, #16213e 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 32px 36px 28px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%; right: -20%;
        width: 400px; height: 400px;
        background: radial-gradient(circle, rgba(22,99,175,0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-container::after {
        content: '';
        position: absolute;
        bottom: -30%; left: -10%;
        width: 300px; height: 300px;
        background: radial-gradient(circle, rgba(0,210,150,0.08) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #e0e6ed 0%, #7eb8da 50%, #00d296 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 8px 0;
        position: relative;
        z-index: 1;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #8899aa;
        font-weight: 400;
        margin: 0;
        position: relative;
        z-index: 1;
    }

    /* ============ PIPELINE STATUS CHIP ============ */
    .pipeline-chip {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(0,210,150,0.08);
        border: 1px solid rgba(0,210,150,0.25);
        border-radius: 24px;
        padding: 6px 16px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #00d296;
        margin-top: 12px;
        position: relative;
        z-index: 1;
    }
    .pipeline-chip .dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #00d296;
        animation: pulse-dot 2s ease-in-out infinite;
    }
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.3); }
    }

    /* ============ COMPLIANCE BADGES ============ */
    .badge-high {
        background: linear-gradient(135deg, #00d296 0%, #00b37a 100%);
        color: #0a1628 !important;
        padding: 6px 18px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.88rem;
        display: inline-block;
        letter-spacing: 0.3px;
        box-shadow: 0 2px 12px rgba(0,210,150,0.3);
    }
    .badge-medium {
        background: linear-gradient(135deg, #f0b429 0%, #d4991a 100%);
        color: #1a1a1a !important;
        padding: 6px 18px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.88rem;
        display: inline-block;
        letter-spacing: 0.3px;
        box-shadow: 0 2px 12px rgba(240,180,41,0.3);
    }
    .badge-low {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white !important;
        padding: 6px 18px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.88rem;
        display: inline-block;
        letter-spacing: 0.3px;
        box-shadow: 0 2px 12px rgba(239,68,68,0.3);
    }
    .badge-na {
        background: linear-gradient(135deg, #64748b 0%, #475569 100%);
        color: white !important;
        padding: 6px 18px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.88rem;
        display: inline-block;
        letter-spacing: 0.3px;
    }

    /* ============ SOURCE CARDS ============ */
    .source-card {
        background: rgba(15,52,96,0.05);
        border-left: 4px solid #0f3460;
        padding: 14px 18px;
        margin: 10px 0;
        border-radius: 0 10px 10px 0;
        font-size: 0.88rem;
        transition: all 0.2s ease;
    }
    .source-card:hover {
        background: rgba(15,52,96,0.1);
        transform: translateX(4px);
    }
    .page-ref {
        background: linear-gradient(135deg, #e8f4fd 0%, #d0e8f7 100%);
        color: #0f3460;
        padding: 3px 12px;
        border-radius: 14px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.3px;
    }

    /* ============ GAP ITEMS ============ */
    .gap-item {
        background: rgba(240,180,41,0.08);
        border-left: 3px solid #f0b429;
        padding: 10px 14px;
        margin: 6px 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.86rem;
    }

    /* ============ COMPARISON OVERVIEW CARDS ============ */
    .comparison-card {
        background: rgba(15,52,96,0.03);
        border: 1px solid rgba(15,52,96,0.12);
        border-radius: 14px;
        padding: 24px 18px;
        text-align: center;
        transition: all 0.25s ease;
    }
    .comparison-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        border-color: rgba(15,52,96,0.25);
    }
    .comparison-card strong {
        font-size: 1rem;
        font-weight: 700;
        display: block;
        margin-bottom: 14px;
    }

    /* ============ COMPARISON TABLE ============ */
    .comparison-table {
        width: 100%;
        border-collapse: collapse;
        border: 1px solid rgba(15,52,96,0.12);
        border-radius: 12px;
        overflow: hidden;
        font-size: 0.9rem;
    }
    .comparison-table thead tr {
        background: linear-gradient(135deg, #0a1628 0%, #132743 100%);
    }
    .comparison-table thead th {
        color: #e0e6ed !important;
        padding: 14px 16px;
        text-align: left;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.3px;
    }
    .comparison-table tbody td {
        padding: 14px 16px;
        border-bottom: 1px solid rgba(15,52,96,0.08);
        vertical-align: top;
        line-height: 1.5;
    }
    .comparison-table tbody tr:hover {
        background: rgba(15,52,96,0.03);
    }
    .comparison-table .aspect-cell {
        font-weight: 600;
        color: #0f3460;
        white-space: nowrap;
        min-width: 140px;
    }

    /* ============ TIMING DISPLAY ============ */
    .timing-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(100,116,139,0.08);
        border: 1px solid rgba(100,116,139,0.15);
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.76rem;
        font-weight: 500;
        color: #64748b;
        margin-top: 8px;
    }

    /* ============ SIDEBAR ELEMENTS ============ */
    .sidebar-divider {
        border: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(126,184,218,0.2), transparent);
        margin: 16px 0;
    }
    .sidebar-section {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #5a8aaa !important;
        font-weight: 600;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* ============ ESRS CONTEXT INDICATOR ============ */
    .esrs-chip {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 10px 16px;
        border-radius: 10px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 8px 0;
    }
    .esrs-chip-enabled {
        background: rgba(0,210,150,0.12);
        border: 1px solid rgba(0,210,150,0.25);
        color: #00d296 !important;
    }
    .esrs-chip-disabled {
        background: rgba(100,116,139,0.1);
        border: 1px solid rgba(100,116,139,0.15);
        color: #8899aa !important;
    }
    .esrs-chip .esrs-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
    }
    .esrs-dot-on { background: #00d296; }
    .esrs-dot-off { background: #8899aa; }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ============ TAB STYLING ============ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Dynamic configuration
# ---------------------------------------------------------------------------
ESRS_CONTEXT_HELPS: bool = load_esrs_context_decision()

# Read pipeline display config (cheap YAML read, separate from pipeline load)
_DISPLAY_CONFIG = load_best_pipeline_config()
_STRATEGY_LABELS = {
    "fixed": "Fixed-size",
    "structure_aware": "Structure-aware",
    "semantic": "Semantic",
}
_METHOD_LABELS = {
    "dense": "Dense",
    "dense_rerank": "Dense + Rerank",
    "sparse": "Sparse (BM25)",
    "sparse_rerank": "Sparse + Rerank",
    "hybrid_rrf": "Hybrid RRF",
    "hybrid_rerank": "Hybrid + Rerank",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _render_company_result(company: str, result: dict) -> None:
    """Render detailed results for a single company."""
    answer_data = result.get("per_company", {}).get(company, {})
    chunks = result.get("per_company_chunks", {}).get(company, [])

    # --- Analysis ---
    st.markdown("#### \U0001f4dd Analysis")
    direct_answer = answer_data.get("direct_answer", "No answer generated.")
    st.markdown(direct_answer)

    findings = answer_data.get("key_findings", [])
    if findings:
        st.markdown("**Key Findings:**")
        for f in findings:
            st.markdown(f"- {f}")

    st.markdown("")

    # --- Sources ---
    st.markdown("#### \U0001f4c4 Sources")

    page_refs = answer_data.get("page_references", [])
    if page_refs:
        for ref in page_refs:
            page_display = ref.get("page_display", "?")
            section = ref.get("section", "Unknown section")
            st.markdown(
                f'<div class="source-card">'
                f'<span class="page-ref">\U0001f4c4 Page {page_display}</span> '
                f'&nbsp; <strong>{section}</strong>'
                f'</div>',
                unsafe_allow_html=True,
            )

    if chunks:
        with st.expander(f"\U0001f4c2 View all {len(chunks)} retrieved passages"):
            for i, chunk in enumerate(chunks, 1):
                pages = chunk.get("page_span", [])
                headers = chunk.get("header_hierarchy", [])
                text = chunk.get("text", "")
                page_str = ""
                if pages:
                    page_str = f"Pages: {', '.join(str(p) for p in pages if p)}"
                st.markdown(f"**[Source {i}]** {' > '.join(headers)}")
                if page_str:
                    st.caption(page_str)
                st.text(text[:500] + ("\u2026" if len(text) > 500 else ""))
                st.divider()

    st.markdown("")

    # --- Compliance Assessment ---
    st.markdown("#### \u2705 Compliance Assessment")
    level = answer_data.get("compliance_level", "N/A")
    rationale = answer_data.get("compliance_rationale", "")
    gaps = answer_data.get("disclosure_gaps", [])

    badge_cls = {
        "High": "badge-high",
        "Medium": "badge-medium",
        "Low": "badge-low",
    }.get(level, "badge-na")

    st.markdown(
        f'Compliance Level: <span class="{badge_cls}">{level}</span>',
        unsafe_allow_html=True,
    )

    if rationale:
        st.markdown("")
        st.markdown(f"**Rationale:** {rationale}")

    if gaps and level != "High":
        st.markdown("**Disclosure Gaps:**")
        for gap in gaps:
            st.markdown(
                f'<div class="gap-item">\u26a0\ufe0f {gap}</div>',
                unsafe_allow_html=True,
            )


def _generate_comparison(
    query: str,
    per_company: dict,
    esrs_standard: str,
) -> dict:
    """Generate a cross-company comparison using the LLM.

    Falls back to a basic data-driven table if the API is unavailable.
    """
    try:
        from openai import OpenAI
        from rag.config.settings import get_openai_api_key, get_generation_model

        key = get_openai_api_key()
        if not key:
            return _basic_comparison(per_company)

        client = OpenAI(api_key=key)

        company_blocks = []
        for company, data in per_company.items():
            company_blocks.append(
                f"Company: {company}\n"
                f"Compliance Level: {data.get('compliance_level', 'N/A')}\n"
                f"Key Findings: {'; '.join(data.get('key_findings', []))}\n"
                f"Disclosure Gaps: {'; '.join(data.get('disclosure_gaps', []))}\n"
                f"Rationale: {data.get('compliance_rationale', '')}"
            )

        system_prompt = (
            "You compare ESRS disclosure compliance across multiple companies. "
            "Based on the individual company assessments provided, generate a "
            "structured comparative analysis. Return a JSON object. "
            "Each row has an 'aspect' name and an 'assessments' array with one "
            "entry per company (company name + concise assessment text). "
            "Include exactly these aspects in this order: "
            "'Key Strengths', 'Key Weaknesses', 'Notable Differences'. "
            "Keep each assessment cell to 1-2 sentences maximum. "
            "The 'summary' field should be a concise 2-3 sentence overall "
            "comparison of the companies' ESRS compliance postures."
        )

        user_prompt = (
            f"Question analyzed: {query}\n"
            f"ESRS Standard: {esrs_standard or 'General'}\n\n"
            + "\n\n".join(company_blocks)
            + "\n\nGenerate the structured comparison."
        )

        response = client.responses.create(
            model=get_generation_model(),
            reasoning={"effort": "medium"},
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "comparison_analysis",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "rows": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "aspect": {"type": "string"},
                                        "assessments": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "company": {
                                                        "type": "string",
                                                    },
                                                    "text": {
                                                        "type": "string",
                                                    },
                                                },
                                                "required": [
                                                    "company",
                                                    "text",
                                                ],
                                                "additionalProperties": False,
                                            },
                                        },
                                    },
                                    "required": ["aspect", "assessments"],
                                    "additionalProperties": False,
                                },
                            },
                            "summary": {"type": "string"},
                        },
                        "required": ["rows", "summary"],
                        "additionalProperties": False,
                    },
                }
            },
            max_output_tokens=2048,
            timeout=120,
        )

        return json.loads(response.output_text or "{}")
    except Exception:
        return _basic_comparison(per_company)


def _basic_comparison(per_company: dict) -> dict:
    """Build a basic comparison table from per-company data (no LLM)."""
    companies = list(per_company.keys())
    rows = [
        {
            "aspect": "Compliance Level",
            "assessments": [
                {
                    "company": c,
                    "text": per_company[c].get("compliance_level", "N/A"),
                }
                for c in companies
            ],
        },
        {
            "aspect": "Key Strengths",
            "assessments": [
                {
                    "company": c,
                    "text": (
                        "; ".join(per_company[c].get("key_findings", [])[:2])
                        or "\u2014"
                    ),
                }
                for c in companies
            ],
        },
        {
            "aspect": "Disclosure Gaps",
            "assessments": [
                {
                    "company": c,
                    "text": (
                        "; ".join(per_company[c].get("disclosure_gaps", [])[:2])
                        or "\u2014"
                    ),
                }
                for c in companies
            ],
        },
    ]
    return {"rows": rows, "summary": ""}


def _render_comparison(comparison: dict, companies: list[str]) -> None:
    """Render the cross-company comparison as an HTML table with summary."""
    rows = comparison.get("rows", [])
    if not rows:
        st.info("No comparison data available.")
        return

    # Build HTML table
    header_cells = "".join(f"<th>{c}</th>" for c in companies)

    _DASH = "\u2014"

    body_html = ""
    for row in rows:
        assessment_map = {
            a["company"]: a["text"] for a in row.get("assessments", [])
        }
        cells = "".join(
            f"<td>{assessment_map.get(c, _DASH)}</td>" for c in companies
        )
        body_html += (
            f'<tr><td class="aspect-cell">{row["aspect"]}</td>'
            f"{cells}</tr>"
        )

    st.markdown(
        f'<div style="overflow-x:auto;">'
        f'<table class="comparison-table">'
        f"<thead><tr><th>Aspect</th>{header_cells}</tr></thead>"
        f"<tbody>{body_html}</tbody>"
        f"</table></div>",
        unsafe_allow_html=True,
    )

    summary = comparison.get("summary", "")
    if summary:
        st.markdown("")
        st.info(f"\U0001f4ca **Comparative Summary:** {summary}")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "pipeline_loaded" not in st.session_state:
    st.session_state.pipeline_loaded = False
    st.session_state.retriever = None
    st.session_state.config = None
    st.session_state.result = None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    # Logo
    st.markdown(
        '<div style="text-align:center; padding: 8px 0 16px;">'
        '<span style="font-size:2rem;">\U0001f4ca</span><br/>'
        '<span style="font-size:1.1rem; font-weight:700; '
        "background: linear-gradient(135deg, #e0e6ed, #7eb8da); "
        "-webkit-background-clip: text; -webkit-text-fill-color: transparent; "
        'background-clip: text;">ESRS Analyzer</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # ---- Company Selector ----
    st.markdown(
        '<p class="sidebar-section">\U0001f3e2 Select Companies</p>',
        unsafe_allow_html=True,
    )
    company_names = sorted(COMPANY_MAP.values())
    selected_companies = st.multiselect(
        "Companies",
        options=company_names,
        default=[],
        help=(
            "Choose one or more companies to analyze. Each company's "
            "sustainability report will be queried independently. "
            "Selecting multiple companies enables a cross-company "
            "comparison table."
        ),
        label_visibility="collapsed",
    )

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # ---- Configuration ----
    st.markdown(
        '<p class="sidebar-section">\u2699\ufe0f Pipeline Configuration</p>',
        unsafe_allow_html=True,
    )

    raw_strategy = _DISPLAY_CONFIG.get("chunk_strategy", "fixed")
    raw_method = _DISPLAY_CONFIG.get("retrieval_method", "hybrid_rerank")
    raw_size = _DISPLAY_CONFIG.get("chunk_size", 1024)

    st.text_input(
        "Chunking Strategy",
        value=_STRATEGY_LABELS.get(raw_strategy, raw_strategy),
        disabled=True,
        help=(
            "Text chunking strategy, optimized in RQ1. "
            "Fixed-size splits text into equal token windows; "
            "Structure-aware respects document headings; "
            "Semantic groups text by embedding similarity. "
            "Read-only \u2014 determined by the best RQ1 configuration."
        ),
    )
    st.text_input(
        "Retrieval Method",
        value=_METHOD_LABELS.get(raw_method, raw_method),
        disabled=True,
        help=(
            "Retrieval method, optimized in RQ1. "
            "Hybrid + Rerank combines dense (embedding) and sparse (BM25) "
            "retrieval, then reranks results with a cross-encoder model "
            "for highest accuracy. "
            "Read-only \u2014 determined by the best RQ1 configuration."
        ),
    )
    st.text_input(
        "Chunk Size (tokens)",
        value=str(raw_size),
        disabled=True,
        help=(
            "Token size for each text chunk, optimized in RQ1. "
            "Larger chunks provide more context per retrieval hit "
            "but may dilute relevance. "
            "Read-only \u2014 determined by the best RQ1 configuration."
        ),
    )

    top_k = st.slider(
        "Chunks to Retrieve",
        min_value=5,
        max_value=30,
        value=int(_DISPLAY_CONFIG.get("top_k", 15)),
        step=5,
        help=(
            "Number of text chunks retrieved per query. Higher values "
            "may capture more evidence but increase processing time "
            "and may include less relevant passages. This parameter was "
            "not part of the RQ1 optimization and can be adjusted freely."
        ),
    )

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # ---- ESRS Context Indicator ----
    if ESRS_CONTEXT_HELPS:
        st.markdown(
            '<div class="esrs-chip esrs-chip-enabled">'
            '<span class="esrs-dot esrs-dot-on"></span>'
            "ESRS Context Injection: Enabled"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="esrs-chip esrs-chip-disabled">'
            '<span class="esrs-dot esrs-dot-off"></span>'
            "ESRS Context Injection: Disabled"
            "</div>",
            unsafe_allow_html=True,
        )

    # ESRS standard selector — only when context injection is active
    selected_esrs = ""
    if ESRS_CONTEXT_HELPS:
        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
        st.markdown(
            '<p class="sidebar-section">\U0001f4cb ESRS Standard</p>',
            unsafe_allow_html=True,
        )
        selected_esrs = st.selectbox(
            "ESRS Standard",
            options=ESRS_STANDARDS,
            index=0,
            help=(
                "Select the ESRS standard to scope the analysis. "
                "The corresponding regulatory requirements will be "
                "injected into the prompt to guide the compliance "
                "assessment."
            ),
            label_visibility="collapsed",
        )



# ---------------------------------------------------------------------------
# Load pipeline (cached across reruns)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _load_pipeline():
    cfg = load_best_pipeline_config()
    pipeline_name = cfg.pop("_best_pipeline_name", "default")
    selection_reason = cfg.pop("_selection_reason", "")
    retriever, cfg = load_pipeline(cfg)
    return retriever, cfg, pipeline_name, selection_reason


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="hero-container">'
    '<h1 class="hero-title">ESRS Sustainability Report Analyzer</h1>'
    '<p class="hero-subtitle">'
    "Analyze corporate sustainability reports against European Sustainability "
    "Reporting Standards using an optimized RAG pipeline."
    "</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Pipeline loading
# ---------------------------------------------------------------------------
try:
    if not st.session_state.pipeline_loaded:
        with st.spinner(
            "\U0001f504 Loading RAG pipeline\u2026 This may take a moment on first run."
        ):
            retriever, config, pipeline_name, selection_reason = _load_pipeline()
        st.session_state.retriever = retriever
        st.session_state.config = config
        st.session_state.pipeline_name = pipeline_name
        st.session_state.selection_reason = selection_reason
        st.session_state.pipeline_loaded = True
    else:
        retriever = st.session_state.retriever
        config = st.session_state.config
        pipeline_name = st.session_state.get("pipeline_name", "default")
        selection_reason = st.session_state.get("selection_reason", "")

    pipeline_ready = True

    # Pipeline status chip
    if BEST_PIPELINE_PATH.exists():
        st.markdown(
            f'<div class="pipeline-chip">'
            f'<span class="dot"></span> Pipeline: {pipeline_name}'
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("Using default pipeline (no RQ1 manifest found)")

except Exception as e:
    st.error(
        f"\u26a0\ufe0f Pipeline not loaded. Please run "
        f"`python run_preprocessing.py` first.\n\n**Error:** {e}"
    )
    pipeline_ready = False

# ---------------------------------------------------------------------------
# Query input
# ---------------------------------------------------------------------------
st.markdown("")

query = st.text_area(
    "Enter your question about sustainability disclosures:",
    placeholder=(
        "e.g., How has the company conducted its double materiality assessment?"
    ),
    height=100,
    key="query_input",
)

col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    submit = st.button(
        "\U0001f50d Analyze",
        type="primary",
        disabled=not pipeline_ready,
        use_container_width=True,
    )
with col2:
    clear = st.button("\U0001f5d1 Clear", use_container_width=True)

if clear:
    st.session_state.result = None
    st.rerun()

# ---------------------------------------------------------------------------
# Run query
# ---------------------------------------------------------------------------
if submit and query and selected_companies:
    run_config = dict(config)
    run_config["top_k"] = top_k

    start_time = time.time()
    with st.spinner(
        "\U0001f50d Retrieving evidence and generating answers\u2026"
    ):
        try:
            result = run_query(
                query=query,
                companies=selected_companies,
                esrs_standard=selected_esrs,
                retriever=retriever,
                config=run_config,
                use_esrs_context=ESRS_CONTEXT_HELPS,
            )
        except Exception as e:
            st.error(f"\u274c Error during analysis: {e}")
            st.stop()

    # Cross-company comparison (extra LLM call)
    if len(selected_companies) > 1:
        with st.spinner(
            "\U0001f4ca Generating cross-company comparison\u2026"
        ):
            comparison = _generate_comparison(
                query=query,
                per_company=result["per_company"],
                esrs_standard=selected_esrs,
            )
            result["comparison"] = comparison

    elapsed = time.time() - start_time
    result["elapsed_seconds"] = round(elapsed, 2)
    st.session_state.result = result

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------
if st.session_state.result:
    result = st.session_state.result

    st.markdown("---")

    # Timing chip
    elapsed = result.get("elapsed_seconds", 0)
    if elapsed:
        st.markdown(
            f'<div class="timing-chip">'
            f"\u23f1 Analysis completed in {elapsed:.1f}s"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("")

    companies_list = result.get("companies", [])

    if len(companies_list) == 1:
        # ---- Single company: direct display ----
        _render_company_result(companies_list[0], result)

    else:
        # ---- Multi-company: overview + tabs + comparison ----

        # Compliance overview cards
        st.markdown("### Compliance Overview")
        cols = st.columns(len(companies_list))
        for i, company in enumerate(companies_list):
            answer_data = result.get("per_company", {}).get(company, {})
            level = answer_data.get("compliance_level", "N/A")
            badge_cls = {
                "High": "badge-high",
                "Medium": "badge-medium",
                "Low": "badge-low",
            }.get(level, "badge-na")
            with cols[i]:
                st.markdown(
                    f'<div class="comparison-card">'
                    f"<strong>{company}</strong>"
                    f'<span class="{badge_cls}">{level}</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("")

        # Company tabs (clickable, horizontal)
        company_tabs = st.tabs(
            [f"\U0001f3e2 {c}" for c in companies_list]
        )
        for tab, company in zip(company_tabs, companies_list):
            with tab:
                _render_company_result(company, result)

        # Cross-company comparison section
        comparison = result.get("comparison")
        if comparison:
            st.markdown("---")
            st.markdown("### \U0001f4ca Cross-Company Comparison")
            _render_comparison(comparison, companies_list)

elif submit and not query:
    st.warning("\u26a0\ufe0f Please enter a question.")
elif submit and not selected_companies:
    st.warning("\u26a0\ufe0f Please select at least one company.")
