"""
ESRS RAG Chatbot — Streamlit Application
==========================================
Interactive frontend for analyzing sustainability reports using the
best-performing RAG pipeline configuration.

Features:
  - Select one or multiple companies
  - Select the relevant ESRS chapter
  - Enter a natural-language question
  - View generated answer with citations
  - View supporting report passages with page references
  - View applicable ESRS requirements
  - View compliance assessment (Low / Medium / High)
"""

import sys
import os
from pathlib import Path

# Ensure the project root is on the Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import json
import time

from rag.config import (
    COMPANY_MAP, ESRS_STANDARDS, get_default_pipeline_config,
    load_best_pipeline_config, BEST_PIPELINE_PATH,
)
from rag.pipeline import load_pipeline, run_query


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ESRS Sustainability Report Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for clean layout
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #1B4332;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #52796F;
        margin-bottom: 2rem;
    }
    .source-card {
        background-color: #F8F9FA;
        border-left: 4px solid #2D6A4F;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    .compliance-high { color: #2D6A4F; font-weight: bold; }
    .compliance-medium { color: #E9C46A; font-weight: bold; }
    .compliance-low { color: #E76F51; font-weight: bold; }
    .page-ref {
        background-color: #D8F3DC;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
if "pipeline_loaded" not in st.session_state:
    st.session_state.pipeline_loaded = False
    st.session_state.retriever = None
    st.session_state.config = None
    st.session_state.result = None


# ---------------------------------------------------------------------------
# Sidebar: Configuration and Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔧 Configuration")
    
    # Company selection
    company_names = sorted(COMPANY_MAP.values())
    selected_companies = st.multiselect(
        "Select Companies",
        options=company_names,
        default=["Aker BP"],
        help="Choose one or more companies to analyze"
    )
    
    # ESRS standard selection
    selected_esrs = st.selectbox(
        "ESRS Chapter",
        options=ESRS_STANDARDS,
        index=0,
        help="Select the ESRS standard to scope the analysis"
    )
    
    # Advanced settings
    with st.expander("⚙️ Advanced Settings"):
        retrieval_method = st.selectbox(
            "Retrieval Method",
            ["hybrid", "hybrid_rerank", "dense", "dense_rerank", "sparse", "sparse_rerank"],
            index=0,
            help="*_rerank variants apply cross-encoder reranking on top of the base retrieval method",
        )
        top_k = st.slider("Retrieved Chunks (top-k)", 5, 30, 15)
    
    st.markdown("---")
    st.markdown("### 📋 About")
    st.markdown(
        "This tool analyzes corporate sustainability reports "
        "against ESRS disclosure requirements using a RAG pipeline."
    )


# ---------------------------------------------------------------------------
# Load pipeline (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading RAG pipeline...")
def _load_pipeline():
    """Load the best-performing pipeline (or default if no manifest)."""
    config = load_best_pipeline_config()
    pipeline_name = config.pop("_best_pipeline_name", "default")
    selection_reason = config.pop("_selection_reason", "")
    retriever, config = load_pipeline(config)
    return retriever, config, pipeline_name, selection_reason


# ---------------------------------------------------------------------------
# Main content area
# ---------------------------------------------------------------------------
st.markdown('<div class="main-header">📊 ESRS Sustainability Report Analyzer</div>', 
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Analyze corporate sustainability reports against '
    'European Sustainability Reporting Standards</div>',
    unsafe_allow_html=True
)

# Load pipeline
try:
    if not st.session_state.pipeline_loaded:
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
    # Show which pipeline is loaded
    if BEST_PIPELINE_PATH.exists():
        st.sidebar.success(f"Pipeline: **{pipeline_name}**")
        if selection_reason:
            st.sidebar.caption(selection_reason[:200])
    else:
        st.sidebar.info("Using default pipeline (no evaluation manifest found)")
except Exception as e:
    st.error(
        f"⚠️ Pipeline not loaded. Please run `python run_preprocessing.py` first.\n\n"
        f"Error: {str(e)}"
    )
    pipeline_ready = False


# Query input
query = st.text_area(
    "Enter your question about sustainability disclosures:",
    placeholder="e.g., How has the company conducted its double materiality assessment?",
    height=100,
)

col1, col2 = st.columns([1, 5])
with col1:
    submit = st.button("🔍 Analyze", type="primary", disabled=not pipeline_ready)

# ---------------------------------------------------------------------------
# Run query and display results
# ---------------------------------------------------------------------------
if submit and query and selected_companies:
    # Update config with sidebar settings
    config["retrieval_method"] = retrieval_method
    config["top_k"] = top_k
    
    with st.spinner("Analyzing sustainability reports..."):
        try:
            result = run_query(
                query=query,
                companies=selected_companies,
                esrs_standard=selected_esrs,
                retriever=retriever,
                config=config,
            )
            st.session_state.result = result
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            st.stop()

# Display results
if st.session_state.result:
    result = st.session_state.result
    
    st.markdown("---")
    
    # --- Per-company results ---
    for company in result.get("companies", []):
        answer_data = result.get("per_company", {}).get(company, {})
        chunks = result.get("per_company_chunks", {}).get(company, [])
        
        st.markdown(f"## 🏢 {company}")
        
        # Tabs for organized display
        tab_answer, tab_sources, tab_esrs, tab_compliance = st.tabs([
            "💡 Answer", "📄 Sources & Pages", "📋 ESRS Requirements", "✅ Compliance"
        ])
        
        with tab_answer:
            direct_answer = answer_data.get("direct_answer", "No answer generated")
            st.markdown(direct_answer)
            
            # Key findings
            findings = answer_data.get("key_findings", [])
            if findings:
                st.markdown("**Key Findings:**")
                for f in findings:
                    st.markdown(f"- {f}")
        
        with tab_sources:
            st.markdown("### Supporting Report Passages")
            
            page_refs = answer_data.get("page_references", [])
            
            if page_refs:
                for ref in page_refs:
                    page_display = ref.get("page_display", "Unknown")
                    section = ref.get("section", "Unknown section")
                    preview = ref.get("text_preview", "")
                    
                    st.markdown(
                        f'<div class="source-card">'
                        f'<span class="page-ref">{page_display}</span> '
                        f'&nbsp; <b>{section}</b><br/>'
                        f'<small>{preview}...</small>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            
            # Full retrieved passages
            if chunks:
                with st.expander(f"View all {len(chunks)} retrieved passages"):
                    for i, chunk in enumerate(chunks, 1):
                        pages = chunk.get("page_span", [])
                        headers = chunk.get("header_hierarchy", [])
                        text = chunk.get("text", "")
                        
                        page_str = ""
                        if pages:
                            page_str = f"Pages: {', '.join(str(p) for p in pages)}"
                        
                        st.markdown(f"**[Source {i}]** {' > '.join(headers)}")
                        if page_str:
                            st.markdown(f"📍 {page_str}")
                        st.text(text[:500] + ("..." if len(text) > 500 else ""))
                        st.markdown("---")
        
        with tab_esrs:
            esrs_req = result.get("esrs_requirements", "")
            if esrs_req:
                st.markdown("### Applicable ESRS Disclosure Requirements")
                st.text(esrs_req)
            else:
                st.info("No ESRS requirements extracted.")
        
        with tab_compliance:
            level = answer_data.get("compliance_level", "Unknown")
            rationale = answer_data.get("compliance_rationale", "")
            gaps = answer_data.get("disclosure_gaps", [])
            
            # Compliance badge
            level_class = {
                "High": "compliance-high",
                "Medium": "compliance-medium",
                "Low": "compliance-low",
            }.get(level, "")
            
            st.markdown(
                f'### Compliance Level: <span class="{level_class}">{level}</span>',
                unsafe_allow_html=True,
            )
            
            if rationale:
                st.markdown(rationale)
            
            if gaps:
                st.markdown("**Disclosure Gaps:**")
                for gap in gaps:
                    st.markdown(f"- ⚠️ {gap}")
    
    # --- Comparative analysis ---
    comparison = result.get("comparison")
    if comparison:
        st.markdown("---")
        st.markdown("## 🔄 Comparative Analysis")
        
        summary = comparison.get("comparative_summary", "")
        if summary:
            st.markdown(summary)
        
        per_co = comparison.get("per_company", {})
        if per_co:
            cols = st.columns(min(len(per_co), 3))
            for i, (co, data) in enumerate(per_co.items()):
                with cols[i % len(cols)]:
                    level = data.get("compliance_level", "?")
                    st.markdown(f"**{co}**: {level}")
                    strengths = data.get("key_strengths", [])
                    if strengths:
                        st.markdown("Strengths: " + "; ".join(strengths[:3]))
                    gaps = data.get("key_gaps", [])
                    if gaps:
                        st.markdown("Gaps: " + "; ".join(gaps[:3]))
        
        common_gaps = comparison.get("common_gaps", [])
        if common_gaps:
            st.markdown("**Common Gaps:**")
            for g in common_gaps:
                st.markdown(f"- {g}")

elif submit and not query:
    st.warning("Please enter a question.")
elif submit and not selected_companies:
    st.warning("Please select at least one company.")
