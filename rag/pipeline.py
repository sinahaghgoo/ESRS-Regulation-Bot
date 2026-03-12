"""
Pipeline Module
===============
Orchestrates the full RAG pipeline from ingestion through answer generation.
Provides functions for:
  - Building the index (parse -> chunk -> embed -> store)
  - Running queries (retrieve -> generate)
  - Loading a pre-built pipeline for the Streamlit app
"""

import json
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

import numpy as np

from rag.config import (
    ARTIFACTS_DIR, REPORTS_MD_DIR,
    get_default_pipeline_config,
    COMPANY_MAP,
    COMPANY_REVERSE_MAP,
    ESRS_STANDARDS,
)
from rag.parser import parse_all_reports, parse_report, ParsedReport
from rag.chunker import chunk_report, Chunk
from rag.embedder import embed_texts, embed_query
from rag.indexer import VectorIndex
from rag.retriever import BM25Index, Retriever
from rag.esrs_context import get_esrs_requirements_text
from rag.generator import generate_answer, generate_comparison


# ---------------------------------------------------------------------------
# Artifact paths
# ---------------------------------------------------------------------------

def _chunks_path(strategy: str) -> Path:
    """Path for cached chunks JSON."""
    return ARTIFACTS_DIR / f"chunks_{strategy}.json"


def _bm25_path(strategy: str) -> Path:
    """Path for cached BM25 index."""
    return ARTIFACTS_DIR / f"bm25_index_{strategy}.json"


def _collection_name(config: Dict) -> str:
    """Generate Qdrant collection name from config."""
    prefix = config.get("qdrant_collection_prefix", "esrs_reports")
    strategy = config.get("chunking_strategy", "structure_aware")
    return f"{prefix}_{strategy}"


def _normalize_company_names(companies: Optional[List[str]]) -> Optional[List[str]]:
    """Normalize company identifiers to display names.

    Accepts both display names (e.g. "Aker BP") and report stems
    (e.g. "AkerBP_2024") to keep CLI/notebook usage backward compatible.
    """
    if not companies:
        return companies

    normalized: List[str] = []
    seen: set[str] = set()

    display_names = list(COMPANY_REVERSE_MAP.keys())
    stem_names = list(COMPANY_MAP.keys())

    for raw in companies:
        name = str(raw or "").strip()
        if not name:
            continue

        if name in COMPANY_REVERSE_MAP:
            display = name
        elif name in COMPANY_MAP:
            display = COMPANY_MAP[name]
        else:
            display = next((d for d in display_names if d.lower() == name.lower()), "")
            if not display:
                stem = next((s for s in stem_names if s.lower() == name.lower()), "")
                display = COMPANY_MAP.get(stem, name)

        if display not in seen:
            normalized.append(display)
            seen.add(display)

    return normalized


# ---------------------------------------------------------------------------
# Index building pipeline
# ---------------------------------------------------------------------------

def build_index(
    config: Optional[Dict] = None,
    force_rebuild: bool = False,
    companies: Optional[List[str]] = None,
) -> Tuple[VectorIndex, BM25Index, List[Chunk]]:
    """
    Build (or load) the complete index for a given pipeline configuration.
    
    Steps:
        1. Parse all markdown reports
        2. Chunk using configured strategy
        3. Embed all chunks via OpenAI API
        4. Build dense (Qdrant/local) and sparse (BM25) indices
        5. Cache all artifacts for resumability
    
    Args:
        config: Pipeline configuration dict (uses default if None)
        force_rebuild: If True, rebuild even if cached artifacts exist
        companies: Optional list of company names to index (default: all)
    
    Returns:
        Tuple of (VectorIndex, BM25Index, list of all Chunks)
    """
    if config is None:
        config = get_default_pipeline_config()
    companies = _normalize_company_names(companies)
    
    strategy = config["chunking_strategy"]
    chunk_size = config["chunk_size_tokens"]
    chunk_overlap = config["chunk_overlap_tokens"]
    collection = _collection_name(config)

    subset_suffix = ""
    if companies:
        # Isolate subset indices to avoid mutating full collections/caches.
        digest = hashlib.md5("|".join(sorted(companies)).encode("utf-8")).hexdigest()[:8]
        subset_suffix = f"_subset_{digest}"
        collection = f"{collection}{subset_suffix}"

    chunks_file = (
        ARTIFACTS_DIR / f"chunks_{strategy}{subset_suffix}.json"
        if subset_suffix
        else _chunks_path(strategy)
    )
    bm25_file = (
        ARTIFACTS_DIR / f"bm25_index_{strategy}{subset_suffix}.json"
        if subset_suffix
        else _bm25_path(strategy)
    )
    
    # --- Step 1 & 2: Parse and chunk (or load from cache) ---
    all_chunks: List[Chunk] = []
    
    if chunks_file.exists() and not force_rebuild:
        print(f"  Loading cached chunks from {chunks_file.name}...")
        with open(chunks_file, "r") as f:
            chunks_data = json.load(f)
        all_chunks = [_dict_to_chunk(d) for d in chunks_data]
        if companies:
            all_chunks = [c for c in all_chunks if c.company in companies]
        print(f"    Loaded {len(all_chunks)} chunks")
    else:
        print("  Parsing reports...")
        reports = {}
        if companies:
            # Cheap smoke/index subsets: only parse selected companies.
            for company in companies:
                stem = COMPANY_REVERSE_MAP.get(company)
                if not stem:
                    print(f"    Warning: unknown company '{company}', skipping")
                    continue
                md_path = REPORTS_MD_DIR / f"{stem}.md"
                if not md_path.exists():
                    print(f"    Warning: missing file for {company}: {md_path.name}")
                    continue
                print(f"  Parsing {md_path.name}...")
                reports[stem] = parse_report(md_path)
                print(
                    f"    -> {len(reports[stem].elements)} elements, "
                    f"{reports[stem].total_pages} pages"
                )
        else:
            reports = parse_all_reports()
        
        print(f"  Chunking with strategy='{strategy}', size={chunk_size}, overlap={chunk_overlap}...")
        for stem, report in reports.items():
            company_display = report.company
            if companies and company_display not in companies:
                continue
            
            chunks = chunk_report(report, strategy, chunk_size, chunk_overlap)
            all_chunks.extend(chunks)
            print(f"    {company_display}: {len(chunks)} chunks")
        
        # Save chunks to cache
        print(f"  Saving {len(all_chunks)} chunks to {chunks_file.name}...")
        with open(chunks_file, "w") as f:
            json.dump([c.to_dict() for c in all_chunks], f)
    
    # --- Step 3: Embed ---
    vec_index = VectorIndex(collection, use_qdrant=True)
    if force_rebuild:
        print(f"  Clearing existing vector index '{collection}'...")
        vec_index.clear()
    
    if vec_index.exists() and vec_index.count() > 0 and not force_rebuild:
        print(f"  Vector index '{collection}' already has {vec_index.count()} vectors")
    else:
        print(f"  Embedding {len(all_chunks)} chunks...")
        texts = [c.text for c in all_chunks]
        
        # Process in batches to show progress
        batch_size = 100
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            print(f"    Embedding batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}...")
            emb = embed_texts(batch)
            all_embeddings.append(emb)
        
        embeddings = np.vstack(all_embeddings)
        
        print(f"  Indexing {len(all_chunks)} chunks into '{collection}'...")
        vec_index.upsert_chunks(all_chunks, embeddings)
        print(f"    Index now has {vec_index.count()} vectors")
    
    # --- Step 4: Build BM25 index ---
    bm25 = BM25Index()
    if bm25_file.exists() and not force_rebuild:
        print(f"  Loading BM25 index from {bm25_file.name}...")
        bm25.load(bm25_file)
        print(f"    Loaded BM25 index with {bm25.n_docs} documents")
    else:
        print(f"  Building BM25 index from {len(all_chunks)} chunks...")
        bm25.build(all_chunks)
        bm25.save(bm25_file)
        print(f"    Saved BM25 index")
    
    return vec_index, bm25, all_chunks


def _dict_to_chunk(d: Dict) -> Chunk:
    """Convert a dictionary back to a Chunk object."""
    return Chunk(
        chunk_id=d.get("chunk_id", ""),
        text=d.get("text", ""),
        company=d.get("company", ""),
        reporting_year=d.get("reporting_year", 2024),
        source_file=d.get("source_file", ""),
        header_hierarchy=d.get("header_hierarchy", []),
        page_number=d.get("page_number", -1),
        page_span=d.get("page_span", []),
        chunk_index=d.get("chunk_index", 0),
        chunking_strategy=d.get("chunking_strategy", ""),
        token_count=d.get("token_count", 0),
        contains_table=d.get("contains_table", False),
        contains_figure=d.get("contains_figure", False),
    )


# ---------------------------------------------------------------------------
# Query pipeline
# ---------------------------------------------------------------------------

def run_query(
    query: str,
    companies: List[str],
    esrs_standard: Optional[str],
    retriever: Optional[Retriever],
    config: Dict,
) -> Dict[str, Any]:
    """
    Run a full RAG query pipeline.
    
    Args:
        query: User's natural language question
        companies: List of company names to analyze
        esrs_standard: Selected ESRS standard (e.g., "ESRS E1")
        retriever: Initialized Retriever instance
        config: Pipeline configuration
    
    Returns:
        Complete result dict with answers, citations, and page references
    """
    normalized_companies = _normalize_company_names(companies) or []
    if not normalized_companies:
        raise ValueError("At least one company must be provided.")

    standard = str(esrs_standard or "").strip()
    if not standard:
        # Backward compatibility for older notebook calls.
        standard = "ESRS 1"
    if standard not in ESRS_STANDARDS:
        raise ValueError(f"Unknown ESRS standard: {standard}")

    active_retriever = retriever
    if active_retriever is None:
        # Backward compatibility for legacy calls that passed retriever=None.
        vec_index, bm25, _ = build_index(config=config, force_rebuild=False, companies=normalized_companies)
        active_retriever = Retriever(vec_index, bm25, config)

    result = {
        "query": query,
        "companies": normalized_companies,
        "esrs_standard": standard,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # --- Extract ESRS context ---
    print("  Extracting ESRS requirements...")
    esrs_context = get_esrs_requirements_text(query, standard)
    result["esrs_requirements"] = esrs_context
    
    # --- Retrieve and generate per company ---
    per_company_chunks = {}
    per_company_answers = {}
    
    for company in normalized_companies:
        print(f"  Retrieving for {company}...")
        chunks = active_retriever.retrieve(query, company=company)
        per_company_chunks[company] = chunks
        
        print(f"  Generating answer for {company}...")
        answer = generate_answer(
            query=query,
            retrieved_chunks=chunks,
            esrs_context=esrs_context,
            company=company,
            config=config,
        )
        per_company_answers[company] = answer
    
    result["per_company"] = per_company_answers
    result["per_company_chunks"] = {
        company: chunks for company, chunks in per_company_chunks.items()
    }

    # Backward-compatible flat fields for single-company calls.
    if len(normalized_companies) == 1:
        only_company = normalized_companies[0]
        answer_obj = per_company_answers.get(only_company, {})
        result["answer"] = answer_obj.get("direct_answer", "")
        result["page_references"] = answer_obj.get("page_references", [])
    
    # --- Comparative analysis for multiple companies ---
    if len(normalized_companies) > 1:
        print("  Generating comparative analysis...")
        comparison = generate_comparison(
            query=query,
            per_company_results=per_company_chunks,
            esrs_context=esrs_context,
            config=config,
        )
        result["comparison"] = comparison
    
    return result


# ---------------------------------------------------------------------------
# Pipeline loader (for Streamlit app)
# ---------------------------------------------------------------------------

def load_pipeline(config: Optional[Dict] = None) -> Tuple[Retriever, Dict]:
    """
    Load a pre-built pipeline for interactive use.
    Loads cached indices without rebuilding.
    
    Args:
        config: Pipeline configuration (uses default if None)
    
    Returns:
        Tuple of (Retriever instance, config dict)
    """
    if config is None:
        config = get_default_pipeline_config()
    
    strategy = config["chunking_strategy"]
    collection = _collection_name(config)
    
    # Load vector index
    vec_index = VectorIndex(collection, use_qdrant=True)
    if not vec_index.exists():
        raise RuntimeError(
            f"Vector index '{collection}' not found. "
            f"Run 'python run_preprocessing.py' first."
        )
    
    # Load BM25 index
    bm25 = BM25Index()
    bm25_file = _bm25_path(strategy)
    if not bm25.load(bm25_file):
        raise RuntimeError(
            f"BM25 index not found at {bm25_file}. "
            f"Run 'python run_preprocessing.py' first."
        )
    
    retriever = Retriever(vec_index, bm25, config)
    
    return retriever, config
