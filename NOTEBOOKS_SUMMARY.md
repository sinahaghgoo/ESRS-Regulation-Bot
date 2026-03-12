# ESRS RAG Thesis - Jupyter Notebooks

Four comprehensive Jupyter notebooks have been created for the ESRS sustainability report analysis project. Each notebook is a properly formatted `.ipynb` file with executable code cells and documentation.

## Notebook 1: Report Structure and Metadata Audit
**File:** `notebooks/01_report_structure_and_metadata_audit.ipynb`

**Purpose:** Parse all 17 Markdown sustainability reports and audit their structural quality.

**Key Analyses:**
- Parse all reports using `rag.parser.parse_all_reports()`
- Generate summary table of document statistics
- Element distribution analysis (headings, paragraphs, tables, figures)
- Page marker coverage validation
- Heading hierarchy depth analysis
- Table and figure audit with HTML integrity checks
- Sample structured elements inspection
- Save audit summary to `data/artifacts/report_audit_summary.csv`

**Cells:** 16 (7 markdown, 9 code)
**Dependencies:** rag.config, rag.parser, pandas

---

## Notebook 2: Chunking Experiments
**File:** `notebooks/02_chunking_experiments.ipynb`

**Purpose:** Compare fixed-size vs structure-aware chunking strategies (Research Question 1).

**Key Analyses:**
- Fixed-size chunking with configurable parameters
- Structure-aware chunking preserving document hierarchy
- Token distribution analysis with histograms
- Page metadata quality comparison
- Side-by-side comparison table
- Save chunks to JSON artifacts for downstream use

**Outputs:**
- Token distribution histogram: `data/artifacts/chunking_token_distribution.png`
- Fixed-size chunks: `data/artifacts/chunks_fixed_size.json`
- Structure-aware chunks: `data/artifacts/chunks_structure_aware.json`

**Cells:** 16 (7 markdown, 9 code)
**Dependencies:** rag.config, rag.parser, rag.chunker, pandas, matplotlib

---

## Notebook 3: Embeddings and Qdrant Indexing
**File:** `notebooks/03_embeddings_and_qdrant_indexing.ipynb`

**Purpose:** Build dense and sparse retrieval indices.

**Key Operations:**
- Embed all chunks using OpenAI embeddings (text-embedding-3-large)
- Create Qdrant vector collections (persisted locally)
- Build BM25 sparse indices
- Index statistics and validation
- Quick BM25 retrieval sanity check with test queries
- Embedding space validation with cosine similarity
- Save comprehensive index summary

**Outputs:**
- Index summary: `data/artifacts/index_summary.json`
- Qdrant collections (local Qdrant instance)
- BM25 indices (persisted as part of build process)

**Cells:** 13 (6 markdown, 7 code)
**Dependencies:** rag.config, rag.pipeline, rag.retriever, pandas

---

## Notebook 4: Retrieval, ESRS Context, and Answer Generation
**File:** `notebooks/04_retrieval_esrs_context_and_generation.ipynb`

**Purpose:** Demonstrate the complete RAG pipeline end-to-end.

**Pipeline Steps:**
1. **Retrieval**: Query chunks using configured retrieval method (hybrid, dense, or sparse)
   - Top-5 retrieval with page references
   - Integration with both vector and BM25 indices

2. **ESRS Context Extraction**: Extract relevant disclosure requirements
   - Load ESRS JSON standards
   - Extract requirements matching query and standard

3. **Answer Generation**: LLM-based answer generation with citations
   - Structured JSON output with strict citation grounding
   - Compliance level assessment
   - Key findings extraction

4. **Citation Resolution**: Verify and display page references
   - Page span validation
   - Header hierarchy context
   - Company information

5. **Multi-Company Comparison** (optional in full mode)
   - Compare answers across multiple companies
   - Identify best practices and gaps

**Cells:** 16 (7 markdown, 9 code)
**Dependencies:** rag.config, rag.pipeline, rag.retriever, rag.esrs_context, rag.generator, pandas

---

## Configuration and Features

### SMOKE_TEST Flag
All notebooks support a `SMOKE_TEST = True` flag (set in early cells) that:
- Processes only 2 companies instead of 17
- Enables quick validation without full computational cost
- Useful for development and debugging

### Artifact Caching
- Chunks are cached as JSON for reuse
- BM25 indices are persisted
- Embedding cache prevents re-embedding
- Index summary JSON for statistics and validation

### Error Handling
All notebooks include try-except blocks to:
- Continue with partial results if components fail
- Print informative error messages
- Allow graceful degradation in smoke test mode

---

## Execution Order

1. **Start with Notebook 1** to validate document parsing and understand data quality
2. **Run Notebook 2** to compare chunking strategies and select preferred approach
3. **Run Notebook 3** to build indices and validate retrieval quality
4. **Run Notebook 4** to demonstrate full RAG pipeline on example queries

---

## Module Imports Used

Each notebook imports relevant modules from the `rag` package:

```
rag.config          # Configuration, paths, ESRS standards
rag.parser          # Document parsing
rag.chunker         # Text chunking strategies
rag.pipeline        # Index building orchestration
rag.retriever       # Retrieval engines (dense, sparse, hybrid)
rag.embedder        # Text embedding (OpenAI)
rag.esrs_context    # ESRS requirement extraction
rag.generator       # LLM answer generation
```

---

## Expected Runtimes

**Smoke Test Mode (2 companies):**
- Notebook 1: ~30 seconds
- Notebook 2: ~1-2 minutes (with caching)
- Notebook 3: ~2-5 minutes (embedding API calls)
- Notebook 4: ~1-3 minutes

**Full Mode (17 companies):**
- Notebook 1: ~2 minutes
- Notebook 2: ~5-10 minutes
- Notebook 3: ~10-30 minutes (API limits apply)
- Notebook 4: ~5-10 minutes

---

## Next Steps

After running the notebooks:
1. Review audit summary for data quality insights
2. Analyze chunking strategy comparison results
3. Validate retrieval quality from index statistics
4. Evaluate answer quality and citation accuracy
5. Use results to inform thesis research questions
