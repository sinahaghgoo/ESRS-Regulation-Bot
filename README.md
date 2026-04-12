# ESRS RAG — Retrieval-Augmented Generation for Corporate Sustainability Reports

**Master's Thesis** — Sina Haghgoo

**Optimizing a Retrieval-Augmented Generation (RAG) Pipeline for a Regulatory Compliance Chatbot: A Case Study on Corporate Adherence to the European Sustainability Reporting Standards (ESRS)**

---

## Project Overview

This repository contains the implementation of a Master's thesis that develops and evaluates a **domain-adapted Retrieval-Augmented Generation (RAG) system** for analyzing corporate sustainability reports. The system answers ESRS-related questions in a transparent, reproducible, and auditable manner.

The project pursues two complementary goals. First, it provides an **experimental research pipeline** that benchmarks different RAG configurations to identify optimal strategies for ESRS-aligned disclosure analysis. Second, it delivers a **deployable analysis tool** (Streamlit application) that implements the best-performing pipeline configuration for interactive use.

The system integrates retrieval from company sustainability reports, ESRS regulatory context injection, LLM-based answer generation, and evaluation against a Golden Standard of expert-verified answers.

---

## Thesis Objective

The central objective of this thesis is to design, implement, and systematically evaluate a RAG pipeline that can analyze long-form corporate sustainability reports and assess their compliance with ESRS disclosure requirements.

Specifically, the pipeline is designed to:

1. **Retrieve** semantically relevant passages from sustainability reports and identify the applicable ESRS disclosure requirements for a given user query.
2. **Generate** structured answers that include a direct response, an ESRS compliance assessment (Low / Medium / High), and — when multiple companies are selected — a comparative analysis.
3. **Evaluate** retrieval and generation quality against a predefined Golden Standard, using both quantitative metrics and qualitative analysis, to determine which pipeline configuration yields the best results.
4. **Deploy** the best-performing pipeline configuration in a Streamlit-based interactive application for end-user analysis.

---

## Research Questions

The experimental evaluation is structured around three sequential research questions. RQ1 identifies the best retrieval configuration, RQ2 tests whether explicit ESRS context injection adds value on top of that configuration, and RQ3 evaluates the final end-to-end system.

**RQ1 — Pipeline Configuration**
What combination of chunking strategy, chunk size, and retrieval method achieves the best retrieval quality for ESRS disclosure queries, and do these factors interact?

RQ1 evaluates a full factorial design over chunking strategy, chunk size, and retrieval method in order to identify the globally best retrieval setup rather than optimising each component in isolation.

**RQ2 — ESRS Domain Adaptation**
To what extent does disclosure-filtered ESRS context injection improve evidence-grounded compliance classification and compliance rationale quality, compared with the same RAG pipeline without injected ESRS context?

RQ2 isolates the contribution of disclosure-filtered ESRS context injection by comparing the best RQ1 pipeline with and without injected ESRS context while holding retrieval fixed. Evaluation uses deterministic compliance-level metrics (macro-F1, balanced accuracy, weighted kappa), LLM-judge rationale evaluation (G-Eval with anti-anchoring design, Claude Sonnet 4.5 as judge), and joint compliance-assessment metrics. The primary composite metric is Assessment Quality Score (AQS). The Wilcoxon signed-rank test on question-level paired deltas (n=12 clusters) in order to find statistically significant effect found.

**RQ3 — Generation Quality**
How accurately does the best-performing pipeline evaluate ESRS compliance, and where does the system fail?

RQ3 evaluates the final system using citation page metrics, a five-dimensional LLM-judge rubric (Claude Sonnet 4.5), and targeted failure analysis across all 204 samples.

## Codebase Structure

The codebase contains the complete **RQ1 retrieval pipeline**, the complete **RQ2 generation/evaluation layer**, and the complete **RQ3 end-to-end generation quality evaluation** under modular folders in `rag/`:

- `rag/config/` — Environment/config constants and 54-config grid builder (3 chunking strategies × 3 chunk sizes × 6 retrieval methods).
- `rag/data/` — Report and golden-standard loading, text normalisation, sentence splitting, and offset mapping utilities.
- `rag/chunking/` — Fixed-size, structure-aware, and semantic chunking strategies with configurable overlap and token ceiling. Semantic chunking uses adjacent-sentence similarity with greedy group merging (including orphan fragment absorption) to produce chunks that approach the target token ceiling while preserving topical coherence.
- `rag/indexing/` — OpenAI embedding service with deterministic caching, Qdrant local vector indexing, and a lightweight in-memory NumPy-backed dense index for RQ2 evaluation.
- `rag/retrieval/` — Dense (vector search with ESRS-specific query expansion), BM25 sparse (Okapi, k1=1.5, b=0.75 with global IDF), hybrid RRF (k=60), and cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`) with section-context-aware reranker input. All six method variants are evaluated. Reranking uses a 5x candidate pool; hybrid fusion uses a 3x candidate pool.
- `rag/generation/` — ESRS standard flattening, disclosure-filtered ESRS context serialization (filters to disclosure-relevant subtrees, falls back to full standard), RQ2 answer generation with structured JSON output (via OpenAI Responses API with JSON Schema enforcement), and prompt builders for the paired ablation conditions with explicit ESRS-context-mode labelling and anti-hallucination constraints.
- `rag/evaluation/` — RQ1 relevance matching (span-based with multi-anchor approximate matching fallback), RQ2 classification and text-overlap metrics (macro-F1, balanced accuracy, exact accuracy, weighted kappa, token-F1, off-by-one accuracy), LLM-judge rationale evaluation (G-Eval v2, Claude Sonnet 4.5 via Anthropic API, with anti-anchoring design and version-stamped caching), paired statistical tests (Wilcoxon signed-rank with exact p-values, sign test, cluster bootstrap, Cohen's d, rank-biserial correlation, McNemar-style discordant test, post-hoc power analysis, prediction bias diagnosis), RQ3 citation page metrics (precision, recall, F1 with range-based ±1 page tolerance), and RQ3 multi-dimensional LLM-judge rubric (factual correctness, completeness, grounding faithfulness, citation usefulness, coherence) with weakness classification and retrieval-generation gap analysis.
- `rag/pipeline/` — RQ1 orchestration with incremental result persistence, plus an RQ2 runner that reuses the RQ1 winner retriever (via lightweight in-memory dense index) and persists paired generation outputs incrementally with thread-parallel condition generation.
- `rag/preprocessing/` — PDF-to-Markdown extraction via Azure Document Intelligence and report structure processing.
- `rag/types.py` — Shared dataclasses forming the pipeline data model (`ReportDocument` → `ChunkRecord` → `RetrievalHit` → `EvalSample` → `ConfigResult` for RQ1; `GenerationSample` → `GeneratedAnswer` → `RQ2SampleResult` → `RQ2Summary` for RQ2; RQ3 reuses the generation and evaluation types).

Six experiment notebooks are included:

- `notebooks/01_rq1_chunking_evaluation.ipynb` — Runs the full 54-config RQ1 evaluation, computes main effects, interaction heatmaps, winner selection, per-standard breakdown, and statistical significance tests.
- `notebooks/02_rq1_chunking_retrieval_walkthrough.ipynb` — Qualitative single-report walkthrough (Eni / ESRS 2): inspect chunk output, compare retrieval stages (dense → sparse → RRF → rerank), visualize ranking shifts via bump chart.
- `notebooks/03_rq2_context_injection_walkthrough.ipynb` — Qualitative walkthrough of the RQ2 ESRS context injection: inspect prompt construction, ESRS context filtering, and side-by-side generation comparisons across 12 illustrative samples (4 questions × 3 companies).
- `notebooks/04_rq2_esrs_context_ablation.ipynb` — Runs the paired RQ2 ESRS-context ablation across all 204 samples, reports AQS, classification metrics, LLM-judge rationale evaluation (G-Eval v2, Claude Sonnet 4.5), confusion matrices, per-standard deltas, and Wilcoxon signed-rank test.
- `notebooks/05_rq3_generation_quality_evaluation.ipynb` — Evaluates end-to-end generation quality of the final system: citation page metrics, five-dimensional LLM-judge rubric scoring, weakness classification, per-standard breakdown, retrieval-generation gap analysis, and failure-type heatmap.
- `notebooks/06_streamlit_frontend_walkthrough.ipynb` — Structured walkthrough of the Streamlit-based frontend: demonstrates automatic pipeline selection from RQ1, ESRS context decision from RQ2, single- and multi-company analysis, cross-company comparison, and visual design.

All notebooks are structured with markdown sections explaining each step.

**Evaluation coverage note:** The Golden Standard contains 12 questions × 17 companies = 204 pairs. All 12 ESRS standards have annotated evidence passages, giving 204 retrieval-evaluable samples.

---

## System Architecture

The RAG pipeline consists of five stages, from document ingestion through to answer generation.

### 1. Data Ingestion and Preprocessing

Source documents (PDF sustainability reports) are processed using **Azure Document Intelligence** with the `prebuilt-layout` model, which extracts text while preserving the structural layout of the document (headings, tables, paragraphs). Charts and figures can be enriched through **multimodal processing with a vision-capable GPT-5 model (configurable via `LLM_MODEL`)**, which generates textual descriptions that replace raw image content with semantic context. A regex-based cleaning step removes artifacts such as repeated page numbers, headers, and footers. The resulting output is structured Markdown that retains the original document hierarchy.

### 2. Chunking

The preprocessing stage produces structured Markdown for each report, which is then segmented into chunks suitable for embedding and retrieval. The pipeline supports multiple chunking strategies that are compared in the evaluation:

- **Fixed-size chunking**: Segments the text into chunks of a predefined token length (256, 512, or 1024 tokens) with 12.5% proportional overlap (32, 64, or 128 tokens), without regard for document structure. Serves as the baseline.
- **Structure-aware chunking**: Leverages the H1–H3 heading hierarchy and section boundaries extracted during preprocessing. Sections smaller than the target ceiling are kept intact; oversized sections are split with the same token-window approach as fixed-size chunking, preserving the heading hierarchy as metadata.
- **Semantic chunking**: Uses `sentence-transformers` (`all-MiniLM-L6-v2`) to compute cosine similarity between adjacent sentences. Splits where similarity drops below 0.5 to detect topic boundaries, then **greedily merges adjacent groups** up to the target token ceiling. An **orphan fragment absorption pass** consolidates groups below 30% of the target size into adjacent groups, avoiding excessive fragmentation. This merging step ensures chunks are both topically coherent and close to the target size, avoiding the excessive fragmentation that raw sentence-level boundaries would produce.

These strategies are evaluated jointly with chunk size and retrieval method in RQ1.

### 3. Embedding and Indexing

Text chunks are embedded using OpenAI's `text-embedding-3-large` model (3072 dimensions). The resulting vectors are stored in **Qdrant**, an open-source vector database configured with local persistence for reproducibility. Each vector entry is stored alongside rich metadata including company name, reporting year, source file, header hierarchy, and associated ESRS identifiers, enabling fine-grained metadata filtering during retrieval. Embeddings are cached deterministically to `data/artifacts/embedding_cache.json` to avoid redundant API calls across re-runs. For RQ2, a lightweight **in-memory NumPy-backed dense index** is used instead of the full Qdrant store, reducing overhead for the fixed evaluation set.

### 4. Retrieval (Hybrid Fusion)

The retrieval stage combines two complementary search strategies:

- **Dense retrieval**: Semantic vector search over the Qdrant index, matching user queries to report chunks based on embedding similarity. Dense queries are expanded with ESRS-specific terminology to improve semantic coverage.
- **Sparse retrieval (BM25)**: Keyword matching using Okapi BM25 (k1=1.5, b=0.75), which captures exact term overlap — particularly important for specific ESRS terminology and regulatory references. IDF statistics are computed globally across all companies to provide stable vocabulary weights.

RQ1 compares six retrieval modes: dense, dense + rerank, sparse, sparse + rerank, hybrid fusion via **Reciprocal Rank Fusion (RRF)** with k=60 (Cormack et al., 2009), and hybrid + rerank. Cross-encoder reranking uses `cross-encoder/ms-marco-MiniLM-L-6-v2` (22 M params) with a 5x candidate pool for reranking depth and section-context-aware input. Hybrid fusion uses a 3x candidate pool. Rerank scores are cached to disk for reproducibility.

### 5. Generation and Domain Adaptation

The generation stage constructs the final answer through a multi-step process:

- **ESRS context injection**: A dedicated module extracts disclosure-filtered ESRS requirement text for each question: it loads the ESRS standard JSON mapped to the question, filters to the subtree(s) matching the target disclosure ID(s) (e.g., E1-1, S4-1), and serializes the matching paragraphs into a prompt-ready context block. If no disclosure-level match is found, it falls back to the full standard. NOTE: If ESRS-Injection is determined to be ineffective, it can be disabled by setting the `ESRS_INJECTION` environment variable to `false` (aim of RESEARCH QUESTION 2).
- **Answer generation**: The retrieved report passages and the injected ESRS standard are provided to the LLM as context for generating a structured answer. Output is enforced via JSON Schema through the OpenAI Responses API.
- **Anti-hallucination constraints**: System prompts explicitly instruct the model to ground its answers in the retrieved context and to refrain from introducing information not present in the source material. The prompt differentiates between ESRS regulatory reference (not company evidence) and retrieved company report evidence (the only source for company-disclosure claims).
- **Compliance assessment**: The generation step requests a fulfillment level (Low / Medium / High), rationale, key findings, disclosure gaps, and cited pages grounded in retrieved passages and the injected ESRS standard.
- **Cross-company comparison**: When multiple companies are selected, specialized generation logic produces a comparative analysis highlighting relative strengths, weaknesses, and differences in ESRS alignment.

---

## Evaluation Framework

### Relevance Matching (RQ1)

A retrieved chunk is deemed relevant to a gold evidence passage when the textual overlap covers the gold passage.The matching this approach:

1. **Primary (span-based)**: Both gold passage and chunk are located as character spans inside the normalised full-report text. Overlap is computed positionally. This is exact and handles partial coverage from chunk boundary effects. When exact substring matching fails, a **multi-anchor approximate matching** approach probes multiple positions within the gold passage and verifies candidates with ≥80% token overlap.

### Retrieval Metrics

| Metric | Description |
|---|---|
| Recall@5, @10, @15 | Fraction of gold evidence passages recovered in the top-k retrieved chunks |
| Precision@5, @10, @15 | Fraction of the top-k retrieved chunks that are relevant |
| F1@5, @10, @15 | Harmonic mean of Precision@k and Recall@k |
| nDCG@10 | Ranking quality that rewards placing multiple relevant chunks near the top |
| MRR@10 | Mean Reciprocal Rank: how highly the first relevant chunk is ranked |
| Budgeted Recall@4096 | Recall within a fixed 4096-token budget (normalises for chunk size) |

**Primary metric:** `Recall@15` — the downstream generation step can only use evidence that retrieval surfaces within the top-15 context window.

### Statistical Validation

- **Bootstrap confidence intervals** (percentile method, 10,000 resamples) for all metrics.
- **Paired permutation tests** (10,000 permutations, two-sided) for pairwise configuration comparisons.
- **Cohen's d effect size** for measuring practical significance of differences.
- **Standard deviations** reported for key metrics to support error bars in thesis tables.

### Answer Quality Metrics (RQ2/RQ3)

RQ2 is evaluated using a combination of deterministic compliance-level metrics (macro-F1, balanced accuracy, exact accuracy, weighted Cohen's kappa, off-by-one accuracy), LLM-judge rationale evaluation (G-Eval v2 with anti-anchoring design using Claude Sonnet 4.5 as judge, 1-5 Likert scale), and joint compliance-assessment metrics (Assessment Quality Score, Overall Assessment Rate, Conditional Rationale Quality). The primary composite metric is Assessment Quality Score (AQS). Statistical significance is assessed via the Wilcoxon signed-rank test on question-level paired deltas (n=12 clusters), with cluster bootstrap CIs (10,000 resamples), Cohen's d effect sizes, rank-biserial correlation, and post-hoc power analysis.

RQ3 evaluates generation quality using citation page metrics (precision, recall, F1 with ±1 page tolerance) and a five-dimensional LLM-judge rubric (factual correctness, completeness, grounding faithfulness, citation usefulness, coherence) scored on a 1–5 Likert scale by Claude Sonnet 4.5. Error analysis includes weakness classification, per-ESRS-standard breakdown, retrieval-generation gap analysis, and a failure-type heatmap. A 20-sample stratified manual audit subset is exported for human calibration.

---

## Output Structure

For each query, the system produces a structured response consisting of three parts:

**1. Direct Answer** — A clear and concise response to the user's question, grounded in the content of the selected sustainability reports.

**2. ESRS Compliance Evaluation** — An assessment of how well the company's disclosures align with the relevant ESRS requirements, including a compliance level (Low / Medium / High), an explanation of which aspects are well covered, and an identification of gaps or missing disclosures.

**3. Company Comparison** (when applicable) — A comparative analysis across selected companies, highlighting differences in how each company addresses the relevant ESRS disclosure requirements and the reasons for stronger or weaker alignment.

---

## Data Sources

### Sustainability Reports

The dataset comprises 17 corporate sustainability reports (reporting year 2024) from European energy and chemical sector companies: Aker BP, BASF, Eni, Equinor, Galp, Helleniq, MOL, Motoroil, Neste, OMV, Petrom, Repsol, SBM Offshore, Shell, Solvay, TotalEnergies, and Vopak. All reports are aligned with the ESRS framework. Source PDFs are stored in `data/Reports_Pdf/`, and their preprocessed Markdown representations in `data/Reports_md_clean/`.

### ESRS Regulation Dataset

The folder `data/ESRS_Standards_JSON/` contains 12 structured JSON files covering the full scope of the ESRS framework: ESRS 1 (General Requirements), ESRS 2 (General Disclosures), ESRS E1–E5 (Environment), ESRS S1–S4 (Social), and ESRS G1 (Governance). Each file includes the disclosure requirements, objectives, and application requirements of the corresponding standard.

### Golden Standard

The Golden Standard is a manually curated evaluation dataset containing ESRS-related questions, the required ESRS disclosures for each question, expert-verified ground-truth answers, page references within the source reports, and expected disclosure fulfillment levels per company. It serves as the reference benchmark against which retrieval and generation quality are measured.

### Models

- **Embedding model**: `text-embedding-3-large` (OpenAI, 3072 dimensions)
- **Language model**: `gpt-5-mini` (OpenAI)
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (22 M params, local)
- **Semantic chunking**: `sentence-transformers/all-MiniLM-L6-v2` (22 M params, local)
- **LLM Judge**: `claude-sonnet-4-5-20250929` (Anthropic, via Anthropic API)
- **Vision language model**: Configurable via `LLM_MODEL` (used for figure descriptions during preprocessing)

### Vector Database

- **Qdrant** — open-source vector search engine, configured with local persistence. Used for storing and retrieving embeddings with metadata-based filtering.

---

## Notebooks

| Notebook | Purpose |
|---|---|
| `01_rq1_chunking_evaluation.ipynb` | Run and analyze the full 54-config RQ1 evaluation: main effects, interaction heatmaps, winner selection (`fixed_1024_hybrid_rerank`, Recall@15 = 0.9616), per-standard breakdown, statistical significance tests. |
| `02_rq1_chunking_retrieval_walkthrough.ipynb` | Qualitative single-report walkthrough (Eni / ESRS 2): inspect chunk output, compare retrieval stages, visualize ranking shifts via bump chart. |
| `03_rq2_context_injection_walkthrough.ipynb` | Qualitative walkthrough of RQ2 ESRS context injection: inspect prompt construction, ESRS context filtering, and side-by-side generation comparisons (4 questions × 3 companies). |
| `04_rq2_esrs_context_ablation.ipynb` | Paired RQ2 ESRS-context ablation: AQS, classification metrics, LLM-judge evaluation, confusion matrices, per-standard deltas, Wilcoxon signed-rank test (p = 0.9688). |
| `05_rq3_generation_quality_evaluation.ipynb` | End-to-end RQ3 generation quality: citation page metrics (F1 = 0.689), five-dimensional LLM-judge rubric (mean = 3.994/5), weakness classification, retrieval-generation gap analysis, per-standard and failure-type breakdown. |
| `06_streamlit_frontend_walkthrough.ipynb` | Streamlit frontend walkthrough: automatic pipeline selection from RQ1, ESRS context decision from RQ2, single/multi-company analysis, cross-company comparison, visual design. |

---

## Streamlit Application

The Streamlit frontend provides an interactive interface for analyzing sustainability reports using the best-performing RAG pipeline configuration identified during evaluation. The application automatically loads the RQ1 winner (`fixed_1024_hybrid_rerank`) and applies the RQ2 finding (ESRS context injection disabled) without requiring manual configuration.

**Features:**

- Select one or multiple companies from the 17 available reports.
- Submit a natural-language question about sustainability disclosures.
- View the generated answer alongside supporting report passages and the compliance fulfillment assessment (Low / Medium / High).
- View the concrete pages from where the information is sourced, with expandable passage detail.
- Multi-company benchmarking with compliance overview cards, per-company tabs, and an LLM-generated cross-company comparison table.
- Research-driven sidebar displaying the active pipeline configuration (chunk strategy, retrieval method, chunk size) as read-only fields and the ESRS context injection status.

The application uses a lightweight in-memory NumPy-backed dense index for retrieval, with BM25 sparse search and cross-encoder reranking. Pipeline initialization is cached via `@st.cache_resource` for efficient page reruns.

---

## Environment Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the project root with the following entries:

```
OPENAI_API_KEY=your_openai_api_key
CLAUDE_API_KEY=your_anthropic_api_key
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=your_azure_endpoint
AZURE_DOCUMENT_INTELLIGENCE=your_azure_key
EMBEDDING_MODEL=text-embedding-3-large
LLM_MODEL=gpt-5-mini
```

### 3. Start Qdrant (required)

```bash
docker run -p 6333:6333 qdrant/qdrant
```

Qdrant can also run in local file-based persistence mode without Docker (the code uses local Qdrant storage under `data/artifacts/qdrant_storage`).

---

## Usage

### Step 1: Preprocessing and Indexing

For RQ1, run preprocessing first:

```bash
python run_preprocessing.py
# Optional smoke test
python run_preprocessing.py --dry-run
```

This loads `data/Reports_md_clean/`, builds chunks, embeds text, creates Qdrant collections, and writes artifacts to `data/artifacts/`.

### Step 2: Evaluation

Run RQ1 evaluation:

```bash
jupyter notebook notebooks/01_rq1_chunking_evaluation.ipynb
python run_evaluation.py
# Optional smoke test
python run_evaluation.py --dry-run
```

Outputs are saved to `data/artifacts/evaluation_results/rq1/` and the best RQ1 config is written to `config/best_pipeline.yaml`.

`notebooks/02_rq1_chunking_retrieval_walkthrough.ipynb` provides a detailed single-report chunking/retrieval walkthrough.

Run RQ2 evaluation:

```bash
jupyter notebook notebooks/04_rq2_esrs_context_ablation.ipynb
python run_rq2_evaluation.py
# Optional smoke test
python run_rq2_evaluation.py --dry-run
```

Outputs are saved to `data/artifacts/evaluation_results/rq2/`, including the ESRS context cache, per-condition responses, sample-level metrics, LLM-judge cache, and the aggregate summary JSON.

Run RQ3 evaluation:

```bash
jupyter notebook notebooks/05_rq3_generation_quality_evaluation.ipynb
```

Outputs are saved to `data/artifacts/evaluation_results/rq3/`, including citation metrics, per-sample judge scores, weakness classifications, per-standard breakdown, and the manual audit template.

### Step 3: Clean Up for Re-Run (after code changes)

```bash
# Delete cached results only (preserves embeddings and Qdrant):
python cleanup_rq1_cache.py

# Also rebuild semantic chunking Qdrant collections:
python cleanup_rq1_cache.py --include-qdrant-semantic

# Full rebuild (deletes all Qdrant collections):
python cleanup_rq1_cache.py --include-all-qdrant

# Alternative: cleanup specifically for re-evaluation after improvements:
python cleanup_for_reeval.py
```

### Step 4: Run Tests

```bash
pytest -q tests/
```

### Step 5: Verify Improvements (optional)

```bash
python verify_improvements.py
```

### Step 6: Interactive Analysis

Launch the Streamlit application:

```bash
streamlit run streamlit_app/app.py
```

The app loads the best pipeline manifest automatically. If no manifest exists, it falls back to the default configuration.

---

## Project Structure

```
ESRS-Regulation-Bot/
├── config/
│   └── best_pipeline.yaml          # Auto-generated best pipeline manifest
├── data/
│   ├── Reports_Pdf/                # Source sustainability reports (PDF)
│   ├── Reports_md/                 # Raw extracted Markdown reports
│   ├── Reports_md_clean/           # Preprocessed reports (cleaned Markdown)
│   ├── ESRS_Standards_JSON/        # ESRS regulation files (JSON)
│   └── artifacts/                  # Generated indices, chunks, embeddings, eval results
│       ├── evaluation_results/rq1/ # Per-config results, chunk stats, winner JSON/CSV
│       ├── evaluation_results/rq2/ # ESRS context cache, responses, sample metrics, summary
│       │   └── judge_cache/        # Version-stamped LLM-judge evaluation cache (RQ2)
│       ├── evaluation_results/rq3/ # Citation metrics, judge scores, weakness analysis, per-standard breakdown
│       │   └── judge_cache/        # Version-stamped LLM-judge evaluation cache (RQ3)
│       ├── qdrant_storage/         # Local Qdrant persistence
│       └── embedding_cache.json    # Deterministic embedding cache
├── golden_standard/
│   └── golden_standard.json        # 12 questions × 17 companies with evidence passages
├── notebooks/
│   ├── 01_rq1_chunking_evaluation.ipynb
│   ├── 02_rq1_chunking_retrieval_walkthrough.ipynb
│   ├── 03_rq2_context_injection_walkthrough.ipynb
│   ├── 04_rq2_esrs_context_ablation.ipynb
│   ├── 05_rq3_generation_quality_evaluation.ipynb
│   └── 06_streamlit_frontend_walkthrough.ipynb
├── rag/                            # Core RAG pipeline modules
│   ├── config/
│   │   └── settings.py             # Paths, constants, experiment grid builder
│   ├── data/
│   │   ├── loaders.py              # Report and golden-standard loading
│   │   └── text_utils.py           # Normalisation, sentence splitting, page offsets
│   ├── preprocessing/
│   │   ├── extract_report.py       # Report structure extraction
│   │   └── pdf_to_json.py          # PDF-to-JSON via Azure Document Intelligence
│   ├── generation/
│   │   ├── esrs_context.py         # ESRS flattening, disclosure-filtered context serialization
│   │   ├── generator.py            # Structured answer generation with JSON Schema enforcement
│   │   └── prompts.py              # Prompt builders for answer generation conditions
│   ├── chunking/
│   │   └── strategies.py           # Fixed, structure-aware, and semantic chunking
│   ├── indexing/
│   │   ├── embedder.py             # OpenAI embedding with deterministic caching
│   │   ├── qdrant_index.py         # Qdrant local vector index
│   │   └── in_memory_index.py      # Lightweight NumPy-backed dense index for RQ2
│   ├── retrieval/
│   │   └── methods.py              # Dense, BM25, RRF fusion, query expansion, cross-encoder rerank
│   ├── evaluation/
│   │   ├── metrics.py              # RQ1 relevance matching with multi-anchor span alignment
│   │   ├── rq2_metrics.py          # RQ2 classification, text-overlap, paired statistics
│   │   ├── rq2_judge.py            # RQ2 LLM-judge rationale evaluation (G-Eval v2, Claude)
│   │   ├── rq3_metrics.py          # RQ3 citation page metrics, weakness classification
│   │   ├── rq3_judge.py            # RQ3 multi-dimensional LLM-judge rubric (5 dimensions, Claude)
│   │   └── statistical_tests.py    # Bootstrap CIs, permutation tests, effect sizes
│   ├── pipeline/
│   │   ├── rq1_runner.py           # RQ1 orchestration and result persistence
│   │   └── rq2_runner.py           # RQ2 paired-generation orchestration with condition parallelism
│   └── types.py                    # Shared dataclasses (pipeline data model)
├── tests/
│   ├── test_evaluation_metrics.py  # Unit tests for RQ1 evaluation logic (36 tests)
│   ├── test_chunking.py            # Unit tests for chunking strategies (18 tests)
│   ├── test_retrieval_methods.py   # Unit tests for retrieval methods (3 tests)
│   ├── test_rq2_metrics.py         # Unit tests for RQ2 metrics, paired statistics (32 tests)
│   ├── test_rq2_context.py         # Unit tests for ESRS context extraction (3 tests)
│   └── test_rq2_judge.py           # Unit tests for LLM-judge evaluation logic (43 tests)
├── streamlit_app/
│   ├── app.py                      # Streamlit frontend (loads best_pipeline.yaml)
│   └── frontend_Images/            # Screenshots for the frontend walkthrough notebook
├── cleanup_rq1_cache.py            # Script to clear cached results before re-run
├── cleanup_for_reeval.py           # Script to prepare for re-evaluation after improvements
├── verify_improvements.py          # Verification suite for RQ1 improvements
├── run_preprocessing.py            # CLI: build chunks, embeddings, Qdrant indices
├── run_evaluation.py               # CLI: run full RQ1 evaluation matrix
├── run_rq2_evaluation.py           # CLI: run the paired RQ2 ESRS-context ablation
├── recompute_rq2_metrics.py        # Recompute RQ2 metrics from saved response JSONs (no LLM calls)
├── recompute_rq2_summary.py        # Recompute RQ2 summary from saved CSVs (no LLM calls)
├── requirements.txt
├── RESEARCH_QUESTIONS.md           # Detailed RQ1/RQ2/RQ3 experimental design
├── .env                            # Environment variables (not tracked)
└── README.md
```

---

## Development Guidelines

- Modular Python-first design: core logic resides in reusable modules, not in notebooks.
- All modules include detailed docstrings explaining both *what* each function does and *why* (for thesis readability).
- Jupyter notebooks serve as orchestration layers for experimentation and evaluation.
- Streamlit provides the user-facing frontend.
- Pipeline configurations are defined via dictionaries for easy switching between experimental conditions.
- All pipeline outputs are designed to be transparent, explainable, and auditable.
- Unit tests cover evaluation metrics, chunking strategies, retrieval methods, ESRS context extraction, RQ2 metrics, and LLM-judge evaluation logic (135 tests total).
- Cached results can be cleared with `cleanup_rq1_cache.py` or `cleanup_for_reeval.py` for fresh re-runs after code changes.

---
