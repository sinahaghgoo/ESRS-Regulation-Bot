# ESRS RAG — Retrieval-Augmented Generation for Corporate Sustainability Reports

**Master's Thesis** — Sina Haghgoo

**Optimizing RAG Pipelines for Analyzing Corporate Sustainability Reports under the European Sustainability Reporting Standards (ESRS)**

---

## Project Overview

This repository contains the implementation of a Master's thesis that develops and evaluates a **domain-adapted Retrieval-Augmented Generation (RAG) system** for analyzing corporate sustainability reports. The system answers ESRS-related questions in a transparent, reproducible, and auditable manner.

The project pursues two complementary goals. First, it provides an **experimental research pipeline** that benchmarks different RAG configurations to identify optimal strategies for ESRS-aligned disclosure analysis. Second, it delivers a **deployable analysis tool** (Streamlit application) that implements the best-performing pipeline configuration for interactive use.

The system integrates retrieval from company sustainability reports, ESRS regulatory context injection, LLM-based answer generation, and evaluation against a Golden Standard of expert-verified answers.

---

## Thesis Objective

The central objective of this thesis is to design, implement, and systematically evaluate a RAG pipeline that can analyze long-form corporate sustainability reports and assess their compliance with ESRS disclosure requirements.

Specifically, the system is designed to:

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
To what extent does full-standard ESRS context injection improve evidence-grounded compliance classification and compliance rationale quality, compared with the same RAG pipeline without injected ESRS context?

RQ2 isolates the contribution of full-standard ESRS context injection by comparing the best RQ1 pipeline with and without injected ESRS context while holding retrieval fixed. Evaluation uses deterministic compliance-level metrics, LLM-judge rationale evaluation, and joint compliance-assessment metrics. The primary metric is Joint Lenient Match Rate (compliance level correct + rationale at least partially aligned).

**RQ3 — Generation Quality**
How accurately does the best-performing pipeline evaluate ESRS compliance, and where does the system fail?

RQ3 evaluates the final system across deterministic metrics and rubric-based judging in order to characterise strengths, weaknesses, and recurring failure modes across ESRS standards.

---

## Implementation Status (Current Codebase)

The current codebase contains a full **RQ1 retrieval pipeline** and a reusable **RQ2 generation/evaluation layer** under modular folders in `rag/`:

- `rag/config/` — Environment/config constants and 54-config grid builder (3 chunking strategies × 3 chunk sizes × 6 retrieval methods).
- `rag/data/` — Report and golden-standard loading, text normalisation, sentence splitting, and offset mapping utilities.
- `rag/chunking/` — Fixed-size, structure-aware, and semantic chunking strategies with configurable overlap and token ceiling. Semantic chunking uses adjacent-sentence similarity with greedy group merging to produce chunks that approach the target token ceiling while preserving topical coherence.
- `rag/indexing/` — OpenAI embedding service with deterministic caching and Qdrant local vector indexing.
- `rag/retrieval/` — Dense (vector search), BM25 sparse (Okapi, k1=1.5, b=0.75 with global IDF), hybrid RRF (k=60), and cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`). All six method variants are evaluated.
- `rag/generation/` — ESRS standard flattening, full-standard ESRS context serialization, RQ2 answer generation, and prompt builders for the paired ablation conditions.
- `rag/evaluation/` — RQ1 relevance matching plus RQ2 classification and text-overlap metrics (macro-F1, balanced accuracy, exact accuracy, weighted kappa, token-F1, evidence-support ratio).
- `rag/pipeline/` — RQ1 orchestration with incremental result persistence, plus an RQ2 runner that reuses the RQ1 winner and persists paired generation outputs incrementally.
- `rag/types.py` — Shared dataclasses forming the pipeline data model (`ReportDocument` → `ChunkRecord` → `RetrievalHit` → `EvalSample` → `ConfigResult`).

Three experiment notebooks are included:

- `notebooks/05_rq1_chunking_evaluation.ipynb` — Runs the full 54-config evaluation, computes main effects, interaction heatmaps, and winner selection.
- `notebooks/06_rq2_esrs_context_ablation.ipynb` — Runs or resumes the paired ESRS-context ablation, reports classification metrics, and analyzes per-standard deltas and qualitative failures.
- `notebooks/09_rq1_chunking_retrieval_walkthrough.ipynb` — Qualitative single-report walkthrough: inspect chunk output, compare retrieval methods, visualize ranking shifts.

Both notebooks are structured with markdown sections explaining each step.

**Evaluation coverage note:** The Golden Standard contains 12 questions × 17 companies = 204 pairs. All 12 ESRS standards have annotated evidence passages, giving 204 retrieval-evaluable samples. Question-company pairs whose evidence passages have empty text are excluded from metric computation (currently none).

---

## System Architecture

The RAG pipeline consists of five stages, from document ingestion through to answer generation.

### 1. Data Ingestion and Preprocessing

Source documents (PDF sustainability reports) are processed using **Azure Document Intelligence** with the `prebuilt-layout` model, which extracts text while preserving the structural layout of the document (headings, tables, paragraphs). Charts and figures can be enriched through **multimodal processing with a vision-capable GPT-5 model (configurable via `LLM_MODEL`)**, which generates textual descriptions that replace raw image content with semantic context. A regex-based cleaning step removes artifacts such as repeated page numbers, headers, and footers. The resulting output is structured Markdown that retains the original document hierarchy.

### 2. Chunking

The preprocessing stage produces structured Markdown for each report, which is then segmented into chunks suitable for embedding and retrieval. The pipeline supports multiple chunking strategies that are compared in the evaluation:

- **Fixed-size chunking**: Segments the text into chunks of a predefined token length (256, 512, or 1024 tokens) with 12.5% proportional overlap (32, 64, or 128 tokens), without regard for document structure. Serves as the baseline.
- **Structure-aware chunking**: Leverages the H1–H3 heading hierarchy and section boundaries extracted during preprocessing. Sections smaller than the target ceiling are kept intact; oversized sections are split with the same token-window approach as fixed-size chunking, preserving the heading hierarchy as metadata.
- **Semantic chunking**: Uses `sentence-transformers` (`all-MiniLM-L6-v2`) to compute cosine similarity between adjacent sentences. Splits where similarity drops below 0.5 to detect topic boundaries, then **greedily merges adjacent groups** up to the target token ceiling. This merging step ensures chunks are both topically coherent and close to the target size, avoiding the excessive fragmentation that raw sentence-level boundaries would produce.

These strategies are evaluated jointly with chunk size and retrieval method in RQ1.

### 3. Embedding and Indexing

Text chunks are embedded using OpenAI's `text-embedding-3-large` model (3072 dimensions). The resulting vectors are stored in **Qdrant**, an open-source vector database configured with local persistence for reproducibility. Each vector entry is stored alongside rich metadata including company name, reporting year, source file, header hierarchy, and associated ESRS identifiers, enabling fine-grained metadata filtering during retrieval. Embeddings are cached deterministically to `data/artifacts/embedding_cache.json` to avoid redundant API calls across re-runs.

### 4. Retrieval (Hybrid Fusion)

The retrieval stage combines two complementary search strategies:

- **Dense retrieval**: Semantic vector search over the Qdrant index, matching user queries to report chunks based on embedding similarity.
- **Sparse retrieval (BM25)**: Keyword matching using Okapi BM25 (k1=1.5, b=0.75), which captures exact term overlap — particularly important for specific ESRS terminology and regulatory references. IDF statistics are computed globally across all companies to provide stable vocabulary weights.

RQ1 compares six retrieval modes: dense, dense + rerank, sparse, sparse + rerank, hybrid fusion via **Reciprocal Rank Fusion (RRF)** with k=60 (Cormack et al., 2009), and hybrid + rerank. Cross-encoder reranking uses `cross-encoder/ms-marco-MiniLM-L-6-v2` (22 M params) with a 3x candidate pool for reranking depth. Rerank scores are cached to disk for reproducibility.

### 5. Generation and Domain Adaptation

The generation stage constructs the final answer through a multi-step process:

- **ESRS context injection**: A dedicated module serializes the full text of the ESRS standard mapped to the evaluation question and injects it into the generation prompt.
- **Answer generation**: The retrieved report passages and the injected ESRS standard are provided to the LLM as context for generating a structured answer.
- **Anti-hallucination constraints**: System prompts explicitly instruct the model to ground its answers in the retrieved context and to refrain from introducing information not present in the source material.
- **Compliance assessment**: The generation step requests a fulfillment level (Low / Medium / High), rationale, and disclosure gaps grounded in retrieved passages and the injected ESRS standard.
- **Cross-company comparison**: When multiple companies are selected, specialized generation logic produces a comparative analysis highlighting relative strengths, weaknesses, and differences in ESRS alignment.

---

## Evaluation Framework

### Relevance Matching (RQ1)

A retrieved chunk is deemed relevant to a gold evidence passage when the textual overlap exceeds a configurable threshold (default 50%) of the gold passage length. The matching uses a two-tier approach:

1. **Primary (span-based)**: Both gold passage and chunk are located as character spans inside the normalised full-report text. Overlap is computed positionally. This is exact and handles partial coverage from chunk boundary effects.
2. **Fallback (token-level)**: When exact substring matching fails (e.g., due to formatting differences between the golden standard and the preprocessed report), a multiplicity-aware bag-of-words overlap is used. Gold passages shorter than 20 characters are skipped to avoid false positives.

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

RQ2 is evaluated using a combination of deterministic compliance-level metrics (macro-F1, balanced accuracy, exact accuracy, weighted Cohen's kappa, off-by-one accuracy), LLM-judge rationale evaluation (G-Eval-style verdicts and scores using Claude Sonnet 4.5 as judge), and joint compliance-assessment metrics (joint strict match, joint lenient match, joint assessment score). The primary metric is Joint Lenient Match Rate. RQ3 extends this with richer qualitative judging and failure analysis.

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
- **Vision language model**: Configurable via `LLM_MODEL` (used for figure descriptions during preprocessing)

### Vector Database

- **Qdrant** — open-source vector search engine, configured with local persistence. Used for storing and retrieving embeddings with metadata-based filtering.

---

## Notebooks

| Notebook | Purpose |
|---|---|
| `05_rq1_chunking_evaluation.ipynb` | Run and analyze the full 54-config RQ1 evaluation: main effects, interaction heatmaps, winner selection, per-standard breakdown. |
| `06_rq2_esrs_context_ablation.ipynb` | Run or resume the paired RQ2 ESRS-context experiment, inspect aggregate metrics, confusion matrices, per-standard deltas, and qualitative examples. |
| `09_rq1_chunking_retrieval_walkthrough.ipynb` | Qualitative single-report walkthrough: inspect chunk output, compare retrieval methods, visualize ranking shifts. |
| `10_rq2_esrs_context_ablation.ipynb` | RQ2 redesigned evaluation with deterministic compliance metrics, LLM-judge rationale evaluation, and joint compliance-assessment metrics. |

Additional notebooks for RQ3 and final pipeline selection remain planned.

---

## Streamlit Application

The Streamlit frontend provides an interactive interface for analyzing sustainability reports using the best-performing RAG pipeline configuration identified during evaluation.

**Features:**

- Select one or multiple companies from the available reports.
- Select the relevant ESRS chapter to scope the analysis.
- Submit a natural-language question about sustainability disclosures.
- View the generated answer alongside supporting report passages, applicable ESRS disclosure requirements, and the compliance fulfillment assessment (Low / Medium / High).
- View the concrete pages from where the info is from.

The application uses Qdrant for persistent per-company embeddings with metadata-based filtering.

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
jupyter notebook notebooks/05_rq1_chunking_evaluation.ipynb
python run_evaluation.py
# Optional smoke test
python run_evaluation.py --dry-run
```

Outputs are saved to `data/artifacts/evaluation_results/rq1/` and the best RQ1 config is written to `config/best_pipeline.yaml`.

`notebooks/09_rq1_chunking_retrieval_walkthrough.ipynb` provides a detailed single-report chunking/retrieval walkthrough.

Run RQ2 evaluation:

```bash
jupyter notebook notebooks/06_rq2_esrs_context_ablation.ipynb
python run_rq2_evaluation.py
# Optional smoke test
python run_rq2_evaluation.py --dry-run
```

Outputs are saved to `data/artifacts/evaluation_results/rq2/`, including the ESRS context cache, per-condition responses, sample-level metrics, and the aggregate summary JSON.

### Step 3: Clean Up for Re-Run (after code changes)

```bash
# Delete cached results only (preserves embeddings and Qdrant):
python cleanup_rq1_cache.py

# Also rebuild semantic chunking Qdrant collections:
python cleanup_rq1_cache.py --include-qdrant-semantic

# Full rebuild (deletes all Qdrant collections):
python cleanup_rq1_cache.py --include-all-qdrant
```

### Step 4: Run Tests

```bash
pytest -q tests/test_evaluation_metrics.py tests/test_chunking.py tests/test_retrieval_methods.py tests/test_rq2_pipeline.py
```

### Step 5: Interactive Analysis

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
│   ├── Reports_md_clean/           # Preprocessed reports (cleaned Markdown)
│   ├── ESRS_Standards_JSON/        # ESRS regulation files (JSON)
│   └── artifacts/                  # Generated indices, chunks, embeddings, eval results
│       ├── evaluation_results/rq1/ # Per-config results, chunk stats, winner JSON/CSV
│       ├── evaluation_results/rq2/ # ESRS context cache, responses, sample metrics, summary
│       ├── qdrant_storage/         # Local Qdrant persistence
│       └── embedding_cache.json    # Deterministic embedding cache
├── golden_standard/
│   └── golden_standard.json        # 12 questions × 17 companies with evidence passages
├── notebooks/
│   ├── 05_rq1_chunking_evaluation.ipynb
│   ├── 06_rq2_esrs_context_ablation.ipynb
│   └── 09_rq1_chunking_retrieval_walkthrough.ipynb
├── rag/                            # Core RAG pipeline modules
│   ├── config/
│   │   └── settings.py             # Paths, constants, experiment grid builder
│   ├── data/
│   │   ├── loaders.py              # Report and golden-standard loading
│   │   └── text_utils.py           # Normalisation, sentence splitting, page offsets
│   ├── generation/
│   │   ├── esrs_context.py         # ESRS flattening and full-standard context serialization
│   │   ├── generator.py            # Structured answer generation for RQ2
│   │   └── prompts.py              # Prompt builders for answer generation
│   ├── chunking/
│   │   └── strategies.py           # Fixed, structure-aware, and semantic chunking
│   ├── indexing/
│   │   ├── embedder.py             # OpenAI embedding with deterministic caching
│   │   └── qdrant_index.py         # Qdrant local vector index
│   ├── retrieval/
│   │   └── methods.py              # Dense, BM25, RRF fusion, cross-encoder rerank
│   ├── evaluation/
│   │   ├── metrics.py              # Relevance matching, metric computation
│   │   ├── rq2_metrics.py          # RQ2 classification and text-overlap metrics
│   │   ├── rq2_judge.py            # RQ2 LLM-judge rationale evaluation (G-Eval style)
│   │   └── statistical_tests.py    # Bootstrap CIs, permutation tests, effect sizes
│   ├── pipeline/
│   │   ├── rq1_runner.py           # RQ1 orchestration and result persistence
│   │   └── rq2_runner.py           # RQ2 paired-generation orchestration
│   └── types.py                    # Shared dataclasses (pipeline data model)
├── tests/
│   ├── test_evaluation_metrics.py  # Unit tests for evaluation logic (28 tests)
│   ├── test_chunking.py            # Unit tests for chunking strategies (14 tests)
│   ├── test_retrieval_methods.py   # Unit tests for retrieval methods
│   └── test_rq2_pipeline.py        # Unit tests for RQ2 generation and runner logic
├── streamlit_app/
│   └── app.py                      # Streamlit frontend (loads best_pipeline.yaml)
├── cleanup_rq1_cache.py            # Script to clear cached results before re-run
├── run_preprocessing.py            # CLI: build chunks, embeddings, Qdrant indices
├── run_evaluation.py               # CLI: run full RQ1 evaluation matrix
├── run_rq2_evaluation.py           # CLI: run the paired RQ2 ESRS-context ablation
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
- Unit tests cover evaluation metrics, chunking strategies, and statistical tests.
- Cached results can be cleared with `cleanup_rq1_cache.py` for fresh re-runs after code changes.

---
