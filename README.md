# ESRS RAG — Retrieval-Augmented Generation for Corporate Sustainability Reports

**Master's Thesis** — Sina Haghgoo

**Optimizing RAG Pipelines for Analyzing Corporate Sustainability Reports under the European Sustainability Reporting Standards (ESRS)**

---

## Project Overview

This repository contains the implementation of a Master's thesis that develops and evaluates a **domain-adapted Retrieval-Augmented Generation (RAG) system** for analyzing corporate sustainability reports. The system answers ESRS-related questions in a transparent, reproducible, and auditable manner.

The project pursues two complementary goals. First, it provides an **experimental research pipeline** that benchmarks different RAG configurations to identify optimal strategies for ESRS-aligned disclosure analysis. Second, it delivers a **deployable analysis tool** (Streamlit application) that implements the best-performing pipeline configuration for interactive use.

The system integrates retrieval from company sustainability reports, ESRS regulatory context extraction, LLM-based answer generation, and evaluation against a Golden Standard of expert-verified answers.

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

The experimental evaluation is structured around three research questions, each targeting a distinct stage of the RAG pipeline.

**RQ1 — Impact of Chunking Strategy on Retrieval Quality**
To what extent do domain-specific chunking strategies — such as structure-aware segmentation based on section headers and hierarchical document layout — improve the retrieval of relevant ESRS disclosure passages compared to fixed-size chunking approaches?

This question investigates whether preserving the semantic structure of sustainability reports during chunking (e.g., respecting section boundaries and heading hierarchies) leads to measurably better retrieval performance than treating reports as unstructured text.

**RQ2 — Effectiveness of Hybrid Search and Reranking**
How does hybrid retrieval — combining dense vector search with sparse keyword matching (BM25) and Reciprocal Rank Fusion — compare to standalone dense or sparse retrieval in terms of precision, recall, and ranking quality for ESRS-specific queries?

This question examines whether the technical and regulatory vocabulary found in ESRS standards and sustainability reports requires keyword-level matching to complement semantic similarity search, and whether reranking via score fusion yields meaningful improvements.

**RQ3 — Downstream Impact of Retrieval Quality on Answer Generation**
To what extent does improved retrieval quality translate into higher-quality generated answers, and does the language model compensate for suboptimal retrieval through its parametric knowledge?

This question assesses the end-to-end relationship between retrieval performance and generation quality, specifically examining whether gains in retrieval metrics (Recall@k, Precision@k) correspond to measurable improvements in answer faithfulness, completeness, and hallucination reduction.

---

## System Architecture

The RAG pipeline consists of five stages, from document ingestion through to answer generation.

### 1. Data Ingestion and Preprocessing

Source documents (PDF sustainability reports) are processed using **Azure Document Intelligence** with the `prebuilt-layout` model, which extracts text while preserving the structural layout of the document (headings, tables, paragraphs). Charts and figures can be enriched through **multimodal processing with a vision-capable GPT-5 model (configurable via `LLM_MODEL`)**, which generates textual descriptions that replace raw image content with semantic context. A regex-based cleaning step removes artifacts such as repeated page numbers, headers, and footers. The resulting output is structured Markdown that retains the original document hierarchy.

### 2. Chunking

The preprocessing stage produces structured Markdown for each report, which is then segmented into chunks suitable for embedding and retrieval. The pipeline supports multiple chunking strategies that are compared in the evaluation:

- **Fixed-size chunking**: segments the text into chunks of a predefined token length with configurable overlap, without regard for document structure.
- **Structure-aware chunking**: leverages the heading hierarchy and section boundaries extracted during preprocessing to produce chunks that respect the logical structure of the report.

Each chunking strategy is evaluated independently to answer RQ1.

### 3. Embedding and Indexing

Text chunks are embedded using OpenAI's `text-embedding-3-large` model (3072 dimensions). The resulting vectors are stored in **Qdrant**, an open-source vector database configured with local persistence for reproducibility. Each vector entry is stored alongside rich metadata including company name, reporting year, source file, header hierarchy, and associated ESRS identifiers, enabling fine-grained metadata filtering during retrieval.

### 4. Retrieval (Hybrid Fusion)

The retrieval stage combines two complementary search strategies:

- **Dense retrieval**: semantic vector search over the Qdrant index, matching user queries to report chunks based on embedding similarity.
- **Sparse retrieval**: BM25 keyword matching, which captures exact term overlap — particularly important for specific ESRS terminology and regulatory references.

Results from both strategies are merged using **Reciprocal Rank Fusion (RRF)** with a constant k=60, producing a unified ranked list. Metadata filters allow retrieval to be scoped to specific companies.

### 5. Generation and Domain Adaptation

The generation stage constructs the final answer through a multi-step process:

- **ESRS context extraction**: a dedicated module queries the LLM with the full text of the relevant ESRS standard to extract only those disclosure requirements that pertain to the user's specific question.
- **Answer generation**: the retrieved report passages and the extracted ESRS requirements are provided to the LLM as context for generating a structured answer.
- **Anti-hallucination constraints**: system prompts explicitly instruct the model to ground its answers in the retrieved context and to refrain from introducing information not present in the source material.
- **Compliance assessment**: the generation step requests a fulfillment level (Low / Medium / High), rationale, and disclosure gaps grounded in retrieved passages and extracted ESRS requirements.
- **Cross-company comparison**: when multiple companies are selected, specialized generation logic produces a comparative analysis highlighting relative strengths, weaknesses, and differences in ESRS alignment.

---

## Output Structure

For each query, the system produces a structured response consisting of three parts:

**1. Direct Answer** — A clear and concise response to the user's question, grounded in the content of the selected sustainability reports.

**2. ESRS Compliance Evaluation** — An assessment of how well the company's disclosures align with the relevant ESRS requirements, including a compliance level (Low / Medium / High), an explanation of which aspects are well covered, and an identification of gaps or missing disclosures.

**3. Company Comparison** (when applicable) — A comparative analysis across selected companies, highlighting differences in how each company addresses the relevant ESRS disclosure requirements and the reasons for stronger or weaker alignment.

---

## Data Sources

### Sustainability Reports

The dataset comprises 17 corporate sustainability reports (reporting year 2024) from European energy and chemical sector companies: Aker BP, BASF, Eni, Equinor, Galp, Helleniq, MOL, Motoroil, Neste, OMV, Petrom, Repsol, SBM Offshore, Shell, Solvay, TotalEnergies, and Vopak. All reports are aligned with the ESRS framework. Source PDFs are stored in `data/Reports_Pdf/`, and their preprocessed Markdown representations in `data/Reports_md/`.

### ESRS Regulation Dataset

The folder `data/ESRS_Standards_JSON/` contains 12 structured JSON files covering the full scope of the ESRS framework: ESRS 1 (General Requirements), ESRS 2 (General Disclosures), ESRS E1–E5 (Environment), ESRS S1–S4 (Social), and ESRS G1 (Governance). Each file includes the disclosure requirements, objectives, and application requirements of the corresponding standard.

### Golden Standard

The Golden Standard is a manually curated evaluation dataset containing ESRS-related questions, the required ESRS disclosures for each question, expert-verified ground-truth answers, page references within the source reports, and expected disclosure fulfillment levels per company. It serves as the reference benchmark against which retrieval and generation quality are measured.

### Models

- **Embedding model**: `text-embedding-3-large` (OpenAI)
- **Language model**: `gpt-5.2` (OpenAI)
- **Vision language model**: configurable via `LLM_MODEL` (OpenAI, used for figure descriptions during preprocessing; default `gpt-5.2`)

### Vector Database

- **Qdrant** — open-source vector search engine, configured with local persistence. Used for storing and retrieving embeddings with metadata-based filtering.

---

## Evaluation Framework

The evaluation benchmarks different RAG pipeline configurations against the Golden Standard across four metric categories. The framework itself works per pipeline configuration. Each configuration is designed to isolate a specific variable that corresponds to one of the research questions. In practice, this means multiple pipeline configurations run through the same evaluation harness (Golden Standard comparison, retrieval metrics, answer quality metrics, etc.), and then the results by research question are compared.

### Retrieval Metrics

Recall@k, Precision@k, MRR@k (Mean Reciprocal Rank), and coverage ratio. These metrics quantify how effectively the retrieval stage surfaces the correct passages from the sustainability reports.

### Answer Quality Metrics

Token-level F1 score, BERTScore, LLM-as-a-judge scoring, and citation correctness. These metrics assess the factual accuracy, semantic similarity, and source attribution quality of the generated answers relative to the Golden Standard.

### Hallucination Metrics

Faithfulness score, supported claim ratio, and hallucination rate. These metrics evaluate whether the generated answers remain grounded in the retrieved context or introduce unsupported claims.

### Efficiency Metrics

Latency, prompt size, token cost, and number of retrieved chunks. These metrics capture the computational and financial cost of each pipeline configuration.

---

## Notebooks

The experimental workflow is organized into eight Jupyter notebooks, each corresponding to a specific pipeline stage or research question.

### Exploration and Pipeline Development (01–04)

| Notebook | Purpose |
|---|---|
| `01_report_structure_and_metadata_audit.ipynb` | Load Markdown reports, inspect page markers, heading hierarchy, and metadata completeness. |
| `02_chunking_experiments.ipynb` | Compare fixed-size vs. structure-aware chunking: distributions, overlap, boundary examples. |
| `03_embeddings_and_qdrant_indexing.ipynb` | Embed chunks, build/verify Qdrant collections, inspect nearest-neighbour quality. |
| `04_retrieval_esrs_context_and_generation.ipynb` | End-to-end single-query walkthrough: retrieve, extract ESRS context, generate answer. |

### Evaluation and Selection (05–08)

| Notebook | Purpose |
|---|---|
| `05_rq1_chunking_evaluation.ipynb` | RQ1 — Evaluate fixed-size vs. structure-aware chunking on retrieval metrics. |
| `06_rq2_retrieval_evaluation.ipynb` | RQ2 — Compare dense, sparse, hybrid, and hybrid+reranker retrieval strategies. |
| `07_rq3_generation_evaluation_and_llm_judge.ipynb` | RQ3 — Evaluate answer generation quality including LLM-as-a-judge (Approach B). |
| `08_best_pipeline_frontend_validation.ipynb` | Select best pipeline via composite scoring, save manifest, validate frontend readiness. |

Notebooks 05–07 save per-RQ evaluation results to `data/artifacts/evaluation_results/`. Notebook 08 calls `select_best_pipeline()` and `save_best_pipeline_manifest()` to persist the winning configuration to `config/best_pipeline.yaml`, which the Streamlit app reads on startup.

---

## Streamlit Application

The Streamlit frontend provides an interactive interface for analyzing sustainability reports using the best-performing RAG pipeline configuration identified during evaluation.

**Features:**

- Select one or multiple companies from the available reports.
- Select the relevant ESRS chapter to scope the analysis.
- Submit a natural-language question about sustainability disclosures.
- View the generated answer alongside supporting report passages, applicable ESRS disclosure requirements, and the compliance fulfillment assessment (Low / Medium / High).
- View the concrete pages from where the info is from!!

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
LLM_MODEL=gpt-5.2
```

### 3. Start Qdrant (if using Docker)

```bash
docker run -p 6333:6333 qdrant/qdrant
```

Alternatively, Qdrant can be configured for local file-based persistence without Docker.

---

## Usage

### Step 1: Preprocessing and Indexing

Run notebooks 01–03 in order (or use the CLI script):

```bash
# Notebooks (recommended for first-time exploration)
jupyter notebook notebooks/01_report_structure_and_metadata_audit.ipynb
jupyter notebook notebooks/02_chunking_experiments.ipynb
jupyter notebook notebooks/03_embeddings_and_qdrant_indexing.ipynb

# Or CLI
python run_preprocessing.py
```

This loads the Markdown reports, applies chunking strategies, generates embeddings via the OpenAI API, builds Qdrant collections, and saves all artifacts to `data/artifacts/`.

### Step 2: Evaluation

Run notebooks 05–08:

```bash
jupyter notebook notebooks/05_rq1_chunking_evaluation.ipynb
jupyter notebook notebooks/06_rq2_retrieval_evaluation.ipynb
jupyter notebook notebooks/07_rq3_generation_evaluation_and_llm_judge.ipynb
jupyter notebook notebooks/08_best_pipeline_frontend_validation.ipynb

# Or CLI
python run_evaluation.py
```

Notebook 08 selects the best pipeline configuration and saves the manifest to `config/best_pipeline.yaml`.  
The CLI `python run_evaluation.py` also writes `config/best_pipeline.yaml` after scoring evaluated configurations.

### Step 3: Interactive Analysis

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
│   ├── pipeline_configs.yaml       # 8 experiment configurations (RQ1×2, RQ2×4, RQ3×2)
│   └── best_pipeline.yaml          # Auto-generated best pipeline manifest
├── data/
│   ├── Reports_Pdf/                # Source sustainability reports (PDF)
│   ├── Reports_md/                 # Preprocessed reports (Markdown)
│   ├── ESRS_Standards_JSON/        # ESRS regulation files (JSON)
│   └── artifacts/                  # Generated indices, chunks, embeddings, eval results
├── golden_standard/                # Evaluation Golden Standard (12 questions × 17 companies)
├── notebooks/                      # 8 Jupyter notebooks (01–08)
├── rag/                            # Core RAG pipeline modules
│   ├── config.py                   # Paths, API keys, experiment configs, manifest I/O
│   ├── parser.py                   # Markdown report parser
│   ├── chunker.py                  # Fixed-size and structure-aware chunking
│   ├── embedder.py                 # OpenAI embedding with file-based caching
│   ├── indexer.py                  # Qdrant vector index + numpy fallback
│   ├── retriever.py                # BM25, RRF, Cohere reranker, unified Retriever
│   ├── esrs_context.py             # ESRS JSON standards extraction with LLM + caching
│   ├── generator.py                # Answer generation with citation resolution
│   ├── evaluator.py                # Full evaluation harness (Approach A + B)
│   ├── llm_client.py               # OpenAI chat completions wrapper
│   └── pipeline.py                 # Pipeline orchestration: build_index, load_pipeline, run_query
├── streamlit_app/
│   └── app.py                      # Streamlit frontend (loads best_pipeline.yaml)
├── tests/                          # Unit tests
├── run_preprocessing.py            # CLI preprocessing script
├── run_evaluation.py               # CLI evaluation script
├── requirements.txt                # Python dependencies
├── .env                            # Environment variables (not tracked)
└── README.md
```

---

## Development Guidelines

- Modular Python-first design: core logic resides in reusable modules, not in notebooks.
- Jupyter notebooks serve as orchestration layers for experimentation and evaluation.
- Streamlit provides the user-facing frontend.
- Pipeline configurations are defined via dictionaries for easy switching between experimental conditions.
- All pipeline outputs are designed to be transparent, explainable, and auditable.

---
