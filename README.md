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

The experimental evaluation is structured around three sequential research questions. RQ1 identifies the best retrieval configuration, RQ2 tests whether explicit ESRS context injection adds value on top of that configuration, and RQ3 evaluates the final end-to-end system.

**RQ1 — Pipeline Configuration**
What combination of chunking strategy, chunk size, and retrieval method achieves the best retrieval quality for ESRS disclosure queries, and do these factors interact?

RQ1 evaluates a full factorial design over chunking strategy, chunk size, and retrieval method in order to identify the globally best retrieval setup rather than optimising each component in isolation.

**RQ2 — ESRS Domain Adaptation**
To what extent does explicitly injecting structured ESRS regulatory requirements into the prompt improve compliance assessment accuracy and answer faithfulness, compared to relying on the language model's parametric knowledge of ESRS alone?

RQ2 isolates the contribution of the ESRS context extraction module by comparing the best RQ1 pipeline with and without explicit regulatory context injection.

**RQ3 — Generation Quality**
How accurately does the best-performing pipeline evaluate ESRS compliance, and where does the system fail?

RQ3 evaluates the final system across deterministic metrics and rubric-based judging in order to characterise strengths, weaknesses, and recurring failure modes across ESRS standards.

---

## RQ1 Implementation Status (Current Codebase)

The current codebase now contains a full **RQ1-only retrieval pipeline** under modular folders in `rag/`:

- `rag/config/` - environment/config constants and 36-config grid builder
- `rag/data/` - report/golden-standard loading and normalization
- `rag/chunking/` - fixed, structure-aware, and semantic chunkers
- `rag/indexing/` - embedding service and Qdrant indexing
- `rag/retrieval/` - dense, BM25 sparse, hybrid RRF, hybrid + cross-encoder rerank (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- `rag/evaluation/` - RQ1 metrics and relevance alignment rules
- `rag/pipeline/` - orchestration for preprocessing and evaluation

Two RQ1 notebooks are included:

- `notebooks/05_rq1_chunking_evaluation.ipynb` — runs the full 36-config evaluation, computes main effects, interaction heatmaps, and winner selection.
- `notebooks/09_rq1_chunking_retrieval_walkthrough.ipynb` — qualitative walkthrough on one sample report showing chunking, retrieval, and ranking behavior.

Both notebooks are structured with markdown sections explaining each step.

**Evaluation coverage note:** The Golden Standard contains 12 questions x 17 companies = 204 pairs, but only 5 of 12 ESRS standards currently have annotated evidence passages (ESRS 1, 2, E1, S1, G1), giving 85 retrieval-evaluable samples. Pairs without evidence text are excluded from metric computation.

---

## System Architecture

The RAG pipeline consists of five stages, from document ingestion through to answer generation.

### 1. Data Ingestion and Preprocessing

Source documents (PDF sustainability reports) are processed using **Azure Document Intelligence** with the `prebuilt-layout` model, which extracts text while preserving the structural layout of the document (headings, tables, paragraphs). Charts and figures can be enriched through **multimodal processing with a vision-capable GPT-5 model (configurable via `LLM_MODEL`)**, which generates textual descriptions that replace raw image content with semantic context. A regex-based cleaning step removes artifacts such as repeated page numbers, headers, and footers. The resulting output is structured Markdown that retains the original document hierarchy.

### 2. Chunking

The preprocessing stage produces structured Markdown for each report, which is then segmented into chunks suitable for embedding and retrieval. The pipeline supports multiple chunking strategies that are compared in the evaluation:

- **Fixed-size chunking**: segments the text into chunks of a predefined token length with configurable overlap, without regard for document structure.
- **Structure-aware chunking**: leverages the heading hierarchy and section boundaries extracted during preprocessing to produce chunks that respect the logical structure of the report.
- **Semantic chunking**: identifies chunk boundaries from sentence-level semantic coherence, subject to a target token ceiling.

These strategies are evaluated jointly with chunk size and retrieval method in RQ1.

### 3. Embedding and Indexing

Text chunks are embedded using OpenAI's `text-embedding-3-large` model (3072 dimensions). The resulting vectors are stored in **Qdrant**, an open-source vector database configured with local persistence for reproducibility. Each vector entry is stored alongside rich metadata including company name, reporting year, source file, header hierarchy, and associated ESRS identifiers, enabling fine-grained metadata filtering during retrieval.

### 4. Retrieval (Hybrid Fusion)

The retrieval stage combines two complementary search strategies:

- **Dense retrieval**: semantic vector search over the Qdrant index, matching user queries to report chunks based on embedding similarity.
- **Sparse retrieval**: BM25 keyword matching, which captures exact term overlap — particularly important for specific ESRS terminology and regulatory references.

RQ1 compares six retrieval modes: dense, dense + rerank, sparse, sparse + rerank, hybrid fusion via **Reciprocal Rank Fusion (RRF)** with k=60, and hybrid retrieval followed by reranking. Each base retrieval method is tested both with and without cross-encoder reranking. Metadata filters allow retrieval to be scoped to specific companies.

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
- **Important:** Qdrant is mandatory in this project.

---

## Evaluation Framework

The evaluation benchmarks different RAG pipeline configurations against the Golden Standard across four metric categories. The framework itself works per pipeline configuration. Each configuration is designed to isolate a specific variable that corresponds to one of the research questions. In practice, this means multiple pipeline configurations run through the same evaluation harness (Golden Standard comparison, retrieval metrics, answer quality metrics, etc.), and then the results by research question are compared.

### Retrieval Metrics

Recall@k, Precision@k, nDCG@k, and MRR@k. These metrics quantify how effectively the retrieval stage surfaces the correct passages from the sustainability reports.

### Answer Quality Metrics

Token-level F1 score, BERTScore, LLM-as-a-judge scoring, and citation correctness. These metrics assess the factual accuracy, semantic similarity, and source attribution quality of the generated answers relative to the Golden Standard.

### Hallucination Metrics

Faithfulness score, supported claim ratio, and hallucination rate. These metrics evaluate whether the generated answers remain grounded in the retrieved context or introduce unsupported claims.

### Efficiency Metrics

Latency, prompt size, token cost, and number of retrieved chunks. These metrics capture the computational and financial cost of each pipeline configuration.

---

## Notebooks

The current codebase includes two RQ1 notebooks:

| Notebook | Purpose |
|---|---|
| `05_rq1_chunking_evaluation.ipynb` | Run and analyze the full 36-config RQ1 evaluation: main effects, interaction heatmaps, winner selection, per-standard breakdown. |
| `09_rq1_chunking_retrieval_walkthrough.ipynb` | Qualitative single-report walkthrough: inspect chunk output, compare retrieval methods, visualize ranking shifts. |

Additional notebooks for RQ2, RQ3, and pipeline selection (06–08) are planned but not yet implemented.

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

This loads `data/Reports_md/`, builds chunks, embeds text, creates Qdrant collections, and writes artifacts to `data/artifacts/`.

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
│   └── best_pipeline.yaml          # Auto-generated best pipeline manifest
├── data/
│   ├── Reports_Pdf/                # Source sustainability reports (PDF)
│   ├── Reports_md/                 # Preprocessed reports (Markdown)
│   ├── ESRS_Standards_JSON/        # ESRS regulation files (JSON)
│   └── artifacts/                  # Generated indices, chunks, embeddings, eval results
│       ├── evaluation_results/rq1/ # Per-config results, chunk stats, winner JSON/CSV
│       └── qdrant_storage/         # Local Qdrant persistence
├── golden_standard/
│   └── golden_standard.json        # 12 questions × 17 companies with evidence passages
├── notebooks/
│   ├── 05_rq1_chunking_evaluation.ipynb
│   └── 09_rq1_chunking_retrieval_walkthrough.ipynb
├── rag/                            # Core RAG pipeline modules
│   ├── config/
│   │   └── settings.py             # Paths, constants, experiment grid builder
│   ├── data/
│   │   ├── loaders.py              # Report and golden-standard loading
│   │   └── text_utils.py           # Normalization, sentence splitting, page offsets
│   ├── chunking/
│   │   └── strategies.py           # Fixed, structure-aware, and semantic chunking
│   ├── indexing/
│   │   ├── embedder.py             # OpenAI embedding with caching
│   │   └── qdrant_index.py         # Qdrant local vector index
│   ├── retrieval/
│   │   └── methods.py              # Dense, BM25, RRF fusion, cross-encoder rerank
│   ├── evaluation/
│   │   └── metrics.py              # RQ1 relevance alignment and metric computation
│   ├── pipeline/
│   │   └── rq1_runner.py           # RQ1 orchestration and result persistence
│   └── types.py                    # Shared dataclasses
├── streamlit_app/
│   └── app.py                      # Streamlit frontend (loads best_pipeline.yaml)
├── run_preprocessing.py            # CLI: build chunks, embeddings, Qdrant indices
├── run_evaluation.py               # CLI: run full RQ1 evaluation matrix
├── requirements.txt
├── RESEARCH_QUESTIONS.md           # Detailed RQ1/RQ2/RQ3 experimental design
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
