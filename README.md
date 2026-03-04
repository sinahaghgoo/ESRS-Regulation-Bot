# 📘 ESRS RAG – Retrieval-Augmented Generation for Corporate Sustainability Reports
Master Thesis — *Sina Haghgoo*
### **Optimizing RAG pipelines for analyzing corporate sustainability reports under the European Sustainability Reporting Standards (ESRS)**  

---

## 🔍 Project Overview

This project develops and evaluates a **domain-adapted Retrieval-Augmented Generation (RAG) system** for analyzing long corporate sustainability reports and answering ESRS-related questions in a **transparent, reproducible, and auditable** way.

The system integrates:

- Retrieval from company sustainability reports  
- ESRS regulatory context extraction  
- LLM-based answer generation  
- Evaluation against a **Golden Standard** (expert-verified answers)  
- A Streamlit frontend for interactive analysis  

The goal is to **benchmark different RAG configurations** and identify optimal strategies for ESRS-aligned disclosure analysis and implementing the best performing one into a streamlit application

---

## Objectives


## Information Retrieval

- Identify the relevant ESRS disclosure requirements related to the user query and create a concise summary of these requirements.
- Retrieve semantically relevant chunks from the selected company reports that correspond to the user query.

## Answer Generation

- Provide the retrieved report excerpts together with the summarized ESRS disclosure requirements to the LLM for answer generation.

## Output Structure

The generated output should follow this structure:

### 1. Direct Answer
Provide a clear and concise answer to the user’s query, based on the content of the company reports.

### 2. ESRS Compliance Evaluation
Assess how well the company’s disclosures align with the relevant ESRS requirements:

- ## 2.1 **Compliance level:** Low / Medium / High  
- ## 2.2 **Reasoning:**  
  - What aspects are well covered?  
  - What information is missing or insufficient (identified gaps or “white spots”)?

### 3. Company Comparison 
If multiple companies are selected, include a comparative analysis that highlights:

- Strengths and weaknesses of each company
- Differences in how well they address the ESRS disclosure requirements
- Reasons for stronger or weaker alignment with ESRS 

2. **Evaluate quality**  
   Benchmark retrieval and answer quality against a predefined **Golden Standard**, based on qualitative and quantitative metrics.

3. **Implementation of best performing Pipeline into Streamlit Frontend**  
   The evaluation forms the basis, for implementing the best performing pipeline into a streamlit chatbot
---

## Research Questions for the Evaluation part

**RQ1 — The Input (Data Structure)**
How do domain-specific chunking strategies impact the retrieval of relevant reporting standards?

Focus: Is understanding the document structure (Parent-Child, Section headers) more important than the raw semantic content?

**RQ2 — The Retrieval (Search Logic)**
To what extent does Hybrid Search (Keyword + Vector) and Reranking improve accuracy over standard Sparse and Dense Retrieval in technical reporting?

Focus: Does the vector database need "keyword help" to find specific ESRS terms?

Metrics: Recall@k, Precision@k, Latency.

**RQ3 — The Validation (Downstream Impact)**
How does the choice of retrieval pipeline affect the generation quality and hallucination rate of the final answer?

Focus: Does better retrieval actually lead to better answers, or does the LLM compensate for bad retrieval?

---

## Data & Resources

### **Sustainability Reports**  
- Corporate sustainability reports (2024) from Oil & Gas companies  
- All aligned with ESRS

### **ESRS Regulation Dataset**  
Folder: `Regulation_Split_JSON/`  
Contains 12 JSON files covering:  
- ESRS 1–2  
- ESRS E1–E5  
- ESRS S1–S4  
- ESRS G1  

Each file includes disclosure requirements, objectives, and application requirements of the corresponding ESRS chapter.

### **Golden Standard** (`golden_standard.json` NOTE: STILL NEEDS TO BE CREATED!!)  
Contains:  
- ESRS-related prompts/questions  
- Required ESRS disclosures  
- Ground-truth answers
- Which pages of the report contain the answer  
- Disclosure fulfillment levels per company

### **Models (from `.env`)**
- Embeddings: **text-embedding-3-large**  
- LLM: **gpt-5.2**

### **Vector Database**
- **[Qdrant](https://qdrant.tech/)**  
  - Used as the core vector search engine for storing and retrieving embeddings.
  - configured with local persistence for reproducibility.

---

## Technical Architecture

### 1. Data Ingestion & Preprocessing (Azure)
- **Source**: Azure Blob Storage (`raw-reports`).
- **Engine**: Azure Document Intelligence with the `prebuilt-layout` model.
- **Multimodal Enrichment**: Integration of **GPT-5-mini Vision (VLM)** to generate textual descriptions for charts and figures, replacing raw images with semantic context.
- **Cleaning**: Regex-based post-processing to remove artifacts (page numbers, headers).

### 2. Chunking
- **Strategy**: Decision on what strategies is open!!

### 3. Embedding & Indexing
- **Embeddings**: `text-embedding-3-large` (3072 dimensions).
- **Vector Database**: **Qdrant** (Local/Docker).
- **Schema**: Stores text vectors alongside rich metadata (Company, Year, Source File, Header Hierarchy, ESRS IDs).

### 4. Retrieval (Hybrid Fusion)
- **Hybrid Search**: Combines **Dense Retrieval** (Semantic vector search) and **Sparse Retrieval** (BM25 keyword matching).
- **Fusion**: Uses **Reciprocal Rank Fusion (RRF)** (k=60) to merge and re-rank results.
- **Filtering**: Supports metadata filtering by Company and Year.

### 5. Generation & Domain Adaptation
- **ESRS Extraction**: Specialized module that queries the LLM with the full ESRS regulation text to extract only requirements relevant to the user's specific question.
- **Anti-Hallucination**: System prompts explicitly constrain answers to retrieved context.
- **Fulfillment Assessment**: Automated evaluator (`Low/Medium/High`) that compares the generated answer against the full ESRS requirements to detect coverage gaps.
- **Cross-Company Comparison**: specialized generation logic for comparing multiple companies.

---

## Jupyter Notebooks --> STILL NEED TO BE CREATED AFTER ALL THE MODULES AND EVEFYTHING ELSE IS CREATED

### **1. Preprocessing & Indexing Notebook**
Responsible for:  
- Loading reports  
- Running different chunking strategies  
- Computing embeddings  
- Building Qdrant index  
- Saving all artifacts for later evaluation and the Streamlit UI  

Heavy logic lives inside `rag/*.py`.

---

### **2. Evaluation Notebook**
Runs systematic benchmarking on the **Golden Standard**:

- Loads chunks, indices, embeddings  
- Runs each RAG configuration  
- Measures retrieval and answer quality  
- Computes similarity metrics  
- Stores evaluation results  

This produces analysis for **RQ1–RQ4**.

---

## Streamlit Frontend

A simple & interactive interface for analysts:

### **Features**
- Select one or multiple companies  
- Select ESRS chapter (required)  
- Ask a question  
- System returns:  
  - Final answer  
  - Supporting report chunks  
  - ESRS disclosure requirements  
  - Disclosure fulfillment (low/medium/high)

### **Backend**
Uses Qdrant for persistent per-company embeddings with metadata filtering.

---

## Evaluation Framework

### **Retrieval Metrics**
- Recall@k  
- Precision@k  
- MRR@k  
- Coverage ratio  

### **Answer Quality Metrics**
- Token-Level F1  
- BERTScore  
- LLM-as-a-judge scoring  
- Citation correctness  

### **Hallucination Metrics**
- Faithfulness score  
- Supported claim ratio  
- Hallucination rate  

### **Efficiency Metrics**
- Latency  
- Prompt size  
- Token cost  
- retrieved chunks  

---

## Development Guidelines

- Python-first modular design  
- Jupyter for experimentation  
- Streamlit for frontend  
- Easy configuration switching via dictionaries  
- Transparent, explainable, auditable RAG pipeline  

---

## Environment Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root containing:

```
OPENAI_API_KEY=your_api_key_here
EMBEDDING_MODEL=text-embedding-3-large
LLM_MODEL=gpt-5.2/ gpt 5 mini
```


---

## Usage

### Step 1: Preprocessing and Indexing

Run the preprocessing notebook to process reports and build indices:

```bash
jupyter notebook notebooks/01_preprocessing_indexing.ipynb
```

Alternatively, use the Python script:

```bash
python run_preprocessing.py
```

This will:
- Load and preprocess PDF reports from OMV, Repsol, and Shell
- Apply fixed-size and semantic chunking
- Generate embeddings using OpenAI
- Build Qdrant index for retrieval
- Save all artifacts to `data/artifacts/`

**Note:** This step requires OpenAI API access and may take some time depending on report sizes.

### Step 2: Evaluation (Optional)

Run the evaluation notebook to benchmark RAG configurations:

```bash
jupyter notebook notebooks/02_evaluation.ipynb
```

Alternatively, use the Python script:

```bash
python run_evaluation.py
```

This will:
- Load the Golden Standard
- Evaluate different chunking strategies
- Compute retrieval and answer quality metrics
- Save results to `data/artifacts/evaluation_results.json`

### Step 3: Interactive Chatbot

Launch the Streamlit app:

```bash
streamlit run streamlit_app/app.py
```

Then open the URL shown in the terminal (typically http://localhost:8501)

**Features:**
- Select one or more companies (OMV, Repsol, Shell)
- Select ESRS chapter (e.g., ESRS E1)
- Ask questions about sustainability reports
- View answers with ESRS fulfillment assessment
- See supporting passages and ESRS requirements

---

## 📁 Project Structure

```
ESRS-Regulation-Bot/
├── rag/                    # Core RAG modules
├── notebooks/              # Jupyter notebooks
├── streamlit_app/          # Streamlit frontend
├── config/                 # Configuration
├── data/
│   ├── company reports/    # PDF reports (OMV, Repsol, Shell)
│   ├── Regulation_Split_JSON/  # ESRS regulation JSON files
│   └── artifacts/         # Generated indices and chunks
└── Golden Standard/        # Evaluation golden standard
```



