# Research Questions — ESRS RAG Master Thesis

**Thesis title:** Optimizing RAG Pipelines for Analyzing Corporate Sustainability Reports under the European Sustainability Reporting Standards (ESRS)

**Author:** Sina Haghgoo

---

## Overview

The experimental evaluation is structured around three research questions, each targeting a distinct stage and concern of the RAG pipeline. The questions are designed to be **non-overlapping** and **sequentially dependent**: the best pipeline configuration identified in RQ1 (including the finding for RQ2) feeds directly into RQ2, and the best overall system from RQ1 is evaluated for generation quality in RQ3.

```
RQ1 → identifies best chunking + chunk size + retrieval combination (retrieval-only evaluation)
RQ2 → uses best config from RQ1, tests whether ESRS context injection improves compliance assessment
RQ3 → uses best config from RQ1, evaluates full generation quality of the final system
```

---

## RQ1 — Pipeline Configuration

### Question

**What combination of chunking strategy, chunk size, and retrieval method achieves optimal retrieval quality for ESRS disclosure queries, and do these factors interact?**

### Motivation

A RAG pipeline's retrieval quality depends on three interdependent design choices: how documents are split into chunks, how large those chunks are, and how those chunks are searched at query time. Prior work typically optimizes these stages independently, risking a local optimum where the best chunking strategy at 512 tokens under one retrieval method is not the best strategy at 256 or 1024 tokens under another. This research question evaluates all combinations jointly to identify the globally optimal configuration and to measure whether interaction effects exist among chunking strategy, chunk size, and retrieval.

### Experimental Design

A full **3 × 3 × 4 factorial experiment** across chunking strategy, chunk size, and retrieval method:

| Chunk size | Fixed-size | Document-structure-aware | Semantic |
|---|---|---|---|
| **256 tokens** | ✓ | ✓ | ✓ |
| **512 tokens** | ✓ | ✓ | ✓ |
| **1024 tokens** | ✓ | ✓ | ✓ |

Each chunking-strategy/size combination is evaluated with all four retrieval methods: Dense, Sparse (BM25), Hybrid (RRF), and Hybrid + Rerank.

**Total: 36 configurations.**

**Chunking strategies defined:**

- **Fixed-size:** Segments text into chunks of 256, 512, or 1024 tokens (measured via `tiktoken.encoding_for_model("text-embedding-3-large")`) with proportional 12.5% overlap (32, 64, or 128 tokens), without regard for document structure.
- **Document-structure-aware:** Splits at H1–H3 section boundaries extracted during preprocessing, producing chunks that respect the logical hierarchy of the sustainability report. The target chunk ceiling is set to 256, 512, or 1024 tokens, with fallback size-based splitting and proportional 12.5% overlap within oversized sections.
- **Semantic:** Uses `sentence-transformers` (`all-MiniLM-L6-v2`) to compute cosine similarity between adjacent sentences. Splits where similarity drops below 0.5, subject to the selected 256-, 512-, or 1024-token ceiling.

**Retrieval methods defined:**

- **Dense:** Semantic vector search over Qdrant using `text-embedding-3-large` embeddings (3072 dimensions).
- **Sparse (BM25):** Keyword matching using Okapi BM25 (k1=1.5, b=0.75), capturing exact ESRS terminology.
- **Hybrid (RRF):** Dense and sparse results fused via Reciprocal Rank Fusion (k=60).
- **Hybrid + Rerank:** Hybrid results reranked using the Cohere `rerank-english-v3.0` API.

### Evaluation

**Evaluation is retrieval-only. No generation step is performed in RQ1.**

Metrics computed per configuration against the Golden Standard (12 questions × 17 companies):

| Metric | Description |
|---|---|
| Recall@5, @10, @15 | Fraction of gold evidence pages recovered in top-k retrieved chunks |
| Precision@5, @10, @15 | Fraction of top-k retrieved chunks that are relevant |
| MRR@10 | Mean Reciprocal Rank — how highly the first relevant chunk is ranked |
| Coverage ratio | Fraction of gold evidence entries matched by any retrieved chunk |

**Analysis:**
- Main effect of chunking strategy (averaged across chunk sizes and retrieval methods)
- Main effect of chunk size (averaged across chunking strategies and retrieval methods)
- Main effect of retrieval method (averaged across chunking strategies and chunk sizes)
- Interaction effects: heatmaps of Recall@15 across all 36 combinations, with particular attention to chunk size × retrieval and chunking strategy × retrieval
- Per-ESRS-standard breakdown to identify whether certain standards benefit more from specific configurations

**Descriptive statistics reported per chunking strategy and chunk size:**
- Total chunk count, mean token count, standard deviation, size distribution histogram for each strategy-size combination
- One qualitative example showing how the same passage is segmented at 256, 512, and 1024 tokens under fixed-size and document-structure-aware chunking

### Expected Output

The winning configuration from RQ1 is passed forward as the fixed pipeline for RQ2 and RQ3.

---

## RQ2 — ESRS Domain Adaptation

### Question

**To what extent does explicitly injecting structured ESRS regulatory requirements into the generation prompt improve compliance assessment accuracy and answer faithfulness, compared to relying on the LLM's general pretraining knowledge of ESRS?**

### Motivation

The system's `esrs_context` module extracts query-specific disclosure requirements from the ESRS standard JSON files and injects them into the generation prompt. This step incurs an additional LLM call and adds significant prompt length. It is not self-evident that this injection is necessary: a large language model trained on publicly available corpora likely has general knowledge of ESRS, and may produce reasonable compliance assessments without explicit regulatory context. This research question isolates the contribution of ESRS context injection by comparing the full system against a condition where injection is disabled.

### Experimental Design

Two conditions, both using the **best pipeline configuration identified in RQ1**:

| Condition | RAG Pipeline | ESRS Context | What it tests |
|---|---|---|---|
| **Full system** | Best config from RQ1 | Injected | Current system — LLM uses retrieved passages + explicit ESRS requirements |
| **No ESRS context** | Best config from RQ1 | Not injected (empty string) | LLM relies on retrieved passages + parametric knowledge of ESRS only |

**Total generation calls: 12 questions × 17 companies × 2 conditions = 408 calls.**

All calls are cached after the first run at zero additional cost for reruns.

> **Why not a "no retrieval" condition?** Without retrieved company-specific passages, the system cannot ground its answer in the actual report content. Such a condition would conflate the absence of ESRS context with the absence of company evidence, making the isolation of the ESRS injection effect impossible. The two-condition design cleanly isolates the single variable of interest.

### Evaluation

The primary dependent variable is **compliance assessment accuracy** — whether the system correctly predicts the Low / Medium / High fulfillment level assigned in the Golden Standard.

| Metric | Type | Description |
|---|---|---|
| **Compliance accuracy** | Deterministic | Exact match of predicted vs gold fulfillment level (Low/Medium/High) |
| **Off-by-one accuracy** | Deterministic | Prediction within one ordinal level of gold label |
| **Token F1** | Deterministic | Lexical overlap between generated answer and gold answer |
| **Faithfulness score** | Deterministic | Fraction of generated claims supported by retrieved passages (lexical) |

> **No LLM judge is used for RQ2.** All four metrics are computed deterministically against the Golden Standard. This ensures full reproducibility across runs and avoids the self-evaluation bias that arises when the generator model also acts as judge.

**Analysis:**
- Per-condition aggregate scores for all four metrics
- Per-ESRS-standard breakdown of compliance accuracy (12 standards) to identify which standards benefit most from explicit injection
- Per-company consistency analysis: do the same companies systematically benefit from injection, or is the effect uniform?

### Expected Output

A quantitative answer to whether the ESRS context injection step is necessary for accurate compliance assessment. If the improvement is large, this validates the `esrs_context` module as a critical component. If marginal, it suggests the LLM's parametric knowledge of ESRS is sufficient and the injection step adds cost without proportional benefit.

---

## RQ3 — Generation Quality

### Question

**How accurately does the best-performing pipeline evaluate ESRS compliance, and where does the system fail?**

### Motivation

RQ1 identifies the optimal retrieval configuration and RQ2 validates the ESRS context injection mechanism. RQ3 evaluates the complete end-to-end system — using the best configuration from RQ1 with ESRS context injected — on the full range of generation quality dimensions: factual accuracy, completeness, answer faithfulness, citation correctness, and compliance assessment quality. Critically, this question goes beyond aggregate performance to identify systematic failure patterns: which ESRS standards are hardest to answer, under what conditions does the system hallucinate, and where does retrieval quality fail to translate into answer quality.

### Experimental Design

Single condition: **best pipeline from RQ1, with ESRS context injected**, evaluated against the full Golden Standard.

- 12 questions × 17 companies = **204 evaluated samples**
- All 12 ESRS standards covered (ESRS 1, 2, E1–E5, S1–S4, G1)

### Evaluation

**Approach A — Deterministic metrics:**

| Metric | Description |
|---|---|
| Token F1 | Lexical overlap between generated answer and gold answer |
| Citation F1 | Precision/recall of cited pages against gold evidence pages |
| Faithfulness score | Fraction of generated claims supported by retrieved passages |
| Hallucination rate | 1 − Faithfulness score |
| Compliance accuracy | Exact match of predicted vs gold fulfillment level |
| Off-by-one compliance accuracy | Prediction within one ordinal level of gold |

**Approach B — LLM-as-judge (multi-dimensional rubric):**

Each generated answer is scored on five dimensions using a structured rubric (1–5 scale):

| Dimension | What it measures |
|---|---|
| Factual correctness | Are stated facts accurate relative to the gold answer? |
| Completeness | Does the answer cover all key points from the gold answer? |
| Grounding faithfulness | Is the answer grounded in retrieved passages only, or does it introduce unsupported claims? |
| Citation quality | Are source pages cited correctly and helpfully? |
| Compliance assessment | Is the Low/Medium/High judgment correct and well-evidenced? |

> **Judge model:** A different model from the generator must be used to avoid self-evaluation bias. If `gpt-5.2` is used for generation, a separate model (e.g., Claude claude-sonnet-4-6 via the Anthropic API) should be used as judge. This is a methodological requirement, not optional. The use of the same model as both generator and judge is acknowledged as a limitation if a different model is unavailable.

**Error analysis (qualitative + quantitative):**

1. **Per-ESRS-standard compliance accuracy table** — which of the 12 standards does the system assess most and least accurately?
2. **Retrieval-generation gap analysis** — identify samples where Recall@15 is high (≥ 0.8) but LLM judge score is low (≤ 2/5). These cases indicate that good retrieval does not guarantee good generation, and suggest the generation step itself is the bottleneck.
3. **Hallucination by standard type** — compare hallucination rates for quantitative standards (ESRS E2–E5: numeric emissions, water, waste) vs qualitative standards (ESRS G1, ESRS 2). Hypothesis: numeric standards have higher hallucination rates because the LLM supplements missing figures from parametric knowledge.
4. **Compliance level confusion matrix** — a 3×3 matrix (predicted × gold) showing whether errors are systematic (e.g., consistently overestimating as High) or random.

### Expected Output

A comprehensive characterisation of the system's strengths and failure modes across the full ESRS framework, providing actionable insight into which pipeline components and which ESRS standards require further development. The compliance confusion matrix and per-standard accuracy breakdown are the primary thesis contributions of RQ3.

---

## Summary Table

| | RQ1 | RQ2 | RQ3 |
|---|---|---|---|
| **Stage tested** | Chunking + Retrieval | ESRS context injection | End-to-end generation |
| **Variable** | Chunking strategy × Chunk size × Retrieval method | ESRS context on/off | None (single best config) |
| **Evaluation type** | Retrieval-only | Generation | Generation + qualitative |
| **LLM calls** | None (embeddings only) | 408 generation calls | 204 generation + judge calls |
| **Primary metric** | Recall@15 | Compliance accuracy | LLM judge overall score |
| **Secondary metrics** | Precision@k, MRR, Coverage | Token F1, Faithfulness | Citation F1, Hallucination rate, Error analysis |
| **Configs evaluated** | 36 | 2 | 1 |
| **Golden Standard coverage** | 12 questions × 17 companies | 12 questions × 17 companies | 12 questions × 17 companies |

---

## Controlled Variables (Fixed Across All RQs Unless Explicitly Varied)

| Variable | Fixed value | Justification |
|---|---|---|
| Embedding model | `text-embedding-3-large` (3072 dims) | State-of-the-art general embedding; consistent across all configs |
| Chunk size | `RQ1:` 256, 512, 1024 tokens; `RQ2–RQ3:` best-performing size from RQ1 | Tests retrieval granularity in RQ1 while keeping downstream evaluation tied to the winning configuration |
| Chunk overlap | 12.5% of chunk size (`RQ1:` 32, 64, 128 tokens) | Prevents boundary information loss while keeping overlap proportional across chunk sizes |
| Top-k retrieved | 15 chunks | Sufficient coverage without excessive context window noise |
| RRF constant | k = 60 | Standard value from Cormack et al. (2009) |
| LLM (generator) | `gpt-5.2` | Consistent across all generation steps |
| Temperature | 0.1 | Near-deterministic generation for reproducibility |
| Dataset | 17 companies, 2024 reports | European energy and chemical sector; all ESRS-aligned |

---

## Limitations

1. **Chunk size sweep is still coarse.** RQ1 tests 256, 512, and 1024 tokens, but the true optimum may lie at intermediate values such as 384 or 768.
2. **Interaction effects within retrieval hyperparameters not explored.** RRF constant k=60 and top-k=15 are fixed; their optimal values under different chunking strategies are not evaluated.
3. **Self-evaluation risk in RQ3.** If the same model is used as both generator and judge, scores may be inflated. A different judge model is strongly recommended; if unavailable, this is stated as an explicit limitation.
4. **Golden Standard is manually curated.** Ground-truth answers and compliance levels were assigned by a single annotator. Inter-annotator agreement was not measured. Findings should be interpreted relative to this benchmark, not as absolute ground truth.
5. **Sequential dependency between RQs.** RQ2 and RQ3 use the best config from RQ1. If the RQ1 winner is marginal (small score differences between configurations), the downstream conclusions inherit that uncertainty.
6. **Cohere reranker requires a paid API.** The `Hybrid + Rerank` condition is not fully reproducible without a Cohere API key. Results for this configuration should be treated as indicative rather than independently verifiable.
