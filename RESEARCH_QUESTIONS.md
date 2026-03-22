# Research Questions — ESRS RAG Master Thesis

**Thesis title:** Optimizing RAG Pipelines for Analyzing Corporate Sustainability Reports under the European Sustainability Reporting Standards (ESRS)

**Author:** Sina Haghgoo

---

## Overview

The experimental evaluation is structured around three research questions, each targeting a distinct stage and concern of the RAG pipeline. The questions are designed to be **non-overlapping** and **sequentially dependent**: the best pipeline configuration identified in RQ1 feeds directly into RQ2, and the best overall system from RQ1 (including the finding for RQ2) is evaluated for generation quality in RQ3.

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

A full **3 × 3 × 6 factorial experiment** across chunking strategy, chunk size, and retrieval method:

| Chunk size | Fixed-size | Document-structure-aware | Semantic |
|---|---|---|---|
| **256 tokens** | ✓ | ✓ | ✓ |
| **512 tokens** | ✓ | ✓ | ✓ |
| **1024 tokens** | ✓ | ✓ | ✓ |

Each chunking-strategy/size combination is evaluated with all six retrieval methods: Dense, Dense + Rerank, Sparse (BM25), Sparse + Rerank, Hybrid (RRF), and Hybrid + Rerank.

**Total: 54 configurations.**

**Chunking strategies defined:**

- **Fixed-size:** Segments text into chunks of 256, 512, or 1024 tokens (measured via `tiktoken.encoding_for_model("text-embedding-3-large")`) with proportional 12.5% overlap (32, 64, or 128 tokens), without regard for document structure. Serves as the baseline strategy.
- **Document-structure-aware:** Splits at H1–H3 section boundaries extracted during preprocessing, producing chunks that respect the logical hierarchy of the sustainability report. The target chunk ceiling is set to 256, 512, or 1024 tokens, with fallback size-based splitting and proportional 12.5% overlap within oversized sections.
- **Semantic:** Uses `sentence-transformers` (`all-MiniLM-L6-v2`) to compute cosine similarity between adjacent sentences. Topic boundaries are detected where similarity drops below 0.5. Adjacent sentence groups are then **greedily merged** up to the target token ceiling (256, 512, or 1024 tokens). This merging step is critical: without it, raw semantic boundaries produce excessive fragmentation (individual sentences averaging ~75 tokens). The merge algorithm preserves topical coherence (groups are still bounded by detected topic transitions) while producing chunks that approach the target size, making the strategy directly comparable to fixed-size and structure-aware chunking under the same token budget.

**Retrieval methods defined:**

- **Dense:** Semantic vector search over Qdrant using `text-embedding-3-large` embeddings (3072 dimensions).
- **Dense + Rerank:** Dense retrieval of 3x candidates followed by cross-encoder reranking.
- **Sparse (BM25):** Keyword matching using Okapi BM25 (k1=1.5, b=0.75) with Robertson et al. (1994) IDF formulation. IDF statistics are computed globally across all companies to provide stable vocabulary weights for ESRS-specific terminology, regardless of which company is being queried.
- **Sparse + Rerank:** Sparse retrieval of 3x candidates followed by cross-encoder reranking.
- **Hybrid (RRF):** Dense and sparse results fused via Reciprocal Rank Fusion with k=60 (Cormack et al., 2009). Each system contributes 2x candidates to the fusion pool.
- **Hybrid + Rerank:** Hybrid RRF of 3x candidates, then reranked using the cross-encoder.

**Cross-encoder model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (22 M params, ~80 MB). This lightweight cross-encoder was chosen for practical evaluation on Apple Silicon; it is the most widely cited compact reranker in IR benchmarks. Reranking scores are cached to disk for deterministic re-runs. A larger model such as `BAAI/bge-reranker-v2-m3` (568 M params) could yield different outcomes but is not tested in this thesis.

### Evaluation

**Evaluation is retrieval-only. No generation step is performed in RQ1.**

#### Relevance Matching

Metrics are computed per configuration against the Golden Standard (12 questions × 17 companies = 204 samples). Because the retrieval unit is a **chunk** while the annotation unit in the Golden Standard is an **evidence passage with page metadata**, relevance is defined explicitly at the passage level using a two-tier matching approach.

**Primary path (span-based):** Both the gold evidence passage and the retrieved chunk are located as character spans inside the normalised full-report text. A chunk is counted as relevant when the overlap covers at least **50% of the gold passage** by character span. This threshold balances strictness (requiring substantial coverage of the annotated evidence) with tolerance for chunking boundary effects where a passage may be split across adjacent chunks.

**Fallback path (token-level):** When the gold passage cannot be found as an exact substring (e.g., due to formatting differences between the manually curated golden standard and the preprocessed report text), a token-level overlap is used. A multiplicity-aware bag-of-words comparison counts each gold token at most as many times as it appears in the chunk, preventing inflated matches from repeated common words. The same 50% threshold applies. Gold passages shorter than 20 characters are excluded from the fallback path to avoid false positives.

**Configurable threshold:** The 50% threshold is a parameter (`threshold` in `rag/evaluation/metrics.py`), not a hardcoded constant. The justification for this threshold choice is discussed in the thesis text.

**References for the matching approach:**
- Adlakha et al. (2023) "Evaluating Correctness and Faithfulness of Instruction-Following Models for Question Answering" — token F1 for passage-level matching.
- TREC passage retrieval evaluation methodology (NIST) — span-based passage alignment.
- Karpukhin et al. (2020) DPR evaluation — passage-level exact match with relaxed string matching.

#### Aggregation Rules

Matching and aggregation follow three rules for reproducibility:

1. **Recall is computed over unique gold evidence passages**: if multiple retrieved chunks match the same gold passage, that passage is counted only once.
2. **Precision is computed over retrieved chunks**: a chunk is counted as relevant if it matches at least one gold evidence passage, even if it matches several.
3. If automatic span alignment fails for a small number of cases because of preprocessing artefacts or minor text differences, the token-overlap fallback handles them transparently. The diagnostic counters track how many checks took each path, ensuring full auditability.

Question-company pairs with no gold evidence passages are excluded from recall computation but retained for precision; no such pairs exist in this dataset.

#### Metrics

| Metric | Description |
|---|---|
| Recall@5, @10, @15 | Fraction of gold evidence passages recovered in the top-k retrieved chunks |
| Precision@5, @10, @15 | Fraction of the top-k retrieved chunks that are relevant |
| F1@5, @10, @15 | Harmonic mean of Precision@k and Recall@k — balances coverage with noise |
| nDCG@10 | Ranking quality with binary relevance, rewarding multiple relevant chunks near the top |
| MRR@10 | Mean Reciprocal Rank: how highly the first relevant chunk is ranked |
| Budgeted Recall@4096 tokens | Fraction of gold evidence passages recovered before cumulative context exceeds 4,096 tokens |

**Primary metric:** `Recall@15`

`Recall@15` is the primary model-selection criterion because the downstream generation step can only use evidence that retrieval surfaces within the final ranked context window. A configuration that retrieves slightly noisier results but captures more relevant evidence in the top 15 is therefore preferable to one that ranks a single relevant chunk very highly but misses additional supporting evidence.

**Secondary metrics:** `Precision@15`, `F1@15`, `nDCG@10`, and `MRR@10`

These metrics distinguish configurations that achieve similar recall but differ in ranking quality and noise level. `F1@15` provides a balanced view of precision-recall trade-offs. `nDCG@10` is especially important because many question-company pairs contain multiple gold evidence passages, making first-hit metrics such as MRR insufficient on their own.

NOTE: MRR and nDCG are reported at k=10 following standard IR benchmark convention (TREC, MS-MARCO); extending to k=15 is omitted as the reciprocal rank contribution beyond rank 10 is negligible (≤ 0.033).

**Robustness metric:** `Budgeted Recall@4096 tokens`

This supplementary analysis tests whether the ranking of configurations remains stable once chunk-size conditions are compared under an approximately equal downstream token budget. Ranked chunks are accumulated in order until a cumulative retrieval budget of **4,096 tokens** is reached; recall is recomputed on that budgeted prefix. The budget check is applied **before** processing each chunk's evidence matches: a chunk that would cause the cumulative count to exceed the budget is excluded entirely. This makes the 256-, 512-, and 1024-token conditions directly comparable under an approximately equal context budget.

If the same winning configuration performs best under both `Recall@15` and `Budgeted Recall@4096 tokens`, the RQ1 conclusion is treated as robust. If the winners differ, both results are reported and the trade-off is discussed.

#### Statistical Validation

To ensure observed performance differences are not due to random variation, statistical significance testing is performed:

- **Bootstrap confidence intervals** (percentile method, 10,000 resamples, seed=42) for all reported metrics. The 95% CI is reported alongside point estimates.
- **Paired permutation tests** (two-sided, 10,000 permutations, seed=42) for pairwise comparisons of top-ranked configurations on the primary metric. P-values are reported with a significance threshold of α=0.05.
- **Cohen's d effect size** for each pairwise comparison, providing a measure of practical significance beyond p-values. Effect sizes are interpreted following standard conventions: small (d=0.2), medium (d=0.5), large (d=0.8).
- **Standard deviations** for key metrics (`recall@15`, `precision@15`, `f1@15`, `ndcg@10`, `mrr@10`) are computed across the 204 evaluation samples to support error bars in thesis visualisations.

These tests are implemented in `rag/evaluation/statistical_tests.py` and can be run via `tests/test_evaluation_metrics.py`.

#### Analysis

- Main effect of chunking strategy, averaged across chunk sizes and retrieval methods
- Main effect of chunk size, averaged across chunking strategies and retrieval methods
- Main effect of retrieval method, averaged across chunking strategies and chunk sizes
- Interaction effects visualised through heatmaps of `Recall@15` across all 54 combinations, with particular attention to chunk size × retrieval and chunking strategy × retrieval
- Robustness comparison of configuration rankings under `Recall@15` versus `Budgeted Recall@4096 tokens` to test whether conclusions are driven by chunk-size budget effects
- Per-ESRS-standard breakdown to identify whether certain standards benefit more from specific configurations
- Statistical significance testing of top-ranked configurations via paired permutation tests with bootstrap confidence intervals

**Descriptive statistics reported per chunking strategy and chunk size:**
- Total chunk count, mean token count, standard deviation, median, 25th/75th percentile, size distribution histogram for each strategy-size combination
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
| **Secondary metrics** | Precision@k, F1@k, MRR, nDCG | Token F1, Faithfulness | Citation F1, Hallucination rate, Error analysis |
| **Statistical validation** | Bootstrap CIs, permutation tests | Per-standard breakdown | Confusion matrix, gap analysis |
| **Configs evaluated** | 54 | 2 | 1 |
| **Golden Standard coverage** | 12 questions × 17 companies | 12 questions × 17 companies | 12 questions × 17 companies |

---

## Controlled Variables (Fixed Across All RQs Unless Explicitly Varied)

| Variable | Fixed value | Justification |
|---|---|---|
| Embedding model | `text-embedding-3-large` (3072 dims) | State-of-the-art general embedding; consistent across all configs |
| Chunk size | `RQ1:` 256, 512, 1024 tokens; `RQ2–RQ3:` best-performing size from RQ1 | Tests retrieval granularity in RQ1 while keeping downstream evaluation tied to the winning configuration |
| Chunk overlap | 12.5% of chunk size (`RQ1:` 32, 64, 128 tokens) | Prevents boundary information loss while keeping overlap proportional across chunk sizes |
| Relevance threshold | 50% gold passage coverage | Balances strictness with tolerance for chunking boundary effects; justification discussed in the thesis text |
| Top-k retrieved | 15 chunks | Sufficient coverage without excessive context window noise |
| RRF constant | k = 60 | Standard value from Cormack et al. (2009) |
| Reranker candidate pool | 3× top-k | Provides sufficient depth for cross-encoder reranking without excessive computation |
| LLM (generator) | `gpt-5.2` | Consistent across all generation steps |
| Temperature | 0.1 | Near-deterministic generation for reproducibility |
| Dataset | 17 companies, 2024 reports | European energy and chemical sector; all ESRS-aligned |
| Statistical tests | Bootstrap 10K resamples, permutation 10K, seed=42 | Ensures reproducibility of significance testing |

---

## Limitations

1. **Chunk size sweep is still coarse.** RQ1 tests 256, 512, and 1024 tokens, but the true optimum may lie at intermediate values such as 384 or 768.
2. **Interaction effects within retrieval hyperparameters not explored.** RRF constant k=60 and top-k=15 are fixed; their optimal values under different chunking strategies are not evaluated.
3. **Self-evaluation risk in RQ3.** If the same model is used as both generator and judge, scores may be inflated. A different judge model is strongly recommended; if unavailable, this is stated as an explicit limitation.
4. **Golden Standard is manually curated.** Ground-truth answers and compliance levels were assigned by a single annotator. Inter-annotator agreement was not measured. Findings should be interpreted relative to this benchmark, not as absolute ground truth.
5. **Sequential dependency between RQs.** RQ2 and RQ3 use the best config from RQ1. If the RQ1 winner is marginal (small score differences between configurations), the downstream conclusions inherit that uncertainty. Statistical significance testing mitigates this by quantifying whether the winner is meaningfully better than runner-up configurations.
6. **Reranker model size.** All rerank conditions use the local `cross-encoder/ms-marco-MiniLM-L-6-v2` cross-encoder (~80 MB, 22 M params). This lightweight model was chosen for practical evaluation on Apple Silicon; a larger model such as `BAAI/bge-reranker-v2-m3` could yield different reranking outcomes but is not tested in this thesis.
7. **Semantic similarity threshold fixed at 0.5.** The cosine similarity threshold for semantic chunking boundary detection is fixed at 0.5. Lower values would produce fewer boundaries (larger initial groups), while higher values would produce more boundaries (finer splitting). This parameter is not varied in the factorial design and is acknowledged as a controlled variable.
