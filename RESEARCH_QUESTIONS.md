# Research Questions — ESRS RAG Master Thesis

**Thesis title:** Optimizing a Retrieval-Augmented Generation (RAG) Pipeline for a Regulatory Compliance Chatbot: A Case Study on Corporate Adherence to the European Sustainability Reporting Standards (ESRS)

**Author:** Sina Haghgoo

---

## Overview

The experimental evaluation is structured around three research questions, each targeting a distinct stage and concern of the RAG pipeline. The questions are designed to be **non-overlapping** and **sequentially dependent**: the best pipeline configuration identified in RQ1 feeds directly into RQ2, and the best overall system from RQ1 — with the ESRS context injection decision informed by the RQ2 finding — is evaluated for generation quality in RQ3.

```
RQ1 → identifies best chunking + chunk size + retrieval combination (retrieval-only evaluation)
RQ2 → uses best config from RQ1, tests whether ESRS context injection improves compliance assessment
RQ3 → uses best config from RQ1 (with or without ESRS context based on RQ2 finding), evaluates full generation quality of the final system
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
- **Dense + Rerank:** Dense retrieval of 5x candidates followed by cross-encoder reranking.
- **Sparse (BM25):** Keyword matching using Okapi BM25 (k1=1.5, b=0.75) with Robertson et al. (1994) IDF formulation. IDF statistics are computed globally across all companies to provide stable vocabulary weights for ESRS-specific terminology, regardless of which company is being queried.
- **Sparse + Rerank:** Sparse retrieval of 5x candidates followed by cross-encoder reranking.
- **Hybrid (RRF):** Dense and sparse results fused via Reciprocal Rank Fusion with k=60 (Cormack et al., 2009). Each system contributes 3x candidates to the fusion pool.
- **Hybrid + Rerank:** Hybrid RRF of 5x candidates, then reranked using the cross-encoder.

**Cross-encoder model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (22 M params, ~80 MB). This lightweight cross-encoder was chosen for practical evaluation on Apple Silicon; it is the most widely cited compact reranker in IR benchmarks. Reranking uses a 5x candidate pool and section-context-aware input (header hierarchy prepended). Reranking scores are cached to disk for deterministic re-runs. A larger model such as `BAAI/bge-reranker-v2-m3` (568 M params) could yield different outcomes but is not tested in this thesis.

### Evaluation

**Evaluation is retrieval-only. No generation step is performed in RQ1.**

#### Relevance Matching

Metrics are computed per configuration against the Golden Standard (12 questions × 17 companies = 204 samples). Because the retrieval unit is a **chunk** while the annotation unit in the Golden Standard is an **evidence passage with page metadata**, relevance is defined explicitly at the passage level using a two-tier matching approach.

**Primary path (span-based):** Both the gold evidence passage and the retrieved chunk are located as character spans inside the normalised full-report text. A chunk is counted as relevant when the overlap covers at least the gold passage by character span. This threshold balances strictness (requiring substantial coverage of the annotated evidence) with tolerance for chunking boundary effects where a passage may be split across adjacent chunks.

**Fallback path (token-level):** When the gold passage cannot be found as an exact substring (e.g., due to formatting differences between the manually curated golden standard and the preprocessed report text), a token-level overlap is used. A multiplicity-aware bag-of-words comparison counts each gold token at most as many times as it appears in the chunk, preventing inflated matches from repeated common words. Gold passages shorter than 20 characters are excluded from the fallback path to avoid false positives.


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
| MRR@10 | Mean Reciprocal Rank: how highly the first relevant chunk is ranked |
| Budgeted Recall@4096 tokens | Fraction of gold evidence passages recovered before cumulative context exceeds 4,096 tokens |

**Primary metric:** `Recall@15`

`Recall@15` is the primary model-selection criterion because the downstream generation step can only use evidence that retrieval surfaces within the final ranked context window. A configuration that retrieves slightly noisier results but captures more relevant evidence in the top 15 is therefore preferable to one that ranks a single relevant chunk very highly but misses additional supporting evidence.

**Secondary metrics:** `Precision@15` and `MRR@10`

These metrics distinguish configurations that achieve similar recall but differ in ranking quality and noise level. `Precision@15` quantifies how much irrelevant noise enters the retrieval context. `MRR@10` captures how quickly the system surfaces the first relevant chunk — an intuitive measure of ranking quality.

NOTE: MRR is reported at k=10 following standard IR benchmark convention (TREC, MS-MARCO); extending to k=15 is omitted as the reciprocal rank contribution beyond rank 10 is negligible (≤ 0.033).

**Robustness metric:** `Budgeted Recall@4096 tokens`

This supplementary analysis tests whether the ranking of configurations remains stable once chunk-size conditions are compared under an approximately equal downstream token budget. Ranked chunks are accumulated in order until a cumulative retrieval budget of **4,096 tokens** is reached; recall is recomputed on that budgeted prefix. The budget check is applied **before** processing each chunk's evidence matches: a chunk that would cause the cumulative count to exceed the budget is excluded entirely. This makes the 256-, 512-, and 1024-token conditions directly comparable under an approximately equal context budget.

If the same winning configuration performs best under both `Recall@15` and `Budgeted Recall@4096 tokens`, the RQ1 conclusion is treated as robust. If the winners differ, both results are reported and the trade-off is discussed.

#### Statistical Validation

To ensure observed performance differences are not due to random variation, statistical significance testing is performed:

- **Bootstrap confidence intervals** (percentile method, 10,000 resamples, seed=42) for all reported metrics. The 95% CI is reported alongside point estimates.
- **Paired permutation tests** (two-sided, 10,000 permutations, seed=42) for pairwise comparisons of top-ranked configurations on the primary metric. P-values are reported with a significance threshold of α=0.05.
- **Cohen's d effect size** for each pairwise comparison, providing a measure of practical significance beyond p-values. Effect sizes are interpreted following standard conventions: small (d=0.2), medium (d=0.5), large (d=0.8).
- **Standard deviations** for key metrics (`recall@15`, `precision@15`, `mrr@10`) are computed across the 204 evaluation samples to support error bars in thesis visualisations.

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

**To what extent does disclosure-filtered ESRS context injection improve evidence-grounded compliance classification and compliance rationale quality, compared with the same RAG pipeline without injected ESRS context?**

### Motivation

The system's `esrs_context` module extracts disclosure-filtered ESRS requirement text for each question: it loads the ESRS standard JSON mapped to the question, filters to the subtree(s) matching the target disclosure ID(s) (e.g., E1-1, S4-1), and serializes the matching paragraphs into a prompt-ready context block. If no disclosure-level match is found, it falls back to the full standard. This disclosure-filtered approach adds significant but targeted prompt length (14K–25K characters depending on the standard), and it is not self-evident that this injection is necessary: a large language model trained on publicly available corpora likely has general knowledge of ESRS, and may produce reasonable compliance assessments without explicit regulatory context. This research question isolates the contribution of ESRS context injection by comparing the full system against a condition where injection is disabled.

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

The primary dependent variable is **evidence-grounded compliance classification quality** — whether the system correctly predicts the Low / Medium / High fulfillment level assigned in the Golden Standard while also generating high-quality compliance rationales, evaluated through a combination of deterministic metrics and LLM-judge assessment.

#### Deterministic Compliance-Level Metrics

| Metric | Description |
|---|---|
| **Macro-F1** | Classification metric across Low / Medium / High with equal weight per class |
| **Balanced accuracy** | Mean recall across the fulfillment classes, robust to the `High`-heavy label distribution |
| **Exact accuracy** | Exact match of predicted vs gold fulfillment level (Low/Medium/High) |
| **Weighted Cohen's kappa** | Ordinal agreement between predicted and gold fulfillment levels |
| **Off-by-one accuracy** | Prediction within one ordinal level of gold fulfillment |

#### LLM-Judge Rationale Evaluation

The compliance rationale quality is evaluated using a G-Eval v2 rubric (Liu et al., 2023) with an anti-anchoring design:

- **Judge model:** Claude Sonnet 4.5 (via Anthropic API), separate from the generator (GPT-5-mini) to avoid self-evaluation bias
- **Judge version:** v2 (`v2_79b8809bad`) — the anti-anchoring redesign withholds the generated compliance level from the judge prompt to prevent anchoring bias. The judge evaluates rationale substance independently of whether the level prediction happened to match. All judge outputs are cached with version-stamped keys to ensure reproducibility across judge iterations
- **Input:** Gold fulfillment level, gold fulfillment rationale, and generated compliance rationale (generated compliance level deliberately withheld)
- **Score:** 1–5 Likert scale following G-Eval's chain-of-thought evaluation design, normalized to 0.0–1.0 for composite metrics
- **Verdicts derived from score:** `aligned` (score 4–5), `partially_aligned` (score 3), `misaligned` (score 1–2)
- **Error tags:** Structured error categorization (`contradiction`, `omission`, `different_level_support`, `none`)
- **Chain-of-thought:** The prompt includes explicit evaluation steps the judge must follow before scoring, following G-Eval methodology

#### Joint Compliance-Assessment Metrics

These metrics evaluate the combined compliance level + rationale quality. The composite uses a **multiplicative** formulation: a sample with the wrong compliance level receives zero credit regardless of rationale quality, because a compliance assessment with an incorrect conclusion is not useful.

| Metric | Description |
|---|---|
| **Assessment Quality Score (AQS)** | Per-sample: `I(level_correct) × rationale_score_normalized`. Averaged across all samples. Primary composite endpoint. |
| **Overall Assessment Rate (OAR)** | Fraction of samples where the compliance level is correct AND the rationale score ≥ 3 (moderate alignment on the 1–5 scale) |
| **Conditional Rationale Quality (CRQ)** | Mean normalized rationale score for correctly-classified samples only — measures rationale quality given correct classification |

**Primary metric:** `Assessment Quality Score (AQS)` — the expected rationale quality weighted by classification correctness. This metric is zero when the level is wrong and scales linearly with rationale quality when correct, providing a genuine joint signal.

**Secondary metrics:** `Macro-F1`, `Balanced accuracy`, `Overall Assessment Rate`, `Mean rationale score`

**Analysis:**
- Per-condition aggregate scores for deterministic metrics and LLM-judge verdicts
- Majority-class baseline comparison to contextualise the `High`-dominant label distribution
- Per-ESRS-standard breakdown of joint lenient match and balanced accuracy to identify which standards benefit most from explicit injection
- Per-company consistency analysis: do the same companies systematically benefit from injection, or is the effect uniform?
- **Wilcoxon signed-rank test** on question-level paired deltas (n=12 clusters) as primary confirmatory test (exact p-values via full enumeration of 2^12 sign assignments), with cluster bootstrap confidence intervals (10,000 resamples), Cohen's d effect sizes, rank-biserial correlation (Kerby, 2014), post-hoc power analysis, and prediction bias diagnosis. A **McNemar-style exact sign test** on sample-level discordant pairs (n=204) provides a complementary sensitivity analysis that ignores clustering.

### Expected Output

A quantitative answer to whether the ESRS context injection step is necessary for accurate compliance assessment. If the improvement is large, this validates the `esrs_context` module as a critical component. If marginal, it suggests the LLM's parametric knowledge of ESRS is sufficient and the injection step adds cost without proportional benefit.

---

## RQ3 — Generation Quality

### Question

**How accurately does the best-performing pipeline evaluate ESRS compliance, and where does the system fail?**

### Motivation

RQ1 identifies the optimal retrieval configuration and RQ2 tests the ESRS context injection mechanism. RQ3 evaluates the complete end-to-end system — using the best configuration from RQ1 without ESRS context injection (based on the RQ2 finding that injection has no statistically detectable benefit) — and characterises how well it performs on the final task that matters in practice: producing a correct ESRS compliance assessment together with a factually grounded and sufficiently complete written answer.

To keep the design **simple but defensible**, RQ3 does **not** attempt to operationalise every quality dimension with a separate complex automatic metric. For long-form answers, lexical-overlap metrics such as Token F1 are too brittle, while full claim-level factuality pipelines would add considerable implementation complexity for limited additional thesis value. Instead, RQ3 uses a **mixed evaluation design**:

- **Deterministic metrics** for the structured parts of the output that can be measured reliably (compliance label and cited pages)
- **LLM-judge rubric scoring** for broader answer quality dimensions (factual correctness, completeness, grounding, citation usefulness, and coherence)
- **Targeted failure analysis** to show where the system breaks down

This design follows recent RAG-evaluation literature that combines structured automatic metrics with rubric-based judging and human sanity checks.

### Experimental Design

Single condition: **best pipeline from RQ1, without ESRS context injection** (based on the RQ2 finding that injection has no significant effect), evaluated against the full Golden Standard.

- 12 questions × 17 companies = **204 evaluated samples**
- All 12 ESRS standards covered (ESRS 1, 2, E1–E5, S1–S4, G1)

### Evaluation

**Approach A — Deterministic metrics:**

| Metric | Description |
|---|---|
| Citation page precision / recall / F1 | Overlap of generated `cited_pages` with gold evidence pages (range-based, ±1 page tolerance); primary deterministic signal for citation quality |

> **Primary deterministic metric:** `Citation page F1`

**Approach B — LLM-as-judge (multi-dimensional rubric):**

Each generated answer is scored on five dimensions using a structured rubric (1–5 scale). The mean score for each dimension is reported separately, and the simple arithmetic mean across all five dimensions is reported as an overall descriptive summary:

| Dimension | What it measures |
|---|---|
| Factual correctness | Are stated facts accurate relative to the gold answer? |
| Completeness | Does the answer cover all key points from the gold answer? |
| Grounding faithfulness | Is the answer grounded in retrieved passages only, or does it introduce unsupported claims? |
| Citation usefulness | Are cited pages relevant and helpful for verifying the answer? |
| Coherence | Is the answer logically structured, internally consistent, and easy to follow? |

> **Judge model:** A different model from the generator must be used to avoid self-evaluation bias. If `gpt-5-mini` is used for generation, a separate model (e.g., Claude Sonnet 4.5 via the Anthropic API) should be used as judge. This is a methodological requirement, not optional. The use of the same model as both generator and judge is acknowledged as a limitation if a different model is unavailable.

**Judge calibration subset:**

To keep the judge-based component defensible without turning RQ3 into a full human-evaluation study, a **stratified manual audit subset of 20 samples** is reviewed using the same 1–5 rubric. The thesis reports:

- exact-or-adjacent agreement between manual and judge scores for each dimension
- qualitative discussion of representative disagreement cases

This subset is used as a **sanity check**, not as a separate replacement benchmark.

**Reporting and validation:**

- RQ3 is a **single-condition characterisation study**, not a paired ablation. Therefore, no pairwise null-hypothesis test is required.
- Aggregate deterministic metrics and mean judge scores are reported with **95% bootstrap confidence intervals** (10,000 resamples).
- The main RQ3 outputs are reported jointly: **mean judge-dimension scores** and **citation page F1**.

**Error analysis (qualitative + quantitative):**

1. **Per-ESRS-standard compliance accuracy table** — which of the 12 standards does the system assess most and least accurately?
2. **Per-ESRS-standard judge-score table** — which standards are factually easier or harder to answer completely and faithfully?
3. **Retrieval-generation gap analysis** — identify samples where Recall@15 is high (≥ 0.8) but grounding-faithfulness judge score is low (≤ 2/5). These cases indicate that good retrieval does not guarantee good generation, and suggest the generation step itself is the bottleneck.
4. **Compliance level confusion matrix** — a 3×3 matrix (predicted × gold) showing whether errors are systematic (e.g., consistently overestimating as High) or random.
5. **Failure-type table** — classify a small set of representative failures into four practical categories: unsupported claim, incomplete answer, wrong compliance level, and weak citation support.

### Expected Output

A comprehensive but manageable characterisation of the system's strengths and failure modes across the full ESRS framework. The main thesis contributions of RQ3 are:

- a robust **compliance-assessment performance profile** 
- a **multi-dimensional quality profile** of the generated answers (factual correctness, completeness, grounding, citation usefulness, coherence)
- a focused analysis of **where retrieval succeeds but generation still fails**

### Method References

- [Liu et al. (2023), *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment*](https://aclanthology.org/2023.emnlp-main.153/)
- [Saad-Falcon et al. (2024), *ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems*](https://aclanthology.org/2024.naacl-long.20/)
- [Min et al. (2023), *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation*](https://aclanthology.org/2023.emnlp-main.741/)
- [Ru et al. (2024), *RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation*](https://arxiv.org/abs/2408.08067)
- [Penzkofer and Baumann (2024), *Evaluating and Fine-Tuning Retrieval-Augmented Language Models to Generate Text with Accurate Citations*](https://aclanthology.org/2024.konvens-main.6/)
- [Wu et al. (2025), *SusGen-GPT: A Data-Centric LLM for Financial NLP and Sustainability Report Generation*](https://aclanthology.org/2025.findings-naacl.66/)
- [He et al. (2025), *ESGenius: Benchmarking LLMs on Environmental, Social, and Governance (ESG) and Sustainability Knowledge*](https://aclanthology.org/2025.emnlp-main.739/)
- [Sun et al. (2026), *ESG-Bench: Benchmarking Long-Context ESG Reports for Hallucination Mitigation*](https://arxiv.org/abs/2603.13154)

---

## Summary Table

| | RQ1  | RQ2  | RQ3  |
|---|---|---|---|
| **Focus** | Retrieval configuration | ESRS context injection | End-to-end generation quality |
| **Design** | 3×3×6 factorial (54 configs) | Paired ablation (2 conditions) | Single-condition characterisation |
| **Samples** | 204 (12 Q × 17 companies) | 408 (204 × 2 conditions) | 204 (12 Q × 17 companies) |
| **Primary metric** | Recall@15 | Assessment Quality Score (AQS) | LLM-judge mean + Citation page F1 |

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
| Reranker candidate pool | 5× top-k | Provides sufficient depth for cross-encoder reranking without excessive computation |
| LLM (generator) | `gpt-5-mini` | Consistent across all generation steps |
| Reasoning effort | `medium` | `gpt-5-mini` is a reasoning model; temperature is not applicable. Reasoning effort controls response determinism. |
| Dataset | 17 companies, 2024 reports | European energy and chemical sector; all ESRS-aligned |
| Statistical tests | Bootstrap 10K resamples, permutation 10K where pairwise comparisons are made, seed=42 | Ensures reproducibility of statistical reporting |

---

## Limitations

1. **Chunk size sweep is still coarse.** RQ1 tests 256, 512, and 1024 tokens, but the true optimum may lie at intermediate values such as 384 or 768.
2. **Interaction effects within retrieval hyperparameters not explored.** RRF constant k=60 and top-k=15 are fixed; their optimal values under different chunking strategies are not evaluated.
3. **Self-evaluation risk in RQ3.** If the same model is used as both generator and judge, scores may be inflated. A different judge model is strongly recommended; if unavailable, this is stated as an explicit limitation.
4. **Golden Standard is manually curated.** Ground-truth answers and compliance levels were assigned by a single annotator. Inter-annotator agreement was not measured. Findings should be interpreted relative to this benchmark, not as absolute ground truth.
5. **Sequential dependency between RQs.** RQ2 and RQ3 use the best config from RQ1. If the RQ1 winner is marginal (small score differences between configurations), the downstream conclusions inherit that uncertainty. Statistical significance testing mitigates this by quantifying whether the winner is meaningfully better than runner-up configurations.
6. **Reranker model size.** All rerank conditions use the local `cross-encoder/ms-marco-MiniLM-L-6-v2` cross-encoder (~80 MB, 22 M params). This lightweight model was chosen for practical evaluation on Apple Silicon; a larger model such as `BAAI/bge-reranker-v2-m3` could yield different reranking outcomes but is not tested in this thesis.
7. **Semantic similarity threshold fixed at 0.5.** The cosine similarity threshold for semantic chunking boundary detection is fixed at 0.5. Lower values would produce fewer boundaries (larger initial groups), while higher values would produce more boundaries (finer splitting). This parameter is not varied in the factorial design and is acknowledged as a controlled variable.
8. **Faithfulness remains difficult to measure deterministically.** A full claim-level factuality pipeline in the style of FACTSCORE or RAGChecker would be stronger than surface overlap, but would add substantial implementation and annotation complexity. RQ3 therefore evaluates broad answer faithfulness mainly through the multi-dimensional judge rubric, with page-level citation overlap reported only as an auxiliary signal.
9. **Low statistical power in RQ2 paired analysis.** With only n=12 question clusters for the Wilcoxon signed-rank test, achieved power is low (approximately 5%) and the minimum detectable effect size at 80% power is large (Cohen's d > 1.5). The study can detect large effects but not small-to-medium effects. This is an inherent constraint of the 12-question Golden Standard design.
