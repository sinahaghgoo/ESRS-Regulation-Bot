# ESRS RAG Evaluation Summary

Configurations evaluated: 8

## Aggregate Results by Configuration

### RQ1_fixed_512 (RQ1)

Evaluated samples: 1

**Retrieval**

| Metric | Value |
|---|---:|
| avg_recall@5 | 1.0000 |
| avg_recall@10 | 1.0000 |
| avg_recall@15 | 1.0000 |
| avg_precision@5 | 1.0000 |
| avg_precision@10 | 1.0000 |
| avg_precision@15 | 0.8000 |
| avg_mrr@5 | 1.0000 |
| avg_mrr@10 | 1.0000 |
| avg_mrr@15 | 1.0000 |
| avg_coverage_ratio | 1.0000 |

**Answer Quality**

| Metric | Value |
|---|---:|
| avg_token_f1 | 0.5444 |
| avg_token_precision | 0.4513 |
| avg_token_recall | 0.6859 |
| avg_bertscore | 0.0000 |
| avg_llm_judge | 0.0000 |
| avg_llm_judge_normalized | 0.0000 |
| compliance_accuracy | 0.0000 |

**Citation**

| Metric | Value |
|---|---:|
| avg_citation_precision | 0.6000 |
| avg_citation_recall | 1.0000 |
| avg_citation_f1 | 0.7500 |

**Hallucination**

| Metric | Value |
|---|---:|
| avg_faithfulness | 0.7469 |
| avg_supported_claim_ratio | 1.0000 |
| avg_hallucination_rate | 0.0000 |

**Efficiency**

| Metric | Value |
|---|---:|
| avg_latency | 0.0128 |
| avg_prompt_tokens | 10789.0000 |
| avg_total_tokens | 12118.0000 |
| avg_cost_usd | 0.0403 |
| avg_chunks_retrieved | 15.0000 |

### RQ1_structure_aware (RQ1)

Evaluated samples: 1

**Retrieval**

| Metric | Value |
|---|---:|
| avg_recall@5 | 1.0000 |
| avg_recall@10 | 1.0000 |
| avg_recall@15 | 1.0000 |
| avg_precision@5 | 0.8000 |
| avg_precision@10 | 0.8000 |
| avg_precision@15 | 0.8667 |
| avg_mrr@5 | 1.0000 |
| avg_mrr@10 | 1.0000 |
| avg_mrr@15 | 1.0000 |
| avg_coverage_ratio | 1.0000 |

**Answer Quality**

| Metric | Value |
|---|---:|
| avg_token_f1 | 0.5152 |
| avg_token_precision | 0.4300 |
| avg_token_recall | 0.6426 |
| avg_bertscore | 0.0000 |
| avg_llm_judge | 0.0000 |
| avg_llm_judge_normalized | 0.0000 |
| compliance_accuracy | 0.0000 |

**Citation**

| Metric | Value |
|---|---:|
| avg_citation_precision | 0.6000 |
| avg_citation_recall | 1.0000 |
| avg_citation_f1 | 0.7500 |

**Hallucination**

| Metric | Value |
|---|---:|
| avg_faithfulness | 0.7814 |
| avg_supported_claim_ratio | 1.0000 |
| avg_hallucination_rate | 0.0000 |

**Efficiency**

| Metric | Value |
|---|---:|
| avg_latency | 0.0026 |
| avg_prompt_tokens | 9356.0000 |
| avg_total_tokens | 10549.0000 |
| avg_cost_usd | 0.0353 |
| avg_chunks_retrieved | 15.0000 |

### RQ2_dense_only (RQ2)

Evaluated samples: 1

**Retrieval**

| Metric | Value |
|---|---:|
| avg_recall@5 | 1.0000 |
| avg_recall@10 | 1.0000 |
| avg_recall@15 | 1.0000 |
| avg_precision@5 | 0.8000 |
| avg_precision@10 | 0.8000 |
| avg_precision@15 | 0.8667 |
| avg_mrr@5 | 1.0000 |
| avg_mrr@10 | 1.0000 |
| avg_mrr@15 | 1.0000 |
| avg_coverage_ratio | 1.0000 |

**Answer Quality**

| Metric | Value |
|---|---:|
| avg_token_f1 | 0.5338 |
| avg_token_precision | 0.4812 |
| avg_token_recall | 0.5993 |
| avg_bertscore | 0.0000 |
| avg_llm_judge | 0.0000 |
| avg_llm_judge_normalized | 0.0000 |
| compliance_accuracy | 0.0000 |

**Citation**

| Metric | Value |
|---|---:|
| avg_citation_precision | 0.5000 |
| avg_citation_recall | 1.0000 |
| avg_citation_f1 | 0.6667 |

**Hallucination**

| Metric | Value |
|---|---:|
| avg_faithfulness | 0.7324 |
| avg_supported_claim_ratio | 1.0000 |
| avg_hallucination_rate | 0.0000 |

**Efficiency**

| Metric | Value |
|---|---:|
| avg_latency | 0.0024 |
| avg_prompt_tokens | 8747.0000 |
| avg_total_tokens | 9903.0000 |
| avg_cost_usd | 0.0334 |
| avg_chunks_retrieved | 15.0000 |

### RQ2_sparse_only (RQ2)

Evaluated samples: 1

**Retrieval**

| Metric | Value |
|---|---:|
| avg_recall@5 | 1.0000 |
| avg_recall@10 | 1.0000 |
| avg_recall@15 | 1.0000 |
| avg_precision@5 | 1.0000 |
| avg_precision@10 | 0.7000 |
| avg_precision@15 | 0.6000 |
| avg_mrr@5 | 1.0000 |
| avg_mrr@10 | 1.0000 |
| avg_mrr@15 | 1.0000 |
| avg_coverage_ratio | 1.0000 |

**Answer Quality**

| Metric | Value |
|---|---:|
| avg_token_f1 | 0.5776 |
| avg_token_precision | 0.5068 |
| avg_token_recall | 0.6715 |
| avg_bertscore | 0.0000 |
| avg_llm_judge | 0.0000 |
| avg_llm_judge_normalized | 0.0000 |
| compliance_accuracy | 0.0000 |

**Citation**

| Metric | Value |
|---|---:|
| avg_citation_precision | 0.5000 |
| avg_citation_recall | 1.0000 |
| avg_citation_f1 | 0.6667 |

**Hallucination**

| Metric | Value |
|---|---:|
| avg_faithfulness | 0.7097 |
| avg_supported_claim_ratio | 1.0000 |
| avg_hallucination_rate | 0.0000 |

**Efficiency**

| Metric | Value |
|---|---:|
| avg_latency | 0.0013 |
| avg_prompt_tokens | 8559.0000 |
| avg_total_tokens | 9821.0000 |
| avg_cost_usd | 0.0340 |
| avg_chunks_retrieved | 15.0000 |

### RQ2_hybrid (RQ2)

Evaluated samples: 1

**Retrieval**

| Metric | Value |
|---|---:|
| avg_recall@5 | 1.0000 |
| avg_recall@10 | 1.0000 |
| avg_recall@15 | 1.0000 |
| avg_precision@5 | 0.8000 |
| avg_precision@10 | 0.8000 |
| avg_precision@15 | 0.8667 |
| avg_mrr@5 | 1.0000 |
| avg_mrr@10 | 1.0000 |
| avg_mrr@15 | 1.0000 |
| avg_coverage_ratio | 1.0000 |

**Answer Quality**

| Metric | Value |
|---|---:|
| avg_token_f1 | 0.5152 |
| avg_token_precision | 0.4300 |
| avg_token_recall | 0.6426 |
| avg_bertscore | 0.0000 |
| avg_llm_judge | 0.0000 |
| avg_llm_judge_normalized | 0.0000 |
| compliance_accuracy | 0.0000 |

**Citation**

| Metric | Value |
|---|---:|
| avg_citation_precision | 0.6000 |
| avg_citation_recall | 1.0000 |
| avg_citation_f1 | 0.7500 |

**Hallucination**

| Metric | Value |
|---|---:|
| avg_faithfulness | 0.7814 |
| avg_supported_claim_ratio | 1.0000 |
| avg_hallucination_rate | 0.0000 |

**Efficiency**

| Metric | Value |
|---|---:|
| avg_latency | 0.0134 |
| avg_prompt_tokens | 9356.0000 |
| avg_total_tokens | 10549.0000 |
| avg_cost_usd | 0.0353 |
| avg_chunks_retrieved | 15.0000 |

### RQ2_hybrid_rerank (RQ2)

Evaluated samples: 1

**Retrieval**

| Metric | Value |
|---|---:|
| avg_recall@5 | 0.6667 |
| avg_recall@10 | 1.0000 |
| avg_recall@15 | 1.0000 |
| avg_precision@5 | 0.6000 |
| avg_precision@10 | 0.8000 |
| avg_precision@15 | 0.8000 |
| avg_mrr@5 | 1.0000 |
| avg_mrr@10 | 1.0000 |
| avg_mrr@15 | 1.0000 |
| avg_coverage_ratio | 1.0000 |

**Answer Quality**

| Metric | Value |
|---|---:|
| avg_token_f1 | 0.5882 |
| avg_token_precision | 0.5648 |
| avg_token_recall | 0.6137 |
| avg_bertscore | 0.0000 |
| avg_llm_judge | 0.0000 |
| avg_llm_judge_normalized | 0.0000 |
| compliance_accuracy | 0.0000 |

**Citation**

| Metric | Value |
|---|---:|
| avg_citation_precision | 0.6000 |
| avg_citation_recall | 1.0000 |
| avg_citation_f1 | 0.7500 |

**Hallucination**

| Metric | Value |
|---|---:|
| avg_faithfulness | 0.6810 |
| avg_supported_claim_ratio | 1.0000 |
| avg_hallucination_rate | 0.0000 |

**Efficiency**

| Metric | Value |
|---|---:|
| avg_latency | 0.0033 |
| avg_prompt_tokens | 6632.0000 |
| avg_total_tokens | 7682.0000 |
| avg_cost_usd | 0.0271 |
| avg_chunks_retrieved | 10.0000 |

### RQ3_best_retrieval (RQ3)

Evaluated samples: 1

**Retrieval**

| Metric | Value |
|---|---:|
| avg_recall@5 | 1.0000 |
| avg_recall@10 | 1.0000 |
| avg_recall@15 | 1.0000 |
| avg_precision@5 | 0.8000 |
| avg_precision@10 | 0.8000 |
| avg_precision@15 | 0.8667 |
| avg_mrr@5 | 1.0000 |
| avg_mrr@10 | 1.0000 |
| avg_mrr@15 | 1.0000 |
| avg_coverage_ratio | 1.0000 |

**Answer Quality**

| Metric | Value |
|---|---:|
| avg_token_f1 | 0.5152 |
| avg_token_precision | 0.4300 |
| avg_token_recall | 0.6426 |
| avg_bertscore | 0.0000 |
| avg_llm_judge | 0.0000 |
| avg_llm_judge_normalized | 0.0000 |
| compliance_accuracy | 0.0000 |

**Citation**

| Metric | Value |
|---|---:|
| avg_citation_precision | 0.6000 |
| avg_citation_recall | 1.0000 |
| avg_citation_f1 | 0.7500 |

**Hallucination**

| Metric | Value |
|---|---:|
| avg_faithfulness | 0.7814 |
| avg_supported_claim_ratio | 1.0000 |
| avg_hallucination_rate | 0.0000 |

**Efficiency**

| Metric | Value |
|---|---:|
| avg_latency | 0.0023 |
| avg_prompt_tokens | 9356.0000 |
| avg_total_tokens | 10549.0000 |
| avg_cost_usd | 0.0353 |
| avg_chunks_retrieved | 15.0000 |

### RQ3_sparse_retrieval (RQ3)

Evaluated samples: 1

**Retrieval**

| Metric | Value |
|---|---:|
| avg_recall@5 | 1.0000 |
| avg_recall@10 | 1.0000 |
| avg_recall@15 | 1.0000 |
| avg_precision@5 | 1.0000 |
| avg_precision@10 | 0.7000 |
| avg_precision@15 | 0.6000 |
| avg_mrr@5 | 1.0000 |
| avg_mrr@10 | 1.0000 |
| avg_mrr@15 | 1.0000 |
| avg_coverage_ratio | 1.0000 |

**Answer Quality**

| Metric | Value |
|---|---:|
| avg_token_f1 | 0.5776 |
| avg_token_precision | 0.5068 |
| avg_token_recall | 0.6715 |
| avg_bertscore | 0.0000 |
| avg_llm_judge | 0.0000 |
| avg_llm_judge_normalized | 0.0000 |
| compliance_accuracy | 0.0000 |

**Citation**

| Metric | Value |
|---|---:|
| avg_citation_precision | 0.5000 |
| avg_citation_recall | 1.0000 |
| avg_citation_f1 | 0.6667 |

**Hallucination**

| Metric | Value |
|---|---:|
| avg_faithfulness | 0.7097 |
| avg_supported_claim_ratio | 1.0000 |
| avg_hallucination_rate | 0.0000 |

**Efficiency**

| Metric | Value |
|---|---:|
| avg_latency | 0.0018 |
| avg_prompt_tokens | 8559.0000 |
| avg_total_tokens | 9821.0000 |
| avg_cost_usd | 0.0340 |
| avg_chunks_retrieved | 15.0000 |

## Compact RQ Comparison

### RQ1

| Config | Recall@15 | Token F1 | Citation F1 | Hallucination Rate | Cost USD |
|---|---:|---:|---:|---:|---:|
| RQ1_fixed_512 | 1.0000 | 0.5444 | 0.7500 | 0.0000 | 0.040262 |
| RQ1_structure_aware | 1.0000 | 0.5152 | 0.7500 | 0.0000 | 0.035320 |

### RQ2

| Config | Recall@15 | Token F1 | Citation F1 | Hallucination Rate | Cost USD |
|---|---:|---:|---:|---:|---:|
| RQ2_dense_only | 1.0000 | 0.5338 | 0.6667 | 0.0000 | 0.033427 |
| RQ2_sparse_only | 1.0000 | 0.5776 | 0.6667 | 0.0000 | 0.034017 |
| RQ2_hybrid | 1.0000 | 0.5152 | 0.7500 | 0.0000 | 0.035320 |
| RQ2_hybrid_rerank | 1.0000 | 0.5882 | 0.7500 | 0.0000 | 0.027080 |

### RQ3

| Config | Recall@15 | Token F1 | Citation F1 | Hallucination Rate | Cost USD |
|---|---:|---:|---:|---:|---:|
| RQ3_best_retrieval | 1.0000 | 0.5152 | 0.7500 | 0.0000 | 0.035320 |
| RQ3_sparse_retrieval | 1.0000 | 0.5776 | 0.6667 | 0.0000 | 0.034017 |

