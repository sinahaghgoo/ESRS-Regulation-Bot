# ESRS RAG Evaluation Summary

Configurations evaluated: 1

## Aggregate Results by Configuration

### RQ2_hybrid_rerank (RQ2)

Evaluated samples: 4

**Retrieval**

| Metric | Value |
|---|---:|
| avg_recall@5 | 0.7917 |
| avg_recall@10 | 1.0000 |
| avg_recall@15 | 1.0000 |
| avg_precision@5 | 0.6500 |
| avg_precision@10 | 0.6250 |
| avg_precision@15 | 0.6250 |
| avg_mrr@5 | 1.0000 |
| avg_mrr@10 | 1.0000 |
| avg_mrr@15 | 1.0000 |
| avg_coverage_ratio | 1.0000 |

**Answer Quality**

| Metric | Value |
|---|---:|
| avg_token_f1 | 0.6504 |
| avg_token_precision | 0.6536 |
| avg_token_recall | 0.6712 |
| avg_bertscore | 0.0000 |
| avg_llm_judge | 0.0000 |
| avg_llm_judge_normalized | 0.0000 |
| compliance_accuracy | 0.0000 |

**Citation**

| Metric | Value |
|---|---:|
| avg_citation_precision | 0.6625 |
| avg_citation_recall | 1.0000 |
| avg_citation_f1 | 0.7738 |

**Hallucination**

| Metric | Value |
|---|---:|
| avg_faithfulness | 0.7388 |
| avg_supported_claim_ratio | 0.9722 |
| avg_hallucination_rate | 0.0278 |

**Efficiency**

| Metric | Value |
|---|---:|
| avg_latency | 13.3472 |
| avg_prompt_tokens | 5894.0000 |
| avg_total_tokens | 6891.5000 |
| avg_cost_usd | 0.0247 |
| avg_chunks_retrieved | 10.0000 |

## Compact RQ Comparison

### RQ2

| Config | Recall@15 | Token F1 | Citation F1 | Hallucination Rate | Cost USD |
|---|---:|---:|---:|---:|---:|
| RQ2_hybrid_rerank | 1.0000 | 0.6504 | 0.7738 | 0.0278 | 0.024710 |

