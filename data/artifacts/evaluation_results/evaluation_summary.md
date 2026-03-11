# ESRS RAG Evaluation Summary

Configurations evaluated: 1

## Aggregate Results by Configuration

### RQ2_hybrid (RQ2)

Evaluated samples: 10

**Retrieval**

| Metric | Value |
|---|---:|
| avg_recall@5 | 0.8500 |
| avg_recall@10 | 0.9000 |
| avg_recall@15 | 0.9250 |
| avg_precision@5 | 0.6800 |
| avg_precision@10 | 0.6600 |
| avg_precision@15 | 0.6067 |
| avg_mrr@5 | 0.8000 |
| avg_mrr@10 | 0.8167 |
| avg_mrr@15 | 0.8167 |
| avg_coverage_ratio | 0.9667 |

**Answer Quality**

| Metric | Value |
|---|---:|
| avg_token_f1 | 0.5218 |
| avg_token_precision | 0.5064 |
| avg_token_recall | 0.5949 |
| avg_bertscore | 0.9207 |
| avg_llm_judge | 0.0000 |
| compliance_accuracy | 0.0000 |

**Citation**

| Metric | Value |
|---|---:|
| avg_citation_precision | 0.5077 |
| avg_citation_recall | 0.9250 |
| avg_citation_f1 | 0.6290 |

**Hallucination**

| Metric | Value |
|---|---:|
| avg_faithfulness | 0.7271 |
| avg_supported_claim_ratio | 0.9800 |
| avg_hallucination_rate | 0.0200 |

**Efficiency**

| Metric | Value |
|---|---:|
| avg_latency | 16.8813 |
| avg_prompt_tokens | 7995.5000 |
| avg_total_tokens | 8930.0000 |
| avg_cost_usd | 0.0293 |
| avg_chunks_retrieved | 15.0000 |

## Compact RQ Comparison

### RQ2

| Config | Recall@15 | Token F1 | Citation F1 | Hallucination Rate | Cost USD |
|---|---:|---:|---:|---:|---:|
| RQ2_hybrid | 0.9250 | 0.5218 | 0.6290 | 0.0200 | 0.029334 |

