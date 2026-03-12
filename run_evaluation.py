#!/usr/bin/env python3
"""
Evaluation Script
==================
Benchmarks RAG pipeline configurations against the Golden Standard.
Evaluates retrieval quality, answer quality, hallucination, and efficiency.

Usage:
    python run_evaluation.py                              # Full evaluation
    python run_evaluation.py --smoke-test                 # Quick 1-company test
    python run_evaluation.py --configs RQ1_fixed_512 RQ2_hybrid  # Specific configs
    python run_evaluation.py --skip-llm-judge             # Skip expensive LLM scoring
    python run_evaluation.py --questions Q1 Q3            # Specific questions only
"""

import argparse
import time

from rag.config import (
    get_experiment_configs,
    load_golden_standard, BENCHMARK_QUESTION_IDS,
    save_best_pipeline_manifest,
)
from rag.pipeline import build_index
from rag.retriever import Retriever
from rag.evaluator import evaluate_pipeline_config, save_evaluation_results, select_best_pipeline


def main():
    parser = argparse.ArgumentParser(description="ESRS RAG Evaluation Pipeline")
    parser.add_argument(
        "--configs", nargs="+", default=None,
        help="Specific config names to evaluate (default: all)"
    )
    parser.add_argument(
        "--questions", nargs="+", default=None,
        help="Specific question IDs to evaluate (default: Q1,Q2,Q3,Q8,Q12)"
    )
    parser.add_argument(
        "--companies", nargs="+", default=None,
        help="Specific companies to evaluate (default: all)"
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Quick test with 1 company and 1 question"
    )
    parser.add_argument(
        "--force-rebuild", action="store_true",
        help="Force rebuilding indices for selected configurations"
    )
    parser.add_argument(
        "--skip-llm-judge", action="store_true",
        help="Skip LLM-as-judge scoring to reduce API costs"
    )
    parser.add_argument(
        "--skip-bertscore", action="store_true",
        help="Skip BERTScore computation"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("ESRS RAG Pipeline — Evaluation")
    print("=" * 60)
    
    # Load golden standard
    golden_standard = load_golden_standard()
    print(f"  Loaded Golden Standard: {len(golden_standard['questions'])} questions")
    
    # Determine what to evaluate
    all_configs = get_experiment_configs()
    
    if args.smoke_test:
        # Minimal test: one config, one question, one company
        config_names = ["RQ1_structure_aware"]
        question_ids = ["Q1"]
        companies = ["Aker BP"]
        skip_judge = True
        skip_bert = True
    else:
        config_names = args.configs if args.configs else list(all_configs.keys())
        question_ids = args.questions if args.questions else BENCHMARK_QUESTION_IDS
        companies = args.companies
        skip_judge = args.skip_llm_judge
        skip_bert = args.skip_bertscore
    
    print(f"  Configs to evaluate: {config_names}")
    print(f"  Questions: {question_ids}")
    print(f"  Companies: {companies or 'all'}")
    pair_count_est = len(question_ids) * (len(companies) if companies else 17)
    print(f"  Estimated company-question pairs per config: ~{pair_count_est}")
    print(f"  LLM judge: {'OFF' if skip_judge else 'ON'} | BERTScore: {'OFF' if skip_bert else 'ON'}")
    
    # Run evaluations
    all_results = {}
    
    for config_name in config_names:
        if config_name not in all_configs:
            print(f"  Warning: Config '{config_name}' not found, skipping")
            continue
        
        config = all_configs[config_name]
        print(f"\n{'─' * 50}")
        print(f"Evaluating: {config_name}")
        print(f"  Strategy: {config['chunking_strategy']}")
        print(f"  Retrieval: {config['retrieval_method']}")
        print(f"  Reranking: {config.get('use_reranker', False)}")
        print(f"{'─' * 50}")
        
        # Load/build the index for this config
        try:
            vec_index, bm25, _chunks = build_index(
                config=config,
                force_rebuild=args.force_rebuild,
                companies=companies,
            )
        except Exception as e:
            print(f"  Error building index: {e}")
            continue
        
        # Create retriever
        retriever = Retriever(vec_index, bm25, config)
        
        # Run evaluation
        t0 = time.time()
        results = evaluate_pipeline_config(
            config_name=config_name,
            retriever=retriever,
            config=config,
            golden_standard=golden_standard,
            question_ids=question_ids,
            companies=companies,
            skip_llm_judge=skip_judge,
            skip_bertscore=skip_bert,
        )
        elapsed = time.time() - t0
        
        results["evaluation_time_seconds"] = elapsed
        all_results[config_name] = results
        
        # Print summary
        agg = results.get("aggregate", {})
        print(f"\n  Summary for {config_name}:")
        print(f"    Evaluated: {agg.get('n_evaluated', 0)} company-question pairs")
        ret = agg.get("retrieval", {})
        print(f"    Avg Recall@15: {ret.get('avg_recall@15', 0):.3f}")
        print(f"    Avg Precision@15: {ret.get('avg_precision@15', 0):.3f}")
        aq = agg.get("answer_quality", {})
        print(f"    Avg Token F1: {aq.get('avg_token_f1', 0):.3f}")
        cit = agg.get("citation", {})
        print(f"    Avg Citation F1: {cit.get('avg_citation_f1', 0):.3f}")
        print(f"    Time: {elapsed:.1f}s")
    
    # Save all results
    save_evaluation_results(all_results)

    if all_results:
        best_name, best_result = select_best_pipeline(all_results)
        best_agg = best_result.get("aggregate", {})

        metrics_summary = {}
        for category in ["retrieval", "answer_quality", "citation", "hallucination", "efficiency"]:
            for key, value in best_agg.get(category, {}).items():
                if isinstance(value, (int, float)):
                    metrics_summary[f"{category}_{key}"] = round(float(value), 6)
                else:
                    metrics_summary[f"{category}_{key}"] = value

        best_config = all_configs.get(best_name, best_result.get("config", {}))
        selection_reason = (
            "Highest weighted composite score from run_evaluation.py "
            f"(n_evaluated={best_agg.get('n_evaluated', 0)})"
        )
        manifest_path = save_best_pipeline_manifest(
            config_name=best_name,
            config=best_config,
            selection_reason=selection_reason,
            metrics_summary=metrics_summary,
        )

        print(f"\nBest pipeline selected: {best_name}")
        print(f"Best pipeline manifest saved to: {manifest_path}")

    print(f"\n{'=' * 60}")
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
