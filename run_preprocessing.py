#!/usr/bin/env python3
"""
Preprocessing and Indexing Script
==================================
Processes sustainability reports and builds vector/BM25 indices.

Usage:
    python run_preprocessing.py                           # Default config
    python run_preprocessing.py --strategy structure_aware # Specific strategy
    python run_preprocessing.py --strategy fixed_size      # Fixed-size chunks
    python run_preprocessing.py --force                    # Force rebuild
    python run_preprocessing.py --all                      # Build all strategies

This step loads the Markdown reports, applies the configured chunking
strategies, generates embeddings via the OpenAI API, builds the indices,
and saves all artifacts to data/artifacts/.
"""

import argparse
import time

from rag.config import get_default_pipeline_config
from rag.pipeline import build_index


def main():
    parser = argparse.ArgumentParser(description="ESRS RAG Preprocessing Pipeline")
    parser.add_argument(
        "--strategy", type=str, default="structure_aware",
        choices=["fixed_size", "structure_aware"],
        help="Chunking strategy to use"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force rebuild even if cached artifacts exist"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Build indices for all chunking strategies"
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Only process one company for quick validation"
    )
    parser.add_argument(
        "--companies", nargs="+", default=None,
        help="Optional explicit list of company names to index"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("ESRS RAG Pipeline — Preprocessing & Indexing")
    print("=" * 60)
    
    strategies = ["fixed_size", "structure_aware"] if args.all else [args.strategy]
    companies = ["Aker BP"] if args.smoke_test else args.companies
    
    for strategy in strategies:
        print(f"\n{'─' * 40}")
        print(f"Strategy: {strategy}")
        print(f"{'─' * 40}")
        
        config = get_default_pipeline_config()
        config["chunking_strategy"] = strategy
        
        t0 = time.time()
        vec_index, bm25, chunks = build_index(
            config=config,
            force_rebuild=args.force,
            companies=companies,
        )
        elapsed = time.time() - t0
        
        print(f"\n  Completed in {elapsed:.1f}s")
        print(f"  Total chunks: {len(chunks)}")
        print(f"  Vector index size: {vec_index.count()}")
        print(f"  BM25 index size: {bm25.n_docs}")
    
    print(f"\n{'=' * 60}")
    print("Preprocessing complete! Artifacts saved to data/artifacts/")
    print("=" * 60)


if __name__ == "__main__":
    main()
