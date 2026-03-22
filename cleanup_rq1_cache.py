#!/usr/bin/env python3
"""
Clean up all cached RQ1 evaluation results for a fresh re-run.

Run this script before re-running the RQ1 evaluation after code changes
(e.g., after fixing semantic chunking or updating metrics).

What gets deleted:
- data/artifacts/evaluation_results/rq1/config_details/*.json  (54 per-config results)
- data/artifacts/evaluation_results/rq1/rq1_config_results.csv
- data/artifacts/evaluation_results/rq1/rq1_config_results.json
- data/artifacts/evaluation_results/rq1/rq1_winner.json
- data/artifacts/evaluation_results/rq1/rq1_errors.log
- data/artifacts/evaluation_results/rq1/chunk_stats_*.json     (9 chunk stat files)
- data/artifacts/rerank_cache.json                              (cross-encoder cache)

What is preserved:
- data/artifacts/qdrant_storage/     (vector DB — only needs rebuild for changed chunking)
- data/artifacts/embedding_cache.json (embedding cache — saves API costs)
- config/best_pipeline.yaml          (will be regenerated)

NOTE: If you changed chunking strategies (e.g., semantic chunking fix),
you should also delete the Qdrant collections for the affected strategies.
Set DELETE_QDRANT_SEMANTIC = True below to also remove semantic collections.

Usage:
    python cleanup_rq1_cache.py
    python cleanup_rq1_cache.py --include-qdrant-semantic
    python cleanup_rq1_cache.py --include-all-qdrant
"""

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RQ1_DIR = PROJECT_ROOT / "data" / "artifacts" / "evaluation_results" / "rq1"
ARTIFACTS_DIR = PROJECT_ROOT / "data" / "artifacts"
QDRANT_DIR = ARTIFACTS_DIR / "qdrant_storage"


def cleanup_rq1(include_qdrant_semantic: bool = False, include_all_qdrant: bool = False):
    deleted = 0

    # 1. Delete config_details directory.
    config_details = RQ1_DIR / "config_details"
    if config_details.exists():
        count = len(list(config_details.glob("*.json")))
        shutil.rmtree(config_details)
        print(f"  Deleted config_details/ ({count} files)")
        deleted += count

    # 2. Delete aggregate result files.
    for name in [
        "rq1_config_results.csv",
        "rq1_config_results.json",
        "rq1_winner.json",
        "rq1_errors.log",
    ]:
        path = RQ1_DIR / name
        if path.exists():
            path.unlink()
            print(f"  Deleted {name}")
            deleted += 1

    # 3. Delete chunk stats.
    for path in RQ1_DIR.glob("chunk_stats_*.json"):
        path.unlink()
        print(f"  Deleted {path.name}")
        deleted += 1

    # 4. Delete rerank cache.
    rerank_cache = ARTIFACTS_DIR / "rerank_cache.json"
    if rerank_cache.exists():
        rerank_cache.unlink()
        print(f"  Deleted rerank_cache.json")
        deleted += 1

    # 5. Optionally delete Qdrant collections for semantic chunking.
    if include_qdrant_semantic or include_all_qdrant:
        if QDRANT_DIR.exists():
            pattern = "rq1_esrs_semantic_*" if not include_all_qdrant else "rq1_esrs_*"
            for path in QDRANT_DIR.glob(pattern):
                if path.is_dir():
                    shutil.rmtree(path)
                    print(f"  Deleted Qdrant collection: {path.name}")
                    deleted += 1

    # 6. Recreate config_details directory.
    config_details.mkdir(parents=True, exist_ok=True)

    print(f"\nCleanup complete: {deleted} items deleted.")
    print("You can now re-run the evaluation with:")
    print("  jupyter notebook notebooks/05_rq1_chunking_evaluation.ipynb")
    print("  # or: python run_evaluation.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean up cached RQ1 results.")
    parser.add_argument(
        "--include-qdrant-semantic",
        action="store_true",
        help="Also delete Qdrant collections for semantic chunking (needs re-embedding).",
    )
    parser.add_argument(
        "--include-all-qdrant",
        action="store_true",
        help="Delete ALL Qdrant collections (full re-embedding required).",
    )
    args = parser.parse_args()
    cleanup_rq1(
        include_qdrant_semantic=args.include_qdrant_semantic,
        include_all_qdrant=args.include_all_qdrant,
    )
