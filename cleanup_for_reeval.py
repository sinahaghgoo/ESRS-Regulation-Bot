#!/usr/bin/env python3
"""
Cleanup script to prepare for re-evaluation after RQ1 improvements.

This script removes stale caches that need to be regenerated after
the code changes to text normalization, retrieval, chunking, and
relevance matching.

IMPORTANT: This preserves the embedding cache and Qdrant vector
collections, which are expensive to regenerate (API calls). Only
result caches and rerank scores are cleared.

Usage:
    python cleanup_for_reeval.py
    python cleanup_for_reeval.py --include-semantic-chunks  # also clear semantic chunk Qdrant collections
"""

import argparse
import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
RQ1_DIR = PROJECT_ROOT / "data" / "artifacts" / "evaluation_results" / "rq1"
ARTIFACTS_DIR = PROJECT_ROOT / "data" / "artifacts"


def backup_results():
    """Backup existing results before cleanup."""
    backup_dir = RQ1_DIR / "pre_improvement_backup"
    if backup_dir.exists():
        print(f"  Backup already exists: {backup_dir}")
        return

    backup_dir.mkdir(parents=True, exist_ok=True)

    # Backup config details
    details_dir = RQ1_DIR / "config_details"
    if details_dir.exists():
        backup_details = backup_dir / "config_details"
        shutil.copytree(details_dir, backup_details)
        print(f"  Backed up {len(list(backup_details.glob('*.json')))} config detail files")

    # Backup result files
    for pattern in ["rq1_config_results.*", "rq1_winner.json"]:
        for f in RQ1_DIR.glob(pattern):
            shutil.copy2(f, backup_dir / f.name)
            print(f"  Backed up {f.name}")

    # Save baseline metrics summary
    results_csv = RQ1_DIR / "rq1_config_results.csv"
    if results_csv.exists():
        import csv
        with open(results_csv) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        summary = {
            "total_configs": len(rows),
            "top_5": [
                {"config_id": r["config_id"], "recall@15": float(r["recall@15"])}
                for r in sorted(rows, key=lambda x: float(x["recall@15"]), reverse=True)[:5]
            ],
        }
        (backup_dir / "baseline_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(f"  Saved baseline summary")

    print(f"  Backup complete: {backup_dir}")


def cleanup_results():
    """Remove stale evaluation results."""
    details_dir = RQ1_DIR / "config_details"
    if details_dir.exists():
        count = len(list(details_dir.glob("*.json")))
        shutil.rmtree(details_dir)
        details_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Cleared {count} config detail files")

    for pattern in ["rq1_config_results.*", "rq1_winner.json"]:
        for f in RQ1_DIR.glob(pattern):
            f.unlink()
            print(f"  Removed {f.name}")


def cleanup_rerank_cache():
    """Remove stale rerank cache (reranker input format changed)."""
    cache_path = ARTIFACTS_DIR / "rerank_cache.json"
    if cache_path.exists():
        size_mb = cache_path.stat().st_size / 1024 / 1024
        cache_path.unlink()
        print(f"  Removed rerank_cache.json ({size_mb:.1f} MB)")
    else:
        print("  No rerank cache to remove")


def cleanup_semantic_qdrant():
    """Remove semantic chunking Qdrant collections (chunk merge changed)."""
    qdrant_dir = ARTIFACTS_DIR / "qdrant_storage" / "collection"
    if not qdrant_dir.exists():
        print("  No Qdrant collections found")
        return

    for coll_dir in sorted(qdrant_dir.iterdir()):
        if "semantic" in coll_dir.name:
            shutil.rmtree(coll_dir)
            print(f"  Removed Qdrant collection: {coll_dir.name}")


def main():
    parser = argparse.ArgumentParser(description="Cleanup for RQ1 re-evaluation")
    parser.add_argument(
        "--include-semantic-chunks",
        action="store_true",
        help="Also clear semantic chunking Qdrant collections (requires re-embedding)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("RQ1 Re-Evaluation Cleanup")
    print("=" * 60)

    print("\n1. Backing up existing results...")
    backup_results()

    print("\n2. Clearing stale evaluation results...")
    cleanup_results()

    print("\n3. Clearing rerank cache (input format changed)...")
    cleanup_rerank_cache()

    if args.include_semantic_chunks:
        print("\n4. Clearing semantic chunking Qdrant collections...")
        cleanup_semantic_qdrant()
    else:
        print("\n4. Skipping semantic Qdrant cleanup (use --include-semantic-chunks)")

    print("\n" + "=" * 60)
    print("Cleanup complete!")
    print()
    print("PRESERVED (reusable):")
    print("  - Embedding cache (data/artifacts/embedding_cache.json)")
    print("  - Fixed/structure-aware Qdrant collections")
    if not args.include_semantic_chunks:
        print("  - Semantic Qdrant collections (chunk merge change is minor)")
    print()
    print("NEXT STEP:")
    print("  Open notebooks/05_rq1_chunking_evaluation.ipynb")
    print("  Set DRY_RUN = False and run all cells")
    print("=" * 60)


if __name__ == "__main__":
    main()
