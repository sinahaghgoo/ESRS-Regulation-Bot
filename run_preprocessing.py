"""RQ1 preprocessing: chunking + embedding + indexing per strategy/size."""

from __future__ import annotations

import argparse
import json

from rag.config.settings import PipelineConfig, build_rq1_experiment_grid
from rag.pipeline.rq1_runner import RQ1PipelineRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare chunking and Qdrant indices for RQ1.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare only one configuration.")
    args = parser.parse_args()

    runner = RQ1PipelineRunner(dry_run=args.dry_run)
    grid = build_rq1_experiment_grid()
    if args.dry_run:
        grid = [grid[0]]

    prepared = set()
    for cfg in grid:
        key = (cfg.chunk_strategy, cfg.chunk_size)
        if key in prepared:
            continue
        runner.prepare_for_config(cfg)
        prepared.add(key)
        print(f"Prepared index for {cfg.chunk_strategy} / {cfg.chunk_size}")

    print(json.dumps({"prepared_indices": len(prepared)}, indent=2))


if __name__ == "__main__":
    main()

