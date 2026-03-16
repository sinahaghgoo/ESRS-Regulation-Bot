"""Run the full RQ1 retrieval evaluation matrix."""

from __future__ import annotations

import argparse
import json

from rag.pipeline.rq1_runner import RQ1PipelineRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RQ1 retrieval evaluation.")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate only one config.")
    args = parser.parse_args()

    runner = RQ1PipelineRunner(dry_run=args.dry_run)
    summary = runner.run()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

