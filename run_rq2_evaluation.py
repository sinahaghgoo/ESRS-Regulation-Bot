"""Run the RQ2 ESRS-context ablation experiment."""

from __future__ import annotations

import argparse
import json

from rag.pipeline.rq2_runner import RQ2PipelineRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RQ2 ESRS-context ablation.")
    parser.add_argument("--dry-run", action="store_true", help="Run only one question-company pair.")
    parser.add_argument("--question-id", type=str, help="Restrict the run to one question ID.")
    parser.add_argument("--company", type=str, help="Restrict the run to one company.")
    args = parser.parse_args()

    runner = RQ2PipelineRunner(
        dry_run=args.dry_run,
        question_id=args.question_id,
        company=args.company,
    )
    summary = runner.run()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
