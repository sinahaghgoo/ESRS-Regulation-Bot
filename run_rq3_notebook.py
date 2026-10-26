#!/usr/bin/env python3
"""Execute the RQ3 notebook end-to-end and persist outputs back to disk.

Uses ``nbclient`` to run every code cell, captures the resulting outputs
into the .ipynb file, and surfaces a summary of any errors.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
NB_PATH = PROJECT_ROOT / "notebooks/05_rq3_generation_quality_evaluation.ipynb"

# Load .env so the kernel inherits API keys
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            _k = _k.strip()
            if _k:
                os.environ.setdefault(_k, _v.strip().strip('"').strip("'"))

import nbformat  # noqa: E402
from nbclient import NotebookClient  # noqa: E402


def main() -> None:
    nb = nbformat.read(NB_PATH, as_version=4)
    client = NotebookClient(
        nb,
        timeout=900,  # 15 min per cell
        kernel_name="python3",
        resources={"metadata": {"path": str(NB_PATH.parent)}},
    )
    print(f"Executing {NB_PATH.name} ({len(nb.cells)} cells)…", flush=True)
    try:
        client.execute()
    except Exception as exc:
        print(f"\n*** Notebook execution FAILED: {exc}", flush=True)
        nbformat.write(nb, NB_PATH)
        sys.exit(1)

    nbformat.write(nb, NB_PATH)
    print("\nNotebook executed and written back successfully.", flush=True)

    # Brief summary of the newly persisted summary JSON, if it exists
    summary_path = PROJECT_ROOT / "data/artifacts/evaluation_results/rq3/rq3_summary.json"
    if summary_path.exists():
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        print("\n=== RQ3 summary ===")
        cm = s.get("citation_metrics", {})
        print(f"Method: {cm.get('method')}, tolerance ±{cm.get('tolerance_pages')}")
        for k in ("precision", "recall", "f1"):
            v = cm.get(k, {})
            if isinstance(v, dict) and "mean" in v:
                print(f"  {k}: {v['mean']:.4f}  [{v.get('ci_lower', 0):.4f}, {v.get('ci_upper', 0):.4f}]")
        cma = s.get("cross_metric_alignment", {})
        if cma:
            print(f"Cross-metric: Pearson r = {cma.get('pearson_cite_p_vs_judge_cu', 0):+.3f}, "
                  f"Spearman ρ = {cma.get('spearman_cite_p_vs_judge_cu', 0):+.3f}")


if __name__ == "__main__":
    main()
