"""Recompute RQ2 metrics from saved response JSONs without any LLM calls.

This script reads the persisted response files, re-evaluates each sample
using the current metric code, and regenerates all output artifacts:
  - rq2_sample_results.csv
  - rq2_paired_results.csv
  - rq2_question_level_results.csv
  - rq2_summary.json
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.config.settings import (
    RQ2_CONTEXT_MODE,
    RQ2_RESULTS_DIR,
    RQ2_RESULTS_VERSION,
    build_rq2_audit_manifest,
    get_generation_model,
)
from rag.data.loaders import load_generation_samples
from rag.evaluation.rq2_metrics import (
    build_context_summary,
    build_question_manifest,
    build_rq2_summary,
    build_sample_result,
    sample_results_to_rows,
)
from rag.generation.esrs_context import ESRSContextExtractor
from rag.types import GeneratedAnswer, GenerationSample, RetrievedEvidenceChunk, RQ2Summary


def _load_response_payloads(responses_dir: Path) -> list[dict]:
    """Load all response JSON files from both condition directories."""
    payloads: list[dict] = []
    for condition_dir in sorted(responses_dir.iterdir()):
        if not condition_dir.is_dir():
            continue
        for path in sorted(condition_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            payloads.append(payload)
    return payloads


def _build_sample_lookup(samples: list[GenerationSample]) -> dict[tuple[str, str], GenerationSample]:
    return {(s.question_id, s.company): s for s in samples}


def main() -> None:
    results_dir = RQ2_RESULTS_DIR / RQ2_CONTEXT_MODE
    responses_dir = results_dir / "responses"

    if not responses_dir.exists():
        print(f"No responses directory found at {responses_dir}")
        sys.exit(1)

    print(f"[Recompute] Loading generation samples from golden standard...")
    samples = load_generation_samples()
    sample_lookup = _build_sample_lookup(samples)
    print(f"[Recompute] Loaded {len(samples)} samples")

    print(f"[Recompute] Loading response payloads from {responses_dir}...")
    payloads = _load_response_payloads(responses_dir)
    print(f"[Recompute] Found {len(payloads)} response files")

    print("[Recompute] Evaluating each response...")
    sample_results = []
    skipped = 0
    for payload in payloads:
        qid = payload.get("question_id", "")
        company = payload.get("company", "")
        condition = payload.get("condition", "")
        key = (qid, company)
        sample = sample_lookup.get(key)
        if sample is None:
            skipped += 1
            continue

        answer_data = payload.get("generated_answer", {})
        answer = GeneratedAnswer(
            direct_answer=str(answer_data.get("direct_answer", "")),
            compliance_level=str(answer_data.get("compliance_level", "")),
            compliance_rationale=str(answer_data.get("compliance_rationale", "")),
            key_findings=list(answer_data.get("key_findings", [])),
            disclosure_gaps=list(answer_data.get("disclosure_gaps", [])),
            cited_pages=[str(p) for p in answer_data.get("cited_pages", [])],
            raw_response=str(answer_data.get("raw_response", "")),
            parse_error=answer_data.get("parse_error"),
        )

        chunks = payload.get("retrieved_chunks", [])
        evidence_texts = [str(c.get("text", "")) for c in chunks]
        retrieval_pages = []
        for c in chunks:
            ps = c.get("page_start")
            pe = c.get("page_end")
            if ps is None and pe is None:
                continue
            s = "" if ps is None else str(ps)
            e = "" if pe is None else str(pe)
            label = s if s == e or not e else f"{s}-{e}"
            if label and label not in retrieval_pages:
                retrieval_pages.append(label)

        result = build_sample_result(
            condition=condition,
            sample=sample,
            answer=answer,
            evidence_texts=evidence_texts,
            retrieval_pages=retrieval_pages,
        )
        sample_results.append(result)

    if skipped:
        print(f"[Recompute] Skipped {skipped} responses (no matching sample)")
    print(f"[Recompute] Evaluated {len(sample_results)} samples")

    print("[Recompute] Building summary, paired results, and statistics...")
    (
        aggregate_metrics,
        per_standard_metrics,
        majority,
        paired_rows,
        question_rows,
        paired_statistics,
    ) = build_rq2_summary(sample_results)

    # Build context summary
    extractor = ESRSContextExtractor(
        cache_path=results_dir / "context_cache.json",
        require_openai=False,
    )
    question_samples = {}
    for s in samples:
        question_samples.setdefault(s.question_id, s)
    context_results = {
        qid: extractor.get_or_extract(s) for qid, s in sorted(question_samples.items())
    }
    context_summary = build_context_summary(context_results)
    question_manifest = build_question_manifest(samples)
    audit_manifest = build_rq2_audit_manifest()

    # Assemble full summary
    summary = RQ2Summary(
        results_version=RQ2_RESULTS_VERSION,
        model=get_generation_model(),
        best_pipeline_name="fixed_1024_hybrid_rerank",
        n_samples=len(samples),
        n_conditions=2,
        aggregate_metrics=aggregate_metrics,
        per_standard_metrics=per_standard_metrics,
        majority_baseline=majority,
    )
    summary_payload = asdict(summary)
    summary_payload["context_mode"] = RQ2_CONTEXT_MODE
    summary_payload["temperature"] = 0.1
    summary_payload["condition_parallelism"] = 2
    summary_payload["audit_manifest"] = audit_manifest
    summary_payload["question_manifest"] = question_manifest
    summary_payload["paired_statistics"] = paired_statistics
    summary_payload["context_summary"] = context_summary

    # Persist
    print("[Recompute] Persisting artifacts...")

    # Sample CSV
    rows = sample_results_to_rows(sample_results)
    csv_path = results_dir / "rq2_sample_results.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    # Paired CSV
    if paired_rows:
        paired_path = results_dir / "rq2_paired_results.csv"
        with paired_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(paired_rows[0].keys()))
            writer.writeheader()
            writer.writerows(paired_rows)

    # Question-level CSV
    if question_rows:
        q_path = results_dir / "rq2_question_level_results.csv"
        with q_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(question_rows[0].keys()))
            writer.writeheader()
            writer.writerows(question_rows)

    # Summary JSON
    (results_dir / "rq2_summary.json").write_text(
        json.dumps(summary_payload, indent=2), encoding="utf-8",
    )

    print("[Recompute] Done.")
    print(f"  Samples: {len(sample_results)}")
    print(f"  Pairs: {len(paired_rows)}")
    print(f"  Questions: {len(question_rows)}")

    # Print key results
    for cond in ("with_esrs_context", "without_esrs_context"):
        agg = aggregate_metrics[cond]
        print(f"\n  {cond}:")
        print(f"    macro_f1={agg['macro_f1']:.4f}  balanced_acc={agg['balanced_accuracy']:.4f}  "
              f"exact_acc={agg['exact_accuracy']:.4f}  kappa={agg['weighted_kappa']:.4f}  "
              f"scored={agg['n_scored']}/{agg['n_total']}")

    ps = paired_statistics
    cb = ps["cluster_bootstrap"]["macro_f1"]
    qt = ps["question_level_tests"]["macro_f1"]
    print(f"\n  Primary endpoint (macro-F1 delta):")
    print(f"    delta={cb['mean_delta']:.4f}  CI=[{cb['ci_lower']:.4f}, {cb['ci_upper']:.4f}]  "
          f"sign_p={qt['p_value']:.4f}")


if __name__ == "__main__":
    main()
