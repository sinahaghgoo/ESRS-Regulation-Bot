"""Evaluation helpers for RQ1 retrieval metrics and RQ2 generation metrics.

Imports are deferred so that individual modules can be loaded without
pulling in every optional dependency (e.g., sklearn).
"""

# Lazy imports — individual modules can still be imported directly.

def __getattr__(name: str):
    """Lazily import symbols on first access."""
    _rq2_symbols = {
        "RQ2AggregateResult",
        "aggregate_condition_metrics",
        "aggregate_results",
        "build_context_summary",
        "build_question_manifest",
        "build_rq2_summary",
        "build_sample_result",
        "evaluate_sample",
        "validate_rq2_artifacts",
    }
    _rq1_symbols = {"evaluate_configuration"}

    if name in _rq2_symbols:
        from . import rq2_metrics
        return getattr(rq2_metrics, name)
    if name in _rq1_symbols:
        from . import metrics
        return getattr(metrics, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
