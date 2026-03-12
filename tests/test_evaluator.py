"""Tests for evaluator metric helpers."""

from rag.evaluator import (
    _chunk_relevant,
    _content_tokens,
    _evidence_text,
    compute_citation_correctness,
    compute_efficiency_metrics,
    compute_faithfulness,
    compute_retrieval_metrics,
    compute_token_f1,
    normalize_page_ref,
    llm_judge_detailed,
    select_best_pipeline,
)


def test_normalize_page_ref_variants():
    assert normalize_page_ref(43) == {43}
    assert normalize_page_ref("43") == {43}
    assert normalize_page_ref("43-45") == {43, 44, 45}
    assert normalize_page_ref("43–44") == {43, 44}
    assert normalize_page_ref("43, 50") == {43, 50}
    assert normalize_page_ref("Not explicitly marked") == set()
    assert normalize_page_ref("") == set()


def test_compute_citation_correctness():
    page_references = [
        {"pages": [42]},
        {"pages": [43, 44]},
    ]
    gold_evidence = [
        {"page_number": "42"},
        {"page_number": "44-45"},
    ]
    metrics = compute_citation_correctness(page_references, gold_evidence)
    # cited={42,43,44}, gold={42,44,45}, overlap={42,44}
    assert 0 < metrics["citation_precision"] < 1
    assert 0 < metrics["citation_recall"] < 1
    assert 0 < metrics["citation_f1"] < 1


def test_compute_faithfulness_supported_claims():
    answer = (
        "The company reports scope 1 and 2 emissions reduction. "
        "It also details methane intensity improvements."
    )
    passages = [
        "The report describes scope 1 and 2 emissions reduction targets and progress.",
        "Methane intensity has improved year over year according to monitoring data.",
    ]
    metrics = compute_faithfulness(answer, passages)
    assert metrics["supported_claim_ratio"] >= 0.5
    assert metrics["hallucination_rate"] <= 0.5


def test_compute_retrieval_metrics_page_overlap():
    retrieved = [
        {"page_span": [42, 43], "page_number": 42, "text": "materiality process", "header_hierarchy": ["DMA"]},
        {"page_span": [60], "page_number": 60, "text": "other section", "header_hierarchy": ["Other"]},
    ]
    gold = [
        {"page_number": "42", "text": "materiality process", "source_section_header": "DMA"},
        {"page_number": "44", "text": "threshold scoring", "source_section_header": "Methodology"},
    ]
    metrics = compute_retrieval_metrics(retrieved, gold, k_values=[1, 2])
    assert metrics["recall@1"] > 0
    assert metrics["precision@1"] > 0
    assert "coverage_ratio" in metrics


def test_compute_token_f1():
    # compute_token_f1 returns a dict with token_precision, token_recall, token_f1
    perfect = compute_token_f1("the cat sat on the mat", "the cat sat on the mat")
    assert perfect["token_f1"] == 1.0
    assert perfect["token_precision"] == 1.0
    assert perfect["token_recall"] == 1.0

    low = compute_token_f1("the cat sat on the mat", "a dog lay on the rug")
    assert low["token_f1"] < 0.5

    empty_pred = compute_token_f1("", "hello world")
    assert empty_pred["token_f1"] == 0.0

    empty_ref = compute_token_f1("hello world", "")
    assert empty_ref["token_f1"] == 0.0


def test_compute_efficiency_metrics():
    # Signature: compute_efficiency_metrics(latency_seconds, token_usage, n_chunks_retrieved)
    token_usage = {"prompt_tokens": 500, "completion_tokens": 200, "total_tokens": 700}
    metrics = compute_efficiency_metrics(
        latency_seconds=2.5,
        token_usage=token_usage,
        n_chunks_retrieved=10,
    )
    assert metrics["latency_seconds"] == 2.5
    assert metrics["prompt_tokens"] == 500
    assert metrics["completion_tokens"] == 200
    assert metrics["total_tokens"] == 700
    assert metrics["n_chunks_retrieved"] == 10
    assert metrics["estimated_cost_usd"] > 0


def test_select_best_pipeline_picks_highest():
    """select_best_pipeline should return the config with the highest composite score."""
    dummy_good = {
        "aggregate": {
            "n_evaluated": 5,
            "retrieval": {"avg_recall@15": 0.9, "avg_precision@15": 0.8, "avg_mrr@15": 0.85},
            "answer_quality": {"avg_token_f1": 0.8, "avg_bertscore": 0.75, "avg_llm_judge": 0.8},
            "citation": {"avg_citation_f1": 0.7},
            "hallucination": {"avg_faithfulness": 0.85},
            "efficiency": {"avg_cost_usd": 0.02, "avg_latency": 1.5},
        }
    }
    dummy_bad = {
        "aggregate": {
            "n_evaluated": 5,
            "retrieval": {"avg_recall@15": 0.3, "avg_precision@15": 0.2, "avg_mrr@15": 0.25},
            "answer_quality": {"avg_token_f1": 0.3, "avg_bertscore": 0.25, "avg_llm_judge": 0.2},
            "citation": {"avg_citation_f1": 0.2},
            "hallucination": {"avg_faithfulness": 0.3},
            "efficiency": {"avg_cost_usd": 0.10, "avg_latency": 5.0},
        }
    }
    all_results = {"good_config": dummy_good, "bad_config": dummy_bad}
    best_name, best_result = select_best_pipeline(all_results)
    assert best_name == "good_config"
    assert best_result is dummy_good


def test_select_best_pipeline_fallback_on_empty():
    """When no config has n_evaluated > 0, fall back to the first config."""
    dummy = {"aggregate": {"n_evaluated": 0}}
    all_results = {"only_config": dummy}
    best_name, best_result = select_best_pipeline(all_results)
    assert best_name == "only_config"


def test_evidence_text_prefers_text_over_exact_text():
    """_evidence_text should return 'text' if present, else 'exact_text'."""
    assert _evidence_text({"text": "hello"}) == "hello"
    assert _evidence_text({"exact_text": "world"}) == "world"
    assert _evidence_text({"text": "hello", "exact_text": "world"}) == "hello"
    assert _evidence_text({}) == ""
    assert _evidence_text({"text": "", "exact_text": "fallback"}) == "fallback"


def test_chunk_relevant_with_exact_text_evidence():
    """Ensure _chunk_relevant works with evidence using 'exact_text' field (e.g. Q3, Q12)."""
    chunk = {
        "page_span": [],
        "page_number": -1,
        "text": "the company conducted a double materiality assessment using ESRS standards",
        "header_hierarchy": [],
    }
    # Evidence using 'exact_text' instead of 'text' (as in Q3/Q12 Golden Standard)
    gold_evidence = [
        {
            "page_number": "",
            "exact_text": "double materiality assessment ESRS",
            "source_section_header": "",
        }
    ]
    assert _chunk_relevant(chunk, gold_evidence) is True


def test_retrieval_metrics_with_exact_text_evidence():
    """Retrieval metrics should work when evidence uses 'exact_text' field."""
    retrieved = [
        {
            "page_span": [],
            "page_number": -1,
            "text": "transition plan for climate change mitigation aligned with targets",
            "header_hierarchy": [],
        },
    ]
    gold = [
        {
            "page_number": "",
            "exact_text": "transition plan for climate change mitigation",
            "source_section_header": "",
        },
    ]
    metrics = compute_retrieval_metrics(retrieved, gold, k_values=[1])
    # Should match via lexical overlap since exact_text is now recognized
    assert metrics["coverage_ratio"] > 0
    assert metrics["recall@1"] > 0


def test_content_tokens_removes_stop_words():
    """_content_tokens should exclude common stop words from matching."""
    tokens = _content_tokens("the company has a plan for climate change mitigation")
    assert "the" not in tokens
    assert "has" not in tokens
    assert "a" not in tokens
    assert "for" not in tokens
    assert "climate" in tokens
    assert "change" in tokens
    assert "mitigation" in tokens
    assert "company" in tokens
    assert "plan" in tokens


def test_chunk_relevant_stop_words_no_false_positive():
    """Chunks should not match evidence purely on stop words."""
    # A chunk about corporate governance (no ESRS/climate overlap)
    chunk = {
        "page_span": [],
        "page_number": -1,
        "text": "The board of directors held a meeting to discuss the annual budget.",
        "header_hierarchy": [],
    }
    # Evidence about climate change (no page overlap, no header overlap)
    gold_evidence = [
        {
            "page_number": "",
            "text": "The company has a transition plan for climate change.",
            "source_section_header": "",
        }
    ]
    # With stop-word filtering, shared words like "the", "a", "has" should not
    # inflate the overlap enough to trigger a match.
    assert _chunk_relevant(chunk, gold_evidence) is False


def test_select_best_pipeline_normalizes_llm_judge_scale():
    """A raw 1-5 LLM judge score should be normalized before composite scoring."""
    judge_heavy_but_weak = {
        "aggregate": {
            "n_evaluated": 5,
            "retrieval": {"avg_recall@15": 0.2},
            "answer_quality": {"avg_token_f1": 0.2, "avg_bertscore": 0.2, "avg_llm_judge": 5.0},
            "citation": {"avg_citation_f1": 0.2},
            "hallucination": {"avg_faithfulness": 0.2},
            "efficiency": {"avg_cost_usd": 0.05},
        }
    }
    consistently_strong = {
        "aggregate": {
            "n_evaluated": 5,
            "retrieval": {"avg_recall@15": 0.8},
            "answer_quality": {"avg_token_f1": 0.8, "avg_bertscore": 0.8, "avg_llm_judge": 0.0},
            "citation": {"avg_citation_f1": 0.8},
            "hallucination": {"avg_faithfulness": 0.8},
            "efficiency": {"avg_cost_usd": 0.05},
        }
    }
    best_name, _best_result = select_best_pipeline(
        {"judge_heavy_but_weak": judge_heavy_but_weak, "consistently_strong": consistently_strong}
    )
    assert best_name == "consistently_strong"


def test_llm_judge_detailed_cache_key_includes_grounding_inputs(tmp_path, monkeypatch):
    """Different retrieved passages should not collide in detailed judge cache."""
    import rag.evaluator as evaluator

    calls = {"count": 0}

    def fake_chat_json(**_kwargs):
        calls["count"] += 1
        return (
            {
                "factual_correctness": {"score": 4, "rationale": "ok"},
                "completeness": {"score": 4, "rationale": "ok"},
                "grounding_faithfulness": {"score": 4, "rationale": "ok"},
                "citation_quality": {"score": 4, "rationale": "ok"},
                "compliance_assessment": {"score": 4, "rationale": "ok"},
                "overall_score": 4.0,
                "overall_rationale": "ok",
            },
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

    monkeypatch.setattr(evaluator, "EVAL_OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(evaluator, "_import_chat_json", lambda: fake_chat_json)

    kwargs = dict(
        query="Q",
        prediction="Generated answer",
        reference="Reference answer",
        predicted_compliance="Medium",
        gold_compliance="High",
    )
    llm_judge_detailed(retrieved_passages=["passage A"], **kwargs)
    llm_judge_detailed(retrieved_passages=["passage B"], **kwargs)
    # Cache hit for the first variant.
    llm_judge_detailed(retrieved_passages=["passage A"], **kwargs)

    assert calls["count"] == 2
