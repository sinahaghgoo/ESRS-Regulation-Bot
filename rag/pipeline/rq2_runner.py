"""RQ2 paired generation experiment on top of the winning RQ1 retriever."""

from __future__ import annotations

import csv
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from rag.chunking.strategies import build_chunks_for_config
from rag.config.settings import (
    RQ2_RESULTS_DIR,
    RQ2_RESULTS_VERSION,
    PipelineConfig,
    ensure_dirs,
    get_generation_model,
    load_best_pipeline_config,
)
from rag.data.loaders import load_generation_samples, load_reports
from rag.evaluation.rq2_metrics import (
    build_rq2_summary,
    build_sample_result,
    sample_results_to_rows,
)
from rag.generation import AnswerGenerator, ESRSContextExtractor
from rag.indexing import EmbeddingService, InMemoryDenseIndex
from rag.retrieval.methods import Retriever
from rag.types import GenerationSample, RetrievedEvidenceChunk, RQ2Summary

if TYPE_CHECKING:
    from rag.pipeline.rq1_runner import RQ1PipelineRunner

_SAFE_RE = re.compile(r"[^a-z0-9]+")


def _slug(value: str) -> str:
    return _SAFE_RE.sub("_", value.lower()).strip("_")


class RQ2PipelineRunner:
    """Execute the RQ2 ESRS-context ablation with resumable persistence."""

    CONDITIONS = ("with_esrs_context", "without_esrs_context")

    def __init__(
        self,
        *,
        dry_run: bool = False,
        question_id: str | None = None,
        company: str | None = None,
        require_openai: bool = True,
        results_dir: Path | None = None,
        samples: list[GenerationSample] | None = None,
        retriever=None,
        best_config: PipelineConfig | None = None,
        best_pipeline_name: str | None = None,
        rq1_runner: "RQ1PipelineRunner" | None = None,
        context_extractor: ESRSContextExtractor | None = None,
        answer_generator: AnswerGenerator | None = None,
        condition_parallelism: int = 2,
    ) -> None:
        ensure_dirs()
        self.require_openai = require_openai
        self.results_dir = results_dir or RQ2_RESULTS_DIR
        self.responses_dir = self.results_dir / "responses"
        self.responses_dir.mkdir(parents=True, exist_ok=True)
        self.context_extractor = context_extractor or ESRSContextExtractor(require_openai=require_openai)
        self.answer_generator = answer_generator or AnswerGenerator(require_openai=require_openai)
        self.condition_parallelism = max(1, int(condition_parallelism))
        self._parallel_answer_generators = answer_generator is None and self.condition_parallelism > 1
        loaded_samples = samples if samples is not None else load_generation_samples()
        self.samples = self._select_samples(loaded_samples, dry_run=dry_run, question_id=question_id, company=company)
        if not self.samples:
            raise RuntimeError("RQ2 runner has no samples to evaluate.")

        if best_config is not None:
            self.best_config = best_config
            self.best_pipeline_name = best_pipeline_name or best_config.config_id
        else:
            best_config_dict = load_best_pipeline_config()
            self.best_pipeline_name = best_pipeline_name or best_config_dict.get("_best_pipeline_name", "best_pipeline")
            self.best_config = PipelineConfig(
                chunk_strategy=best_config_dict["chunk_strategy"],
                chunk_size=int(best_config_dict["chunk_size"]),
                retrieval_method=best_config_dict["retrieval_method"],
                chunk_overlap_ratio=float(best_config_dict.get("chunk_overlap_ratio", 0.125)),
                top_k=int(best_config_dict.get("top_k", 15)),
                rrf_k=int(best_config_dict.get("rrf_k", 60)),
                rerank_model=best_config_dict.get("rerank_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            )

        self.rq1_runner = rq1_runner
        self._retrieval_backend = None
        if retriever is not None:
            self.retriever = retriever
        elif self.rq1_runner is not None:
            self.retriever = self.rq1_runner.prepare_for_config(self.best_config)
        else:
            self.retriever = self._build_lightweight_retriever(require_openai=require_openai)

    @staticmethod
    def _select_samples(
        samples: list[GenerationSample],
        *,
        dry_run: bool,
        question_id: str | None,
        company: str | None,
    ) -> list[GenerationSample]:
        selected = samples
        if question_id:
            selected = [sample for sample in selected if sample.question_id == question_id]
        if company:
            selected = [sample for sample in selected if sample.company == company]
        selected = sorted(selected, key=lambda sample: (sample.question_id, sample.company))
        if dry_run and selected:
            first = selected[0]
            selected = [sample for sample in selected if sample.question_id == first.question_id and sample.company == first.company]
        return selected

    def close(self) -> None:
        if self.rq1_runner is not None:
            self.rq1_runner.close()
        if self._retrieval_backend is not None and hasattr(self._retrieval_backend, "close"):
            self._retrieval_backend.close()

    def _build_lightweight_retriever(self, *, require_openai: bool) -> Retriever:
        """Build the winning RQ1 retriever without opening the local Qdrant store.

        RQ2 evaluates the same fixed question set repeatedly. For that setup,
        a compact NumPy-backed dense index is materially lighter than the
        persisted local Qdrant database while preserving the same retrieval
        method, chunking, BM25 fusion, and reranking logic.
        """
        print(f"[RQ2] Building lightweight retriever for {self.best_pipeline_name}")
        reports = load_reports()
        chunks, _ = build_chunks_for_config(reports, self.best_config)

        embedder = EmbeddingService(
            require_openai=require_openai,
            load_cache=False,
        )
        dense_index = InMemoryDenseIndex(dims=embedder.dims)
        collection_name = dense_index.collection_name(
            self.best_config.chunk_strategy,
            self.best_config.chunk_size,
        )

        if dense_index.collection_point_count(collection_name) == 0:
            print(f"[RQ2] Creating compact dense vector cache for {collection_name}")
            vectors = []
            batch_size = 64
            texts = [chunk.retrieval_text for chunk in chunks]
            for start in range(0, len(texts), batch_size):
                end = min(start + batch_size, len(texts))
                print(f"[RQ2] Embedding chunk batch {start + 1}-{end} / {len(texts)}")
                vectors.extend(embedder.embed_texts(texts[start:end]))
            dense_index.recreate_collection(collection_name)
            dense_index.upsert_chunks(collection_name, chunks, vectors)
        else:
            print(f"[RQ2] Reusing compact dense vector cache for {collection_name}")

        self._retrieval_backend = dense_index
        return Retriever(
            chunks=chunks,
            config=self.best_config,
            embedder=embedder,
            qdrant=dense_index,
        )

    def _response_path(self, condition: str, sample: GenerationSample) -> Path:
        return self.responses_dir / condition / f"{sample.question_id}__{_slug(sample.company)}.json"

    @staticmethod
    def _serialize_hits(hits) -> list[RetrievedEvidenceChunk]:
        out: list[RetrievedEvidenceChunk] = []
        for hit in hits:
            out.append(
                RetrievedEvidenceChunk(
                    chunk_id=hit.chunk.chunk_id,
                    text=hit.chunk.text,
                    score=float(hit.score),
                    rank=int(hit.rank),
                    page_start=hit.chunk.page_start,
                    page_end=hit.chunk.page_end,
                    header_hierarchy=list(hit.chunk.header_hierarchy),
                )
            )
        return out

    def _retrieve_chunks(self, sample: GenerationSample) -> list[RetrievedEvidenceChunk]:
        hits = self.retriever.retrieve(sample.question, company=sample.company, top_k=self.best_config.top_k)
        return self._serialize_hits(hits)

    def _write_response_payload(self, condition: str, sample: GenerationSample, payload: dict) -> None:
        path = self._response_path(condition, sample)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load_response_payload(self, condition: str, sample: GenerationSample) -> dict | None:
        path = self._response_path(condition, sample)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("results_version") != RQ2_RESULTS_VERSION:
            return None
        return payload

    def _generate_or_load_condition(self, *, condition: str, sample: GenerationSample, retrieved_chunks: list[RetrievedEvidenceChunk], esrs_context: str) -> dict:
        cached = self._load_response_payload(condition, sample)
        if cached is not None:
            return cached

        answer = self.answer_generator.generate(
            sample=sample,
            retrieved_chunks=retrieved_chunks,
            esrs_context=esrs_context,
            condition=condition,
        )
        payload = {
            "results_version": RQ2_RESULTS_VERSION,
            "condition": condition,
            "model": get_generation_model(),
            "question_id": sample.question_id,
            "question": sample.question,
            "esrs_standard": sample.esrs_standard,
            "esrs_disclosure_id": sample.esrs_disclosure_id,
            "company": sample.company,
            "retrieved_chunks": [asdict(chunk) for chunk in retrieved_chunks],
            "esrs_context": esrs_context,
            "generated_answer": asdict(answer),
        }
        self._write_response_payload(condition, sample, payload)
        return payload

    def _generate_or_load_condition_with_fresh_generator(
        self,
        *,
        condition: str,
        sample: GenerationSample,
        retrieved_chunks: list[RetrievedEvidenceChunk],
        esrs_context: str,
    ) -> dict:
        cached = self._load_response_payload(condition, sample)
        if cached is not None:
            return cached

        generator = AnswerGenerator(
            model=self.answer_generator.model,
            temperature=self.answer_generator.temperature,
            max_completion_tokens=self.answer_generator.max_completion_tokens,
            require_openai=self.require_openai,
        )
        answer = generator.generate(
            sample=sample,
            retrieved_chunks=retrieved_chunks,
            esrs_context=esrs_context,
            condition=condition,
        )
        payload = {
            "results_version": RQ2_RESULTS_VERSION,
            "condition": condition,
            "model": get_generation_model(),
            "question_id": sample.question_id,
            "question": sample.question,
            "esrs_standard": sample.esrs_standard,
            "esrs_disclosure_id": sample.esrs_disclosure_id,
            "company": sample.company,
            "retrieved_chunks": [asdict(chunk) for chunk in retrieved_chunks],
            "esrs_context": esrs_context,
            "generated_answer": asdict(answer),
        }
        self._write_response_payload(condition, sample, payload)
        return payload

    def _evaluate_payload(self, sample: GenerationSample, payload: dict):
        answer_payload = payload.get("generated_answer", {})
        # Reconstruct from persisted payload to keep the evaluator independent
        # from the live generation call path.
        from rag.types import GeneratedAnswer

        generated_answer = GeneratedAnswer(
            direct_answer=str(answer_payload.get("direct_answer", "")),
            compliance_level=str(answer_payload.get("compliance_level", "")),
            compliance_rationale=str(answer_payload.get("compliance_rationale", "")),
            key_findings=list(answer_payload.get("key_findings", [])),
            disclosure_gaps=list(answer_payload.get("disclosure_gaps", [])),
            cited_pages=[str(page) for page in answer_payload.get("cited_pages", [])],
            raw_response=str(answer_payload.get("raw_response", "")),
            parse_error=answer_payload.get("parse_error"),
        )
        return build_sample_result(
            condition=str(payload.get("condition", "")),
            sample=sample,
            answer=generated_answer,
        )

    def _persist_evaluation(
        self,
        sample_results,
        paired_rows: list[dict],
        question_rows: list[dict],
        summary_payload: dict,
    ) -> None:
        # Sample-level CSV
        csv_path = self.results_dir / "rq2_sample_results.csv"
        rows = sample_results_to_rows(sample_results)
        if rows:
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

        # Paired results CSV
        if paired_rows:
            paired_path = self.results_dir / "rq2_paired_results.csv"
            with paired_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0].keys()))
                writer.writeheader()
                writer.writerows(paired_rows)

        # Question-level results CSV
        if question_rows:
            question_path = self.results_dir / "rq2_question_level_results.csv"
            with question_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(question_rows[0].keys()))
                writer.writeheader()
                writer.writerows(question_rows)

        # Summary JSON
        (self.results_dir / "rq2_summary.json").write_text(
            json.dumps(summary_payload, indent=2),
            encoding="utf-8",
        )

    def run(self) -> dict:
        """Execute RQ2 and persist all per-sample and aggregate artifacts."""
        try:
            grouped_by_qid: dict[str, GenerationSample] = {}
            for sample in self.samples:
                grouped_by_qid.setdefault(sample.question_id, sample)

            contexts = {}
            for question_id, sample in grouped_by_qid.items():
                print(f"[RQ2] ESRS context -> {question_id} / {sample.esrs_standard}")
                contexts[question_id] = self.context_extractor.get_or_extract(sample)

            sample_results = []
            for sample in self.samples:
                print(f"[RQ2] Retrieving evidence for {sample.question_id} / {sample.company}")
                retrieved_chunks = self._retrieve_chunks(sample)
                with_context = contexts[sample.question_id].context_text
                without_context = ""
                condition_specs = (
                    ("with_esrs_context", with_context),
                    ("without_esrs_context", without_context),
                )

                if self._parallel_answer_generators:
                    with ThreadPoolExecutor(max_workers=self.condition_parallelism) as executor:
                        futures = {}
                        for condition, context_text in condition_specs:
                            print(f"[RQ2] {condition} -> {sample.question_id} / {sample.company}")
                            futures[
                                executor.submit(
                                    self._generate_or_load_condition_with_fresh_generator,
                                    condition=condition,
                                    sample=sample,
                                    retrieved_chunks=retrieved_chunks,
                                    esrs_context=context_text,
                                )
                            ] = condition
                        for future in as_completed(futures):
                            payload = future.result()
                            sample_results.append(self._evaluate_payload(sample, payload))
                else:
                    for condition, context_text in condition_specs:
                        print(f"[RQ2] {condition} -> {sample.question_id} / {sample.company}")
                        payload = self._generate_or_load_condition(
                            condition=condition,
                            sample=sample,
                            retrieved_chunks=retrieved_chunks,
                            esrs_context=context_text,
                        )
                        sample_results.append(self._evaluate_payload(sample, payload))

            (
                aggregate_metrics,
                per_standard_metrics,
                majority,
                paired_rows,
                question_rows,
                paired_statistics,
            ) = build_rq2_summary(sample_results)

            summary = RQ2Summary(
                results_version=RQ2_RESULTS_VERSION,
                model=get_generation_model(),
                best_pipeline_name=self.best_pipeline_name,
                n_samples=len(self.samples),
                n_conditions=len(self.CONDITIONS),
                aggregate_metrics=aggregate_metrics,
                per_standard_metrics=per_standard_metrics,
                majority_baseline=majority,
            )
            summary_payload = asdict(summary)
            summary_payload["paired_statistics"] = paired_statistics
            self._persist_evaluation(
                sample_results, paired_rows, question_rows, summary_payload,
            )
            return summary_payload
        finally:
            self.close()
