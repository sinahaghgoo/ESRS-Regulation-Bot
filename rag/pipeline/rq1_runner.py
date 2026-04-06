"""
RQ1 experiment orchestration and runtime retrieval interface.

Runs a full 3 x 3 x 6 factorial experiment (54 configurations):
- 3 chunking strategies: fixed, structure_aware, semantic
- 3 chunk sizes: 256, 512, 1024 tokens
- 6 retrieval methods: dense, dense_rerank, sparse, sparse_rerank,
  hybrid_rrf, hybrid_rerank

For each configuration, the pipeline:
1. Builds chunks for the strategy/size pair (cached across retrieval methods).
2. Embeds chunks and indexes them in Qdrant.
3. Retrieves top-15 chunks per golden-standard sample.
4. Computes Recall@k, Precision@k, MRR@10, and Budgeted Recall@4096.

Results are persisted incrementally so interrupted runs can resume.
"""

from __future__ import annotations

import csv
import gc
import json
import logging
import re
import traceback
from dataclasses import asdict
from pathlib import Path

from rag.chunking.strategies import build_chunks_for_config
from rag.config.settings import (
    PipelineConfig,
    RQ1_RESULTS_VERSION,
    RQ1_RESULTS_DIR,
    build_rq1_experiment_grid,
    ensure_dirs,
    pipeline_config_to_dict,
    save_best_pipeline_config,
)
from rag.data.loaders import load_golden_samples, load_reports
from rag.evaluation.metrics import evaluate_configuration
from rag.indexing.embedder import EmbeddingService
from rag.indexing.qdrant_index import QdrantChunkIndex
from rag.retrieval.methods import Retriever
from rag.types import ConfigResult

_logger = logging.getLogger(__name__)
_RESULTS_VERSION = RQ1_RESULTS_VERSION

# Page marker regex used to strip markers before normalising report text.
_PAGE_RE = re.compile(r"<!--\s*Report Page\s+\d+\s*-->")


class RQ1PipelineRunner:
    """Build indexes and evaluate all RQ1 configurations.

    Usage:
        runner = RQ1PipelineRunner()
        summary = runner.run()  # evaluates all 54 configurations
    """

    def __init__(
        self,
        dry_run: bool = False,
        require_openai: bool = True,
        load_embedding_cache: bool = True,
    ) -> None:
        ensure_dirs()
        self.dry_run = dry_run
        self.reports = load_reports()
        self.samples = load_golden_samples()
        self.embedder = EmbeddingService(
            require_openai=require_openai and not dry_run,
            load_cache=load_embedding_cache,
        )
        self.qdrant = QdrantChunkIndex(dims=self.embedder.dims)

        # Caches: chunks are shared across retrieval methods with the same
        # strategy/size; retrievers are config-specific.
        self._chunks_cache: dict[tuple[str, int], list] = {}
        self._retriever_cache: dict[str, Retriever] = {}
        self._config_details_dir = RQ1_RESULTS_DIR / "config_details"
        self._config_details_dir.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        """Release local resources such as the Qdrant file lock."""
        try:
            self.qdrant.close()
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()

    def prepare_for_config(self, config: PipelineConfig) -> Retriever:
        """Build or retrieve the Retriever for a given configuration.

        Chunks and Qdrant collections are cached by (strategy, size) so
        they are only built once per pair, even across multiple retrieval
        methods.
        """
        dense_methods = {"dense", "dense_rerank", "hybrid_rrf", "hybrid_rerank"}
        key = (config.chunk_strategy, config.chunk_size)
        if key not in self._chunks_cache:
            chunks, stats = build_chunks_for_config(self.reports, config)
            self._chunks_cache[key] = chunks
            stats_path = (
                RQ1_RESULTS_DIR
                / f"chunk_stats_{config.chunk_strategy}_{config.chunk_size}.json"
            )
            stats_path.write_text(
                json.dumps(asdict(stats), indent=2), encoding="utf-8"
            )

        if config.retrieval_method in dense_methods:
            collection = self.qdrant.collection_name(
                config.chunk_strategy, config.chunk_size
            )
            existing_count = self.qdrant.collection_point_count(collection)
            if existing_count > 0:
                print(
                    f"  Qdrant collection {collection} already has "
                    f"{existing_count} vectors — reusing"
                )
            else:
                print(
                    f"  Building Qdrant collection {collection} "
                    f"({len(self._chunks_cache[key])} chunks)"
                )
                vectors = self.embedder.embed_texts(
                    [c.retrieval_text for c in self._chunks_cache[key]]
                )
                self.qdrant.recreate_collection(collection)
                self.qdrant.upsert_chunks(collection, self._chunks_cache[key], vectors)

        rid = config.config_id
        if rid not in self._retriever_cache:
            self._retriever_cache[rid] = Retriever(
                chunks=self._chunks_cache[key],
                config=config,
                embedder=self.embedder,
                qdrant=self.qdrant,
            )
        return self._retriever_cache[rid]

    def _build_report_text_index(self) -> dict[str, str]:
        """Build normalised report text index for evaluation matching.

        Strips page markers BEFORE normalising so the normalised text
        matches the chunks (which are built from page-marker-free text).
        """
        from rag.data.text_utils import normalize_text

        return {
            k: normalize_text(_PAGE_RE.sub("", v.raw_text))
            for k, v in self.reports.items()
        }

    def run(self) -> dict:
        """Execute the full RQ1 evaluation matrix and persist results.

        Returns a summary dict with the winning configuration and its metrics.
        """
        try:
            grid = build_rq1_experiment_grid()
            if self.dry_run:
                grid = [grid[0]]

            report_text_by_company = self._build_report_text_index()
            config_results = self._load_existing_results()
            completed_ids = {res.config_id for res in config_results}
            error_log = RQ1_RESULTS_DIR / "rq1_errors.log"
            failed_configs: list[str] = []

            for config in grid:
                if config.config_id in completed_ids:
                    print(f"Skipping already completed config: {config.config_id}")
                    continue

                print(f"Running config: {config.config_id}")
                try:
                    retriever = self.prepare_for_config(config)
                    result = evaluate_configuration(
                        config_id=config.config_id,
                        chunk_strategy=config.chunk_strategy,
                        chunk_size=config.chunk_size,
                        retrieval_method=config.retrieval_method,
                        samples=self.samples,
                        retrieve_fn=lambda q, company, top_k, _r=retriever: _r.retrieve(
                            q, company, top_k
                        ),
                        report_text_by_company=report_text_by_company,
                    )
                    config_results.append(result)
                    completed_ids.add(config.config_id)
                    self._save_config_detail(result)

                    # Persist incremental ranking so interrupted runs can resume.
                    ranked_partial = sorted(
                        config_results,
                        key=lambda x: x.metrics["recall@15"],
                        reverse=True,
                    )
                    self._persist_results(ranked_partial, ranked_partial[0])

                    # Print progress with diagnostics.
                    diag = result.details.get("match_diagnostics", {})
                    span_ratio = diag.get("span_path_ratio", 0.0)
                    print(
                        f"  -> completed: recall@15 = {result.metrics['recall@15']:.4f}"
                        f"  (span match: {span_ratio:.1%}, "
                        f"token fallback: {diag.get('token_fallback_checks', 0)} checks)"
                    )
                except Exception as exc:
                    tb = traceback.format_exc()
                    msg = f"[FAILED] {config.config_id}: {exc}\n{tb}\n"
                    print(msg)
                    failed_configs.append(config.config_id)
                    with error_log.open("a", encoding="utf-8") as f:
                        f.write(msg)
                finally:
                    # Free memory between configs — critical on Apple Silicon.
                    if config.config_id in self._retriever_cache:
                        del self._retriever_cache[config.config_id]
                    gc.collect()

            if failed_configs:
                print(f"\n{len(failed_configs)} configs failed: {failed_configs}")
                print(f"  Error details: {error_log}")

            # --- Final ranking and winner selection ---
            ranked = sorted(
                config_results, key=lambda x: x.metrics["recall@15"], reverse=True
            )
            winner = ranked[0]
            self._persist_results(ranked, winner)

            save_best_pipeline_config(
                {
                    **pipeline_config_to_dict(
                        PipelineConfig(
                            chunk_strategy=winner.chunk_strategy,
                            chunk_size=winner.chunk_size,
                            retrieval_method=winner.retrieval_method,
                        )
                    ),
                    "_best_pipeline_name": winner.config_id,
                    "_selection_reason": (
                        f"Highest recall@15 ({winner.metrics['recall@15']:.4f}) in RQ1."
                    ),
                }
            )

            return {
                "winner": winner.config_id,
                "winner_metrics": winner.metrics,
                "evaluated_configs": len(ranked),
            }
        finally:
            self.close()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _save_config_detail(self, result: ConfigResult) -> None:
        """Save detailed per-config results (metrics + diagnostics)."""
        payload = {
            "config_id": result.config_id,
            "chunk_strategy": result.chunk_strategy,
            "chunk_size": result.chunk_size,
            "retrieval_method": result.retrieval_method,
            "results_version": _RESULTS_VERSION,
            "metrics": result.metrics,
            "per_standard_metrics": result.per_standard_metrics,
            "details": result.details,
        }
        path = self._config_details_dir / f"{result.config_id}.json"
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def _load_existing_results(self) -> list[ConfigResult]:
        """Load previously computed results for incremental evaluation."""
        out: list[ConfigResult] = []
        for path in sorted(self._config_details_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("results_version") != _RESULTS_VERSION:
                    continue
                out.append(
                    ConfigResult(
                        config_id=payload["config_id"],
                        chunk_strategy=payload["chunk_strategy"],
                        chunk_size=int(payload["chunk_size"]),
                        retrieval_method=payload["retrieval_method"],
                        metrics=payload["metrics"],
                        per_standard_metrics=payload.get("per_standard_metrics", {}),
                        details=payload.get("details", {}),
                    )
                )
            except Exception:
                continue
        return out

    def _persist_results(self, ranked: list[ConfigResult], winner: ConfigResult) -> None:
        """Persist ranked results as CSV, JSON, and winner manifest."""
        if not ranked:
            return

        rows = []
        for res in ranked:
            row = {
                "config_id": res.config_id,
                "chunk_strategy": res.chunk_strategy,
                "chunk_size": res.chunk_size,
                "retrieval_method": res.retrieval_method,
                "results_version": _RESULTS_VERSION,
                **res.metrics,
            }
            rows.append(row)

        # CSV and JSON for all configurations.
        json_path = RQ1_RESULTS_DIR / "rq1_config_results.json"
        csv_path = RQ1_RESULTS_DIR / "rq1_config_results.csv"
        json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        # Winner manifest.
        (RQ1_RESULTS_DIR / "rq1_winner.json").write_text(
            json.dumps(
                {
                    "config_id": winner.config_id,
                    "results_version": _RESULTS_VERSION,
                    "metrics": winner.metrics,
                    "per_standard_metrics": winner.per_standard_metrics,
                    "evaluation_coverage": winner.details,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Runtime compatibility loaders (for Streamlit app)
# ---------------------------------------------------------------------------

def load_pipeline(config: dict):
    """Lightweight pipeline loader optimised for the Streamlit app.

    Key optimisations vs the full ``RQ1PipelineRunner``:

    * **Chunk pickle cache** — Chunks are built once, then serialised to
      a pickle file.  Subsequent launches load the pickle (~seconds)
      instead of re-chunking 17 reports (~tens of seconds).
    * **InMemoryDenseIndex (NPZ)** — A compact NumPy matrix (~31 MB)
      replaces the Qdrant file-lock store.  The 7.5 GB embedding-cache
      JSON is only touched on the very first run to produce the NPZ.
    * **No golden-sample loading** — The evaluation dataset is not needed
      at query time.
    * **Query-only embedder** — The ``EmbeddingService`` is instantiated
      *without* loading the multi-GB cache.  Only live API calls for
      short query strings are made.
    """
    import gc
    import pickle

    from rag.chunking.strategies import build_chunks_for_config
    from rag.data.loaders import load_reports
    from rag.indexing.embedder import EmbeddingService
    from rag.indexing.in_memory_index import InMemoryDenseIndex
    from rag.retrieval.methods import Retriever

    ensure_dirs()

    cfg = PipelineConfig(
        chunk_strategy=config.get("chunk_strategy", "structure_aware"),
        chunk_size=int(config.get("chunk_size", 512)),
        retrieval_method=config.get("retrieval_method", "hybrid_rrf"),
        top_k=int(config.get("top_k", 15)),
        rrf_k=int(config.get("rrf_k", 60)),
    )

    from rag.config.settings import ARTIFACTS_DIR, RQ1_INDEX_VERSION

    # ------------------------------------------------------------------
    # 1. Chunks — pickle cache keyed by (strategy, size, code version)
    #    The RQ1_INDEX_VERSION fingerprint changes when chunking logic,
    #    text_utils, or types change, so stale caches are automatically
    #    invalidated when the best pipeline changes after re-running
    #    notebooks.
    # ------------------------------------------------------------------
    chunk_cache_path = (
        ARTIFACTS_DIR / f"app_chunks_{cfg.chunk_strategy}_{cfg.chunk_size}_{RQ1_INDEX_VERSION}.pkl"
    )

    if chunk_cache_path.exists():
        _logger.info("Loading cached chunks from %s", chunk_cache_path)
        with open(chunk_cache_path, "rb") as fh:
            chunks = pickle.load(fh)
    else:
        _logger.info("Building chunks for %s_%s …", cfg.chunk_strategy, cfg.chunk_size)
        reports = load_reports()
        chunks, _ = build_chunks_for_config(reports, cfg)
        chunk_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(chunk_cache_path, "wb") as fh:
            pickle.dump(chunks, fh, protocol=pickle.HIGHEST_PROTOCOL)
        del reports
        gc.collect()

    # ------------------------------------------------------------------
    # 2. Dense index — InMemoryDenseIndex (NPZ file)
    # ------------------------------------------------------------------
    dense_index = InMemoryDenseIndex(dims=3072)
    collection = dense_index.collection_name(cfg.chunk_strategy, cfg.chunk_size)

    dense_methods = {"dense", "dense_rerank", "hybrid_rrf", "hybrid_rerank"}
    if cfg.retrieval_method in dense_methods and dense_index.collection_point_count(collection) == 0:
        _logger.info(
            "NPZ index missing for %s — building (one-time, requires embedding cache) …",
            collection,
        )
        # Load full embedding cache only for this one-time build.
        heavy_embedder = EmbeddingService(load_cache=True, require_openai=False)
        vectors = heavy_embedder.embed_texts([c.retrieval_text for c in chunks])
        dense_index.upsert_chunks(collection, chunks, vectors)
        del heavy_embedder, vectors
        gc.collect()

    # ------------------------------------------------------------------
    # 3. Query-only embedder (no giant cache load)
    # ------------------------------------------------------------------
    query_embedder = EmbeddingService(load_cache=False, require_openai=False)

    # ------------------------------------------------------------------
    # 4. Build Retriever (BM25 + dense backed by NPZ)
    # ------------------------------------------------------------------
    retriever = Retriever(
        chunks=chunks,
        config=cfg,
        embedder=query_embedder,
        qdrant=dense_index,  # same search() interface as QdrantChunkIndex
    )

    out_cfg = dict(config)
    out_cfg.setdefault("top_k", cfg.top_k)
    return retriever, out_cfg


def run_query(
    query: str,
    companies: list[str],
    esrs_standard: str,
    retriever,
    config: dict,
    *,
    use_esrs_context: bool = False,
) -> dict:
    """Full pipeline query: retrieval + optional ESRS context + LLM generation.

    When an OpenAI API key is available the generator produces structured
    JSON answers.  Falls back to retrieval-only output otherwise.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from rag.generation.esrs_context import load_esrs_standards, extract_esrs_context
    from rag.generation.generator import AnswerGenerator
    from rag.types import GenerationSample, RetrievedEvidenceChunk

    top_k = int(config.get("top_k", 15))

    # --- ESRS context (shared across companies) ---
    esrs_context = ""
    if use_esrs_context and esrs_standard:
        try:
            standards = load_esrs_standards()
            esrs_context = extract_esrs_context(
                esrs_standard, standards, question=query,
            )
        except Exception:
            esrs_context = ""

    # --- Generator (may be None if no API key) ---
    generator: AnswerGenerator | None = None
    try:
        generator = AnswerGenerator(require_openai=True)
    except Exception:
        generator = None

    condition = "full_system" if use_esrs_context else "no_esrs_context"

    # Pre-load cross-encoder before any parallel retrieval so threads
    # don't race to initialise it (avoids loading the model N times).
    if hasattr(retriever, "_ensure_cross_encoder"):
        try:
            retriever._ensure_cross_encoder()
        except Exception:
            pass

    def _process_company(company: str) -> tuple[str, dict, list[dict]]:
        hits = retriever.retrieve(query=query, company=company, top_k=top_k)
        chunks_payload = [
            {
                "chunk_id": hit.chunk.chunk_id,
                "text": hit.chunk.text,
                "score": hit.score,
                "rank": hit.rank,
                "page_span": [hit.chunk.page_start, hit.chunk.page_end],
                "header_hierarchy": hit.chunk.header_hierarchy,
            }
            for hit in hits
        ]

        if generator is not None:
            sample = GenerationSample(
                question_id="", question=query,
                esrs_standard=esrs_standard or "",
                esrs_disclosure_id="", company=company,
                ground_truth_answer="", fulfillment_level="",
                fulfillment_rationale="", evidences=[],
            )
            retrieved_chunks = [
                RetrievedEvidenceChunk(
                    chunk_id=hit.chunk.chunk_id,
                    text=hit.chunk.text,
                    score=hit.score,
                    rank=hit.rank,
                    page_start=hit.chunk.page_start,
                    page_end=hit.chunk.page_end,
                    header_hierarchy=hit.chunk.header_hierarchy,
                )
                for hit in hits
            ]
            answer = generator.generate(
                sample=sample,
                retrieved_chunks=retrieved_chunks,
                esrs_context=esrs_context,
                condition=condition,
            )
            answer_data = {
                "direct_answer": answer.direct_answer,
                "key_findings": answer.key_findings,
                "compliance_level": answer.compliance_level,
                "compliance_rationale": answer.compliance_rationale,
                "disclosure_gaps": answer.disclosure_gaps,
                "cited_pages": answer.cited_pages,
                "page_references": [
                    {
                        "page_display": (
                            str(c["page_span"][0])
                            if c["page_span"][0] == c["page_span"][1]
                            else f"{c['page_span'][0]}-{c['page_span'][1]}"
                        ),
                        "section": " > ".join(c["header_hierarchy"]),
                        "text_preview": c["text"][:220],
                    }
                    for c in chunks_payload[:5]
                ],
            }
        else:
            answer_data = {
                "direct_answer": "Generation unavailable (no OpenAI API key).",
                "key_findings": [],
                "compliance_level": "N/A",
                "compliance_rationale": "Generation not available.",
                "disclosure_gaps": [],
                "cited_pages": [],
                "page_references": [
                    {
                        "page_display": (
                            str(c["page_span"][0])
                            if c["page_span"][0] == c["page_span"][1]
                            else f"{c['page_span'][0]}-{c['page_span'][1]}"
                        ),
                        "section": " > ".join(c["header_hierarchy"]),
                        "text_preview": c["text"][:220],
                    }
                    for c in chunks_payload[:5]
                ],
            }

        return company, answer_data, chunks_payload

    # --- Run companies in parallel for speed ---
    per_company: dict[str, dict] = {}
    per_company_chunks: dict[str, list] = {}

    if len(companies) == 1:
        company, answer_data, chunks_payload = _process_company(companies[0])
        per_company[company] = answer_data
        per_company_chunks[company] = chunks_payload
    else:
        with ThreadPoolExecutor(max_workers=min(len(companies), 4)) as pool:
            futures = {
                pool.submit(_process_company, co): co for co in companies
            }
            for future in as_completed(futures):
                company, answer_data, chunks_payload = future.result()
                per_company[company] = answer_data
                per_company_chunks[company] = chunks_payload

    return {
        "companies": companies,
        "esrs_standard": esrs_standard,
        "query": query,
        "per_company_chunks": per_company_chunks,
        "per_company": per_company,
        "esrs_requirements": esrs_context or "",
    }
