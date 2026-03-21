"""RQ1 experiment orchestration (54 configs: 3 chunk strategies × 3 sizes × 6 retrieval methods) and minimal runtime retrieval interface."""

from __future__ import annotations

import csv
import gc
import json
import re
import traceback
from dataclasses import asdict
from pathlib import Path

from rag.chunking.strategies import build_chunks_for_config
from rag.config.settings import (
    PipelineConfig,
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


class RQ1PipelineRunner:
    """Build indexes and evaluate all RQ1 configurations."""

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
        self._chunks_cache: dict[tuple[str, int], list] = {}
        self._retriever_cache: dict[str, Retriever] = {}
        self._config_details_dir = RQ1_RESULTS_DIR / "config_details"
        self._config_details_dir.mkdir(parents=True, exist_ok=True)

    def prepare_for_config(self, config: PipelineConfig) -> Retriever:
        key = (config.chunk_strategy, config.chunk_size)
        if key not in self._chunks_cache:
            chunks, stats = build_chunks_for_config(self.reports, config)
            self._chunks_cache[key] = chunks
            collection = self.qdrant.collection_name(config.chunk_strategy, config.chunk_size)
            existing_count = self.qdrant.collection_point_count(collection)
            if existing_count > 0:
                print(f"  Qdrant collection {collection} already has {existing_count} vectors — reusing")
            else:
                print(f"  Building Qdrant collection {collection} ({len(chunks)} chunks)")
                vectors = self.embedder.embed_texts([c.text for c in chunks])
                self.qdrant.recreate_collection(collection)
                self.qdrant.upsert_chunks(collection, chunks, vectors)
            stats_path = RQ1_RESULTS_DIR / f"chunk_stats_{config.chunk_strategy}_{config.chunk_size}.json"
            stats_path.write_text(json.dumps(asdict(stats), indent=2), encoding="utf-8")

        rid = config.config_id
        if rid not in self._retriever_cache:
            self._retriever_cache[rid] = Retriever(
                chunks=self._chunks_cache[key],
                config=config,
                embedder=self.embedder,
                qdrant=self.qdrant,
            )
        return self._retriever_cache[rid]

    def run(self) -> dict:
        """Execute RQ1 matrix and persist results."""
        grid = build_rq1_experiment_grid()
        if self.dry_run:
            grid = [grid[0]]
        # Strip page markers BEFORE normalizing so the normalized text
        # matches the chunks (which are built from page-marker-free text).
        _page_re = re.compile(r"<!--\s*Report Page\s+\d+\s*-->")
        from rag.data.text_utils import normalize_text
        report_text_by_company = {
            k: normalize_text(_page_re.sub("", v.raw_text))
            for k, v in self.reports.items()
        }
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
                    retrieve_fn=lambda q, company, top_k, _r=retriever: _r.retrieve(q, company, top_k),
                    report_text_by_company=report_text_by_company,
                )
                config_results.append(result)
                completed_ids.add(config.config_id)
                self._save_config_detail(result)
                # Persist incremental ranking so interrupted runs can resume.
                ranked_partial = sorted(config_results, key=lambda x: x.metrics["recall@15"], reverse=True)
                self._persist_results(ranked_partial, ranked_partial[0])
                diag = result.details.get("match_diagnostics", {})
                span_ratio = diag.get("span_path_ratio", 0.0)
                print(
                    f"  -> completed: recall@15 = {result.metrics['recall@15']:.4f}"
                    f"  (span match: {span_ratio:.1%}, token fallback: {diag.get('token_fallback_checks', 0)} checks)"
                )
            except Exception as exc:
                tb = traceback.format_exc()
                msg = f"[FAILED] {config.config_id}: {exc}\n{tb}\n"
                print(msg)
                failed_configs.append(config.config_id)
                with error_log.open("a", encoding="utf-8") as f:
                    f.write(msg)
            finally:
                # Free memory between configs — critical on Apple Silicon
                # with the 2.3 GB cross-encoder model.
                if config.config_id in self._retriever_cache:
                    del self._retriever_cache[config.config_id]
                gc.collect()
        if failed_configs:
            print(f"\n⚠ {len(failed_configs)} configs failed: {failed_configs}")
            print(f"  Error details: {error_log}")
        ranked = sorted(config_results, key=lambda x: x.metrics["recall@15"], reverse=True)
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
                "_selection_reason": f"Highest recall@15 ({winner.metrics['recall@15']:.4f}) in RQ1.",
            }
        )
        return {
            "winner": winner.config_id,
            "winner_metrics": winner.metrics,
            "evaluated_configs": len(ranked),
        }

    def _save_config_detail(self, result: ConfigResult) -> None:
        payload = {
            "config_id": result.config_id,
            "chunk_strategy": result.chunk_strategy,
            "chunk_size": result.chunk_size,
            "retrieval_method": result.retrieval_method,
            "metrics": result.metrics,
            "per_standard_metrics": result.per_standard_metrics,
            "details": result.details,
        }
        path = self._config_details_dir / f"{result.config_id}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load_existing_results(self) -> list[ConfigResult]:
        out: list[ConfigResult] = []
        for path in sorted(self._config_details_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
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

    def _persist_results(self, ranked, winner) -> None:
        if not ranked:
            return
        rows = []
        for res in ranked:
            row = {
                "config_id": res.config_id,
                "chunk_strategy": res.chunk_strategy,
                "chunk_size": res.chunk_size,
                "retrieval_method": res.retrieval_method,
                **res.metrics,
            }
            rows.append(row)
        json_path = RQ1_RESULTS_DIR / "rq1_config_results.json"
        csv_path = RQ1_RESULTS_DIR / "rq1_config_results.csv"
        json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        (RQ1_RESULTS_DIR / "rq1_winner.json").write_text(
            json.dumps(
                {
                    "config_id": winner.config_id,
                    "metrics": winner.metrics,
                    "per_standard_metrics": winner.per_standard_metrics,
                    "evaluation_coverage": winner.details,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def load_pipeline(config: dict):
    """
    Compatibility loader for Streamlit app.

    For now this initializes the retrieval pipeline using one config.
    """
    cfg = PipelineConfig(
        chunk_strategy=config.get("chunk_strategy", "structure_aware"),
        chunk_size=int(config.get("chunk_size", 512)),
        retrieval_method=config.get("retrieval_method", "hybrid_rrf"),
        top_k=int(config.get("top_k", 15)),
        rrf_k=int(config.get("rrf_k", 60)),
    )
    runner = RQ1PipelineRunner(dry_run=True)
    retriever = runner.prepare_for_config(cfg)
    out_cfg = dict(config)
    out_cfg.setdefault("top_k", cfg.top_k)
    return retriever, out_cfg


def run_query(query: str, companies: list[str], esrs_standard: str, retriever, config: dict) -> dict:
    """
    Minimal retrieval-only query output for compatibility.

    This repository's current focus is RQ1 retrieval, so generation fields are placeholders.
    """
    top_k = int(config.get("top_k", 15))
    per_company_chunks = {}
    per_company = {}
    for company in companies:
        hits = retriever.retrieve(query=query, company=company, top_k=top_k)
        chunks_payload = []
        for hit in hits:
            chunks_payload.append(
                {
                    "chunk_id": hit.chunk.chunk_id,
                    "text": hit.chunk.text,
                    "score": hit.score,
                    "rank": hit.rank,
                    "page_span": [hit.chunk.page_start, hit.chunk.page_end],
                    "header_hierarchy": hit.chunk.header_hierarchy,
                }
            )
        per_company_chunks[company] = chunks_payload
        per_company[company] = {
            "direct_answer": "RQ1 mode: retrieval-only pipeline. Generation is not part of this run.",
            "key_findings": [],
            "compliance_level": "N/A",
            "compliance_rationale": "Not computed in RQ1 retrieval-only mode.",
            "disclosure_gaps": [],
            "page_references": [
                {
                    "page_display": f"{c['page_span'][0]}-{c['page_span'][1]}",
                    "section": " > ".join(c["header_hierarchy"]),
                    "text_preview": c["text"][:220],
                }
                for c in chunks_payload[:5]
            ],
        }
    return {
        "companies": companies,
        "esrs_standard": esrs_standard,
        "query": query,
        "per_company_chunks": per_company_chunks,
        "per_company": per_company,
        "esrs_requirements": "RQ1 mode: no ESRS context extraction.",
    }

