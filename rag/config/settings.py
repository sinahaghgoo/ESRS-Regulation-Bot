"""Central configuration for RQ1 experiments."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path

from dotenv import load_dotenv

# Avoid noisy warnings when .env contains non KEY=VALUE lines.
logging.getLogger("dotenv.main").setLevel(logging.ERROR)
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_MD_DIR = DATA_DIR / "Reports_md_clean"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
RQ1_RESULTS_DIR = ARTIFACTS_DIR / "evaluation_results" / "rq1"
GOLDEN_STANDARD_PATH = PROJECT_ROOT / "golden_standard" / "golden_standard.json"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
CONFIG_DIR = PROJECT_ROOT / "config"
BEST_PIPELINE_PATH = CONFIG_DIR / "best_pipeline.yaml"

QDRANT_STORAGE_PATH = ARTIFACTS_DIR / "qdrant_storage"
QDRANT_COLLECTION_PREFIX = "rq1_esrs"

CHUNK_STRATEGIES = ("fixed", "structure_aware", "semantic")
CHUNK_SIZES = (256, 512, 1024)
RETRIEVAL_METHODS = ("dense", "dense_rerank", "sparse", "sparse_rerank", "hybrid_rrf", "hybrid_rerank")

DEFAULT_TOP_K = 15
RRF_K = 60
# Local cross-encoder reranker via sentence-transformers.
# Using the lightweight MiniLM-L-6 (~80 MB, 22 M params) instead of
# BAAI/bge-reranker-v2-m3 (2.27 GB, 568 M params) for practical
# evaluation on Apple Silicon.  MiniLM-L-6 is the most widely cited
# lightweight reranker in IR benchmarks and runs ~30x faster on CPU.
BGE_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Streamlit app compatibility constants.
ESRS_STANDARDS = [
    "ESRS 1",
    "ESRS 2",
    "ESRS E1",
    "ESRS E2",
    "ESRS E3",
    "ESRS E4",
    "ESRS E5",
    "ESRS S1",
    "ESRS S2",
    "ESRS S3",
    "ESRS S4",
    "ESRS G1",
]
COMPANY_MAP = {
    "AkerBP": "Aker BP",
    "BASF": "BASF",
    "Eni": "Eni",
    "Equinor": "Equinor",
    "Galp": "Galp",
    "Helleniq": "Helleniq",
    "Mol": "MOL",
    "Motoroil": "Motoroil",
    "Neste": "Neste",
    "OMV": "OMV",
    "Petrom": "Petrom",
    "Repsol": "Repsol",
    "SBMOffshore": "SBM Offshore",
    "Shell": "Shell",
    "Solvay": "Solvay",
    "TotalEnergies": "TotalEnergies",
    "Vopak": "Vopak",
}


@dataclass
class PipelineConfig:
    """Single experiment configuration for RQ1."""

    chunk_strategy: str
    chunk_size: int
    retrieval_method: str
    chunk_overlap_ratio: float = 0.125
    top_k: int = DEFAULT_TOP_K
    rrf_k: int = RRF_K
    rerank_model: str = BGE_RERANK_MODEL

    @property
    def config_id(self) -> str:
        return f"{self.chunk_strategy}_{self.chunk_size}_{self.retrieval_method}"


def build_rq1_experiment_grid() -> list[PipelineConfig]:
    """Return the full 54-condition RQ1 experiment matrix."""
    grid: list[PipelineConfig] = []
    for strategy in CHUNK_STRATEGIES:
        for size in CHUNK_SIZES:
            for method in RETRIEVAL_METHODS:
                grid.append(
                    PipelineConfig(
                        chunk_strategy=strategy,
                        chunk_size=size,
                        retrieval_method=method,
                    )
                )
    return grid


def get_default_pipeline_config() -> dict:
    """Default app config compatible with the current Streamlit frontend."""
    return {
        "chunk_strategy": "structure_aware",
        "chunk_size": 512,
        "retrieval_method": "hybrid_rrf",
        "chunk_overlap_ratio": 0.125,
        "top_k": DEFAULT_TOP_K,
        "rrf_k": RRF_K,
        "rerank_model": BGE_RERANK_MODEL,
    }


def load_best_pipeline_config() -> dict:
    """Load best pipeline manifest if available; else use defaults."""
    if BEST_PIPELINE_PATH.exists():
        content = BEST_PIPELINE_PATH.read_text(encoding="utf-8")
        try:
            # YAML is optional: JSON-compatible parser fallback.
            import yaml  # type: ignore

            data = yaml.safe_load(content) or {}
        except Exception:
            data = json.loads(content)
        if isinstance(data, dict):
            data.setdefault("top_k", DEFAULT_TOP_K)
            return data
    return get_default_pipeline_config()


def save_best_pipeline_config(best_config: dict) -> None:
    """Persist the winning RQ1 config for downstream usage."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    payload = dict(best_config)
    try:
        import yaml  # type: ignore

        BEST_PIPELINE_PATH.write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )
    except Exception:
        BEST_PIPELINE_PATH.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )


def ensure_dirs() -> None:
    """Create expected artifact directories."""
    for p in (ARTIFACTS_DIR, RQ1_RESULTS_DIR, NOTEBOOKS_DIR, QDRANT_STORAGE_PATH, CONFIG_DIR):
        p.mkdir(parents=True, exist_ok=True)


def get_openai_api_key(required: bool = False) -> str | None:
    """Read OpenAI API key from env."""
    key = os.getenv("OPENAI_API_KEY")
    if required and not key:
        raise RuntimeError("OPENAI_API_KEY is required but not set.")
    return key


def pipeline_config_to_dict(config: PipelineConfig) -> dict:
    """Serialize dataclass config."""
    return asdict(config)

