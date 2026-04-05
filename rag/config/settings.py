"""Central configuration for RQ1 and RQ2 experiments."""

from __future__ import annotations

import hashlib
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
ESRS_JSON_DIR = DATA_DIR / "ESRS_Standards_JSON"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
RQ1_RESULTS_DIR = ARTIFACTS_DIR / "evaluation_results" / "rq1"
RQ2_RESULTS_DIR = ARTIFACTS_DIR / "evaluation_results" / "rq2"
RQ3_RESULTS_DIR = ARTIFACTS_DIR / "evaluation_results" / "rq3"
DENSE_INDEX_DIR = ARTIFACTS_DIR / "dense_index_cache"
GOLDEN_STANDARD_PATH = PROJECT_ROOT / "golden_standard" / "golden_standard.json"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
CONFIG_DIR = PROJECT_ROOT / "config"
BEST_PIPELINE_PATH = CONFIG_DIR / "best_pipeline.yaml"

QDRANT_STORAGE_PATH = ARTIFACTS_DIR / "qdrant_storage"
QDRANT_COLLECTION_PREFIX = "rq1_esrs"


def _fingerprint_files(*relative_paths: str) -> str:
    """Return a short content fingerprint for the given project files.

    RQ1 artifacts are easy to invalidate accidentally: retrieval logic,
    chunking, and evaluation matching evolve independently, while the saved
    CSV/JSON outputs still look valid to the notebook. Fingerprinting the
    source files that define those behaviours makes stale results explicit
    and prevents the notebook from silently analysing old experiments.
    """
    digest = hashlib.sha1()
    for rel in relative_paths:
        path = PROJECT_ROOT / rel
        digest.update(rel.encode("utf-8"))
        try:
            digest.update(path.read_bytes())
        except FileNotFoundError:
            digest.update(b"<missing>")
    return digest.hexdigest()[:10]


# Qdrant collections only depend on chunk construction and the indexed text
# representation, so we keep the fingerprint scoped to those inputs.
RQ1_INDEX_VERSION = _fingerprint_files(
    "rag/chunking/strategies.py",
    "rag/data/text_utils.py",
    "rag/types.py",
)

# Evaluation results depend on the matching logic plus the chunk/index version.
RQ1_EVAL_VERSION = _fingerprint_files(
    "rag/evaluation/metrics.py",
    "rag/data/text_utils.py",
    "rag/types.py",
)

# Rerank scores must be refreshed whenever the reranker input construction or
# score blending changes.
RERANK_CACHE_VERSION = _fingerprint_files(
    "rag/retrieval/methods.py",
    "rag/types.py",
)

RQ1_RESULTS_VERSION = (
    f"index={RQ1_INDEX_VERSION};eval={RQ1_EVAL_VERSION};rerank={RERANK_CACHE_VERSION}"
)
RQ2_RESULTS_VERSION = _fingerprint_files(
    "rag/generation/esrs_context.py",
    "rag/generation/generator.py",
    "rag/generation/prompts.py",
    "rag/evaluation/rq2_metrics.py",
    "rag/pipeline/rq2_runner.py",
    "rag/data/loaders.py",
    "rag/types.py",
)

# RQ2 injects the full mapped ESRS standard for each question.
RQ2_CONTEXT_MODE = "substantive_full_standard"

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
    for p in (
        ARTIFACTS_DIR,
        RQ1_RESULTS_DIR,
        RQ2_RESULTS_DIR,
        RQ3_RESULTS_DIR,
        DENSE_INDEX_DIR,
        NOTEBOOKS_DIR,
        QDRANT_STORAGE_PATH,
        CONFIG_DIR,
        ESRS_JSON_DIR,
    ):
        p.mkdir(parents=True, exist_ok=True)


def get_openai_api_key(required: bool = False) -> str | None:
    """Read OpenAI API key from env."""
    key = os.getenv("OPENAI_API_KEY")
    if required and not key:
        raise RuntimeError("OPENAI_API_KEY is required but not set.")
    return key


def get_generation_model(default: str = "gpt-5-mini") -> str:
    """Read the configured generation model from env."""
    return os.getenv("LLM_MODEL", default).strip() or default


def get_local_model_thread_limit(default_cap: int = 4) -> int:
    """Thread cap for local transformer inference on Apple Silicon.

    The evaluation is intentionally sequential, but PyTorch can still fan out
    aggressively inside each rerank or semantic-chunking call. A modest thread
    cap keeps CPU and memory pressure predictable on laptops while remaining
    overridable via ``RQ1_TORCH_THREADS``.
    """
    raw = os.getenv("RQ1_TORCH_THREADS")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass

    cpu_count = os.cpu_count() or default_cap
    return max(1, min(default_cap, max(1, cpu_count // 2)))


def configure_local_torch_runtime() -> None:
    """Apply conservative PyTorch thread limits for local sentence models."""
    try:
        import torch
    except Exception:
        return

    thread_limit = get_local_model_thread_limit()
    try:
        torch.set_num_threads(thread_limit)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(max(1, min(2, thread_limit)))
    except Exception:
        # PyTorch only allows this to be set once per process.
        pass


def get_rerank_batch_size(default: int = 8) -> int:
    """Batch size for cross-encoder reranking with a safe laptop-friendly default."""
    raw = os.getenv("RQ1_RERANK_BATCH_SIZE")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return default


def pipeline_config_to_dict(config: PipelineConfig) -> dict:
    """Serialize dataclass config."""
    return asdict(config)


def build_rq2_audit_manifest() -> dict:
    """Return a manifest of file fingerprints for RQ2 artifact validation.

    The manifest captures the content hash of every source file that
    influences RQ2 results.  If any of these files change after a run,
    the saved artifacts are stale and must be regenerated.
    """
    files = [
        "rag/generation/esrs_context.py",
        "rag/generation/generator.py",
        "rag/generation/prompts.py",
        "rag/evaluation/rq2_metrics.py",
        "rag/pipeline/rq2_runner.py",
        "rag/data/loaders.py",
        "rag/types.py",
        "golden_standard/golden_standard.json",
    ]
    return {
        "results_version": RQ2_RESULTS_VERSION,
        "context_mode": RQ2_CONTEXT_MODE,
        "file_fingerprints": {
            rel: _fingerprint_files(rel) for rel in files
        },
    }
