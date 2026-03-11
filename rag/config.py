"""Project configuration, environment loading, and experiment definitions."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import yaml

# ---------------------------------------------------------------------------
# Project paths (resolved relative to repo root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_MD_DIR = DATA_DIR / "Reports_md"
REPORTS_PDF_DIR = DATA_DIR / "Reports_Pdf"
ESRS_JSON_DIR = DATA_DIR / "ESRS_Standards_JSON"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
GOLDEN_STANDARD_PATH = PROJECT_ROOT / "golden_standard" / "golden_standard.json"
CONFIG_DIR = PROJECT_ROOT / "config"

# Ensure artifacts directory exists
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load environment variables from .env (tolerant parser)
# ---------------------------------------------------------------------------


def _load_env_file(env_path: Path) -> None:
    """Load KEY=VALUE lines from .env without failing on non-standard lines."""
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            # Ignore malformed lines to avoid noisy parser warnings.
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            os.environ.setdefault(key, value)


_load_env_file(PROJECT_ROOT / ".env")

# API keys and endpoints (never print these)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
AZURE_DOC_INTELLIGENCE_ENDPOINT = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "")
AZURE_DOC_INTELLIGENCE_KEY = os.getenv("AZURE_DOCUMENT_INTELLIGENCE", "")
AZURE_STORAGE_CONNECTION = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")

# Model configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5.2")
EMBEDDING_DIMENSIONS = 3072  # text-embedding-3-large produces 3072-dim vectors

# ---------------------------------------------------------------------------
# Company mapping: filename stem -> display name
# ---------------------------------------------------------------------------
COMPANY_MAP: Dict[str, str] = {
    "AkerBP_2024": "Aker BP",
    "BASF_2024": "BASF",
    "Eni_2024": "Eni",
    "Equinor_2024": "Equinor",
    "Galp_2024": "Galp",
    "Helleniq_2024": "Helleniq",
    "Mol_2024": "MOL",
    "Motoroil_2024": "Motoroil",
    "Neste_2024": "Neste",
    "OMV_2024": "OMV",
    "Petrom_2024": "Petrom",
    "Repsol_2024": "Repsol",
    "SBMOffshore_2024": "SBM Offshore",
    "Shell_2024": "Shell",
    "Solvay_2024": "Solvay",
    "TotalEnergies_2024": "TotalEnergies",
    "Vopak_2024": "Vopak",
}

# Reverse mapping: display name -> filename stem
COMPANY_REVERSE_MAP: Dict[str, str] = {v: k for k, v in COMPANY_MAP.items()}

# ---------------------------------------------------------------------------
# ESRS Standards mapping
# ---------------------------------------------------------------------------
ESRS_STANDARDS = [
    "ESRS 1", "ESRS 2",
    "ESRS E1", "ESRS E2", "ESRS E3", "ESRS E4", "ESRS E5",
    "ESRS S1", "ESRS S2", "ESRS S3", "ESRS S4",
    "ESRS G1",
]

ESRS_JSON_FILES: Dict[str, str] = {
    "ESRS 1": "ESRS_1.json",
    "ESRS 2": "ESRS_2.json",
    "ESRS E1": "ESRS_E1.json",
    "ESRS E2": "ESRS_E2.json",
    "ESRS E3": "ESRS_E3.json",
    "ESRS E4": "ESRS_E4.json",
    "ESRS E5": "ESRS_E5.json",
    "ESRS S1": "ESRS_S1.json",
    "ESRS S2": "ESRS_S2.json",
    "ESRS S3": "ESRS_S3.json",
    "ESRS S4": "ESRS_S4.json",
    "ESRS G1": "ESRS_G1.json",
}

# ---------------------------------------------------------------------------
# Golden Standard benchmark questions (complete answers for all 17 companies)
# ---------------------------------------------------------------------------
BENCHMARK_QUESTION_IDS = ["Q1", "Q2", "Q3", "Q8", "Q12"]

# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

_DEFAULT_PIPELINE_CONFIG: Dict[str, Any] = {
    # Chunking
    "chunking_strategy": "structure_aware",  # fixed_size | structure_aware
    "chunk_size_tokens": 512,
    "chunk_overlap_tokens": 64,
    # Embedding
    "embedding_model": EMBEDDING_MODEL,
    "embedding_dimensions": EMBEDDING_DIMENSIONS,
    # Retrieval
    "retrieval_method": "hybrid",  # dense | sparse | hybrid
    "top_k": 15,
    "rrf_k": 60,
    "use_reranker": False,
    "rerank_top_n": 10,
    # Generation
    "llm_model": LLM_MODEL,
    "max_generation_tokens": 2048,
    "temperature": 0.1,
    # Collection naming
    "qdrant_collection_prefix": "esrs_reports",
}


def _load_pipeline_config_yaml(path: Path) -> Dict[str, Any]:
    """Load pipeline configurations from YAML if available."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return {}
    return data


def get_default_pipeline_config() -> Dict[str, Any]:
    """Return default runtime configuration, merged with YAML defaults."""
    yaml_cfg = _load_pipeline_config_yaml(CONFIG_DIR / "pipeline_configs.yaml")
    yaml_default = yaml_cfg.get("default", {}) if isinstance(yaml_cfg.get("default", {}), dict) else {}
    merged = {**_DEFAULT_PIPELINE_CONFIG, **yaml_default}
    # Ensure env-sourced model values remain authoritative.
    merged["embedding_model"] = EMBEDDING_MODEL
    merged["embedding_dimensions"] = EMBEDDING_DIMENSIONS
    merged["llm_model"] = LLM_MODEL
    return merged


def get_experiment_configs() -> Dict[str, Dict[str, Any]]:
    """Return named experiment configurations, merged over defaults."""
    base = get_default_pipeline_config()
    yaml_cfg = _load_pipeline_config_yaml(CONFIG_DIR / "pipeline_configs.yaml")
    configs: Dict[str, Dict[str, Any]] = {}

    for name, override in yaml_cfg.items():
        if name == "default":
            continue
        if not isinstance(override, dict):
            continue
        configs[name] = {**base, **override}

    # Fallback matrix if YAML is absent or incomplete.
    if not configs:
        configs = {
            "RQ1_fixed_512": {**base, "chunking_strategy": "fixed_size"},
            "RQ1_structure_aware": {**base, "chunking_strategy": "structure_aware"},
            "RQ2_dense_only": {**base, "retrieval_method": "dense", "use_reranker": False},
            "RQ2_sparse_only": {**base, "retrieval_method": "sparse", "use_reranker": False},
            "RQ2_hybrid": {**base, "retrieval_method": "hybrid", "use_reranker": False},
            "RQ2_hybrid_rerank": {**base, "retrieval_method": "hybrid", "use_reranker": True},
            "RQ3_best_retrieval": {**base, "retrieval_method": "hybrid", "use_reranker": False},
            "RQ3_sparse_retrieval": {**base, "retrieval_method": "sparse", "use_reranker": False},
        }
    return configs


def load_golden_standard() -> dict:
    """Load and return the Golden Standard evaluation dataset."""
    with open(GOLDEN_STANDARD_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
