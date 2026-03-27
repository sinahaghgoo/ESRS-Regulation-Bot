"""Lightweight dense index used by RQ2 to avoid local Qdrant startup costs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rag.config.settings import DENSE_INDEX_DIR, QDRANT_COLLECTION_PREFIX, RQ1_INDEX_VERSION
from rag.types import ChunkRecord


class InMemoryDenseIndex:
    """Dense-vector search with a small on-disk cache.

    The winning RQ1 retriever needs dense search, but opening the persisted
    local Qdrant store can be prohibitively expensive on laptops. This index
    keeps the retrieval logic identical at the algorithm level while swapping
    the storage backend for a compact NumPy cache that loads quickly.
    """

    def __init__(self, dims: int = 3072, storage_dir: Path | None = None) -> None:
        self.dims = dims
        self.storage_dir = storage_dir or DENSE_INDEX_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._collections: dict[str, dict[str, np.ndarray]] = {}

    @staticmethod
    def collection_name(strategy: str, size: int) -> str:
        version = RQ1_INDEX_VERSION.replace(".", "_")
        return f"{QDRANT_COLLECTION_PREFIX}_{version}_{strategy}_{size}".lower()

    def _collection_path(self, collection_name: str) -> Path:
        return self.storage_dir / f"{collection_name}.npz"

    def _load_collection(self, collection_name: str) -> dict[str, np.ndarray] | None:
        if collection_name in self._collections:
            return self._collections[collection_name]

        path = self._collection_path(collection_name)
        if not path.exists():
            return None

        data = np.load(path)
        collection = {
            "vectors": np.asarray(data["vectors"], dtype=np.float32),
            "chunk_ids": np.asarray(data["chunk_ids"], dtype=str),
            "companies": np.asarray(data["companies"], dtype=str),
        }
        self._collections[collection_name] = collection
        return collection

    def collection_point_count(self, collection_name: str) -> int:
        collection = self._load_collection(collection_name)
        if collection is None:
            return 0
        return int(collection["chunk_ids"].shape[0])

    def recreate_collection(self, collection_name: str) -> None:
        self._collections.pop(collection_name, None)
        path = self._collection_path(collection_name)
        if path.exists():
            path.unlink()

    def upsert_chunks(
        self,
        collection_name: str,
        chunks: list[ChunkRecord],
        vectors: list[list[float]],
    ) -> None:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("Dense index expects a 2D vector matrix.")

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        matrix = matrix / norms

        chunk_ids = np.asarray([chunk.chunk_id for chunk in chunks], dtype=str)
        companies = np.asarray([chunk.company for chunk in chunks], dtype=str)
        path = self._collection_path(collection_name)
        np.savez(path, vectors=matrix, chunk_ids=chunk_ids, companies=companies)

        self._collections[collection_name] = {
            "vectors": matrix,
            "chunk_ids": chunk_ids,
            "companies": companies,
        }

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int,
        company: str | None = None,
    ) -> list[tuple[str, float, dict]]:
        collection = self._load_collection(collection_name)
        if collection is None:
            return []

        vectors = collection["vectors"]
        chunk_ids = collection["chunk_ids"]
        companies = collection["companies"]

        qvec = np.asarray(query_vector, dtype=np.float32)
        qnorm = float(np.linalg.norm(qvec))
        if qnorm == 0.0:
            return []
        qvec = qvec / qnorm

        if company is not None:
            candidate_idx = np.flatnonzero(companies == company)
        else:
            candidate_idx = np.arange(vectors.shape[0])

        if candidate_idx.size == 0:
            return []

        scores = vectors[candidate_idx] @ qvec
        order = np.argsort(scores)[::-1][:top_k]
        ranked_idx = candidate_idx[order]
        ranked_scores = scores[order]

        rows: list[tuple[str, float, dict]] = []
        for idx, score in zip(ranked_idx, ranked_scores):
            rows.append(
                (
                    str(chunk_ids[idx]),
                    float(score),
                    {
                        "chunk_id": str(chunk_ids[idx]),
                        "company": str(companies[idx]),
                    },
                )
            )
        return rows

    def close(self) -> None:
        """Mirror the Qdrant interface used by the shared retriever."""
        self._collections.clear()
