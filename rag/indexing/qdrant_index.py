"""Qdrant-backed dense index for chunks."""

from __future__ import annotations

from dataclasses import asdict

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from rag.config.settings import QDRANT_COLLECTION_PREFIX, QDRANT_STORAGE_PATH
from rag.types import ChunkRecord


class QdrantChunkIndex:
    """Dense vector index with local persistence."""

    def __init__(self, dims: int = 3072) -> None:
        try:
            self.client = QdrantClient(path=str(QDRANT_STORAGE_PATH))
        except RuntimeError as exc:
            if "already accessed" in str(exc):
                # A previous Jupyter kernel left a stale file-lock.
                # Remove it and retry once.
                lock_file = QDRANT_STORAGE_PATH / ".lock"
                if lock_file.exists():
                    lock_file.unlink(missing_ok=True)
                self.client = QdrantClient(path=str(QDRANT_STORAGE_PATH))
            else:
                raise
        self.dims = dims

    def close(self) -> None:
        """Explicitly close the Qdrant client to release the file lock."""
        if hasattr(self, "client") and self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass

    @staticmethod
    def collection_name(strategy: str, size: int) -> str:
        return f"{QDRANT_COLLECTION_PREFIX}_{strategy}_{size}".lower()

    def recreate_collection(self, collection_name: str) -> None:
        """Create or replace collection."""
        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(size=self.dims, distance=qmodels.Distance.COSINE),
        )

    def upsert_chunks(
        self,
        collection_name: str,
        chunks: list[ChunkRecord],
        vectors: list[list[float]],
    ) -> None:
        """Insert chunks with payload."""
        points: list[qmodels.PointStruct] = []
        for chunk, vec in zip(chunks, vectors):
            point_id = int(chunk.chunk_id, 16) % (2**63 - 1)
            payload = {
                "chunk_id": chunk.chunk_id,
                "company": chunk.company,
                "year": chunk.year,
                "strategy": chunk.strategy,
                "chunk_size": chunk.chunk_size,
                "text": chunk.text,
                "normalized_text": chunk.normalized_text,
                "token_count": chunk.token_count,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "header_hierarchy": chunk.header_hierarchy,
            }
            points.append(qmodels.PointStruct(id=point_id, vector=vec, payload=payload))
        self.client.upsert(collection_name=collection_name, points=points, wait=True)

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int,
        company: str | None = None,
    ) -> list[tuple[str, float, dict]]:
        """Dense similarity search with optional company filter."""
        flt = None
        if company:
            flt = qmodels.Filter(
                must=[qmodels.FieldCondition(key="company", match=qmodels.MatchValue(value=company))]
            )
        if hasattr(self.client, "search"):
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=flt,
                limit=top_k,
            )
        else:
            response = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                query_filter=flt,
                limit=top_k,
            )
            results = response.points
        out = []
        for res in results:
            payload = res.payload or {}
            chunk_id = str(payload.get("chunk_id", res.id))
            out.append((chunk_id, float(res.score), payload))
        return out

