"""Qdrant-backed vector index implementation.

This module is intentionally Qdrant-only. If Qdrant is not available,
initialization fails fast so no local fallback backend is used.
"""

import hashlib
import numpy as np
from typing import List, Dict, Optional, Any

from rag.config import ARTIFACTS_DIR, EMBEDDING_DIMENSIONS
from rag.chunker import Chunk

# ---------------------------------------------------------------------------
# Qdrant dependency
# ---------------------------------------------------------------------------
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct,
        Filter, FieldCondition, MatchValue,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
QDRANT_LOCAL_PATH = str(ARTIFACTS_DIR / "qdrant_storage")

# ===================================================================
# Qdrant Index API
# ===================================================================

class VectorIndex:
    """
    Vector index backed by local-persistent Qdrant only.
    """
    
    def __init__(self, collection_name: str, use_qdrant: bool = True):
        """
        Initialize the vector index.
        
        Args:
            collection_name: Name of the vector collection
            use_qdrant: Must be True. Kept for backward compatibility.
        """
        self.collection_name = collection_name
        if not use_qdrant:
            raise ValueError("VectorIndex is Qdrant-only; use_qdrant must be True.")
        if not QDRANT_AVAILABLE:
            raise RuntimeError(
                "qdrant-client is not installed. Install it to run this project "
                "(no local fallback backend is supported)."
            )

        self.client = QdrantClient(path=QDRANT_LOCAL_PATH)
        collections = [c.name for c in self.client.get_collections().collections]
        if collection_name not in collections:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIMENSIONS,
                    distance=Distance.COSINE,
                ),
            )
            print(f"  Created Qdrant collection '{collection_name}'")
        else:
            count = self.client.count(collection_name).count
            print(f"  Loaded Qdrant collection '{collection_name}' ({count} points)")
        self.backend = "qdrant"
    
    def upsert_chunks(self, chunks: List[Chunk], embeddings: np.ndarray):
        """
        Insert chunks with their embeddings into the vector store.
        
        Args:
            chunks: List of Chunk objects
            embeddings: numpy array of shape (len(chunks), dim)
        """
        points = []
        for i, chunk in enumerate(chunks):
            payload = chunk.to_dict()
            point_id = int(hashlib.sha1(chunk.chunk_id.encode("utf-8")).hexdigest()[:16], 16)
            points.append(PointStruct(
                # Qdrant local mode requires numeric ID or UUID string.
                id=point_id,
                vector=embeddings[i].tolist(),
                payload=payload,
            ))

        batch_size = 100
        for bs in range(0, len(points), batch_size):
            batch = points[bs:bs + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )
    
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 15,
        company_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for the most similar chunks.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results
            company_filter: Optional company name filter
        
        Returns:
            List of dicts with keys: chunk_id, score, text, and all metadata
        """
        query_filter = None
        if company_filter:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="company",
                        match=MatchValue(value=company_filter),
                    )
                ]
            )

        if hasattr(self.client, "search"):
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector.tolist(),
                query_filter=query_filter,
                limit=top_k,
            )
        else:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector.tolist(),
                query_filter=query_filter,
                limit=top_k,
            )
            results = getattr(response, "points", response)

        output = []
        for r in results:
            item = dict(r.payload)
            item["score"] = r.score
            output.append(item)
        return output
    
    def count(self) -> int:
        """Return number of indexed vectors."""
        return self.client.count(self.collection_name).count
    
    def exists(self) -> bool:
        """Check if the collection has data."""
        try:
            return self.client.count(self.collection_name).count > 0
        except Exception:
            return False

    def clear(self) -> None:
        """Remove all vectors from the configured collection."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=EMBEDDING_DIMENSIONS,
                distance=Distance.COSINE,
            ),
        )
