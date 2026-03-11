"""
Indexer Module
==============
Manages vector storage and retrieval using either:
  - Qdrant (preferred, if qdrant-client is available)
  - Local numpy-based fallback (for environments without Qdrant)

Both backends store embeddings alongside rich metadata per chunk,
enabling metadata-filtered retrieval.
"""

import json
import os
import shutil
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import asdict

from rag.config import ARTIFACTS_DIR, EMBEDDING_DIMENSIONS
from rag.chunker import Chunk

# ---------------------------------------------------------------------------
# Try to import Qdrant; fall back to local implementation
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
LOCAL_INDEX_DIR = ARTIFACTS_DIR / "local_index"


# ===================================================================
# Local Numpy-based Vector Store (fallback)
# ===================================================================

class LocalVectorStore:
    """
    Simple file-based vector store using numpy.
    Stores embeddings as .npy and metadata as .json.
    Supports cosine similarity search with metadata filtering.
    """
    
    def __init__(self, collection_name: str, storage_dir: Path = LOCAL_INDEX_DIR):
        self.collection_name = collection_name
        self.storage_dir = storage_dir / collection_name
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self._embeddings_path = self.storage_dir / "embeddings.npy"
        self._metadata_path = self.storage_dir / "metadata.json"
        self._ids_path = self.storage_dir / "ids.json"
        
        # In-memory state
        self.embeddings: Optional[np.ndarray] = None
        self.metadata: List[Dict] = []
        self.ids: List[str] = []
        
        # Load existing data if available
        self._load()
    
    def _load(self):
        """Load existing index from disk."""
        if self._embeddings_path.exists():
            self.embeddings = np.load(str(self._embeddings_path))
            with open(self._metadata_path, "r") as f:
                self.metadata = json.load(f)
            with open(self._ids_path, "r") as f:
                self.ids = json.load(f)
            print(f"    Loaded {len(self.ids)} vectors from local store '{self.collection_name}'")
    
    def _save(self):
        """Persist index to disk."""
        if self.embeddings is not None:
            np.save(str(self._embeddings_path), self.embeddings)
            with open(self._metadata_path, "w") as f:
                json.dump(self.metadata, f)
            with open(self._ids_path, "w") as f:
                json.dump(self.ids, f)
    
    def upsert(self, ids: List[str], embeddings: np.ndarray, metadata_list: List[Dict]):
        """
        Insert or update vectors with their metadata.
        
        Args:
            ids: List of unique vector IDs
            embeddings: numpy array of shape (n, dim)
            metadata_list: List of metadata dictionaries
        """
        if self.embeddings is None or len(self.ids) == 0:
            self.embeddings = embeddings
            self.metadata = list(metadata_list)
            self.ids = list(ids)
            self._save()
            return

        # Update existing ids in place and append new ones.
        id_to_idx = {id_: i for i, id_ in enumerate(self.ids)}
        emb_rows = [row for row in self.embeddings]

        for id_, emb, meta in zip(ids, embeddings, metadata_list):
            if id_ in id_to_idx:
                idx = id_to_idx[id_]
                emb_rows[idx] = emb
                self.metadata[idx] = meta
            else:
                id_to_idx[id_] = len(emb_rows)
                emb_rows.append(emb)
                self.metadata.append(meta)
                self.ids.append(id_)

        self.embeddings = np.vstack(emb_rows)
        
        self._save()

    def clear(self):
        """Delete all vectors from the local collection."""
        if self.storage_dir.exists():
            shutil.rmtree(self.storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings = None
        self.metadata = []
        self.ids = []
    
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        company_filter: Optional[str] = None,
    ) -> List[Tuple[str, float, Dict]]:
        """
        Search for nearest vectors using cosine similarity.
        
        Args:
            query_vector: 1D query embedding
            top_k: Number of results to return
            company_filter: Optional company name to filter by
        
        Returns:
            List of (chunk_id, score, metadata) tuples sorted by score desc
        """
        if self.embeddings is None or len(self.ids) == 0:
            return []
        
        # Apply company filter if specified
        if company_filter:
            mask = np.array([
                m.get("company", "") == company_filter 
                for m in self.metadata
            ])
            if not mask.any():
                return []
            filtered_emb = self.embeddings[mask]
            filtered_ids = [self.ids[i] for i in range(len(self.ids)) if mask[i]]
            filtered_meta = [self.metadata[i] for i in range(len(self.metadata)) if mask[i]]
        else:
            filtered_emb = self.embeddings
            filtered_ids = self.ids
            filtered_meta = self.metadata
        
        # Cosine similarity
        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-10)
        emb_norms = filtered_emb / (np.linalg.norm(filtered_emb, axis=1, keepdims=True) + 1e-10)
        scores = emb_norms @ query_norm
        
        # Top-k
        k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            results.append((filtered_ids[idx], float(scores[idx]), filtered_meta[idx]))
        
        return results
    
    def count(self) -> int:
        """Return the number of vectors in the store."""
        return len(self.ids)
    
    def exists(self) -> bool:
        """Check if this collection has any data."""
        return self._embeddings_path.exists()


# ===================================================================
# Unified Indexer API
# ===================================================================

class VectorIndex:
    """
    Unified vector index that uses Qdrant if available,
    or falls back to the local numpy-based store.
    
    Provides a consistent API for upserting chunks and searching.
    """
    
    def __init__(self, collection_name: str, use_qdrant: bool = True):
        """
        Initialize the vector index.
        
        Args:
            collection_name: Name of the vector collection
            use_qdrant: Whether to attempt using Qdrant
        """
        self.collection_name = collection_name
        self.use_qdrant = use_qdrant and QDRANT_AVAILABLE
        
        if self.use_qdrant:
            try:
                self.client = QdrantClient(path=QDRANT_LOCAL_PATH)
                # Create collection if it doesn't exist
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
            except Exception as e:
                print(f"  Qdrant init failed ({e}), falling back to local store")
                self.local_store = LocalVectorStore(collection_name)
                self.backend = "local"
        else:
            self.local_store = LocalVectorStore(collection_name)
            self.backend = "local"
    
    def upsert_chunks(self, chunks: List[Chunk], embeddings: np.ndarray):
        """
        Insert chunks with their embeddings into the vector store.
        
        Args:
            chunks: List of Chunk objects
            embeddings: numpy array of shape (len(chunks), dim)
        """
        if self.backend == "qdrant":
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
            
            # Upsert in batches
            batch_size = 100
            for bs in range(0, len(points), batch_size):
                batch = points[bs:bs + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                )
            
        else:
            ids = [c.chunk_id for c in chunks]
            metadata = [c.to_dict() for c in chunks]
            self.local_store.upsert(ids, embeddings, metadata)
    
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
        if self.backend == "qdrant":
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
        
        else:
            results = self.local_store.search(
                query_vector, top_k=top_k, company_filter=company_filter
            )
            output = []
            for chunk_id, score, meta in results:
                item = dict(meta)
                item["score"] = score
                output.append(item)
            return output
    
    def count(self) -> int:
        """Return number of indexed vectors."""
        if self.backend == "qdrant":
            return self.client.count(self.collection_name).count
        return self.local_store.count()
    
    def exists(self) -> bool:
        """Check if the collection has data."""
        if self.backend == "qdrant":
            try:
                return self.client.count(self.collection_name).count > 0
            except Exception:
                return False
        return self.local_store.exists()

    def clear(self) -> None:
        """Remove all vectors from the configured collection."""
        if self.backend == "qdrant":
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
            return
        self.local_store.clear()
