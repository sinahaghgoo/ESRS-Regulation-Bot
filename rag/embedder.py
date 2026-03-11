"""
Embedding Module
================
Generates embeddings using the OpenAI API (text-embedding-3-large).
Includes caching to avoid redundant API calls and support resumability.
Uses the requests library directly to avoid dependency on the openai package.
"""

import json
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import requests

from rag.config import OPENAI_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS, ARTIFACTS_DIR


# ---------------------------------------------------------------------------
# Embedding cache (file-based for persistence across runs)
# ---------------------------------------------------------------------------

EMBEDDING_CACHE_DIR = ARTIFACTS_DIR / "embedding_cache"
EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_key(text: str, model: str) -> str:
    """Generate a deterministic cache key for a text+model pair."""
    raw = f"{model}::{text}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _get_cached_embedding(text: str, model: str) -> Optional[np.ndarray]:
    """Look up a cached embedding. Returns None if not found."""
    key = _cache_key(text, model)
    cache_file = EMBEDDING_CACHE_DIR / f"{key}.npy"
    if cache_file.exists():
        return np.load(str(cache_file))
    return None


def _save_embedding_to_cache(text: str, model: str, embedding: np.ndarray):
    """Save an embedding to the file cache."""
    key = _cache_key(text, model)
    cache_file = EMBEDDING_CACHE_DIR / f"{key}.npy"
    np.save(str(cache_file), embedding)


# ---------------------------------------------------------------------------
# OpenAI API embedding (via requests)
# ---------------------------------------------------------------------------

OPENAI_EMBEDDING_URL = "https://api.openai.com/v1/embeddings"


def embed_texts(
    texts: List[str],
    model: str = EMBEDDING_MODEL,
    batch_size: int = 50,
    max_retries: int = 3,
) -> np.ndarray:
    """
    Generate embeddings for a list of texts using the OpenAI API.
    
    Uses file-based caching to avoid redundant API calls.
    Processes texts in batches for efficiency.
    
    Args:
        texts: List of text strings to embed
        model: OpenAI embedding model name
        batch_size: Number of texts per API call
        max_retries: Maximum retry attempts on failure
    
    Returns:
        numpy array of shape (len(texts), embedding_dimensions)
    """
    if not texts:
        return np.array([])
    
    embeddings = [None] * len(texts)
    uncached_indices: List[int] = []
    
    # Check cache first
    for i, text in enumerate(texts):
        cached = _get_cached_embedding(text, model)
        if cached is not None:
            embeddings[i] = cached
        else:
            uncached_indices.append(i)
    
    cached_count = len(texts) - len(uncached_indices)
    if cached_count > 0:
        print(f"    Embeddings: {cached_count}/{len(texts)} from cache, "
              f"{len(uncached_indices)} to compute")
    
    # Batch process uncached texts
    if uncached_indices:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        
        for batch_start in range(0, len(uncached_indices), batch_size):
            batch_idx = uncached_indices[batch_start:batch_start + batch_size]
            batch_texts = [texts[i] for i in batch_idx]
            
            # Truncate very long texts (API limit ~8191 tokens)
            batch_texts = [t[:30000] if len(t) > 30000 else t for t in batch_texts]
            
            payload = {
                "model": model,
                "input": batch_texts,
                "dimensions": EMBEDDING_DIMENSIONS,
            }
            
            for attempt in range(max_retries):
                try:
                    resp = requests.post(
                        OPENAI_EMBEDDING_URL,
                        headers=headers,
                        json=payload,
                        timeout=120,
                    )

                    # Handle rate limiting with exponential backoff
                    if resp.status_code == 429:
                        retry_after = int(resp.headers.get("Retry-After", 2 ** (attempt + 2)))
                        wait_time = min(retry_after, 60)
                        print(f"    Rate limited (429). Waiting {wait_time}s...")
                        time.sleep(wait_time)
                        if attempt < max_retries - 1:
                            continue

                    resp.raise_for_status()
                    data = resp.json()

                    # Extract embeddings from response
                    for item in data["data"]:
                        idx_in_batch = item["index"]
                        emb_vec = np.array(item["embedding"], dtype=np.float32)
                        original_idx = batch_idx[idx_in_batch]
                        embeddings[original_idx] = emb_vec
                        # Cache the result
                        _save_embedding_to_cache(texts[original_idx], model, emb_vec)

                    break  # success

                except requests.exceptions.RequestException as e:
                    if attempt < max_retries - 1:
                        wait_time = min(2 ** (attempt + 1), 30)
                        print(f"    Embedding API error (attempt {attempt+1}): {e}")
                        print(f"    Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        raise RuntimeError(
                            f"Embedding API failed after {max_retries} attempts: {e}"
                        )
            
            # Rate limiting: brief pause between batches
            if batch_start + batch_size < len(uncached_indices):
                time.sleep(0.5)
    
    # Stack into a single numpy array
    result = np.stack(embeddings)
    return result


def embed_query(query: str, model: str = EMBEDDING_MODEL) -> np.ndarray:
    """
    Embed a single query string. Uses cache if available.
    
    Args:
        query: The query text
        model: Embedding model name
    
    Returns:
        1D numpy array of embedding dimensions
    """
    result = embed_texts([query], model=model)
    return result[0]
