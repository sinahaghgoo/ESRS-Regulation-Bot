"""Embeddings for chunk and query vectors with cache support."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from rag.config.settings import ARTIFACTS_DIR, get_openai_api_key


class EmbeddingService:
    """Embedding wrapper with OpenAI primary path and deterministic fallback."""

    def __init__(self, model: str = "text-embedding-3-large", dims: int = 3072) -> None:
        self.model = model
        self.dims = dims
        self.cache_path = ARTIFACTS_DIR / "embedding_cache.json"
        self.cache: dict[str, list[float]] = {}
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        if self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                self.cache = {}
        self._openai = None
        key = get_openai_api_key(required=False)
        if key:
            try:
                from langchain_openai import OpenAIEmbeddings  # type: ignore

                self._openai = OpenAIEmbeddings(model=self.model, api_key=key)
            except Exception:
                self._openai = None

    def _persist(self) -> None:
        self.cache_path.write_text(json.dumps(self.cache), encoding="utf-8")

    def _key(self, text: str) -> str:
        return hashlib.sha1(f"{self.model}:{text}".encode("utf-8")).hexdigest()

    def _fallback_embedding(self, text: str) -> list[float]:
        # Deterministic hashed bag-of-words style embedding for offline runs.
        vec = np.zeros(self.dims, dtype=np.float32)
        for tok in text.lower().split():
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dims
            vec[idx] += 1.0
        norm = np.linalg.norm(vec) or 1.0
        return (vec / norm).astype(float).tolist()

    def embed_text(self, text: str) -> list[float]:
        """Embed one text string with caching."""
        key = self._key(text)
        if key in self.cache:
            return self.cache[key]
        if self._openai is not None:
            emb = self._openai.embed_query(text)
        else:
            emb = self._fallback_embedding(text)
        self.cache[key] = emb
        if len(self.cache) % 50 == 0:
            self._persist()
        return emb

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        out: list[list[float]] = []
        uncached_idx: list[int] = []
        uncached_texts: list[str] = []
        for i, txt in enumerate(texts):
            key = self._key(txt)
            if key in self.cache:
                out.append(self.cache[key])
            else:
                out.append([])
                uncached_idx.append(i)
                uncached_texts.append(txt)
        if uncached_texts:
            if self._openai is not None:
                vectors = self._openai.embed_documents(uncached_texts)
            else:
                vectors = [self._fallback_embedding(t) for t in uncached_texts]
            for idx, vec in zip(uncached_idx, vectors):
                txt = texts[idx]
                key = self._key(txt)
                out[idx] = vec
                self.cache[key] = vec
            self._persist()
        return out

