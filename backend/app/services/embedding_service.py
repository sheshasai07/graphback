"""
GraphRAG Backend - Voyage AI Embedding Service
Uses Voyage AI API (voyage-4-lite) for high-performance embeddings.
Supports input_type="document" for indexing and input_type="query" for retrieval.
Includes resilient fallback when VOYAGE_API_KEY is not yet configured.
"""

from typing import List, Union, Optional
import asyncio
import hashlib
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger

from app.config import settings


class EmbeddingService:
    """
    Voyage AI embedding service using the voyage-4-lite model.
    Thread-safe and async-first with automatic batching and retry logic.
    """

    def __init__(self):
        self._async_client = None
        self._sync_client = None
        self._lock = asyncio.Lock()

    def _has_api_key(self) -> bool:
        return bool(
            settings.voyage_api_key
            and settings.voyage_api_key.strip()
            and "your_voyage_api_key" not in settings.voyage_api_key
        )

    def _get_async_client(self):
        if not self._has_api_key():
            return None
        if self._async_client is None:
            try:
                import voyageai
                self._async_client = voyageai.AsyncClient(api_key=settings.voyage_api_key)
            except Exception as e:
                logger.error(f"Failed to initialize Voyage AI AsyncClient: {e}")
                return None
        return self._async_client

    def _get_sync_client(self):
        if not self._has_api_key():
            return None
        if self._sync_client is None:
            try:
                import voyageai
                self._sync_client = voyageai.Client(api_key=settings.voyage_api_key)
            except Exception as e:
                logger.error(f"Failed to initialize Voyage AI Client: {e}")
                return None
        return self._sync_client

    @property
    def dimension(self) -> int:
        return settings.embedding_dimension

    @property
    def model_name(self) -> str:
        return settings.embedding_model_name

    # -------------------------------------------------------------------------
    # Deterministic Local Fallback (Used when API key is not yet set)
    # -------------------------------------------------------------------------

    def _deterministic_hash_embed(self, text: str) -> List[float]:
        """
        Produce a deterministic 1024-dimensional dense normalized embedding
        using token n-grams and md5 hashing.
        Guarantees:
        - Consistent vector for identical text
        - Strictly L2-normalized
        - Prevents system crash when API key is pending
        """
        dim = self.dimension
        vec = np.zeros(dim, dtype=np.float32)
        words = text.lower().strip().split()
        if not words:
            vec[0] = 1.0
            return vec.tolist()

        tokens = list(words)
        for i in range(len(words) - 1):
            tokens.append(f"{words[i]}_{words[i+1]}")

        for tok in tokens:
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % dim
            sign = 1.0 if ((h >> 16) & 1) else -1.0
            weight = 1.0 / (1.0 + np.log(1.0 + len(tok)))
            vec[idx] += sign * weight

        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        else:
            vec[0] = 1.0
        return vec.tolist()

    # -------------------------------------------------------------------------
    # Core Voyage AI Calls
    # -------------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _call_voyage_async(
        self, texts: List[str], input_type: Optional[str] = "document"
    ) -> List[List[float]]:
        client = self._get_async_client()
        if client is None:
            raise RuntimeError("Voyage AI API key is not configured.")

        # Voyage AI API accepts texts, model, input_type, output_dimension
        kwargs = {
            "texts": texts,
            "model": self.model_name,
        }
        if input_type in ("document", "query"):
            kwargs["input_type"] = input_type
        if self.dimension in (256, 512, 1024, 2048):
            kwargs["output_dimension"] = self.dimension

        try:
            result = await client.embed(**kwargs)
            return result.embeddings
        except Exception as e:
            err_msg = str(e).lower()
            if "rate limit" in err_msg or "3 rpm" in err_msg or "429" in err_msg:
                logger.warning(f"Voyage AI 3 RPM rate limit encountered. Cooling off for 20s before retry...")
                await asyncio.sleep(20)
                result = await client.embed(**kwargs)
                return result.embeddings
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _call_voyage_sync(
        self, texts: List[str], input_type: Optional[str] = "document"
    ) -> List[List[float]]:
        client = self._get_sync_client()
        if client is None:
            raise RuntimeError("Voyage AI API key is not configured.")

        kwargs = {
            "texts": texts,
            "model": self.model_name,
        }
        if input_type in ("document", "query"):
            kwargs["input_type"] = input_type
        if self.dimension in (256, 512, 1024, 2048):
            kwargs["output_dimension"] = self.dimension

        try:
            result = client.embed(**kwargs)
            return result.embeddings
        except Exception as e:
            err_msg = str(e).lower()
            if "rate limit" in err_msg or "3 rpm" in err_msg or "429" in err_msg:
                import time
                logger.warning(f"Voyage AI 3 RPM rate limit encountered. Cooling off for 20s before retry...")
                time.sleep(20)
                result = client.embed(**kwargs)
                return result.embeddings
            raise

    # -------------------------------------------------------------------------
    # Public Embedding APIs
    # -------------------------------------------------------------------------

    async def aembed(
        self, texts: Union[str, List[str]], input_type: Optional[str] = "document"
    ) -> List[List[float]]:
        """
        Async batch embedding with Voyage AI.
        input_type: "document" for ingested chunks/entities, "query" for retrieval.
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return []

        if not self._has_api_key():
            logger.debug(
                f"VOYAGE_API_KEY not configured. Using deterministic local fallback embeddings (dim={self.dimension})."
            )
            return [self._deterministic_hash_embed(t) for t in texts]

        # Batch in chunks of up to 128 (Voyage maximum batch size)
        batch_size = 128
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                emb = await self._call_voyage_async(batch, input_type=input_type)
                all_embeddings.extend(emb)
            except Exception as e:
                logger.warning(f"Voyage AI async call failed ({e}). Falling back to local hash embeddings.")
                all_embeddings.extend([self._deterministic_hash_embed(t) for t in batch])

        return all_embeddings

    async def aembed_single(
        self, text: str, input_type: Optional[str] = "document"
    ) -> List[float]:
        """Async single text embedding."""
        res = await self.aembed([text], input_type=input_type)
        return res[0]

    def embed(
        self, texts: Union[str, List[str]], input_type: Optional[str] = "document"
    ) -> List[List[float]]:
        """
        Synchronous batch embedding with Voyage AI.
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return []

        if not self._has_api_key():
            return [self._deterministic_hash_embed(t) for t in texts]

        batch_size = 128
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                emb = self._call_voyage_sync(batch, input_type=input_type)
                all_embeddings.extend(emb)
            except Exception as e:
                logger.warning(f"Voyage AI sync call failed ({e}). Falling back to local hash embeddings.")
                all_embeddings.extend([self._deterministic_hash_embed(t) for t in batch])

        return all_embeddings

    def embed_single(
        self, text: str, input_type: Optional[str] = "document"
    ) -> List[float]:
        """Synchronous single text embedding."""
        return self.embed(text, input_type=input_type)[0]

    def cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a = np.array(vec_a, dtype=np.float32)
        b = np.array(vec_b, dtype=np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-6 or norm_b < 1e-6:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


# Singleton
embedding_service = EmbeddingService()
