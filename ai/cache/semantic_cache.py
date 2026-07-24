"""
ai/cache/semantic_cache.py

Semantic cache using Redis + BGE embeddings + cosine similarity.

How it works:
  WRITE (cache miss):
    1. Embed the question with BGE (local, free)
    2. Store embedding as bytes in Redis  →  key: {prefix}:emb:{hash}
    3. Store answer as string in Redis    →  key: {prefix}:ans:{hash}
    4. Store metadata (doc_id, strategy)  →  key: {prefix}:meta:{hash}
    5. Add hash to a Redis sorted set for LRU eviction tracking

  READ (cache lookup):
    1. Embed the incoming question
    2. Fetch all cached embedding keys (batched scan)
    3. Compute cosine similarity between new embedding and each cached one
    4. If best similarity >= threshold (0.92) → return cached answer (HIT)
    5. Else → return None (MISS) → call Mistral → write to cache

  EVICTION:
    TTL-based (24h default) — Redis handles expiry automatically
    Max size enforced by evicting lowest-scored entries when limit reached

Why 0.92 threshold:
  "What are the payment terms?" and "What is the payment schedule?"
  have similarity ~0.88 → different enough to re-query
  "What are the payment terms?" asked twice has similarity ~0.99 → cache hit
  "Payment terms?" has similarity ~0.94 → cache hit (same intent)

Location: ai/cache/semantic_cache.py
"""

import json
import hashlib
import logging
import struct
import time
from typing import Optional

import numpy as np

from ai.config import settings

logger = logging.getLogger(__name__)

EMB_KEY  = lambda h: f"{settings.cache_key_prefix}:emb:{h}"
ANS_KEY  = lambda h: f"{settings.cache_key_prefix}:ans:{h}"
META_KEY = lambda h: f"{settings.cache_key_prefix}:meta:{h}"
LRU_KEY  = f"{settings.cache_key_prefix}:lru"


def _embed(text: str) -> np.ndarray:
    """
    Embed a text string using BGE (same model used for document chunks).
    Returns a normalised float32 numpy array.
    """
    from ai.rag.embeddings import get_embeddings
    embeddings = get_embeddings()
    vec = embeddings.embed_query(text)
    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr


def _arr_to_bytes(arr: np.ndarray) -> bytes:
    """Serialise numpy float32 array to raw bytes for Redis storage."""
    return arr.astype(np.float32).tobytes()


def _bytes_to_arr(b: bytes) -> np.ndarray:
    """Deserialise raw bytes back to numpy float32 array."""
    return np.frombuffer(b, dtype=np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two normalised vectors."""
    return float(np.dot(a, b))


def _cache_key_hash(question: str, document_id: Optional[str]) -> str:
    """
    Generate a short hash for the (question, document_id) pair.
    Used as a namespace prefix — not the lookup key (similarity handles that).
    """
    raw = f"{question}||{document_id or 'ALL'}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def get(
    question: str,
    document_id: Optional[str] = None,
    strategy: str = "hybrid",
) -> Optional[dict]:
    """
    Look up a question in the semantic cache.

    Args:
        question:    user question string
        document_id: optional document filter (cache is per-document)
        strategy:    retrieval strategy (included in cache key namespace)

    Returns:
        Cached result dict if hit:
          { answer, document_id, question, strategy, cache_hit, similarity }
        None if miss.
    """
    if not settings.enable_semantic_cache:
        return None

    from ai.cache.redis_client import get_redis_client
    redis = get_redis_client()

    try:
        # Scope cache by document_id — different docs, different answers
        scope    = f"{document_id or 'ALL'}:{strategy}"
        scan_pat = f"{settings.cache_key_prefix}:emb:{scope}:*"

        # Embed incoming question
        q_vec = _embed(question)

        best_sim  = -1.0
        best_hash = None

        # Scan all cached embeddings for this scope
        cursor = 0
        while True:
            cursor, keys = redis.scan(cursor, match=scan_pat, count=200)
            for key in keys:
                raw = redis.get(key)
                if raw is None:
                    continue
                cached_vec = _bytes_to_arr(raw)
                sim = _cosine(q_vec, cached_vec)
                if sim > best_sim:
                    best_sim  = sim
                    best_hash = key.decode() if isinstance(key, bytes) else key

            if cursor == 0:
                break

        if best_hash and best_sim >= settings.cache_similarity_threshold:
            # Extract hash suffix from key
            h = best_hash.split(":")[-1]
            ans_key  = f"{settings.cache_key_prefix}:ans:{scope}:{h}"
            meta_key = f"{settings.cache_key_prefix}:meta:{scope}:{h}"

            raw_ans  = redis.get(ans_key)
            raw_meta = redis.get(meta_key)

            if raw_ans:
                answer = raw_ans.decode() if isinstance(raw_ans, bytes) else raw_ans
                meta   = json.loads(raw_meta) if raw_meta else {}

                # Update LRU score
                redis.zadd(LRU_KEY, {best_hash: time.time()})

                logger.info(
                    f"[CACHE HIT]  sim={best_sim:.4f}  "
                    f"doc={document_id or 'ALL'}  "
                    f"q='{question[:50]}'"
                )

                return {
                    "answer":      answer,
                    "document_id": document_id,
                    "question":    question,
                    "strategy":    strategy,
                    "cache_hit":   True,
                    "similarity":  round(best_sim, 4),
                    "meta":        meta,
                }

        logger.info(
            f"[CACHE MISS]  best_sim={best_sim:.4f}  "
            f"threshold={settings.cache_similarity_threshold}  "
            f"q='{question[:50]}'"
        )
        return None

    except Exception as e:
        logger.warning(f"Cache get failed — proceeding without cache: {e}")
        return None


def set(
    question: str,
    answer: str,
    document_id: Optional[str] = None,
    strategy: str = "hybrid",
    extra_meta: dict = None,
) -> bool:
    """
    Store a question-answer pair in the semantic cache.

    Args:
        question:    user question
        answer:      LLM-generated answer to cache
        document_id: document this answer belongs to
        strategy:    retrieval strategy used
        extra_meta:  optional metadata dict to store alongside

    Returns:
        True if stored successfully, False otherwise
    """
    if not settings.enable_semantic_cache:
        return False

    from ai.cache.redis_client import get_redis_client
    redis = get_redis_client()

    try:
        scope = f"{document_id or 'ALL'}:{strategy}"
        h     = _cache_key_hash(question, document_id)

        emb_key  = f"{settings.cache_key_prefix}:emb:{scope}:{h}"
        ans_key  = f"{settings.cache_key_prefix}:ans:{scope}:{h}"
        meta_key = f"{settings.cache_key_prefix}:meta:{scope}:{h}"

        # Embed and store
        q_vec = _embed(question)
        ttl   = settings.cache_ttl_seconds

        redis.setex(emb_key,  ttl, _arr_to_bytes(q_vec))
        redis.setex(ans_key,  ttl, answer.encode())
        redis.setex(meta_key, ttl, json.dumps({
            "question":    question,
            "document_id": document_id,
            "strategy":    strategy,
            "stored_at":   time.time(),
            **(extra_meta or {}),
        }).encode())

        # Track in LRU sorted set
        redis.zadd(LRU_KEY, {emb_key: time.time()})

        # Evict oldest if over max size
        _evict_if_needed(redis)

        logger.info(
            f"[CACHE SET]  doc={document_id or 'ALL'}  "
            f"strategy={strategy}  "
            f"q='{question[:50]}'  "
            f"ttl={ttl}s"
        )
        return True

    except Exception as e:
        logger.warning(f"Cache set failed: {e}")
        return False


def invalidate(document_id: str) -> int:
    """
    Invalidate all cached answers for a specific document.
    Called when a document is deleted or re-ingested.

    Args:
        document_id: MongoDB document ID

    Returns:
        number of cache entries deleted
    """
    from ai.cache.redis_client import get_redis_client
    redis = get_redis_client()

    try:
        pattern = f"{settings.cache_key_prefix}:*:{document_id}:*"
        deleted = 0
        cursor  = 0

        while True:
            cursor, keys = redis.scan(cursor, match=pattern, count=200)
            if keys:
                redis.delete(*keys)
                deleted += len(keys)
            if cursor == 0:
                break

        logger.info(f"Cache invalidated {deleted} entries for doc={document_id}")
        return deleted

    except Exception as e:
        logger.warning(f"Cache invalidation failed: {e}")
        return 0


def flush_all() -> bool:
    """
    Flush the entire LexMind cache namespace.
    Does NOT flush other Redis data — only lexmind:cache:* keys.
    """
    from ai.cache.redis_client import get_redis_client
    redis = get_redis_client()

    try:
        pattern = f"{settings.cache_key_prefix}:*"
        cursor  = 0
        deleted = 0

        while True:
            cursor, keys = redis.scan(cursor, match=pattern, count=200)
            if keys:
                redis.delete(*keys)
                deleted += len(keys)
            if cursor == 0:
                break

        logger.info(f"Cache flushed: {deleted} keys deleted")
        return True

    except Exception as e:
        logger.warning(f"Cache flush failed: {e}")
        return False


def get_stats() -> dict:
    """
    Return cache statistics.
    Used by health check and admin endpoints.
    """
    from ai.cache.redis_client import get_redis_client, is_redis_available
    redis = get_redis_client()

    if not is_redis_available():
        return {
            "redis_connected": False,
            "cache_enabled":   settings.enable_semantic_cache,
            "total_entries":   0,
            "threshold":       settings.cache_similarity_threshold,
            "ttl_seconds":     settings.cache_ttl_seconds,
        }

    try:
        pattern = f"{settings.cache_key_prefix}:ans:*"
        count   = 0
        cursor  = 0

        while True:
            cursor, keys = redis.scan(cursor, match=pattern, count=200)
            count += len(keys)
            if cursor == 0:
                break

        return {
            "redis_connected": True,
            "cache_enabled":   settings.enable_semantic_cache,
            "total_entries":   count,
            "threshold":       settings.cache_similarity_threshold,
            "ttl_seconds":     settings.cache_ttl_seconds,
            "max_size":        settings.cache_max_size,
        }

    except Exception as e:
        return {"redis_connected": False, "error": str(e)}


def _evict_if_needed(redis) -> None:
    """
    Evict oldest entries from LRU set when cache exceeds max size.
    Removes the 10% oldest entries to avoid constant eviction churn.
    """
    try:
        total = redis.zcard(LRU_KEY)
        if total > settings.cache_max_size:
            evict_count = max(1, int(settings.cache_max_size * 0.1))
            oldest_keys = redis.zrange(LRU_KEY, 0, evict_count - 1)
            if oldest_keys:
                redis.delete(*oldest_keys)
                redis.zrem(LRU_KEY, *oldest_keys)
                logger.info(f"Cache evicted {len(oldest_keys)} oldest entries")
    except Exception as e:
        logger.warning(f"Cache eviction failed: {e}")
