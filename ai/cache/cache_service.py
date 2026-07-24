"""
ai/cache/cache_service.py

Single entry point for all cache operations.
All routes import from here — not from semantic_cache directly.

Functions:
  cache_get(question, document_id, strategy)   check cache before LLM
  cache_set(question, answer, document_id, ..) store after LLM responds
  cache_invalidate(document_id)                clear cache when doc deleted
  cache_stats()                                Redis stats for health check
  cache_flush()                                flush entire cache (admin)

Location: ai/cache/cache_service.py
"""

import logging
from typing import Optional

from ai.config import settings

logger = logging.getLogger(__name__)


def cache_get(
    question: str,
    document_id: Optional[str] = None,
    strategy: str = "hybrid",
) -> Optional[dict]:
    """
    Check semantic cache before calling Mistral.

    Returns cached result dict on HIT, None on MISS.
    Silently returns None if Redis is down or cache disabled.
    """
    if not settings.enable_semantic_cache:
        return None

    from ai.cache.semantic_cache import get as sc_get
    return sc_get(question, document_id=document_id, strategy=strategy)


def cache_set(
    question: str,
    answer: str,
    document_id: Optional[str] = None,
    strategy: str = "hybrid",
    extra_meta: dict = None,
) -> bool:
    """
    Store a question-answer pair in the semantic cache after Mistral responds.

    Returns True if stored, False if Redis down or cache disabled.
    Never raises — cache failures are silent.
    """
    if not settings.enable_semantic_cache:
        return False

    if not answer or len(answer.strip()) < 10:
        return False

    from ai.cache.semantic_cache import set as sc_set
    return sc_set(
        question=question,
        answer=answer,
        document_id=document_id,
        strategy=strategy,
        extra_meta=extra_meta,
    )


def cache_invalidate(document_id: str) -> int:
    """
    Invalidate all cache entries for a document.
    Called automatically when a contract is deleted.

    Returns number of entries cleared.
    """
    if not settings.enable_semantic_cache:
        return 0

    from ai.cache.semantic_cache import invalidate
    return invalidate(document_id)


def cache_stats() -> dict:
    """Return cache statistics for health check and admin endpoints."""
    from ai.cache.semantic_cache import get_stats
    return get_stats()


def cache_flush() -> bool:
    """Flush the entire LexMind cache namespace (admin use only)."""
    from ai.cache.semantic_cache import flush_all
    return flush_all()
