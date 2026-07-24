"""
ai/api/routes/cache_admin.py

Cache management endpoints.

GET  /cache/stats              Redis stats and cache metrics
POST /cache/invalidate         invalidate cache for a document
POST /cache/flush              flush entire cache (admin only)

Location: ai/api/routes/cache_admin.py
"""

import logging
from fastapi import APIRouter
from pydantic import BaseModel

from ai.cache.cache_service import cache_stats, cache_invalidate, cache_flush

router = APIRouter()
logger = logging.getLogger(__name__)


class InvalidateRequest(BaseModel):
    document_id: str


@router.get("/stats")
async def get_cache_stats():
    """
    Return Redis connection status and cache metrics.
    Useful for monitoring and health dashboards.
    """
    stats = cache_stats()
    return {"status": "ok", "cache": stats}


@router.post("/invalidate")
async def invalidate_document_cache(req: InvalidateRequest):
    """
    Invalidate all cached answers for a specific document.
    Call this after re-ingesting a contract or updating its content.
    """
    deleted = cache_invalidate(req.document_id)
    logger.info(f"Cache invalidated for doc={req.document_id}: {deleted} entries")
    return {
        "status":     "ok",
        "document_id": req.document_id,
        "deleted":    deleted,
    }


@router.post("/flush")
async def flush_cache():
    """
    Flush the entire LexMind cache namespace.
    Use with caution — clears all cached answers.
    """
    success = cache_flush()
    return {"status": "ok" if success else "failed", "flushed": success}