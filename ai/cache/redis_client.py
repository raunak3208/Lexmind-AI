"""
ai/cache/redis_client.py

Redis connection singleton with graceful fallback.

If Redis is not running, ALL cache operations silently no-op.
The system works exactly as before — just without caching.
No crashes, no errors shown to the user.

Location: ai/cache/redis_client.py
"""

import logging
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)


class _NoOpRedis:
    """
    Drop-in Redis replacement used when Redis is unavailable.
    Every method is a no-op — system works normally without cache.
    """

    def get(self, *a, **kw):         return None
    def set(self, *a, **kw):         return None
    def setex(self, *a, **kw):       return None
    def delete(self, *a, **kw):      return None
    def exists(self, *a, **kw):      return 0
    def keys(self, *a, **kw):        return []
    def flushdb(self, *a, **kw):     return None
    def ping(self, *a, **kw):        raise ConnectionError("NoOpRedis")
    def dbsize(self, *a, **kw):      return 0
    def pipeline(self, *a, **kw):    return self
    def execute(self, *a, **kw):     return []
    def __enter__(self):             return self
    def __exit__(self, *a):          pass


@lru_cache(maxsize=1)
def get_redis_client():
    """
    Return a Redis client instance (or NoOpRedis if unavailable).
    Cached — one connection pool shared across the whole app.

    Returns:
        redis.Redis instance, or _NoOpRedis if connection fails
    """
    from ai.config import settings

    try:
        import redis

        client = redis.from_url(
            settings.redis_url,
            decode_responses=False,   # we handle bytes ourselves for numpy arrays
            socket_connect_timeout=2,
            socket_timeout=2,
            retry_on_timeout=False,
        )
        client.ping()
        logger.info(f"Redis connected: {settings.redis_url}")
        return client

    except Exception as e:
        logger.warning(
            f"Redis unavailable ({e}) — semantic cache disabled, "
            f"system works normally without it"
        )
        return _NoOpRedis()


def is_redis_available() -> bool:
    """Check if Redis is actually connected (not NoOp)."""
    client = get_redis_client()
    if isinstance(client, _NoOpRedis):
        return False
    try:
        client.ping()
        return True
    except Exception:
        return False
