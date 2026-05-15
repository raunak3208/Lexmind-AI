"""
ai/rag/embeddings.py

Mistral Embeddings wrapper — FREE tier, no cost.
Model: mistral-embed  (1024-dim, great for legal text)

Usage:
    from ai.rag.embeddings import get_embeddings
    embed = get_embeddings()
"""

import logging
from functools import lru_cache

from langchain_mistralai import MistralAIEmbeddings

from ai.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embeddings() -> MistralAIEmbeddings:
    """
    Returns a singleton MistralAIEmbeddings instance.
    Cached so we don't create a new HTTP client on every call.

    Mistral free tier limits:
      - 1 req/sec  (we handle this via LangChain's built-in retry)
      - 500k tokens/month  (plenty for dev / small production)
    """
    logger.info(f"Initialising Mistral embeddings: {settings.mistral_embed_model}")

    embeddings = MistralAIEmbeddings(
        model=settings.mistral_embed_model,
        mistral_api_key=settings.mistral_api_key,
    )
    return embeddings