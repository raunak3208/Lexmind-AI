"""
ai/rag/embeddings.py

Embedding model selector.
Supports two modes controlled by settings.use_bge_embeddings:

  BGE (default, recommended):
    Model  : BAAI/bge-small-en-v1.5  (fast) or BAAI/bge-large-en-v1.5 (best)
    Cost   : FREE — downloads once from HuggingFace, runs on local CPU
    Quality: significantly better than mistral-embed on legal/technical text
    Dims   : 384 (small) / 1024 (large)

  Mistral embed (fallback):
    Model  : mistral-embed
    Cost   : FREE tier — 500k tokens/month
    Quality: good general purpose
    Dims   : 1024
"""

import logging
from functools import lru_cache

from ai.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embeddings():
    """
    Return singleton embedding model.
    BGE is used by default (better quality, free, local).
    Falls back to Mistral embed if use_bge_embeddings=False in config.
    """
    if settings.use_bge_embeddings:
        return _get_bge_embeddings()
    return _get_mistral_embeddings()


def _get_bge_embeddings():
    """
    BGE embeddings via HuggingFace sentence-transformers.
    Downloads model on first run (~130MB for small, ~1.3GB for large).
    Subsequent runs load from local cache instantly.

    encode_kwargs normalize_embeddings=True is required for BGE —
    the model card explicitly states this for correct cosine similarity.
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings

    logger.info(f"Loading BGE embeddings: {settings.bge_model_name}")

    embeddings = HuggingFaceEmbeddings(
        model_name=settings.bge_model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    logger.info(f"BGE embeddings ready: {settings.bge_model_name}")
    return embeddings


def _get_mistral_embeddings():
    """
    Mistral embed fallback — requires MISTRAL_API_KEY and internet.
    Used when use_bge_embeddings=False.
    """
    from langchain_mistralai import MistralAIEmbeddings

    logger.info(f"Loading Mistral embeddings: {settings.mistral_embed_model}")

    embeddings = MistralAIEmbeddings(
        model=settings.mistral_embed_model,
        mistral_api_key=settings.mistral_api_key,
    )
    return embeddings