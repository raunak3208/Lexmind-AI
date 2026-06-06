"""
ai/rag/reranker.py

Cross-encoder reranking for LexMind retrieval pipeline.

How it works:
  1. Retriever fetches top-k candidates from Chroma (fast, approximate)
  2. Reranker scores each candidate against the query (slow, exact)
  3. Returns top-n reranked results — much higher precision

Why this matters:
  Vector similarity is fast but approximate.
  Cross-encoders see (query, document) together — far more accurate.
  Example: query "can the contract be ended early?" matches
  "termination for convenience" better via cross-encoder than vector distance.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Runs fully local on CPU (no API cost, no internet after first download)
  - ~90MB download once
  - Fast inference even on CPU (~50ms per batch)
  - Trained on MS MARCO passage ranking — strong general retrieval
"""

import logging
from functools import lru_cache
from typing import Optional

from langchain_core.documents import Document

from ai.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_cross_encoder():
    """
    Load cross-encoder model once and cache it.
    Downloads ~90MB on first run, loads from cache after.
    """
    from sentence_transformers import CrossEncoder

    logger.info(f"Loading reranker: {settings.reranker_model_name}")
    model = CrossEncoder(settings.reranker_model_name)
    logger.info("Reranker ready")
    return model


def rerank(
    query: str,
    documents: list[Document],
    top_n: int = None,
) -> list[Document]:
    """
    Rerank a list of retrieved documents using cross-encoder scoring.

    Args:
        query:     the user's search query or question
        documents: candidate documents from Chroma retriever
        top_n:     how many top documents to return (default from config)

    Returns:
        top_n documents sorted by cross-encoder score (best first)
        Each document gets a rerank_score added to its metadata.
    """
    top_n = top_n or settings.reranker_top_n

    if not documents:
        logger.info("Reranker: no documents to rerank")
        return []

    if not settings.use_reranker:
        logger.info("Reranker disabled — returning original order")
        return documents[:top_n]

    model = _get_cross_encoder()

    # Build (query, passage) pairs for the cross-encoder
    pairs = [(query, doc.page_content) for doc in documents]

    # Score all pairs in one batch — fast even on CPU
    scores = model.predict(pairs)

    # Zip scores with documents and sort descending
    scored_docs = sorted(
        zip(scores, documents),
        key=lambda x: x[0],
        reverse=True,
    )

    # Attach score to metadata for transparency
    reranked = []
    for rank, (score, doc) in enumerate(scored_docs[:top_n]):
        doc.metadata["rerank_score"] = round(float(score), 4)
        doc.metadata["rerank_position"] = rank + 1
        reranked.append(doc)

    logger.info(
        f"Reranker: {len(documents)} candidates → top {len(reranked)} "
        f"scores={[round(float(s), 3) for s, _ in scored_docs[:top_n]]}"
    )
    return reranked


def rerank_with_threshold(
    query: str,
    documents: list[Document],
    threshold: float = 0.0,
    top_n: int = None,
) -> list[Document]:
    """
    Rerank and also filter out documents below a minimum score threshold.
    Useful for search where you want to return nothing rather than bad results.

    Args:
        query:      search query
        documents:  candidate documents
        threshold:  minimum cross-encoder score to keep (0.0 = keep all)
        top_n:      max results to return

    Returns:
        documents above threshold, sorted by score
    """
    top_n = top_n or settings.reranker_top_n

    if not documents or not settings.use_reranker:
        return documents[:top_n]

    model = _get_cross_encoder()
    pairs = [(query, doc.page_content) for doc in documents]
    scores = model.predict(pairs)

    scored_docs = sorted(
        zip(scores, documents),
        key=lambda x: x[0],
        reverse=True,
    )

    filtered = [
        (score, doc) for score, doc in scored_docs
        if float(score) >= threshold
    ][:top_n]

    reranked = []
    for rank, (score, doc) in enumerate(filtered):
        doc.metadata["rerank_score"] = round(float(score), 4)
        doc.metadata["rerank_position"] = rank + 1
        reranked.append(doc)

    logger.info(
        f"Reranker (threshold={threshold}): "
        f"{len(documents)} in → {len(reranked)} out"
    )
    return reranked