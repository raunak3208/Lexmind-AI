"""
ai/rag/hybrid_retriever.py

Hybrid retrieval using Reciprocal Rank Fusion (RRF).

Pipeline:
  1. Dense retrieval  — Chroma vector search (BGE embeddings)
  2. Sparse retrieval — BM25 keyword search
  3. RRF fusion       — merge both ranked lists into one

RRF formula:  score(d) = Σ  1 / (k + rank(d))
  where k=60 is a smoothing constant (standard value from original paper).
  Documents appearing in both lists get combined scores — naturally boosted.
  Documents only in one list still contribute their single-list score.

Why RRF over weighted sum:
  - No tuning needed — k=60 works well across domains
  - Rank-based not score-based — handles scale differences between
    BM25 scores (unbounded) and cosine similarity (0 to 1)
  - Consistently outperforms weighted averaging in benchmarks

Location: ai/rag/hybrid_retriever.py
"""

import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
import pydantic

from ai.config import settings
from ai.rag.bm25_store import get_bm25_store

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    dense_docs: list[Document],
    sparse_results: list[tuple[float, Document]],
    k: int = None,
    top_n: int = None,
) -> list[Document]:
    """
    Merge dense and sparse results using Reciprocal Rank Fusion.

    Args:
        dense_docs:     ranked list from Chroma (index 0 = best)
        sparse_results: list of (bm25_score, doc) from BM25Store.search()
        k:              RRF smoothing constant (default 60)
        top_n:          number of final results to return

    Returns:
        Deduplicated list of Documents sorted by combined RRF score,
        with rrf_score, dense_rank, sparse_rank added to metadata.
    """
    k      = k or settings.hybrid_rrf_k
    top_n  = top_n or settings.retriever_k

    rrf_scores: dict[str, float]    = {}
    doc_map:    dict[str, Document] = {}
    dense_rank: dict[str, int]      = {}
    sparse_rank: dict[str, int]     = {}

    # Process dense results
    for rank, doc in enumerate(dense_docs, start=1):
        key = _doc_key(doc)
        rrf_scores[key]  = rrf_scores.get(key, 0.0) + (1.0 / (k + rank))
        doc_map[key]     = doc
        dense_rank[key]  = rank

    # Process sparse results
    for rank, (score, doc) in enumerate(sparse_results, start=1):
        key = _doc_key(doc)
        rrf_scores[key]   = rrf_scores.get(key, 0.0) + (1.0 / (k + rank))
        sparse_rank[key]  = rank
        if key not in doc_map:
            doc_map[key] = doc

    # Sort by combined RRF score
    sorted_keys = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

    results = []
    for key in sorted_keys[:top_n]:
        doc = doc_map[key]
        doc.metadata["rrf_score"]    = round(rrf_scores[key], 6)
        doc.metadata["dense_rank"]   = dense_rank.get(key, -1)
        doc.metadata["sparse_rank"]  = sparse_rank.get(key, -1)
        doc.metadata["in_both"]      = key in dense_rank and key in sparse_rank
        results.append(doc)

    logger.info(
        f"RRF fusion: dense={len(dense_docs)}  sparse={len(sparse_results)}  "
        f"merged={len(results)}  in_both={sum(1 for d in results if d.metadata.get('in_both'))}"
    )
    return results


def _doc_key(doc: Document) -> str:
    """Unique key for a chunk — document_id + chunk_index."""
    meta = doc.metadata
    doc_id  = meta.get("document_id", "unknown")
    chunk_i = meta.get("chunk_index", hash(doc.page_content[:100]))
    return f"{doc_id}:{chunk_i}"


def hybrid_search(
    query: str,
    document_id: Optional[str] = None,
    k: int = None,
) -> list[Document]:
    """
    Standalone hybrid search function.
    Runs dense + BM25, merges with RRF, returns final list.

    Used directly by tools and comparison routes.

    Args:
        query:       search query
        document_id: optional document filter
        k:           number of final results

    Returns:
        RRF-merged Documents with rrf_score in metadata
    """
    from ai.rag.vector_store import similarity_search

    k = k or settings.retriever_k

    # Dense search — fetch more than k so RRF has good candidates
    fetch_k = k * 3
    dense_docs = similarity_search(query, document_id=document_id, k=fetch_k)

    # Sparse BM25 search
    bm25 = get_bm25_store()
    sparse_results = bm25.search(query, k=fetch_k, document_id=document_id)

    if not dense_docs and not sparse_results:
        logger.warning("Hybrid search: both dense and sparse returned nothing")
        return []

    return reciprocal_rank_fusion(
        dense_docs=dense_docs,
        sparse_results=sparse_results,
        top_n=k,
    )


class HybridRetriever(BaseRetriever):
    """
    LangChain-compatible retriever wrapping hybrid_search().
    Plugs directly into RAG chains and agents.
    """
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    document_id: Optional[str] = None
    k: int = 5

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        return hybrid_search(query, document_id=self.document_id, k=self.k)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        return hybrid_search(query, document_id=self.document_id, k=self.k)


def get_hybrid_retriever(
    document_id: Optional[str] = None,
    k: int = None,
) -> HybridRetriever:
    """
    Return a HybridRetriever instance.
    Called by get_retriever() factory when strategy='hybrid'.

    Args:
        document_id: optional document filter
        k:           number of results

    Returns:
        HybridRetriever (LangChain BaseRetriever)
    """
    k = k or settings.retriever_k

    logger.info(
        f"[Hybrid Retriever]  k={k}  "
        f"doc_filter={document_id or 'ALL'}  "
        f"rrf_k={settings.hybrid_rrf_k}"
    )
    return HybridRetriever(document_id=document_id, k=k)