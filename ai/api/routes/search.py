"""
ai/api/routes/search.py

POST /search
Semantic search across all ingested contracts (or filtered to one document).
Returns top-k matching chunks with metadata.
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ai.rag.vector_store import similarity_search
from ai.guardrails.guardrail_service import protect_query
from ai.cache.cache_service import cache_get, cache_set

router = APIRouter()
logger = logging.getLogger(__name__)


class SearchRequest(BaseModel):
    query:       str
    document_id: Optional[str] = None  # None = search all documents
    k:           int = 5


class SearchHit(BaseModel):
    document_id: str
    filename:    str
    chunk_index: int
    page:        Optional[int]
    text:        str
    score_rank:  int   # 1 = most relevant


class SearchResponse(BaseModel):
    query:     str
    hits:      list[SearchHit]
    total:     int
    cache_hit: bool = False


@router.post("", response_model=SearchResponse)
async def semantic_search(req: SearchRequest):
    """
    Semantic search for clauses matching the query.

    Examples:
      - "limitation of liability clause"
      - "termination without cause"
      - "payment terms net 30"
      - "intellectual property ownership"
    """
    if not req.query.strip():
        raise HTTPException(status_code=422, detail="Query cannot be empty.")

    logger.info(
        f"Semantic search: '{req.query[:60]}'  "
        f"doc={req.document_id or 'ALL'}  k={req.k}"
    )

    try:
        # Guard query input
        guard = protect_query(req.query)
        if not guard["allowed"]:
            raise HTTPException(status_code=400, detail=guard["blocked_reason"])
        search_query = guard["text"]

        # Check semantic cache for search results
        cache_key_q = f"search:{search_query}"
        cached = cache_get(cache_key_q, document_id=req.document_id, strategy="search")
        if cached and "hits" in cached.get("meta", {}):
            logger.info(f"Search cache HIT: '{search_query[:50]}'")
            return SearchResponse(
                query=search_query,
                hits=[SearchHit(**h) for h in cached["meta"]["hits"]],
                total=cached["meta"]["total"],
                cache_hit=True,
            )

        strategy = req.strategy.lower()

        if strategy == "hybrid":
            from ai.rag.hybrid_retriever import hybrid_search
            results = hybrid_search(
                query=search_query,
                document_id=req.document_id,
                k=req.k,
            )
        else:
            results = similarity_search(
                query=search_query,
                document_id=req.document_id,
                k=req.k * 3 if req.rerank else req.k,
            )

        if req.rerank and results and strategy != "hybrid":
            from ai.rag.reranker import rerank
            results = rerank(search_query, results, top_n=req.k)

    except Exception as e:
        logger.exception(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

    hits = [
        SearchHit(
            document_id=doc.metadata.get("document_id", "unknown"),
            filename=doc.metadata.get("filename", "unknown"),
            chunk_index=doc.metadata.get("chunk_index", 0),
            page=doc.metadata.get("page"),
            text=doc.page_content,
            score_rank=rank + 1,
        )
        for rank, doc in enumerate(results)
    ]

    # Cache the search results
    cache_set(
        question=f"search:{search_query}",
        answer="__search__",
        document_id=req.document_id,
        strategy="search",
        extra_meta={"hits": [h.model_dump() for h in hits], "total": len(hits)},
    )

    return SearchResponse(query=search_query, hits=hits, total=len(hits), cache_hit=False)
