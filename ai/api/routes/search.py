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
    query:   str
    hits:    list[SearchHit]
    total:   int


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
        results = similarity_search(
            query=req.query,
            document_id=req.document_id,
            k=req.k,
        )
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

    return SearchResponse(query=req.query, hits=hits, total=len(hits))