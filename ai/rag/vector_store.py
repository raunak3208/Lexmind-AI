"""
ai/rag/vector_store.py

Local Chroma vector store — 100% free, persists to disk.
Handles:
  - adding document chunks after ingestion
  - deleting all chunks belonging to a document
  - returning a retriever for RAG queries
  - returning a raw collection for similarity comparisons

Collection name: "lexmind_contracts"
Each chunk is stored with metadata: document_id, filename, chunk_index, etc.
"""

import logging
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from ai.config import settings
from ai.rag.embeddings import get_embeddings
from ai.rag.bm25_store import get_bm25_store

logger = logging.getLogger(__name__)

COLLECTION_NAME = "lexmind_contracts"


def _get_store() -> Chroma:
    """
    Return (or create) the persistent Chroma collection.
    Data is written to disk at settings.chroma_persist_dir.
    """
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_persist_dir,
    )


def add_chunks(chunks: List[Document]) -> List[str]:
    """
    Embed and store a list of document chunks in Chroma.

    Args:
        chunks: output of enrich_metadata() — must have document_id in metadata

    Returns:
        list of Chroma-assigned IDs for the stored vectors
    """
    if not chunks:
        logger.warning("add_chunks called with empty list — nothing stored")
        return []

    store = _get_store()
    doc_id = chunks[0].metadata.get("document_id", "unknown")
    logger.info(f"Embedding & storing {len(chunks)} chunks for document_id={doc_id}")

    ids = store.add_documents(chunks)
    logger.info(f"Stored {len(ids)} vectors in Chroma")

    # Mirror to BM25 sparse index
    bm25 = get_bm25_store()
    bm25.add_chunks(chunks)

    return ids


def delete_document(document_id: str) -> None:
    """
    Remove all chunks belonging to a specific document from the vector store.
    Called when a user deletes a contract.

    Args:
        document_id: the document's unique ID (from MongoDB / Node backend)
    """
    store = _get_store()

    # Chroma supports metadata filtering on delete
    store._collection.delete(where={"document_id": document_id})
    logger.info(f"Deleted all vectors for document_id={document_id}")

    # Mirror deletion to BM25 sparse index
    bm25 = get_bm25_store()
    bm25.delete_document(document_id)

    # Invalidate all semantic cache entries for this document
    try:
        from ai.cache.cache_service import cache_invalidate
        cache_invalidate(document_id)
    except Exception as e:
        logger.warning(f"Cache invalidation failed: {e}")

    # Delete knowledge graph for this document
    try:
        from ai.knowledge_graph.graph_service import delete_document_graph
        delete_document_graph(document_id)
    except Exception as e:
        logger.warning(f"Graph deletion failed: {e}")


def get_retriever(
    document_id: Optional[str] = None,
    k: int = None,
):
    """
    Return a LangChain retriever for semantic search.

    Args:
        document_id: if provided, restrict search to this document only.
                     Pass None to search across ALL documents.
        k:           number of top chunks to return (default from config)

    Returns:
        LangChain BaseRetriever — pass directly to RAG chains
    """
    k = k or settings.retriever_k
    store = _get_store()

    search_kwargs: dict = {"k": k}

    if document_id:
        # Filter to a single document — used for per-doc chat
        search_kwargs["filter"] = {"document_id": document_id}

    retriever = store.as_retriever(
        search_type="mmr",           # Maximal Marginal Relevance — reduces redundancy
        search_kwargs=search_kwargs,
    )
    logger.info(
        f"Retriever ready  k={k}  "
        f"filter={'document_id=' + document_id if document_id else 'all docs'}"
    )
    return retriever


def similarity_search(
    query: str,
    document_id: Optional[str] = None,
    k: int = None,
) -> List[Document]:
    """
    Raw similarity search — returns Document objects with scores ignored.
    Used by the comparison tool and risk agent.

    Args:
        query:       search string
        document_id: optional filter
        k:           number of results

    Returns:
        list of matching Document objects
    """
    k = k or settings.retriever_k
    store = _get_store()

    filter_dict = {"document_id": document_id} if document_id else None

    results = store.similarity_search(query, k=k, filter=filter_dict)
    logger.info(f"similarity_search returned {len(results)} results for: '{query[:60]}'")
    return results


def get_collection_stats() -> dict:
    """
    Return basic stats about the vector store.
    Useful for health checks and the Node backend's status endpoint.
    """
    store = _get_store()
    count = store._collection.count()
    return {
        "collection": COLLECTION_NAME,
        "total_vectors": count,
        "persist_dir": settings.chroma_persist_dir,
    }
