"""
ai/rag/retriever.py

Dedicated retriever module for LexMind.
Sits between vector_store.py and rag_chain.py — all retrieval logic lives here.

Strategies available:
  1. MMR Retriever        — default, reduces redundancy (best for chat)
  2. Similarity Retriever — raw cosine similarity (best for search/compare)
  3. Clause-Type Retriever— filtered to a specific clause type (e.g. only payment clauses)
  4. Multi-Query Retriever — generates query variants via Mistral, merges results (best recall)
  5. Contextual Retriever — for a given chunk, also fetches its neighbours (best for analysis)

All retrievers respect document_id filtering — pass None to search all docs.
All use Mistral embeddings (free) + local Chroma (free).
"""

import logging
from typing import Optional

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_chroma             import Chroma
from langchain_core.documents     import Document
from langchain_mistralai          import ChatMistralAI

from ai.config           import settings
from ai.rag.embeddings   import get_embeddings
from ai.rag.reranker       import rerank, rerank_with_threshold
from ai.rag.hybrid_retriever import get_hybrid_retriever, hybrid_search

logger = logging.getLogger(__name__)

COLLECTION_NAME = "lexmind_contracts"

# ── Internal helper ───────────────────────────────────────────────────────────

def _get_chroma() -> Chroma:
    """Return the persistent Chroma instance (shared across all retrievers)."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_persist_dir,
    )


def _get_llm() -> ChatMistralAI:
    """Mistral LLM used by multi-query retriever (free tier)."""
    return ChatMistralAI(
        model=settings.mistral_llm_model,
        mistral_api_key=settings.mistral_api_key,
        temperature=0,
        max_retries=3,
    )


def _build_filter(
    document_id: Optional[str] = None,
    clause_type: Optional[str] = None,
) -> Optional[dict]:
    """
    Build a Chroma metadata filter dict.

    Supports:
      - document_id only   → {"document_id": "..."}
      - clause_type only   → {"clause_type": "..."}   (if you store it in metadata)
      - both               → {"$and": [...]}
      - neither            → None  (no filter = search all docs)
    """
    conditions = []

    if document_id:
        conditions.append({"document_id": {"$eq": document_id}})
    if clause_type:
        conditions.append({"clause_type": {"$eq": clause_type}})

    if len(conditions) == 0:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


# ── 1. MMR Retriever ──────────────────────────────────────────────────────────

def get_mmr_retriever(
    document_id: Optional[str] = None,
    k: int = None,
    fetch_k: int = None,
    lambda_mult: float = 0.6,
):
    """
    Maximal Marginal Relevance retriever.
    Balances relevance vs diversity — avoids returning near-duplicate chunks.
    Best for: per-document chat, general Q&A.

    Args:
        document_id:  restrict to one doc (None = all docs)
        k:            number of chunks to return to the chain
        fetch_k:      candidates fetched before MMR re-ranking (should be >> k)
        lambda_mult:  0.0 = max diversity, 1.0 = pure similarity (0.6 is balanced)

    Returns:
        LangChain BaseRetriever
    """
    k       = k       or settings.retriever_k
    fetch_k = fetch_k or k * 4   # fetch 4x then re-rank

    store  = _get_chroma()
    filter_dict = _build_filter(document_id=document_id)

    search_kwargs = {
        "k":           k,
        "fetch_k":     fetch_k,
        "lambda_mult": lambda_mult,
    }
    if filter_dict:
        search_kwargs["filter"] = filter_dict

    retriever = store.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs,
    )

    logger.info(
        f"[MMR Retriever]  k={k}  fetch_k={fetch_k}  λ={lambda_mult}  "
        f"doc_filter={document_id or 'ALL'}"
    )
    return retriever


# ── 2. Similarity Retriever ───────────────────────────────────────────────────

def get_similarity_retriever(
    document_id: Optional[str] = None,
    k: int = None,
    score_threshold: float = 0.0,
):
    """
    Pure cosine similarity retriever (no diversity re-ranking).
    Returns the k most similar chunks by embedding distance.
    Best for: semantic search endpoint, finding duplicate clauses.

    Args:
        document_id:     restrict to one doc (None = all docs)
        k:               number of results
        score_threshold: minimum similarity score (0.0 = no threshold)

    Returns:
        LangChain BaseRetriever
    """
    k = k or settings.retriever_k
    store = _get_chroma()
    filter_dict = _build_filter(document_id=document_id)

    search_kwargs: dict = {"k": k}
    if filter_dict:
        search_kwargs["filter"] = filter_dict
    if score_threshold > 0.0:
        search_kwargs["score_threshold"] = score_threshold

    search_type = "similarity_score_threshold" if score_threshold > 0.0 else "similarity"

    retriever = store.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )

    logger.info(
        f"[Similarity Retriever]  k={k}  threshold={score_threshold}  "
        f"doc_filter={document_id or 'ALL'}"
    )
    return retriever


# ── 3. Clause-Type Filtered Retriever ────────────────────────────────────────

def get_clause_type_retriever(
    clause_type: str,
    document_id: Optional[str] = None,
    k: int = None,
):
    """
    Retriever filtered to a specific clause type.
    Best for: "show me all payment clauses across all contracts"
              or risk agent fetching only liability clauses.

    Args:
        clause_type:  e.g. "payment", "termination", "liability"
                      must match values in ClauseType enum
        document_id:  optional — also restrict to one document
        k:            number of results

    Returns:
        LangChain BaseRetriever
    """
    k = k or settings.retriever_k
    store = _get_chroma()
    filter_dict = _build_filter(document_id=document_id, clause_type=clause_type)

    search_kwargs: dict = {"k": k}
    if filter_dict:
        search_kwargs["filter"] = filter_dict

    retriever = store.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs,
    )

    logger.info(
        f"[ClauseType Retriever]  clause_type={clause_type}  k={k}  "
        f"doc_filter={document_id or 'ALL'}"
    )
    return retriever


# ── 4. Multi-Query Retriever ──────────────────────────────────────────────────

def get_multi_query_retriever(
    document_id: Optional[str] = None,
    k: int = None,
):
    """
    Multi-Query Retriever — uses Mistral to generate 3 query variants,
    retrieves results for each, then deduplicates and merges.

    Why: Legal questions are often phrased ambiguously. Generating variants
    ("termination clause", "contract cancellation terms", "end of agreement")
    dramatically improves recall.

    Cost: 1 extra Mistral call per retrieval (still free tier).
    Best for: the /analyze pipeline and /chat where recall matters most.

    Args:
        document_id:  restrict to one doc (None = all docs)
        k:            chunks per query variant (total results = k * num_variants)

    Returns:
        LangChain MultiQueryRetriever
    """
    k = k or settings.retriever_k

    # Base retriever (similarity, since MMR is applied per-variant already)
    base_retriever = get_similarity_retriever(document_id=document_id, k=k)

    llm = _get_llm()

    retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
        # Custom prompt instructs Mistral to generate legal query variants
        # Uses LangChain's default MultiQueryRetriever prompt (works well out of box)
    )

    logger.info(
        f"[MultiQuery Retriever]  base_k={k}  "
        f"doc_filter={document_id or 'ALL'}"
    )
    return retriever


# ── 5. Contextual Retriever (window retrieval) ────────────────────────────────

def get_contextual_retriever(
    document_id: Optional[str] = None,
    k: int = None,
    window_size: int = 1,
):
    """
    Contextual / sliding-window retriever.
    For each matched chunk, also fetches its immediate neighbours
    (chunk_index - window_size  to  chunk_index + window_size).

    Why: Legal clauses often span multiple chunks. Fetching neighbours
    gives the LLM more complete clause context.

    Args:
        document_id:  restrict to one doc (required for neighbour lookup)
        k:            number of seed chunks to retrieve
        window_size:  number of neighbours on each side (1 = prev + next chunk)

    Returns:
        list of Document objects (not a LangChain retriever — returns docs directly)
        Call retrieve_with_context() instead of using this as a chain retriever.
    """
    # We return the base retriever here — contextual expansion happens in
    # retrieve_with_context() below, since Chroma doesn't natively support
    # neighbour lookups (we do it manually via chunk_index metadata).
    logger.info(
        f"[Contextual Retriever]  k={k or settings.retriever_k}  "
        f"window={window_size}  doc={document_id or 'ALL'}"
    )
    return get_similarity_retriever(document_id=document_id, k=k)


def retrieve_with_context(
    query: str,
    document_id: str,
    k: int = None,
    window_size: int = 1,
) -> list[Document]:
    """
    Retrieve chunks for a query AND their neighbours for richer context.
    Requires document_id (neighbour lookup only makes sense within one doc).

    Args:
        query:        search query
        document_id:  the document to search in (required)
        k:            number of seed chunks
        window_size:  neighbour window on each side

    Returns:
        deduplicated list of Document objects sorted by chunk_index
    """
    k = k or settings.retriever_k
    store = _get_chroma()
    filter_doc = _build_filter(document_id=document_id)

    # Step 1: find seed chunks
    seed_results = store.similarity_search(
        query, k=k, filter=filter_doc
    )

    if not seed_results:
        logger.info(f"[ContextualRetriever] No seed results for query: '{query[:60]}'")
        return []

    # Step 2: collect chunk indices to fetch
    seed_indices: set[int] = set()
    for doc in seed_results:
        idx = doc.metadata.get("chunk_index", 0)
        for offset in range(-window_size, window_size + 1):
            seed_indices.add(idx + offset)

    seed_indices = {i for i in seed_indices if i >= 0}  # no negative indices

    # Step 3: fetch each neighbour by chunk_index
    all_docs: dict[int, Document] = {}
    for idx in seed_indices:
        results = store.similarity_search(
            query,
            k=1,
            filter={
                "$and": [
                    {"document_id": {"$eq": document_id}},
                    {"chunk_index": {"$eq": idx}},
                ]
            },
        )
        if results:
            all_docs[idx] = results[0]

    # Step 4: sort by chunk_index for natural reading order
    sorted_docs = [all_docs[i] for i in sorted(all_docs.keys())]

    logger.info(
        f"[ContextualRetriever]  query='{query[:50]}'  "
        f"seeds={len(seed_results)}  total_with_context={len(sorted_docs)}"
    )
    return sorted_docs


# ── 6. Reranked Retriever ─────────────────────────────────────────────────────

def get_reranked_retriever(
    document_id=None,
    k: int = None,
    reranker_top_n: int = None,
    fetch_strategy: str = "similarity",
):
    """
    Two-stage retrieval: fetch more candidates than needed, then rerank.

    Stage 1 (fast): Chroma vector search fetches fetch_k candidates
    Stage 2 (precise): cross-encoder reranks and returns top reranker_top_n

    This is the highest quality retriever in LexMind.
    Use for: final answer generation, high-stakes clause lookup.

    Args:
        document_id:     optional document filter
        k:               candidates to fetch from Chroma (fetch wide)
        reranker_top_n:  final results after reranking (keep narrow)
        fetch_strategy:  "similarity" or "mmr" for the fetch stage

    Returns:
        LangChain BaseRetriever wrapping reranked results
    """
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    import pydantic

    k = k or (settings.retriever_k * 3)  # fetch 3x more than needed
    top_n = reranker_top_n or settings.reranker_top_n

    # Choose fetch strategy
    if fetch_strategy == "mmr":
        base_retriever = get_mmr_retriever(document_id=document_id, k=k)
    else:
        base_retriever = get_similarity_retriever(document_id=document_id, k=k)

    class RerankedRetriever(BaseRetriever):
        model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
        _base: object = pydantic.PrivateAttr()
        _top_n: int = pydantic.PrivateAttr()

        def __init__(self, base, top_n):
            super().__init__()
            self._base = base
            self._top_n = top_n

        def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun):
            candidates = self._base.invoke(query)
            return rerank(query, candidates, top_n=self._top_n)

        async def _aget_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun):
            candidates = await self._base.ainvoke(query)
            return rerank(query, candidates, top_n=self._top_n)

    logger.info(
        f"[Reranked Retriever]  fetch_k={k}  top_n={top_n}  "
        f"strategy={fetch_strategy}  doc={document_id or 'ALL'}"
    )
    return RerankedRetriever(base=base_retriever, top_n=top_n)


# ── Convenience factory ───────────────────────────────────────────────────────

def get_retriever(
    strategy: str = "mmr",
    document_id: Optional[str] = None,
    clause_type: Optional[str] = None,
    k: int = None,
):
    """
    Factory function — get a retriever by strategy name.
    Used by rag_chain.py and FastAPI routes.

    Args:
        strategy:     "mmr" | "similarity" | "clause_type" | "multi_query" | "reranked" | "hybrid"
        document_id:  optional document filter
        clause_type:  required when strategy="clause_type"
        k:            number of results

    Returns:
        LangChain BaseRetriever
    """
    strategy = strategy.lower()

    if strategy == "mmr":
        return get_mmr_retriever(document_id=document_id, k=k)

    elif strategy == "similarity":
        return get_similarity_retriever(document_id=document_id, k=k)

    elif strategy == "clause_type":
        if not clause_type:
            raise ValueError("clause_type is required when strategy='clause_type'")
        return get_clause_type_retriever(
            clause_type=clause_type,
            document_id=document_id,
            k=k,
        )

    elif strategy == "multi_query":
        return get_multi_query_retriever(document_id=document_id, k=k)

    elif strategy == "reranked":
        return get_reranked_retriever(
            document_id=document_id,
            k=k,
            reranker_top_n=settings.reranker_top_n,
        )

    elif strategy == "hybrid":
        return get_hybrid_retriever(document_id=document_id, k=k)

    else:
        logger.warning(f"Unknown strategy '{strategy}' — defaulting to MMR")
        return get_mmr_retriever(document_id=document_id, k=k)